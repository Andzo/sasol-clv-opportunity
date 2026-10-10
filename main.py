"""Main end-to-end execution pipeline for Sasol CLV & Opportunity prediction."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, f1_score

from src.data_processing import (
    load_config,
    load_cleaned_transactions,
    get_snapshot_labels,
    get_test_customer_ids
)
from src.feature_engineering import (
    extract_customer_features,
    compute_historical_eligibility_mask
)
from src.models import (
    CLVRegressionEnsemble,
    OpportunityClassificationEnsemble
)
from src.post_processing import (
    apply_eligibility_mask,
    optimize_class_weights_for_weighted_f1,
    tune_zero_threshold
)

PROJECT_DIR = Path(__file__).resolve().parent
SUBMISSION_PATH = PROJECT_DIR / 'submission.csv'


def main():
    start_time = time.time()
    print("=" * 60)
    print("SASOL CLV & OPPORTUNITY PREDICTION PIPELINE")
    print("=" * 60)

    config = load_config()
    opportunity_labels = config['opportunity_labels']
    label_to_idx = {l: i for i, l in enumerate(opportunity_labels)}
    idx_to_label = {i: l for i, l in enumerate(opportunity_labels)}
    num_classes = len(opportunity_labels)

    # 1. Load Cleaned Transactions
    print("\n[Step 1/6] Loading cleaned transactions...")
    transactions = load_cleaned_transactions()
    print(f"Transactions loaded successfully ({len(transactions):,} rows).")

    # 2. Build Historical Snapshots for Training & Validation
    print("\n[Step 2/6] Building historical snapshots...")
    train_cutoffs = ['2024-09-01', '2024-12-01', '2025-03-01', '2025-06-01']
    val_cutoff = '2025-09-01'
    test_cutoff = '2025-12-01'

    # Cache directory for features
    cache_dir = PROJECT_DIR / 'data_cache'
    cache_dir.mkdir(exist_ok=True)

    def get_or_create_snapshot(cutoff_str, is_test=False):
        feat_path = cache_dir / f"feats_{cutoff_str}.parquet"
        label_path = cache_dir / f"labels_{cutoff_str}.parquet"

        if feat_path.exists():
            feats = pd.read_parquet(feat_path)
        else:
            print(f"  Extracting features for cutoff {cutoff_str}...")
            feats = extract_customer_features(transactions, cutoff_str)
            feats.to_parquet(feat_path, index=False)

        if is_test:
            return feats, None

        if label_path.exists():
            labels = pd.read_parquet(label_path)
        else:
            print(f"  Generating ground truth labels for cutoff {cutoff_str}...")
            labels = get_snapshot_labels(transactions, cutoff_str, config)
            labels.to_parquet(label_path, index=False)

        merged = pd.merge(feats, labels[['ID', 'CLV_fuel', 'CLV_nonfuel', 'Opportunity']], on='ID')
        return merged

    train_dfs = []
    for c in train_cutoffs:
        print(f"Processing training cutoff: {c}...")
        df_c = get_or_create_snapshot(c)
        train_dfs.append(df_c)
    train_df = pd.concat(train_dfs, ignore_index=True)
    print(f"Total training instances (pre-holdout): {len(train_df):,}")

    print(f"Processing validation cutoff: {val_cutoff}...")
    val_df = get_or_create_snapshot(val_cutoff)
    print(f"Holdout validation instances: {len(val_df):,}")

    feature_cols = [c for c in val_df.columns if c not in ['ID', 'CLV_fuel', 'CLV_nonfuel', 'Opportunity']]
    print(f"Feature set size: {len(feature_cols)} features")

    X_train = train_df[feature_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0).to_numpy()
    y_train_fuel = train_df['CLV_fuel'].to_numpy()
    y_train_nonfuel = train_df['CLV_nonfuel'].to_numpy()
    y_train_opp = train_df['Opportunity'].map(label_to_idx).to_numpy()

    X_val = val_df[feature_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0).to_numpy()
    y_val_fuel = val_df['CLV_fuel'].to_numpy()
    y_val_nonfuel = val_df['CLV_nonfuel'].to_numpy()
    y_val_opp = val_df['Opportunity'].map(label_to_idx).to_numpy()

    # 3. Model Training & Evaluation on Holdout Snapshot (2025-09-01)
    print("\n[Step 3/6] Training and evaluating on Holdout Snapshot (2025-09-01)...")

    # A. Fuel Regression
    print("  Training CLV_fuel ensemble...")
    fuel_model = CLVRegressionEnsemble(target_name='CLV_fuel')
    fuel_model.fit(X_train, y_train_fuel)
    val_pred_fuel_raw = fuel_model.predict(X_val)
    raw_rmse_fuel = root_mean_squared_error(y_val_fuel, val_pred_fuel_raw)
    tau_fuel = tune_zero_threshold(val_pred_fuel_raw, y_val_fuel, lower=0.0, upper=0.5)
    val_pred_fuel = np.where(val_pred_fuel_raw < tau_fuel, 0.0, val_pred_fuel_raw)
    rmse_fuel = root_mean_squared_error(y_val_fuel, val_pred_fuel)
    print(f"  Holdout CLV_fuel RMSE: raw={raw_rmse_fuel:.4f}, calibrated (tau={tau_fuel:.2f})={rmse_fuel:.4f}")

    # B. Non-fuel Regression
    print("  Training CLV_nonfuel ensemble...")
    nonfuel_model = CLVRegressionEnsemble(target_name='CLV_nonfuel')
    nonfuel_model.fit(X_train, y_train_nonfuel)
    val_pred_nonfuel_raw = nonfuel_model.predict(X_val)
    raw_rmse_nonfuel = root_mean_squared_error(y_val_nonfuel, val_pred_nonfuel_raw)
    tau_nonfuel = tune_zero_threshold(val_pred_nonfuel_raw, y_val_nonfuel, lower=0.0, upper=0.6)
    val_pred_nonfuel = np.where(val_pred_nonfuel_raw < tau_nonfuel, 0.0, val_pred_nonfuel_raw)
    rmse_nonfuel = root_mean_squared_error(y_val_nonfuel, val_pred_nonfuel)
    print(f"  Holdout CLV_nonfuel RMSE: raw={raw_rmse_nonfuel:.4f}, calibrated (tau={tau_nonfuel:.2f})={rmse_nonfuel:.4f}")

    # C. Opportunity Classification
    print("  Training Opportunity multiclass ensemble...")
    opp_model = OpportunityClassificationEnsemble(num_classes=num_classes)
    opp_model.fit(X_train, y_train_opp)
    val_probs_raw = opp_model.predict_proba(X_val)
    raw_f1 = f1_score(y_val_opp, np.argmax(val_probs_raw, axis=1), average='weighted')

    print("  Computing and applying historical eligibility mask...")
    val_mask = compute_historical_eligibility_mask(transactions, val_cutoff, val_df['ID'].tolist())
    val_probs_masked = apply_eligibility_mask(val_probs_raw, val_mask)
    masked_f1 = f1_score(y_val_opp, np.argmax(val_probs_masked, axis=1), average='weighted')
    print(f"  Holdout Opportunity Weighted F1: raw={raw_f1:.4f}, with Hard Constraint Masking={masked_f1:.4f}")

    print("  Optimizing class threshold weights for Weighted F1...")
    opt_weights = optimize_class_weights_for_weighted_f1(val_probs_masked, y_val_opp, num_classes=num_classes)
    val_probs_weighted = val_probs_masked * opt_weights
    calibrated_val_preds = np.argmax(val_probs_weighted, axis=1)
    calibrated_f1 = f1_score(y_val_opp, calibrated_val_preds, average='weighted')
    print(f"  Holdout Opportunity Weighted F1 after Prior Calibration: {calibrated_f1:.4f}")

    # Evaluate Holdout Metrics
    norm_fuel = max(0, 1.0 - (rmse_fuel / 1.8))
    norm_nonfuel = max(0, 1.0 - (rmse_nonfuel / 1.0))
    est_score = 0.3 * norm_fuel + 0.3 * norm_nonfuel + 0.4 * calibrated_f1
    print(f"\n  ---> ESTIMATED HOLDOUT SCORE: {est_score:.4f} (Fuel RMSE: {rmse_fuel:.4f}, Nonfuel RMSE: {rmse_nonfuel:.4f}, Weighted F1: {calibrated_f1:.4f})")

    # 4. Retrain Models on All Available Historical Data (including Holdout)
    print("\n[Step 4/6] Retraining final models on ALL historical cutoffs...")
    all_train_df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"Full training dataset size: {len(all_train_df):,} samples.")

    X_all = all_train_df[feature_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0).to_numpy()
    y_all_fuel = all_train_df['CLV_fuel'].to_numpy()
    y_all_nonfuel = all_train_df['CLV_nonfuel'].to_numpy()
    y_all_opp = all_train_df['Opportunity'].map(label_to_idx).to_numpy()

    final_fuel_model = CLVRegressionEnsemble(target_name='CLV_fuel')
    final_fuel_model.fit(X_all, y_all_fuel)

    final_nonfuel_model = CLVRegressionEnsemble(target_name='CLV_nonfuel')
    final_nonfuel_model.fit(X_all, y_all_nonfuel)

    final_opp_model = OpportunityClassificationEnsemble(num_classes=num_classes)
    final_opp_model.fit(X_all, y_all_opp)

    # 5. Generate Test Predictions (2025-12-01 Cutoff)
    print("\n[Step 5/6] Generating predictions for test customers (cutoff: 2025-12-01)...")
    test_ids = get_test_customer_ids()
    print(f"Number of test customers: {len(test_ids):,}")

    test_feats, _ = get_or_create_snapshot(test_cutoff, is_test=True)
    # Ensure correct ordering to match test_ids exactly
    test_feats = test_feats.set_index('ID').reindex(test_ids).reset_index()

    X_test = test_feats[feature_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0).to_numpy()

    test_pred_fuel = final_fuel_model.predict(X_test)
    test_pred_fuel = np.where(test_pred_fuel < tau_fuel, 0.0, test_pred_fuel)

    test_pred_nonfuel = final_nonfuel_model.predict(X_test)
    test_pred_nonfuel = np.where(test_pred_nonfuel < tau_nonfuel, 0.0, test_pred_nonfuel)

    test_probs = final_opp_model.predict_proba(X_test)
    test_mask = compute_historical_eligibility_mask(transactions, test_cutoff, test_ids)
    test_probs_masked = apply_eligibility_mask(test_probs, test_mask)
    test_probs_weighted = test_probs_masked * opt_weights
    test_opp_preds_int = np.argmax(test_probs_weighted, axis=1)

    test_opp_labels = [idx_to_label[i] for i in test_opp_preds_int]

    # 6. Verify and Save Submission File
    print("\n[Step 6/6] Verifying submission format and saving...")
    sub_df = pd.DataFrame({
        'ID': test_ids,
        'CLV_fuel': test_pred_fuel,
        'CLV_nonfuel': test_pred_nonfuel,
        'Opportunity': test_opp_labels
    })

    # Validation checks
    sample_sub = pd.read_csv(PROJECT_DIR / 'SampleSubmission.csv')
    assert len(sub_df) == len(sample_sub), f"Row count mismatch: {len(sub_df)} vs {len(sample_sub)}"
    assert list(sub_df.columns) == list(sample_sub.columns), f"Columns mismatch: {sub_df.columns} vs {sample_sub.columns}"
    assert (sub_df['ID'].astype(str) == sample_sub['ID'].astype(str)).all(), "ID order mismatch!"
    assert not sub_df.isnull().any().any(), "Submission contains null values!"
    assert (sub_df['CLV_fuel'] >= 0).all(), "CLV_fuel has negative values!"
    assert (sub_df['CLV_nonfuel'] >= 0).all(), "CLV_nonfuel has negative values!"
    assert set(sub_df['Opportunity']).issubset(set(opportunity_labels)), "Invalid Opportunity labels present!"

    sub_df.to_csv(SUBMISSION_PATH, index=False, float_format='%.17g')
    print(f"Submission saved to {SUBMISSION_PATH}")
    print("\nSubmission summary:")
    print(f"  Rows: {len(sub_df)}")
    print(f"  CLV_fuel: mean={sub_df['CLV_fuel'].mean():.4f}, zeros={(sub_df['CLV_fuel'] == 0).mean():.2%}")
    print(f"  CLV_nonfuel: mean={sub_df['CLV_nonfuel'].mean():.4f}, zeros={(sub_df['CLV_nonfuel'] == 0).mean():.2%}")
    print("\nOpportunity Class Distribution in Submission:")
    print(sub_df['Opportunity'].value_counts())

    elapsed = time.time() - start_time
    print(f"\nPipeline successfully completed in {elapsed / 60:.2f} minutes!")


if __name__ == '__main__':
    main()
