"""Model trainers and ensembling for Sasol CLV & Opportunity challenge."""
import numpy as np
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import root_mean_squared_error, f1_score


class CLVRegressionEnsemble:
    """Ensemble of LightGBM, XGBoost, and CatBoost for CLV regression."""
    def __init__(self, target_name='CLV_fuel'):
        self.target_name = target_name
        self.lgb = LGBMRegressor(
            n_estimators=450,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        self.xgb = XGBRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        self.cat = CatBoostRegressor(
            iterations=500,
            learning_rate=0.04,
            depth=6,
            random_seed=42,
            verbose=0,
            thread_count=-1
        )
        self.weights = [0.4, 0.3, 0.3]

    def fit(self, X_train, y_train):
        self.lgb.fit(X_train, y_train)
        self.xgb.fit(X_train, y_train)
        self.cat.fit(X_train, y_train)
        return self

    def predict(self, X):
        p_lgb = np.clip(self.lgb.predict(X), 0, None)
        p_xgb = np.clip(self.xgb.predict(X), 0, None)
        p_cat = np.clip(self.cat.predict(X), 0, None)
        
        preds = (
            self.weights[0] * p_lgb +
            self.weights[1] * p_xgb +
            self.weights[2] * p_cat
        )
        return np.clip(preds, 0, None)


class OpportunityClassificationEnsemble:
    """Ensemble of LightGBM, CatBoost, and XGBoost for Opportunity multiclass classification."""
    def __init__(self, num_classes=17):
        self.num_classes = num_classes
        self.lgb = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        self.cat = CatBoostClassifier(
            iterations=450,
            learning_rate=0.04,
            depth=6,
            random_seed=42,
            verbose=0,
            thread_count=-1
        )
        self.xgb = XGBClassifier(
            n_estimators=350,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        self.weights = [0.4, 0.35, 0.25]

    def fit(self, X_train, y_train):
        self.lgb.fit(X_train, y_train)
        self.cat.fit(X_train, y_train)
        self.xgb.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        prob_lgb = self.lgb.predict_proba(X)
        prob_cat = self.cat.predict_proba(X)
        prob_xgb = self.xgb.predict_proba(X)

        # Handle any possible class dimension mismatches if a rare class had 0 examples in a subset
        def ensure_shape(p):
            if p.shape[1] < self.num_classes:
                full = np.zeros((p.shape[0], self.num_classes))
                full[:, :p.shape[1]] = p
                return full
            return p

        prob_lgb = ensure_shape(prob_lgb)
        prob_cat = ensure_shape(prob_cat)
        prob_xgb = ensure_shape(prob_xgb)

        blended = (
            self.weights[0] * prob_lgb +
            self.weights[1] * prob_cat +
            self.weights[2] * prob_xgb
        )
        # Renormalize rows
        return blended / blended.sum(axis=1, keepdims=True)
