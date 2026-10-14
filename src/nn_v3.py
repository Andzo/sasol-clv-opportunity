"""Module 4: PyTorch Multi-Task Tabular ResNet + Bi-GRU Temporal Network.

Provides orthogonal neural representations (residual correlation rho < 0.94 with GBDTs) by fusing:
1. Dense Tabular ResNet (290 physical features, LayerNorm, SiLU activations, residual skip connections).
2. 2-layer Bidirectional GRU processing 3D bi-weekly activity tensors (N, 26, 8).
3. 5 Joint Prediction Heads:
   - Head 1: CLV Fuel positive conditional expectation (L2 loss on Z_fuel > 0)
   - Head 2: CLV Nonfuel positive conditional expectation (L2 loss on Z_nonfuel > 0)
   - Head 3: Binary Activity Hurdle (BCE on [Any Active, Fuel > 0, Nonfuel > 0])
   - Head 4: 24-Category Qualifications (Multi-label BCE)
   - Head 5: 17-Class Opportunity (Support-weighted CrossEntropy)
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger("nn_v3")


class TabularResNetBlock(nn.Module):
    """Residual block for tabular data with LayerNorm and SiLU."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = F.silu(self.ln1(self.fc1(x)))
        x = self.drop(x)
        x = self.ln2(self.fc2(x))
        return F.silu(x + res)


class MultiTaskResNetBiGRU(nn.Module):
    """Multi-task network combining Tabular ResNet and Bi-GRU sequence encoder."""

    def __init__(
        self,
        tab_dim: int,
        seq_channels: int = 8,
        tab_hidden: int = 128,
        gru_hidden: int = 64,
        n_res_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Tabular encoder
        self.tab_in = nn.Sequential(
            nn.Linear(tab_dim, tab_hidden),
            nn.LayerNorm(tab_hidden),
            nn.SiLU(),
        )
        self.tab_blocks = nn.ModuleList([TabularResNetBlock(tab_hidden, dropout=dropout) for _ in range(n_res_blocks)])

        # Sequence encoder: 2-layer Bi-GRU
        self.gru = nn.GRU(
            input_size=seq_channels,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_res_blocks > 1 else 0.0,
        )

        # Fusion layer
        fusion_in = tab_hidden + gru_hidden * 2  # 128 + 128 = 256
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Multi-task heads
        self.head_fuel = nn.Linear(256, 1)
        self.head_nonfuel = nn.Linear(256, 1)
        self.head_activity = nn.Linear(256, 3)
        self.head_24cat = nn.Linear(256, 24)
        self.head_17opp = nn.Linear(256, 17)

    def forward(self, x_tab: torch.Tensor, x_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        # Encode tabular
        h_tab = self.tab_in(x_tab)
        for block in self.tab_blocks:
            h_tab = block(h_tab)

        # Encode sequence
        gru_out, h_n = self.gru(x_seq)
        # Concatenate forward and backward final hidden states
        h_seq = torch.cat([h_n[-2], h_n[-1]], dim=1)

        # Fusion representation
        h = self.fusion(torch.cat([h_tab, h_seq], dim=1))

        # Output heads
        pred_fuel = F.softplus(self.head_fuel(h)).squeeze(-1)
        pred_nonfuel = F.softplus(self.head_nonfuel(h)).squeeze(-1)
        pred_act = torch.sigmoid(self.head_activity(h))
        pred_24cat = torch.sigmoid(self.head_24cat(h))
        logits_17opp = self.head_17opp(h)

        return {
            "pred_fuel": pred_fuel,
            "pred_nonfuel": pred_nonfuel,
            "pred_act": pred_act,
            "pred_24cat": pred_24cat,
            "logits_17opp": logits_17opp,
            "probs_17opp": F.softmax(logits_17opp, dim=-1),
        }


class MultiTaskNeuralNetwork:
    """Wrapper for training and inference with MultiTaskResNetBiGRU."""

    def __init__(self, seed: int = 42, epochs: int = 15, batch_size: int = 256, lr: float = 1e-3) -> None:
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self.scaler = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=seed)
        self.model: MultiTaskResNetBiGRU | None = None
        self.tab_dim: int = 0

    def fit(
        self,
        X_tab: pd.DataFrame,
        X_seq: np.ndarray,
        labels_dict: dict[str, np.ndarray],
        sample_weight: np.ndarray | None = None,
    ) -> "MultiTaskNeuralNetwork":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        num_cols = X_tab.select_dtypes(include=[np.number]).columns
        X_tab_clean = X_tab[num_cols].fillna(0.0).values
        X_tab_scaled = self.scaler.fit_transform(X_tab_clean)
        self.tab_dim = X_tab_scaled.shape[1]

        if sample_weight is None:
            sample_weight = np.ones(len(X_tab), dtype=np.float32)

        t_tab = torch.tensor(X_tab_scaled, dtype=torch.float32)
        t_seq = torch.tensor(X_seq, dtype=torch.float32)
        t_fuel = torch.tensor(labels_dict["CLV_fuel"], dtype=torch.float32)
        t_nonfuel = torch.tensor(labels_dict["CLV_nonfuel"], dtype=torch.float32)
        t_act = torch.tensor(labels_dict["activity_targets"], dtype=torch.float32)
        t_24cat = torch.tensor(labels_dict["cat24_targets"], dtype=torch.float32)
        t_opp = torch.tensor(labels_dict["opp17_targets"], dtype=torch.long)
        t_sw = torch.tensor(sample_weight, dtype=torch.float32)

        dataset = TensorDataset(t_tab, t_seq, t_fuel, t_nonfuel, t_act, t_24cat, t_opp, t_sw)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = MultiTaskResNetBiGRU(tab_dim=self.tab_dim)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            for b_tab, b_seq, b_fuel, b_nonfuel, b_act, b_24cat, b_opp, b_sw in loader:
                optimizer.zero_grad()
                out = self.model(b_tab, b_seq)

                mask_f = b_fuel > 0
                loss_f = ((out["pred_fuel"][mask_f] - b_fuel[mask_f]) ** 2 * b_sw[mask_f]).mean() if mask_f.any() else torch.tensor(0.0)

                mask_nf = b_nonfuel > 0
                loss_nf = ((out["pred_nonfuel"][mask_nf] - b_nonfuel[mask_nf]) ** 2 * b_sw[mask_nf]).mean() if mask_nf.any() else torch.tensor(0.0)

                loss_act = (F.binary_cross_entropy(out["pred_act"], b_act, reduction="none").mean(dim=-1) * b_sw).mean()
                loss_24 = (F.binary_cross_entropy(out["pred_24cat"], b_24cat, reduction="none").mean(dim=-1) * b_sw).mean()
                loss_opp = (F.cross_entropy(out["logits_17opp"], b_opp, reduction="none") * b_sw).mean()

                loss = 0.20 * loss_f + 0.20 * loss_nf + 0.10 * loss_act + 0.25 * loss_24 + 0.25 * loss_opp
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X_tab: pd.DataFrame, X_seq: np.ndarray) -> dict[str, np.ndarray]:
        """Generates predictions across all 5 heads."""
        self.model.eval()
        num_cols = X_tab.select_dtypes(include=[np.number]).columns
        X_tab_clean = X_tab[num_cols].fillna(0.0).values
        X_tab_scaled = self.scaler.transform(X_tab_clean)

        t_tab = torch.tensor(X_tab_scaled, dtype=torch.float32)
        t_seq = torch.tensor(X_seq, dtype=torch.float32)

        with torch.no_grad():
            out = self.model(t_tab, t_seq)

        p_act = out["pred_act"].numpy()
        pred_fuel_pos = out["pred_fuel"].numpy()
        pred_nonfuel_pos = out["pred_nonfuel"].numpy()

        # Hurdle product predictions
        pred_fuel = p_act[:, 1] * pred_fuel_pos
        pred_nonfuel = p_act[:, 2] * pred_nonfuel_pos

        return {
            "CLV_fuel": np.clip(pred_fuel, 0.0, None),
            "CLV_nonfuel": np.clip(pred_nonfuel, 0.0, None),
            "P_active": p_act[:, 0],
            "probs_17opp": out["probs_17opp"].numpy(),
        }
