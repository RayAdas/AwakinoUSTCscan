import torch
from torch.utils.data import Dataset
import math

class NormalSignalDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=100, max_peaks=3):
        self.seq_len = seq_len
        self.max_peaks = max_peaks
        self.inputs = []
        self.targets = []  # list of tau tensors (variable length)

        t_axis = torch.linspace(0, 100, seq_len)
        a = 1 / math.sqrt(2 * math.pi)

        for _ in range(num_samples):
            ns = torch.randint(0, max_peaks + 1, (1,)).item()
            taus = torch.rand(ns) * 80 + 10  # [10, 90]

            signal = torch.zeros(seq_len)
            for tau in taus:
                signal += a * torch.exp(-0.5 * ((t_axis - tau) / 2) ** 2)

            self.inputs.append(signal)
            self.targets.append(taus)  # 不排序，不 padding

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

import torch
import math

class PositionalEncoding1D(torch.nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, C, T)
        T = x.size(-1)
        return x + self.pe[:T].T.unsqueeze(0)

import torch.nn as nn
import torch.nn.functional as F

class SetPredictionNet(nn.Module):
    def __init__(self, seq_len=100, num_queries=3, hidden_dim=256):
        super().__init__()
        self.num_queries = num_queries

        # CNN backbone (保留时间维度)
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )

        self.pos_enc = PositionalEncoding1D(hidden_dim, max_len=seq_len)

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim))

        # Cross-attention (queries attend to signal)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)

        # Output heads
        self.tau_head = nn.Linear(hidden_dim, 1)
        self.exist_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: (B, T)
        """
        B, T = x.shape
        x = x.unsqueeze(1)                  # (B,1,T)
        feats = self.backbone(x)             # (B,C,T)
        feats = self.pos_enc(feats)
        feats = feats.permute(0, 2, 1)       # (B,T,C)

        queries = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B,Q,C)

        # queries attend to time features
        q_out, _ = self.attn(queries, feats, feats)

        tau = torch.sigmoid(self.tau_head(q_out)).squeeze(-1)  # (B,Q) in [0,1]
        exist_logit = self.exist_head(q_out).squeeze(-1)       # (B,Q)

        return tau, exist_logit

import torch
from scipy.optimize import linear_sum_assignment

class SetPredictionLoss(nn.Module):
    def __init__(self, tau_weight=1.0, exist_weight=1.0):
        super().__init__()
        self.tau_weight = tau_weight
        self.exist_weight = exist_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, tau_pred, exist_logit, targets):
        """
        tau_pred: (B,Q) in [0,1]
        exist_logit: (B,Q)
        targets: list of tensors, each (Ni,)
        """
        B, Q = tau_pred.shape
        total_loss = 0.0

        for b in range(B):
            tgt = targets[b] / 100.0  # normalize to [0,1]
            n = len(tgt)

            if n == 0:
                # all should be "non-existent"
                exist_tgt = torch.zeros(Q, device=tau_pred.device)
                total_loss += self.exist_weight * self.bce(exist_logit[b], exist_tgt)
                continue

            # cost matrix: (Q, n)
            cost = torch.cdist(tau_pred[b].unsqueeze(1), tgt.unsqueeze(0), p=2)

            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

            # regression loss (matched)
            tau_loss = F.mse_loss(
                tau_pred[b, row_ind],
                tgt[col_ind],
                reduction="sum"
            ) / n

            # existence targets
            exist_tgt = torch.zeros(Q, device=tau_pred.device)
            exist_tgt[row_ind] = 1.0

            exist_loss = self.bce(exist_logit[b], exist_tgt)

            total_loss += self.tau_weight * tau_loss + self.exist_weight * exist_loss

        return total_loss / B

@torch.no_grad()
def infer_peaks(model, signal, thresh=0.5):
    model.eval()
    tau, exist_logit = model(signal.unsqueeze(0))
    prob = torch.sigmoid(exist_logit)[0]
    tau = tau[0] * 100.0

    keep = prob > thresh
    return tau[keep].cpu().numpy()
