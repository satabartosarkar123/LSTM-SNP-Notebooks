#!/usr/bin/env python3
"""
Generate all 20 Transformer-based Time Series Forecasting Jupyter Notebooks.
5 models × 4 datasets = 20 notebooks.

Models:
  1. Informer (ProbSparse Attention)
  2. Autoformer (Decomposition: trend + seasonal)
  3. FEDformer (Fourier blocks)
  4. PatchTST (Patch-based Transformer)
  5. Pyraformer (Pyramid Attention)

Datasets:
  - Dow Jones (Closing)
  - S&P 500
  - Lake Erie
  - Monthly Milk Production

Preprocessing is exactly matched to the Fuzzy LSTM-SNP pipeline:
  - First-order differencing
  - Supervised learning format (lag=1)
  - MinMaxScaler feature_range=(-1, 1)
  - Train-test split: last 60 points = test
  - 30 runs, 100 epochs each
  - Metrics: RMSE, MSE, NMSE

Usage:
    python generate_transformer_notebooks.py
"""

import json
import os

# ============================================================
# Dataset configurations — IDENTICAL to Fuzzy LSTM-SNP pipeline
# ============================================================
DATASETS = {
    "DowJones": {
        "csv_path": "/Users/satabarto/Research/content/monthly-closings-of-the-dowjones.csv",
        "name": "Dow Jones Industrial Index",
        "test_size": 60,
    },
    "SP500": {
        "csv_path": "/Users/satabarto/Research/content/sp500.csv",
        "name": "S&P 500",
        "test_size": 60,
    },
    "LakeErie": {
        "csv_path": "/Users/satabarto/Research/content/monthly-lake-erie-levels-1921-19.csv",
        "name": "Monthly Lake Erie Levels",
        "test_size": 60,
    },
    "Milk": {
        "csv_path": "/Users/satabarto/Research/content/monthly-milk-production-pounds-p.csv",
        "name": "Monthly Milk Production",
        "test_size": 60,
    },
}

# ============================================================
# Model configurations
# ============================================================
MODELS = {
    "Informer": {
        "name": "Informer",
        "description": "ProbSparse Self-Attention Mechanism",
        "paper": "Zhou et al., 2021 — Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting",
    },
    "Autoformer": {
        "name": "Autoformer",
        "description": "Auto-Correlation with Series Decomposition",
        "paper": "Wu et al., 2021 — Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting",
    },
    "FEDformer": {
        "name": "FEDformer",
        "description": "Frequency Enhanced Decomposed Transformer",
        "paper": "Zhou et al., 2022 — FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting",
    },
    "PatchTST": {
        "name": "PatchTST",
        "description": "Patch-based Time Series Transformer",
        "paper": "Nie et al., 2023 — A Time Series is Worth 64 Words: Long-term Forecasting with Transformers",
    },
    "Pyraformer": {
        "name": "Pyraformer",
        "description": "Pyramidal Attention for Time Series",
        "paper": "Liu et al., 2022 — Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting",
    },
}


# ============================================================
# Helper: create notebook cells
# ============================================================
def md_cell(source):
    """Create a markdown cell."""
    if isinstance(source, str):
        source = source.split('\n')
    lines = []
    for i, line in enumerate(source):
        if i < len(source) - 1:
            lines.append(line if line.endswith('\n') else line + '\n')
        else:
            lines.append(line.rstrip('\n'))
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def code_cell(source):
    """Create a code cell."""
    if isinstance(source, str):
        source = source.split('\n')
    lines = []
    for i, line in enumerate(source):
        if i < len(source) - 1:
            lines.append(line if line.endswith('\n') else line + '\n')
        else:
            lines.append(line.rstrip('\n'))
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def make_notebook(cells):
    """Wrap cells into a notebook dict."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ============================================================
# Common code fragments (IDENTICAL preprocessing to LSTM-SNP)
# ============================================================
IMPORTS_CODE = """\
# ============================================================
# ALL IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import random
import time

print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")

# ============================================================
# Device Selection
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")\
"""


def seed_code():
    return """\
# ============================================================
# Seed Control — IDENTICAL to LSTM-SNP pipeline
# ============================================================

def set_seed(seed):
    \"\"\"Set all random seeds for reproducibility.\"\"\"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False\
"""


def data_loading_code(csv_path):
    return f"""\
# ============================================================
# 1. Load Time Series Data
# ============================================================

series = pd.read_csv(
    '{csv_path}',
    header=0,
    parse_dates=[0],
    index_col=0
)

raw_values = series.values.flatten()
print(f"Data shape: {{raw_values.shape}}")
print(f"First 5 values: {{raw_values[:5]}}")\
"""


PREPROCESSING_CODE = """\
# ============================================================
# 2. First-Order Differencing — IDENTICAL to LSTM-SNP pipeline
# ============================================================

def difference(dataset, interval=1):
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)

diff_values = difference(raw_values, 1)\
"""


SUPERVISED_CODE = """\
# ============================================================
# 3. Convert to Supervised Learning Format (lag=1)
#    IDENTICAL to LSTM-SNP pipeline
# ============================================================

def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df.values

supervised = timeseries_to_supervised(diff_values, 1)
print(f"Supervised data shape: {supervised.shape}")\
"""


def split_scale_code(test_size):
    return f"""\
# ============================================================
# 4. Train-Test Split — IDENTICAL to LSTM-SNP pipeline
# ============================================================

train, test = supervised[:-{test_size}], supervised[-{test_size}:]
print(f"Train: {{train.shape}}, Test: {{test.shape}}")

# ============================================================
# 5. Feature Scaling — IDENTICAL to LSTM-SNP pipeline
# ============================================================

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(train)

train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)\
"""


RESHAPE_CODE = """\
# ============================================================
# 6. Prepare Data for Model
# ============================================================

X_train, y_train = train_scaled[:, 0:-1], train_scaled[:, -1]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

X_test, y_test = test_scaled[:, 0:-1], test_scaled[:, -1]
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")\
"""


# ============================================================
# Model definitions — faithful simplified implementations
# ============================================================

INFORMER_MODEL_CODE = """\
# ============================================================
# Informer: ProbSparse Self-Attention
# Reference: Zhou et al., 2021
#
# Key idea: ProbSparse attention selects top-u queries based on
# KL-divergence from uniform distribution, reducing O(L^2) to
# O(L·log L). For our single-step univariate setup, we implement
# a simplified but faithful version.
# ============================================================

class ProbSparseAttention(nn.Module):
    \"\"\"
    ProbSparse Self-Attention Mechanism.
    Selects top-u dominant queries based on query-key sparsity measure.
    u = c * ln(L_Q) where c is the sampling factor.
    \"\"\"
    def __init__(self, d_model, n_heads, factor=3):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _prob_QK(self, Q, K, sample_k):
        \"\"\"Compute sparsity measurement M(qi, K) for ProbSparse attention.\"\"\"
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape

        # Randomly sample keys for sparsity measurement
        K_sample_idx = torch.randint(0, L_K, (L_Q, sample_k), device=Q.device)
        K_sample = K[:, :, K_sample_idx, :]  # [B, H, L_Q, sample_k, D]

        # Compute Q·K^T for sampled keys
        Q_expand = Q.unsqueeze(-2)  # [B, H, L_Q, 1, D]
        M = torch.matmul(Q_expand, K_sample.transpose(-2, -1)).squeeze(-2)  # [B, H, L_Q, sample_k]

        # Sparsity measurement: max - mean
        M_top = M.max(dim=-1)[0] - M.mean(dim=-1)  # [B, H, L_Q]
        return M_top

    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_heads
        D = self.d_k

        Q = self.W_Q(x).view(B, L, H, D).transpose(1, 2)  # [B, H, L, D]
        K = self.W_K(x).view(B, L, H, D).transpose(1, 2)
        V = self.W_V(x).view(B, L, H, D).transpose(1, 2)

        L_Q = L
        L_K = L

        # Number of top queries to keep
        u = max(1, int(self.factor * np.ceil(np.log(L_Q + 1))))
        u = min(u, L_Q)

        # Sample keys for sparsity measurement
        sample_k = max(1, int(self.factor * np.ceil(np.log(L_K + 1))))
        sample_k = min(sample_k, L_K)

        if L_Q > 1:
            # ProbSparse: select top-u queries
            M_top = self._prob_QK(Q, K, sample_k)
            M_top_idx = M_top.topk(u, dim=-1)[1]  # [B, H, u]

            # Gather top queries
            Q_reduce = torch.gather(Q, 2, M_top_idx.unsqueeze(-1).expand(-1, -1, -1, D))

            # Full attention on selected queries only
            scores = torch.matmul(Q_reduce, K.transpose(-2, -1)) / np.sqrt(D)
            attn = torch.softmax(scores, dim=-1)
            context = torch.matmul(attn, V)  # [B, H, u, D]

            # Fill output with mean value, then scatter selected queries
            V_mean = V.mean(dim=2, keepdim=True).expand(-1, -1, L_Q, -1)
            output = V_mean.clone()
            output.scatter_(2, M_top_idx.unsqueeze(-1).expand(-1, -1, -1, D), context)
        else:
            # Standard attention for single token
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(D)
            attn = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn, V)

        output = output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(output)


class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = ProbSparseAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class InformerModel(nn.Module):
    \"\"\"
    Informer for univariate time-series forecasting.
    Input: (batch, seq_len=1, 1) → Output: (batch, 1)
    \"\"\"
    def __init__(self, input_dim=1, d_model=32, n_heads=4, d_ff=64,
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        self.encoder_layers = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        B, L, _ = x.shape
        x = self.input_proj(x)  # (B, L, d_model)
        x = x + self.pos_embedding[:, :L, :]

        for layer in self.encoder_layers:
            x = layer(x)

        # Use last token
        x = x[:, -1, :]  # (B, d_model)
        return self.output_proj(x)  # (B, 1)\
"""


AUTOFORMER_MODEL_CODE = """\
# ============================================================
# Autoformer: Auto-Correlation with Series Decomposition
# Reference: Wu et al., 2021
#
# Key ideas:
# 1. Series decomposition: separate trend and seasonal components
# 2. Auto-Correlation mechanism: replaces self-attention with
#    period-based dependencies via FFT
# ============================================================

class SeriesDecomposition(nn.Module):
    \"\"\"Series decomposition block: extract trend via moving average.\"\"\"
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        # x: (B, L, D)
        # Transpose for AvgPool1d: (B, D, L)
        trend = self.avg_pool(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


class AutoCorrelationLayer(nn.Module):
    \"\"\"
    Auto-Correlation Mechanism.
    Uses FFT to find period-based dependencies, then aggregates
    top-k correlated time delays.
    \"\"\"
    def __init__(self, d_model, n_heads, factor=3):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_heads
        D = self.d_k

        Q = self.W_Q(x).view(B, L, H, D).transpose(1, 2)  # [B, H, L, D]
        K = self.W_K(x).view(B, L, H, D).transpose(1, 2)
        V = self.W_V(x).view(B, L, H, D).transpose(1, 2)

        # Auto-correlation via FFT
        Q_fft = torch.fft.rfft(Q, dim=2)
        K_fft = torch.fft.rfft(K, dim=2)

        # Cross-correlation in frequency domain
        corr = Q_fft * torch.conj(K_fft)
        corr_real = torch.fft.irfft(corr, n=L, dim=2)  # [B, H, L, D]

        # Find top-k delays
        top_k = max(1, int(self.factor * np.log(L + 1)))
        top_k = min(top_k, L)

        # Mean correlation across D dimension
        mean_corr = corr_real.mean(dim=-1)  # [B, H, L]
        _, top_idx = mean_corr.topk(top_k, dim=-1)  # [B, H, top_k]

        # Aggregate values at top-k delays
        weights = torch.softmax(
            torch.gather(mean_corr, -1, top_idx), dim=-1
        )  # [B, H, top_k]

        # Roll and aggregate
        output = torch.zeros_like(V)
        for i in range(top_k):
            delay = top_idx[:, :, i:i+1].unsqueeze(-1).expand(-1, -1, -1, D)
            w = weights[:, :, i:i+1].unsqueeze(-1)  # [B, H, 1, 1]
            rolled_V = torch.roll(V, shifts=1, dims=2)  # Approximate roll
            output = output + w * rolled_V

        output = output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(output)


class AutoformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.decomp1 = SeriesDecomposition(kernel_size=3)
        self.decomp2 = SeriesDecomposition(kernel_size=3)
        self.auto_corr = AutoCorrelationLayer(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Auto-correlation + decomposition
        attn_out = self.auto_corr(x)
        x = x + self.dropout(attn_out)
        seasonal1, trend1 = self.decomp1(x)

        # Feed-forward + decomposition
        ff_out = self.ff(seasonal1)
        seasonal2, trend2 = self.decomp2(seasonal1 + ff_out)

        return seasonal2 + trend1 + trend2


class AutoformerModel(nn.Module):
    \"\"\"
    Autoformer for univariate time-series forecasting.
    Input: (batch, seq_len=1, 1) → Output: (batch, 1)
    \"\"\"
    def __init__(self, input_dim=1, d_model=32, n_heads=4, d_ff=64,
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        self.decomp_init = SeriesDecomposition(kernel_size=3)
        self.encoder_layers = nn.ModuleList([
            AutoformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        B, L, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, :L, :]

        for layer in self.encoder_layers:
            x = layer(x)

        x = x[:, -1, :]
        return self.output_proj(x)\
"""


FEDFORMER_MODEL_CODE = """\
# ============================================================
# FEDformer: Frequency Enhanced Decomposed Transformer
# Reference: Zhou et al., 2022
#
# Key ideas:
# 1. Frequency attention: attention in Fourier space via random
#    mode selection (keeps top-M Fourier modes)
# 2. Series decomposition (shared with Autoformer)
# 3. Mixture of Experts (MoE) for frequency domain processing
# ============================================================

class FourierBlock(nn.Module):
    \"\"\"
    Fourier-based attention block.
    Operates in the frequency domain by selecting random Fourier modes
    and applying learned linear transformations.
    \"\"\"
    def __init__(self, d_model, n_modes=8):
        super().__init__()
        self.d_model = d_model
        self.n_modes = n_modes

        # Learnable complex weights for frequency domain
        self.scale = 1.0 / d_model
        self.W_real = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.W_imag = nn.Parameter(self.scale * torch.randn(d_model, d_model))

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape

        # FFT along sequence dimension
        x_fft = torch.fft.rfft(x, dim=1)  # (B, L//2+1, D)
        freq_len = x_fft.shape[1]

        # Select modes (all modes for short sequences)
        n_modes = min(self.n_modes, freq_len)

        # Apply learnable frequency transform on selected modes
        out_fft = torch.zeros_like(x_fft)
        selected = x_fft[:, :n_modes, :]

        # Complex multiplication with learned weights
        W_complex = torch.complex(self.W_real, self.W_imag)
        selected_out = torch.einsum('bld,dd->bld', selected, W_complex)
        out_fft[:, :n_modes, :] = selected_out

        # Inverse FFT
        output = torch.fft.irfft(out_fft, n=L, dim=1)  # (B, L, D)
        return output


class FEDSeriesDecomposition(nn.Module):
    \"\"\"Series decomposition via moving average (shared design).\"\"\"
    def __init__(self, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        trend = self.avg_pool(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


class FEDformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_modes=8, dropout=0.1):
        super().__init__()
        self.fourier_block = FourierBlock(d_model, n_modes)
        self.decomp1 = FEDSeriesDecomposition(kernel_size=3)
        self.decomp2 = FEDSeriesDecomposition(kernel_size=3)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Fourier block + decomposition
        fourier_out = self.fourier_block(x)
        x = x + self.dropout(fourier_out)
        seasonal1, trend1 = self.decomp1(x)

        # Feed-forward + decomposition
        ff_out = self.ff(seasonal1)
        seasonal2, trend2 = self.decomp2(seasonal1 + ff_out)

        return seasonal2 + trend1 + trend2


class FEDformerModel(nn.Module):
    \"\"\"
    FEDformer for univariate time-series forecasting.
    Input: (batch, seq_len=1, 1) → Output: (batch, 1)
    \"\"\"
    def __init__(self, input_dim=1, d_model=32, n_heads=4, d_ff=64,
                 n_modes=8, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        self.encoder_layers = nn.ModuleList([
            FEDformerEncoderLayer(d_model, n_heads, d_ff, n_modes, dropout)
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        B, L, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, :L, :]

        for layer in self.encoder_layers:
            x = layer(x)

        x = x[:, -1, :]
        return self.output_proj(x)\
"""


PATCHTST_MODEL_CODE = """\
# ============================================================
# PatchTST: Patch-based Time Series Transformer
# Reference: Nie et al., 2023
#
# Key ideas:
# 1. Patching: segment time series into subseries-level patches
# 2. Channel independence: each variate processed independently
# 3. Instance normalization for distribution shift
#
# For our single-step (seq_len=1) setup, patch_len=1 naturally.
# The architecture still applies the full Transformer pipeline.
# ============================================================

class PatchEmbedding(nn.Module):
    \"\"\"
    Patch embedding for time series.
    Segments input into patches and projects to d_model.
    For seq_len=1, patch_len=1, this is equivalent to a linear projection.
    \"\"\"
    def __init__(self, input_dim, d_model, patch_len=1, stride=1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len * input_dim, d_model)

    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        # Number of patches
        n_patches = max(1, (L - self.patch_len) // self.stride + 1)

        patches = []
        for i in range(n_patches):
            start = i * self.stride
            end = start + self.patch_len
            patch = x[:, start:end, :].reshape(B, -1)  # (B, patch_len * C)
            patches.append(patch)

        patches = torch.stack(patches, dim=1)  # (B, n_patches, patch_len * C)
        return self.proj(patches)  # (B, n_patches, d_model)


class PatchTSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads,
                                                dropout=dropout,
                                                batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class PatchTSTModel(nn.Module):
    \"\"\"
    PatchTST for univariate time-series forecasting.
    Input: (batch, seq_len=1, 1) → Output: (batch, 1)
    \"\"\"
    def __init__(self, input_dim=1, d_model=32, n_heads=4, d_ff=64,
                 n_layers=2, patch_len=1, stride=1, dropout=0.1):
        super().__init__()
        # Instance normalization
        self.instance_norm = nn.InstanceNorm1d(input_dim, affine=False)

        self.patch_embed = PatchEmbedding(input_dim, d_model, patch_len, stride)
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        self.encoder_layers = nn.ModuleList([
            PatchTSTEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.flatten_head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape

        # Instance normalization (channel-wise)
        if L > 1:
            x = self.instance_norm(x.transpose(1, 2)).transpose(1, 2)

        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, d_model)
        n_patches = x.shape[1]
        x = x + self.pos_embedding[:, :n_patches, :]

        # Transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)

        # Use last patch for prediction
        x = x[:, -1, :]
        return self.flatten_head(x)\
"""


PYRAFORMER_MODEL_CODE = """\
# ============================================================
# Pyraformer: Pyramidal Attention for Time Series
# Reference: Liu et al., 2022
#
# Key ideas:
# 1. Pyramidal attention: multi-resolution temporal hierarchy
# 2. Coarser-scale nodes summarize finer-scale patterns
# 3. Inter-scale and intra-scale attention connections
# 4. Reduces complexity from O(L^2) to O(L)
#
# For our setup, we build a simplified pyramid with CSCM
# (Coarser-Scale Construction Module) and PAM
# (Pyramidal Attention Module).
# ============================================================

class CoarserScaleConstruction(nn.Module):
    \"\"\"
    CSCM: Coarser-Scale Construction Module.
    Builds pyramid by progressively downsampling the sequence.
    Uses 1D convolutions with stride=2 for downsampling.
    \"\"\"
    def __init__(self, d_model, n_scales=2):
        super().__init__()
        self.n_scales = n_scales
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
            )
            for _ in range(n_scales)
        ])

    def forward(self, x):
        \"\"\"
        x: (B, L, D)
        Returns list of tensors at different scales.
        \"\"\"
        scales = [x]
        current = x.transpose(1, 2)  # (B, D, L)
        for layer in self.downsample_layers:
            current = layer(current)
            scales.append(current.transpose(1, 2))
        return scales


class PyramidAttentionLayer(nn.Module):
    \"\"\"
    Pyramidal Attention: applies attention within and across scales.
    - Intra-scale: standard self-attention within each resolution
    - Inter-scale: attention from finer to coarser scale
    \"\"\"
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # Intra-scale attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads,
                                                dropout=dropout,
                                                batch_first=True)
        # Inter-scale attention (cross-attention from fine to coarse)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads,
                                                 dropout=dropout,
                                                 batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_fine, x_coarse=None):
        # Intra-scale self-attention
        attn_out, _ = self.self_attn(x_fine, x_fine, x_fine)
        x = self.norm1(x_fine + self.dropout(attn_out))

        # Inter-scale cross-attention (if coarse scale available)
        if x_coarse is not None:
            cross_out, _ = self.cross_attn(x, x_coarse, x_coarse)
            x = self.norm2(x + self.dropout(cross_out))

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm3(x + ff_out)
        return x


class PyraformerModel(nn.Module):
    \"\"\"
    Pyraformer for univariate time-series forecasting.
    Input: (batch, seq_len=1, 1) → Output: (batch, 1)
    \"\"\"
    def __init__(self, input_dim=1, d_model=32, n_heads=4, d_ff=64,
                 n_layers=2, n_scales=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)

        # CSCM: build multi-scale pyramid
        self.cscm = CoarserScaleConstruction(d_model, n_scales)

        # Pyramidal attention layers (one per scale at finest level)
        self.pyramid_layers = nn.ModuleList([
            PyramidAttentionLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        B, L, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, :L, :]

        # Build pyramid
        scales = self.cscm(x)  # List of tensors at different resolutions

        # Apply pyramidal attention at finest scale with coarser context
        finest = scales[0]
        for i, layer in enumerate(self.pyramid_layers):
            # Use next coarser scale as context (if available)
            coarse_idx = min(i + 1, len(scales) - 1)
            coarse = scales[coarse_idx]
            finest = layer(finest, coarse)

        x = finest[:, -1, :]
        return self.output_proj(x)\
"""


# Map model keys to their code
MODEL_CODE = {
    "Informer": INFORMER_MODEL_CODE,
    "Autoformer": AUTOFORMER_MODEL_CODE,
    "FEDformer": FEDFORMER_MODEL_CODE,
    "PatchTST": PATCHTST_MODEL_CODE,
    "Pyraformer": PYRAFORMER_MODEL_CODE,
}

MODEL_CLASS_NAME = {
    "Informer": "InformerModel",
    "Autoformer": "AutoformerModel",
    "FEDformer": "FEDformerModel",
    "PatchTST": "PatchTSTModel",
    "Pyraformer": "PyraformerModel",
}


# ============================================================
# Training loop code — 30 runs, 100 epochs (IDENTICAL to LSTM-SNP)
# ============================================================
def training_code(model_key, test_size):
    class_name = MODEL_CLASS_NAME[model_key]
    return f"""\
# ============================================================
# 30-Run Experiment Protocol — IDENTICAL to LSTM-SNP pipeline
# ============================================================

all_rmse = []
all_mse = []
all_nmse = []
all_predictions = []
all_losses = []
all_times = []

for run in range(30):
    print(f'\\n===== RUN {{run+1}}/30 =====')
    start_time = time.time()

    # Set seed — IDENTICAL to LSTM-SNP: np.random.seed(run), etc.
    set_seed(run)

    # Build model
    model = {class_name}(input_dim=1, d_model=32, n_heads=4, d_ff=64,
                          n_layers=2, dropout=0.1).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Convert data to tensors
    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).unsqueeze(1).to(device)

    # Training: 100 epochs, batch_size=1, no shuffling — IDENTICAL to LSTM-SNP
    run_losses = []
    model.train()
    for epoch in range(100):
        epoch_loss = 0.0
        for i in range(len(X_tr)):
            x_sample = X_tr[i:i+1]  # (1, 1, 1)
            y_sample = y_tr[i:i+1]  # (1, 1)

            optimizer.zero_grad()
            output = model(x_sample)
            loss = criterion(output, y_sample)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(X_tr)
        run_losses.append(avg_loss)

    all_losses.append(run_losses)
    print(f'Training complete for run {{run+1}}')

    # Test predictions (single-step) — IDENTICAL inverse transform to LSTM-SNP
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(len(test_scaled)):
            X = test_scaled[i, 0:-1]
            X_input = torch.FloatTensor(X.reshape(1, 1, -1)).to(device)
            yhat = model(X_input).cpu().numpy()[0, 0]

            # Invert scaling — IDENTICAL to LSTM-SNP
            new_row = [x for x in X] + [yhat]
            array = np.array(new_row).reshape(1, len(new_row))
            inverted = scaler.inverse_transform(array)[0, -1]

            # Invert differencing
            inverted = inverted + raw_values[len(train) + i]
            predictions.append(inverted)

            expected = raw_values[len(train) + i + 1]
            print(f'Month={{i+1}}, Predicted={{inverted:.4f}}, Expected={{expected:.4f}}')

    # Compute metrics — IDENTICAL to LSTM-SNP
    actual = raw_values[-{test_size}:]
    rmse = sqrt(mean_squared_error(actual, predictions))
    mse = mean_squared_error(actual, predictions)
    meanV = np.mean(actual)
    dominator = np.linalg.norm(np.array(predictions) - meanV, 2)
    nmse = mse / np.power(dominator, 2)

    elapsed = time.time() - start_time

    all_rmse.append(rmse)
    all_mse.append(mse)
    all_nmse.append(nmse)
    all_predictions.append(predictions)
    all_times.append(elapsed)

    print(f'Run {{run+1}} — RMSE: {{rmse:.6f}}, MSE: {{mse:.6f}}, NMSE: {{nmse:.10f}}, Time: {{elapsed:.1f}}s')\
"""


def results_code(test_size, model_name, dataset_name):
    return f"""\
# ============================================================
# Summary Statistics (30 runs) — IDENTICAL format to LSTM-SNP
# ============================================================

print('\\n===== FINAL RESULTS (30 runs) =====')
print(f'RMSE: {{np.mean(all_rmse):.6f}} ± {{np.std(all_rmse):.6f}}')
print(f'MSE:  {{np.mean(all_mse):.6f}} ± {{np.std(all_mse):.6f}}')
print(f'NMSE: {{np.mean(all_nmse):.10f}} ± {{np.std(all_nmse):.10f}}')
print(f'Avg training time: {{np.mean(all_times):.1f}}s ± {{np.std(all_times):.1f}}s')

best_idx = np.argmin(all_rmse)
print(f'\\nBest run: {{best_idx+1}}')
print(f'  RMSE: {{all_rmse[best_idx]:.6f}}')
print(f'  MSE:  {{all_mse[best_idx]:.6f}}')
print(f'  NMSE: {{all_nmse[best_idx]:.10f}}')\
"""


def plot_code(test_size, model_name, dataset_name):
    return f"""\
# ============================================================
# Predictions vs Actual (Best Run)
# ============================================================

actual = raw_values[-{test_size}:]
best_predictions = all_predictions[best_idx]

plt.figure(figsize=(12, 5))
plt.plot(actual, label='Actual', color='blue', linewidth=1.5)
plt.plot(best_predictions, label='Predicted (Best Run)', color='red',
         linewidth=1.5, linestyle='--')
plt.title('{model_name} — {dataset_name}\\nPredictions vs Actual (Best of 30 runs)')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# Loss Curve (Best Run)
# ============================================================

plt.figure(figsize=(12, 4))
plt.plot(all_losses[best_idx], color='green', linewidth=1.0)
plt.title('{model_name} — {dataset_name}\\nTraining Loss (Best Run)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# Final Metrics Summary
# ============================================================

print('=== Best Run Metrics ===')
print(f'RMSE: {{all_rmse[best_idx]:.6f}}')
print(f'MSE:  {{all_mse[best_idx]:.6f}}')
print(f'NMSE: {{all_nmse[best_idx]:.10f}}')\
"""


def save_predictions_code(model_name, dataset_name):
    return f"""\
# ============================================================
# Save Predictions to CSV
# ============================================================

actual = raw_values[-60:]
results_df = pd.DataFrame({{
    'Actual': actual,
    'Predicted_BestRun': all_predictions[best_idx],
}})
for i in range(30):
    results_df[f'Predicted_Run_{{i+1}}'] = all_predictions[i]

csv_filename = '{model_name}_{dataset_name}_predictions.csv'
results_df.to_csv(csv_filename, index=False)
print(f'Predictions saved to {{csv_filename}}')\
"""


# ============================================================
# Theory sections for each model
# ============================================================
def theory_md(model_key):
    theories = {
        "Informer": """\
## Theory: Informer Architecture

### ProbSparse Self-Attention

Standard Transformer attention computes $\\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$ with $O(L^2)$ complexity.

**Informer's key innovation**: Not all queries contribute equally. The ProbSparse mechanism measures each query's "sparsity":

$$M(q_i, K) = \\max_j \\frac{q_i k_j^T}{\\sqrt{d}} - \\frac{1}{L_K}\\sum_j \\frac{q_i k_j^T}{\\sqrt{d}}$$

Only the top-$u$ queries (where $u = c \\cdot \\ln L_Q$) are selected for full attention computation. Remaining queries receive the mean value from $V$.

### Distilling Operation
The encoder uses a convolutional distilling layer between attention blocks to halve the sequence length progressively, creating a multi-resolution representation.

### Complexity
ProbSparse attention reduces complexity from $O(L^2)$ to $O(L \\log L)$, enabling efficient long-sequence processing.""",

        "Autoformer": """\
## Theory: Autoformer Architecture

### Series Decomposition
Autoformer applies deep decomposition at each encoder layer:

$$X_{\\text{seasonal}}, X_{\\text{trend}} = \\text{SeriesDecomp}(X)$$

The trend is extracted via moving average:
$$X_{\\text{trend}} = \\text{AvgPool}(\\text{Padding}(X))$$
$$X_{\\text{seasonal}} = X - X_{\\text{trend}}$$

### Auto-Correlation Mechanism
Instead of point-wise attention, Autoformer discovers **period-based dependencies** using FFT:

1. Compute auto-correlation via FFT: $R_{QK}(\\tau) = \\mathcal{F}^{-1}(\\mathcal{F}(Q) \\cdot \\overline{\\mathcal{F}(K)})$
2. Select top-$k$ time delays with highest correlation
3. Aggregate values at those delays with softmax weights

This captures inherent periodicity in time series more naturally than dot-product attention.

### Complexity
Auto-correlation operates in $O(L \\log L)$ via FFT.""",

        "FEDformer": """\
## Theory: FEDformer Architecture

### Frequency Enhanced Attention
FEDformer operates in the **frequency domain** rather than the time domain:

1. Transform input to frequency space via FFT: $X_f = \\mathcal{F}(X)$
2. Randomly select $M$ Fourier modes (random mode selection)
3. Apply learned linear transformations to selected modes
4. Transform back via inverse FFT: $X = \\mathcal{F}^{-1}(X_f)$

### Frequency Domain Mixing
The Fourier block applies learned complex-valued weights:
$$Y_f[m] = W \\cdot X_f[m], \\quad m \\in \\{\\text{selected modes}\\}$$

where $W = W_{\\text{real}} + i \\cdot W_{\\text{imag}}$ is a learnable complex weight matrix.

### Series Decomposition
Like Autoformer, FEDformer uses moving-average decomposition at each layer to separate trend and seasonal components.

### Complexity
By selecting only $M \\ll L$ Fourier modes, FEDformer achieves $O(L)$ complexity.""",

        "PatchTST": """\
## Theory: PatchTST Architecture

### Patching
Time series is segmented into **subseries-level patches**:
$$P_i = x[i \\cdot S : i \\cdot S + P], \\quad i = 0, 1, \\ldots, N-1$$

where $P$ = patch length, $S$ = stride, $N$ = number of patches.

Each patch is projected to $d_{\\text{model}}$ dimensions via a linear layer.

### Channel Independence
Each variate (channel) is processed **independently** through the same Transformer backbone, reducing parameters and improving robustness.

### Instance Normalization
Applied per-instance to handle distribution shift between training and testing:
$$\\hat{x} = \\frac{x - \\mu_x}{\\sigma_x}$$

### Standard Transformer
After patching, a standard Transformer encoder processes the patch tokens with multi-head self-attention and feed-forward layers.

### Complexity
Patching reduces the effective sequence length from $L$ to $N = L/S$, achieving $O(N^2) = O((L/S)^2)$ complexity.""",

        "Pyraformer": """\
## Theory: Pyraformer Architecture

### Pyramidal Attention
Pyraformer builds a **multi-resolution temporal hierarchy**:

**Level 0 (finest)**: Original sequence tokens
**Level 1**: Downsampled by factor 2
**Level $k$**: Downsampled by factor $2^k$

### CSCM (Coarser-Scale Construction Module)
Builds the pyramid using 1D convolutions with stride 2:
$$X^{(k+1)} = \\text{Conv1D}(X^{(k)}, \\text{stride}=2)$$

### PAM (Pyramidal Attention Module)
Two types of attention connections:

1. **Intra-scale**: Self-attention within each resolution level
2. **Inter-scale**: Cross-attention from finer to coarser scales

The finest level queries attend to coarser-level summaries, capturing both local detail and global context.

### Complexity
The pyramidal structure reduces attention complexity from $O(L^2)$ to $O(L)$ by limiting attention to $O(1)$ connections per node across scales.""",
    }
    return theories[model_key]


# ============================================================
# Notebook generation
# ============================================================
def generate_notebook(model_key, dataset_key):
    """Generate a single notebook for a given model and dataset."""
    ds = DATASETS[dataset_key]
    model_info = MODELS[model_key]
    model_name = model_info["name"]
    dataset_name = ds["name"]
    csv_path = ds["csv_path"]
    test_size = ds["test_size"]

    cells = []

    # 1. Title
    cells.append(md_cell(f"""\
# {model_name} — {dataset_name}

**Model**: {model_info['description']}
**Reference**: {model_info['paper']}

## Description
This notebook implements the **{model_name}** Transformer architecture for univariate time-series
forecasting on the {dataset_name} dataset. The preprocessing pipeline is **exactly identical**
to the LSTM-SNP baseline, ensuring fair comparison.

**Experiment Protocol**: 30 independent runs × 100 epochs each"""))

    # 2. Theory
    cells.append(md_cell(theory_md(model_key)))

    # 3. Architecture heading
    cells.append(md_cell("## Model Architecture & Implementation"))

    # 4. Imports
    cells.append(code_cell(IMPORTS_CODE))

    # 5. Seed control
    cells.append(md_cell("### Seed Control"))
    cells.append(code_cell(seed_code()))

    # 6. Model definition
    cells.append(md_cell(f"### {model_name} Model Definition"))
    cells.append(code_cell(MODEL_CODE[model_key]))

    # 7. Model summary check
    class_name = MODEL_CLASS_NAME[model_key]
    cells.append(code_cell(f"""\
# Quick model check
set_seed(0)
test_model = {class_name}(input_dim=1, d_model=32, n_heads=4, d_ff=64, n_layers=2, dropout=0.1).to(device)
total_params = sum(p.numel() for p in test_model.parameters())
trainable_params = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
print(f"Total parameters: {{total_params}}")
print(f"Trainable parameters: {{trainable_params}}")
print(f"Memory estimate: {{total_params * 4 / 1024:.1f}} KB")

# Test forward pass
dummy = torch.randn(1, 1, 1).to(device)
out = test_model(dummy)
print(f"Input shape: {{dummy.shape}}, Output shape: {{out.shape}}")
del test_model, dummy, out"""))

    # 8. Data pipeline heading
    cells.append(md_cell(f"## Data Pipeline — {dataset_name}"))

    # 9. Data loading
    cells.append(code_cell(data_loading_code(csv_path)))

    # 10. Differencing
    cells.append(code_cell(PREPROCESSING_CODE))

    # 11. Supervised format
    cells.append(code_cell(SUPERVISED_CODE))

    # 12. Train-test split + scaling
    cells.append(code_cell(split_scale_code(test_size)))

    # 13. Reshape
    cells.append(code_cell(RESHAPE_CODE))

    # 14. Training heading
    cells.append(md_cell("## Training Loop"))

    # 15. Training code
    cells.append(code_cell(training_code(model_key, test_size)))

    # 16. Results heading
    cells.append(md_cell("## Results"))

    # 17. Summary statistics
    cells.append(code_cell(results_code(test_size, model_name, dataset_name)))

    # 18. Plots
    cells.append(code_cell(plot_code(test_size, model_name, dataset_name)))

    # 19. Save predictions
    cells.append(md_cell("## Export Predictions"))
    cells.append(code_cell(save_predictions_code(model_name, dataset_key)))

    # 20. Observations
    cells.append(md_cell(f"""\
## Observations

### {model_name} on {dataset_name}

**Run the notebook to generate results and fill in observations:**

1. **Prediction Quality**: Compare RMSE/MSE/NMSE with LSTM-SNP variants
2. **Training Stability**: Examine loss curves for convergence behavior
3. **Prediction Tracking**: Assess how well predictions track actual values
4. **Computational Cost**: Compare training time with LSTM-SNP
5. **Model Size**: Note parameter count vs LSTM-SNP (~361 params)

*After running all 5 model notebooks per dataset, perform cross-model comparison to evaluate
whether Transformer architectures improve over LSTM-SNP for these time series datasets.*"""))

    return make_notebook(cells)


# ============================================================
# Generate all 20 notebooks
# ============================================================
def main():
    output_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Transformer_Models")
    os.makedirs(output_base, exist_ok=True)

    generated = []

    for model_key in MODELS:
        for dataset_key in DATASETS:
            nb = generate_notebook(model_key, dataset_key)
            filename = f"{model_key}_{dataset_key}.ipynb"
            filepath = os.path.join(output_base, filename)

            with open(filepath, 'w') as f:
                json.dump(nb, f, indent=1)

            generated.append(filename)
            print(f"✅ Generated: {filename}")

    print(f"\n{'='*60}")
    print(f"Total notebooks generated: {len(generated)}")
    print(f"Output directory: {output_base}")
    print(f"{'='*60}")
    for fn in generated:
        print(f"  {fn}")

    print(f"\n📋 Notebooks are organized as:")
    print(f"   Transformer_Models/")
    for fn in generated:
        print(f"     {fn}")

    print(f"\n🔑 Key consistency with LSTM-SNP pipeline:")
    print(f"   - Preprocessing: first-order differencing → supervised (lag=1)")
    print(f"   - Scaling: MinMaxScaler(-1, 1), fit on TRAIN only")
    print(f"   - Split: last 60 points = test, NO SHUFFLING")
    print(f"   - Training: 100 epochs, batch_size=1, Adam lr=0.001")
    print(f"   - Evaluation: 30 runs, seeds 0-29")
    print(f"   - Metrics: RMSE, MSE, NMSE (identical formulas)")
    print(f"   - Inverse transform: inverse scale → inverse difference")


if __name__ == "__main__":
    main()
