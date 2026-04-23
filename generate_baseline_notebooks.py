#!/usr/bin/env python3
"""Generate SNN-LSTM, Pure LSTM, and Pure GRU notebooks for all 4 datasets."""
import json, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    'dow_jones': {'csv': 'monthly-closings-of-the-dowjones.csv', 'title': 'Dow Jones Monthly Closings'},
    'lake_erie': {'csv': 'monthly-lake-erie-levels-1921-19.csv', 'title': 'Lake Erie Monthly Levels'},
    'milk_production': {'csv': 'monthly-milk-production-pounds-p.csv', 'title': 'Monthly Milk Production'},
    'sp500': {'csv': 'sp500.csv', 'title': 'S&P 500'},
}

KERNEL = {"display_name": "SNN Transformer", "language": "python", "name": "snn_transformer"}
LANG_INFO = {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py",
             "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python",
             "pygments_lexer": "ipython3", "version": "3.11.14"}

def cell(source_str):
    lines = [l + '\n' for l in source_str.split('\n')]
    if lines and lines[-1] == '\n': lines[-1] = ''
    lines = [l for l in lines if l != '']
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}

def md_cell(source_str):
    lines = [l + '\n' for l in source_str.split('\n')]
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def pid_cell():
    return cell("# ============================================================\n# PROCESS IDENTIFICATION\n# ============================================================\nimport os\nprint(f\"Process ID (PID): {os.getpid()}\")")

def timer_start():
    return cell("# ============================================================\n# NOTEBOOK TIMER — START\n# ============================================================\nimport time as _timer_module\n_NOTEBOOK_START_TIME = _timer_module.time()\nprint(f\"Notebook execution started at: {_timer_module.strftime('%Y-%m-%d %H:%M:%S')}\")")

def timer_end():
    return cell("# ============================================================\n# NOTEBOOK TIMER — END\n# ============================================================\nimport time as _timer_module\n_NOTEBOOK_END_TIME = _timer_module.time()\n_NOTEBOOK_ELAPSED = _NOTEBOOK_END_TIME - _NOTEBOOK_START_TIME\n_hours, _rem = divmod(_NOTEBOOK_ELAPSED, 3600)\n_minutes, _seconds = divmod(_rem, 60)\nprint(f\"\\nTotal notebook execution time: {int(_hours)}h {int(_minutes)}m {_seconds:.2f}s\")\nprint(f\"Total seconds: {_NOTEBOOK_ELAPSED:.2f}\")")

def gpu_cell():
    return cell("""# ============================================================
# GPU Acceleration Settings (CUDA + Apple Metal Support)
# ============================================================
import torch
import platform

# Device priority: CUDA > Apple MPS (Metal) > CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"CUDA GPU Enabled: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
    print(f"Apple Metal GPU Enabled via MPS backend")
    print(f"  Chipset: Apple {platform.machine()} (M-series)")
    print(f"  MPS built: {torch.backends.mps.is_built()}")
else:
    device = torch.device('cpu')
    if platform.system() == 'Darwin' and platform.processor() == 'arm':
        print("Apple Silicon detected but MPS not available.")
        print("  Upgrade PyTorch: pip install --upgrade torch")
    else:
        print("No GPU found. Falling back to CPU.")
        print("  For CUDA: ensure NVIDIA drivers + CUDA toolkit are installed.")

print(f"\\nUsing device: {device}")
print(f"PyTorch version: {torch.__version__}")""")

def imports_cell():
    return cell("""# ============================================================
# ALL IMPORTS
# ============================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import random
import time
import warnings
warnings.filterwarnings('ignore')

print(f"NumPy version: {np.__version__}")""")

def seed_cell():
    return cell("""# ============================================================
# Seed Control — Reproducibility
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
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS uses torch.manual_seed for seeding
        pass""")

def preprocess_cell(csv_name, dataset_title):
    return cell(f"""# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================
# Pipeline (identical to LSTM-SNP):
#   1. Load CSV
#   2. First-order differencing
#   3. Lag-1 supervised learning format
#   4. Train/Test split: last 60 observations for test
#   5. MinMaxScaler(-1, 1)

CSV_PATH = "../content/{csv_name}"
series = pd.read_csv(CSV_PATH, header=0, parse_dates=[0], index_col=0)
raw_values = series.values.flatten()
print(f"Dataset: {dataset_title}")
print(f"Data shape: {{raw_values.shape}}")
print(f"First 5 values: {{raw_values[:5]}}")

# First-order differencing
def difference(dataset, interval=1):
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)

diff_values = difference(raw_values, 1)

# Supervised learning format (lag=1)
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df.values

supervised = timeseries_to_supervised(diff_values, 1)

# Train/test split
N_TEST = 60
train, test = supervised[:-N_TEST], supervised[-N_TEST:]

# Scale to [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(train)
train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)

# Reshape for PyTorch: (samples, 1, features)
X_train = torch.tensor(train_scaled[:, 0:-1], dtype=torch.float32).unsqueeze(1).to(device)
y_train = torch.tensor(train_scaled[:, -1], dtype=torch.float32).to(device)
X_test_np = test_scaled[:, 0:-1]
y_test_np = test_scaled[:, -1]

print(f"X_train shape: {{X_train.shape}}")
print(f"Train samples: {{len(train_scaled)}}, Test samples: {{len(test_scaled)}}")""")

def train_eval_cells():
    c1 = cell("""# ============================================================
# TRAINING FUNCTION
# ============================================================
# Protocol (identical to LSTM-SNP):
#   - Optimizer: Adam (lr=0.001)
#   - Loss: MSE
#   - Epochs: 200
#   - Batch size: 1 (sample-by-sample, no shuffle)

def train_model(model, X_train, y_train, epochs=200, lr=0.001, verbose=True):
    \"\"\"Train model following the LSTM-SNP protocol exactly.\"\"\"
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n_samples = X_train.shape[0]
    epoch_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(n_samples):
            x_i = X_train[i:i+1]
            y_i = y_train[i:i+1]
            optimizer.zero_grad()
            pred = model(x_i)
            loss = loss_fn(pred.squeeze(-1), y_i)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_samples
        epoch_losses.append(avg_loss)
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    return epoch_losses""")

    c2 = cell("""# ============================================================
# EVALUATION — Rolling One-Step-Ahead Prediction
# ============================================================
# Identical to LSTM-SNP evaluation:
#   1. Predict scaled diff value
#   2. Inverse scale to get diff value
#   3. Inverse difference: predicted_raw(t) = predicted_diff(t) + raw(t-1)
#   4. Compute metrics: RMSE, MSE, NMSE

def inverse_scale(scaler, X, predicted_diff):
    row = np.array([X, predicted_diff]).reshape(1, 2)
    inverted = scaler.inverse_transform(row)
    return inverted[0, -1]

def inverse_difference(last_obs, predicted_diff):
    return predicted_diff + last_obs

def rolling_predict(model, X_test_np, y_test_np, raw_values, scaler, n_test=60):
    \"\"\"Rolling one-step-ahead prediction on the test set.\"\"\"
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(len(X_test_np)):
            X = X_test_np[i, 0:-1] if X_test_np.ndim > 1 and X_test_np.shape[1] > 1 else X_test_np[i]
            x_i = torch.tensor(X_test_np[i:i+1, 0:-1] if X_test_np.ndim > 1 else X_test_np[i:i+1],
                               dtype=torch.float32).unsqueeze(1).to(device)
            yhat_scaled = model(x_i).cpu().numpy().flatten()[0]

            X_val = X_test_np[i, 0] if X_test_np.ndim > 1 else X_test_np[i]
            yhat_diff = inverse_scale(scaler, X_val, yhat_scaled)

            raw_idx = len(raw_values) - n_test - 1 + i
            last_obs = raw_values[raw_idx]
            yhat_raw = inverse_difference(last_obs, yhat_diff)
            predictions.append(yhat_raw)

    return predictions

def compute_metrics(actual, predictions):
    rmse = sqrt(mean_squared_error(actual, predictions))
    mse = mean_squared_error(actual, predictions)
    meanV = np.mean(actual)
    dominator = np.linalg.norm(np.array(predictions) - meanV, 2)
    nmse = mse / np.power(dominator, 2) if dominator != 0 else float('inf')
    return rmse, mse, nmse""")
    return [c1, c2]

def experiment_cell(model_name, dataset_title):
    return cell(f"""# ============================================================
# RUN EXPERIMENT — 30 Independent Runs
# ============================================================

N_RUNS = 30
N_EPOCHS = 200

all_rmse, all_mse, all_nmse = [], [], []
all_predictions = []
all_losses = []

print(f"Model: {model_name}")
print(f"Dataset: {dataset_title}")
print(f"Runs: {{N_RUNS}}, Epochs: {{N_EPOCHS}}")
print(f"Device: {{device}}")
print(f"{{\'=\' * 60}}")

for run_idx in range(N_RUNS):
    print(f"\\nRun {{run_idx+1}}/{{N_RUNS}}", end=" — ", flush=True)
    start = time.time()

    set_seed(run_idx)
    model = build_model().to(device)

    losses = train_model(model, X_train, y_train, epochs=N_EPOCHS, lr=0.001, verbose=False)
    all_losses.append(losses)

    predictions = rolling_predict(model, test_scaled, y_test_np, raw_values, scaler, n_test=N_TEST)
    actual = raw_values[-N_TEST:]
    rmse, mse, nmse = compute_metrics(actual, predictions)

    all_rmse.append(rmse)
    all_mse.append(mse)
    all_nmse.append(nmse)
    all_predictions.append(predictions)

    elapsed = time.time() - start
    print(f"RMSE: {{rmse:.6f}}, MSE: {{mse:.6f}}, NMSE: {{nmse:.10f}} ({{elapsed:.1f}}s)")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()""")

def results_cell(model_name, dataset_title):
    return cell(f"""# ============================================================
# AGGREGATE RESULTS ACROSS 30 RUNS
# ============================================================

print(f"\\n{{\'=\' * 60}}")
print(f"{model_name} — {dataset_title}")
print(f"{{\'=\' * 60}}")
print(f"RMSE: {{np.mean(all_rmse):.10f}} ± {{np.var(all_rmse):.15f}}")
print(f"MSE:  {{np.mean(all_mse):.10f}} ± {{np.var(all_mse):.15f}}")
print(f"NMSE: {{np.mean(all_nmse):.10f}} ± {{np.var(all_nmse):.15f}}")

best_idx = all_rmse.index(min(all_rmse))
print(f"\\nBest run: {{best_idx+1}}")
print(f"  RMSE: {{all_rmse[best_idx]:.15f}}")
print(f"  MSE:  {{all_mse[best_idx]:.15f}}")
print(f"  NMSE: {{all_nmse[best_idx]:.15f}}")
print(f"\\nAll RMSE values: {{all_rmse}}")""")

def plot_cell(model_name, dataset_title):
    return cell(f"""# ============================================================
# PLOTS — Best Run Prediction vs Actual
# ============================================================

actual = raw_values[-N_TEST:]
best_predictions = all_predictions[best_idx]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Prediction vs Actual
axes[0].plot(actual, 'k-o', markersize=3, label='Original data')
axes[0].plot(best_predictions, 'r+-', markersize=3, label='Predicted data')
axes[0].set_xlabel('Time', fontsize=12)
axes[0].set_ylabel('Magnitude', fontsize=12)
axes[0].legend()
axes[0].set_title(f'{model_name} — {dataset_title} (Best Run)')

# Error plot
error = abs(np.array(actual) - np.array(best_predictions))
axes[1].plot(error, 'k-o', markersize=3, label='|Actual - Predicted|')
axes[1].set_xlabel('Time', fontsize=12)
axes[1].set_ylabel('Absolute Error', fontsize=12)
axes[1].legend()
axes[1].set_title(f'{model_name} — Prediction Error (Best Run)')

plt.tight_layout()
plt.show()

# Loss convergence (best run)
plt.figure(figsize=(8, 4))
plt.plot(all_losses[best_idx], 'b-', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title(f'{model_name} — Training Loss (Best Run)')
plt.grid(True, alpha=0.3)
plt.show()""")

# ── Model Architecture Cells ────────────────────────────────────────────────

def snn_lstm_model_cell():
    return cell("""# ============================================================
# SNN-LSTM: Spiking Neural Network LSTM
# ============================================================
# Combines LSTM gating with Leaky Integrate-and-Fire (LIF) spiking
# neuron dynamics. The LIF neuron replaces standard activations
# with membrane potential accumulation, threshold-based firing,
# and surrogate gradient for backpropagation.
#
# Equations (LIF neuron):
#   U[t] = beta * U[t-1] + X[t]        (membrane potential)
#   S[t] = Heaviside(U[t] - u_th)      (spike generation)
#   H[t] = V_reset*S[t] + beta*U[t]*(1-S[t])  (output)

class SurrogateHeaviside(torch.autograd.Function):
    \"\"\"Heaviside step with surrogate gradient for backpropagation.\"\"\"
    @staticmethod
    def forward(ctx, input_tensor, alpha):
        ctx.save_for_backward(input_tensor)
        ctx.alpha = alpha
        return (input_tensor >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input_tensor,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output / (1 + alpha * input_tensor.abs()) ** 2
        return grad_input, None

def surrogate_heaviside(x, alpha=2.0):
    return SurrogateHeaviside.apply(x, alpha)


class LIFNeuron(nn.Module):
    \"\"\"Leaky Integrate-and-Fire neuron with surrogate gradient.\"\"\"
    def __init__(self, size, beta=0.9, threshold=1.0, reset=0.0):
        super().__init__()
        self.size = size
        self.beta = beta
        self.threshold = threshold
        self.reset = reset

    def forward(self, x):
        # x: (batch, seq_len, features)
        batch, seq_len, features = x.shape
        membrane = torch.zeros(batch, features, device=x.device)
        spikes_out = []

        for t in range(seq_len):
            membrane = self.beta * membrane + x[:, t, :]
            spike = surrogate_heaviside(membrane - self.threshold)
            membrane = self.reset * spike + self.beta * membrane * (1 - spike)
            spikes_out.append(spike)

        return torch.stack(spikes_out, dim=1)  # (batch, seq_len, features)


class SNN_LSTM(nn.Module):
    \"\"\"
    SNN-LSTM: LSTM with spiking neuron dynamics.
    The LSTM processes temporal dependencies, then a LIF neuron
    converts the continuous hidden states to spike trains.
    \"\"\"
    def __init__(self, input_dim=1, hidden_dim=8, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # LSTM layer for temporal processing
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # LIF spiking layer
        self.lif = LIFNeuron(hidden_dim, beta=0.9, threshold=0.5)

        # Output projection
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)       # (batch, seq_len, hidden_dim)
        spike_out = self.lif(lstm_out)   # (batch, seq_len, hidden_dim)

        # Use last timestep
        out = self.fc(spike_out[:, -1, :])  # (batch, output_dim)
        return out


def build_model():
    return SNN_LSTM(input_dim=1, hidden_dim=8, output_dim=1)

# Quick architecture check
set_seed(0)
test_model = build_model().to(device)
total_params = sum(p.numel() for p in test_model.parameters())
print(f"SNN-LSTM Parameters: {total_params}")
print(f"Memory estimate: {total_params * 4 / 1024:.1f} KB")
dummy = torch.randn(1, 1, 1).to(device)
out = test_model(dummy)
print(f"Input: {dummy.shape} → Output: {out.shape}")
del test_model, dummy, out""")

def pure_lstm_model_cell():
    return cell("""# ============================================================
# Pure LSTM — Standard 2nd Generation Neural Network
# ============================================================
# Classic Hochreiter & Schmidhuber (1997) LSTM.
# Uses standard nn.LSTM with identical hyperparameters
# to LSTM-SNP for fair comparison.
#   - Hidden units: 8
#   - Single layer
#   - No dropout

class PureLSTM(nn.Module):
    \"\"\"Standard LSTM for time series forecasting.\"\"\"
    def __init__(self, input_dim=1, hidden_dim=8, output_dim=1, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        out = self.fc(lstm_out[:, -1, :])  # (batch, output_dim)
        return out


def build_model():
    return PureLSTM(input_dim=1, hidden_dim=8, output_dim=1)

# Quick architecture check
set_seed(0)
test_model = build_model().to(device)
total_params = sum(p.numel() for p in test_model.parameters())
print(f"Pure LSTM Parameters: {total_params}")
print(f"Memory estimate: {total_params * 4 / 1024:.1f} KB")
dummy = torch.randn(1, 1, 1).to(device)
out = test_model(dummy)
print(f"Input: {dummy.shape} → Output: {out.shape}")
del test_model, dummy, out""")

def pure_gru_model_cell():
    return cell("""# ============================================================
# Pure GRU — Gated Recurrent Unit
# ============================================================
# Cho et al. (2014) GRU.
# Simpler than LSTM (2 gates vs 3), fewer parameters.
# Same hyperparameters as LSTM-SNP for fair comparison.
#   - Hidden units: 8
#   - Single layer
#   - No dropout

class PureGRU(nn.Module):
    \"\"\"Standard GRU for time series forecasting.\"\"\"
    def __init__(self, input_dim=1, hidden_dim=8, output_dim=1, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        gru_out, h_n = self.gru(x)
        # Use last hidden state
        out = self.fc(gru_out[:, -1, :])  # (batch, output_dim)
        return out


def build_model():
    return PureGRU(input_dim=1, hidden_dim=8, output_dim=1)

# Quick architecture check
set_seed(0)
test_model = build_model().to(device)
total_params = sum(p.numel() for p in test_model.parameters())
print(f"Pure GRU Parameters: {total_params}")
print(f"Memory estimate: {total_params * 4 / 1024:.1f} KB")
dummy = torch.randn(1, 1, 1).to(device)
out = test_model(dummy)
print(f"Input: {dummy.shape} → Output: {out.shape}")
del test_model, dummy, out""")

# ── Model configs ────────────────────────────────────────────────────────────

MODELS = {
    'SNN_LSTM': {'dir': 'SNN_LSTM', 'name': 'SNN-LSTM', 'model_cell_fn': snn_lstm_model_cell},
    'Pure_LSTM': {'dir': 'Pure_LSTM', 'name': 'Pure LSTM', 'model_cell_fn': pure_lstm_model_cell},
    'Pure_GRU': {'dir': 'Pure_GRU', 'name': 'Pure GRU', 'model_cell_fn': pure_gru_model_cell},
}

def generate_notebook(model_key, dataset_key):
    model_cfg = MODELS[model_key]
    ds_cfg = DATASETS[dataset_key]
    model_name = model_cfg['name']
    dataset_title = ds_cfg['title']

    cells = [
        md_cell(f"# {model_name} — {dataset_title}\n\nTime series forecasting using {model_name} architecture.\\\nFollows the identical experimental protocol as the LSTM-SNP baseline."),
        pid_cell(),
        timer_start(),
        gpu_cell(),
        imports_cell(),
        seed_cell(),
        model_cfg['model_cell_fn'](),
        preprocess_cell(ds_cfg['csv'], dataset_title),
    ]
    cells.extend(train_eval_cells())
    cells.extend([
        experiment_cell(model_name, dataset_title),
        results_cell(model_name, dataset_title),
        plot_cell(model_name, dataset_title),
        timer_end(),
    ])

    nb = {
        "cells": cells,
        "metadata": {"kernelspec": KERNEL, "language_info": LANG_INFO},
        "nbformat": 4, "nbformat_minor": 5
    }

    outdir = os.path.join(BASE_DIR, model_cfg['dir'])
    os.makedirs(outdir, exist_ok=True)
    fname = f"{model_cfg['dir']}_{dataset_key}.ipynb"
    outpath = os.path.join(outdir, fname)
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')
    return outpath

def main():
    print("Generating baseline notebooks...\n")
    generated = []
    for model_key in MODELS:
        for ds_key in DATASETS:
            path = generate_notebook(model_key, ds_key)
            rel = os.path.relpath(path, BASE_DIR)
            print(f"  ✓ {rel}")
            generated.append(path)

    print(f"\nGenerated {len(generated)} notebooks")

    # Verify
    print("\nVerifying...")
    for p in generated:
        with open(p) as f:
            nb = json.load(f)
        full = ''.join(''.join(c.get('source',[])) for c in nb['cells'])
        checks = {
            'GPU cell': 'Apple Metal GPU Enabled' in full,
            'MPS': 'mps' in full,
            'CSV path': 'content/' in full,
            '30 runs': 'N_RUNS = 30' in full,
            'Seed': 'set_seed' in full,
        }
        failed = [k for k,v in checks.items() if not v]
        rel = os.path.relpath(p, BASE_DIR)
        if failed:
            print(f"  ✗ {rel}: MISSING {', '.join(failed)}")
        else:
            print(f"  ✓ {rel}: all checks passed")

if __name__ == '__main__':
    main()
