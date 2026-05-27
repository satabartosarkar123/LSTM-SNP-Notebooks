#!/usr/bin/env python3
"""
Generate Membership LSTM-SNP Jupyter Notebooks.
Topologies:
  - MF_Gates_Only: Baseline architecture using DoG/SG/GB replacing c(t) and o(t).
  - MF_Gates_Plus_FuzzyInput: NumPy Fuzzy Augmented Input + MF Gates.
  - MF_Gates_Plus_FuzzyOutput: MF Gates + FuzzyOutputLayer final prediction.
"""

import json
import os

# ============================================================
# Dataset configurations
# ============================================================
DATASETS = {
    "sp500": {"csv_path": "content/sp500.csv", "name": "S&P 500", "test_size": 60},
    "dow_jones": {"csv_path": "content/monthly-closings-of-the-dowjones.csv", "name": "Dow Jones", "test_size": 60},
    "lake_erie": {"csv_path": "content/monthly-lake-erie-levels-1921-19.csv", "name": "Lake Erie", "test_size": 60},
    "milk_production": {"csv_path": "content/monthly-milk-production-pounds-p.csv", "name": "Milk Production", "test_size": 60},
}

def md_cell(source):
    lines = [line if line.endswith('\n') else line + '\n' for line in (source.split('\n') if isinstance(source, str) else source)]
    if lines: lines[-1] = lines[-1].rstrip('\n')
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def code_cell(source):
    lines = [line if line.endswith('\n') else line + '\n' for line in (source.split('\n') if isinstance(source, str) else source)]
    if lines: lines[-1] = lines[-1].rstrip('\n')
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.10.0"}},
        "nbformat": 4, "nbformat_minor": 5,
    }

IMPORTS_CODE = """\
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
"""

def data_loading_code(csv_path):
    return f"series = pd.read_csv('{csv_path}', header=0, parse_dates=[0], index_col=0)\nraw_values = series.values.flatten()"

PREPROCESSING_CODE = """\
def difference(dataset, interval=1):
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)

diff_values = difference(raw_values, 1)
def timeseries_to_supervised(data, lag):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df.values
"""

def split_scale_code(test_size):
    return f"""\
train, test = supervised[:-{test_size}], supervised[-{test_size}:]
scaler = MinMaxScaler(feature_range=(-1, 1))
train_scaled, test_scaled = scaler.fit_transform(train), scaler.transform(test)
"""

FUZZY_INFERENCE_NUMPY_CODE = """\
def gaussian_mf(x, center, sigma=0.5):
    return np.exp(-(x - center)**2 / (2 * sigma**2))

def fuzzy_inference_numpy(x_t, x_tm1):
    mu_low_xt, mu_high_xt = gaussian_mf(x_t, -1.0), gaussian_mf(x_t, 1.0)
    mu_low_xtm1, mu_high_xtm1 = gaussian_mf(x_tm1, -1.0), gaussian_mf(x_tm1, 1.0)

    w1, w2 = mu_low_xt * mu_low_xtm1, mu_low_xt * mu_high_xtm1
    w3, w4 = mu_high_xt * mu_low_xtm1, mu_high_xt * mu_high_xtm1

    y1 = 0.5 * x_t + 0.5 * x_tm1
    y2 = 0.7 * x_t + 0.3 * x_tm1 - 0.1
    y3 = 0.3 * x_t + 0.7 * x_tm1 + 0.1
    y4 = 0.5 * x_t + 0.5 * x_tm1

    return (w1*y1 + w2*y2 + w3*y3 + w4*y4) / (w1 + w2 + w3 + w4 + 1e-8)
"""

MEMBERSHIP_CELL_CODE = """\
def dog(x, mu1=-1.0, mu2=1.0, sigma1=0.5, sigma2=0.5):
    return tf.exp(-tf.square(x - mu1)/ (2 * sigma1**2)) - tf.exp(-tf.square(x - mu2)/ (2 * sigma2**2))

def signed_gaussian(x, sigma=0.5):
    return x * tf.exp(-tf.square(x)/ (2 * sigma**2))

def generalized_bell(x, a=1.0, b=2.0, c_val=0.0):
    return 1.0 / (1.0 + tf.pow(tf.abs((x - c_val) / a), 2*b))

@tf.keras.utils.register_keras_serializable()
class MembershipLSTMSNPCell(layers.Layer):
    def __init__(self, units, mf_type='dog', **kwargs):
        super().__init__(**kwargs)
        self.units, self.mf_type = units, mf_type
        self.state_size, self.output_size = (units, units, units), units
        self.mf = dog if mf_type == 'dog' else signed_gaussian if mf_type == 'sg' else generalized_bell

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units * 4), initializer='glorot_uniform', name='kernel')
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 4), initializer='orthogonal', name='recurrent_kernel')
        self.bias = self.add_weight(shape=(self.units * 4,), initializer='zeros', name='bias')

    def call(self, inputs, states):
        u_tm1 = states[0]  
        z = tf.matmul(inputs, self.kernel) + tf.matmul(u_tm1, self.recurrent_kernel) + self.bias
        z0, z1, z2, z3 = z[:, :self.units], z[:, self.units:2*self.units], z[:, 2*self.units:3*self.units], z[:, 3*self.units:]

        r = tf.tanh(z0)
        c = tf.clip_by_value(self.mf(z1), -1.0, 1.0)
        o = tf.clip_by_value(self.mf(z2), -1.0, 1.0)
        a = tf.tanh(z3)

        u = r * u_tm1 - c * a
        h = o * a
        return h, [u, c, o]

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units, 'mf_type': self.mf_type})
        return config
"""

FUZZY_OUTPUT_LAYER_CODE = """\
@tf.keras.utils.register_keras_serializable()
class FuzzyOutputLayer(layers.Layer):
    def __init__(self, units_in, **kwargs):
        super().__init__(**kwargs)
        self.units_in, self.sigma = units_in, 0.5
    def build(self, input_shape):
        self.rule_a = self.add_weight(shape=(4,), initializer='glorot_uniform', name='rule_a')
        self.rule_b = self.add_weight(shape=(4,), initializer='glorot_uniform', name='rule_b')
        self.rule_c = self.add_weight(shape=(4,), initializer='zeros', name='rule_c')
    def call(self, inputs):
        half = self.units_in // 2
        s1 = tf.reduce_mean(inputs[:, :half], axis=-1, keepdims=True)
        s2 = tf.reduce_mean(inputs[:, half:], axis=-1, keepdims=True)
        
        m_l1, m_h1 = tf.exp(-tf.square(s1 - (-1.0))/ (2.0*0.25)), tf.exp(-tf.square(s1 - 1.0)/ (2.0*0.25))
        m_l2, m_h2 = tf.exp(-tf.square(s2 - (-1.0))/ (2.0*0.25)), tf.exp(-tf.square(s2 - 1.0)/ (2.0*0.25))
        w1, w2, w3, w4 = m_l1*m_l2, m_l1*m_h2, m_h1*m_l2, m_h1*m_h2
        
        y1, y2 = self.rule_a[0]*s1 + self.rule_b[0]*s2 + self.rule_c[0], self.rule_a[1]*s1 + self.rule_b[1]*s2 + self.rule_c[1]
        y3, y4 = self.rule_a[2]*s1 + self.rule_b[2]*s2 + self.rule_c[2], self.rule_a[3]*s1 + self.rule_b[3]*s2 + self.rule_c[3]
        return (w1*y1 + w2*y2 + w3*y3 + w4*y4) / (w1 + w2 + w3 + w4 + 1e-8)
    def get_config(self):
        config = super().get_config()
        config.update({'units_in': self.units_in})
        return config
"""

def build_model_code(topology, mf_type):
    out_layer = "FuzzyOutputLayer(units_in=units)(x)" if topology == "MF_Gates_Plus_FuzzyOutput" else "layers.Dense(1)(x)"
    return f"""\
def build_model(input_dim, units, batch_size):
    cell = MembershipLSTMSNPCell(units, mf_type='{mf_type}')
    rnn = layers.RNN(cell, return_sequences=False, return_state=True, stateful=True)

    inputs = tf.keras.Input(batch_shape=(batch_size, 1, input_dim))
    x, u_out, c_out, o_out = rnn(inputs)
    outputs = {out_layer}

    model = tf.keras.Model(inputs=inputs, outputs=[outputs, c_out, o_out])
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(clipnorm=1.0), loss=['mean_squared_error', None, None])
    return model
"""

def build_experiment_code(topology, test_size):
    needs_fuzzy_in = (topology == "MF_Gates_Plus_FuzzyInput")
    prep_train_test = f"""
supervised = timeseries_to_supervised(diff_values, {2 if needs_fuzzy_in else 1})
train, test = supervised[:-{test_size}], supervised[-{test_size}:]
scaler = MinMaxScaler(feature_range=(-1, 1))
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

X_train_raw, y_train = train_scaled[:, 0:-1], train_scaled[:, -1]
X_test_raw, y_test = test_scaled[:, 0:-1], test_scaled[:, -1]

{"X_train_fuzzy = np.zeros((X_train_raw.shape[0], 2)); X_test_fuzzy = np.zeros((X_test_raw.shape[0], 2))" if needs_fuzzy_in else ""}
{"for i in range(X_train_raw.shape[0]): X_train_fuzzy[i] = [X_train_raw[i, 1], fuzzy_inference_numpy(X_train_raw[i, 1], X_train_raw[i, 0])]" if needs_fuzzy_in else ""}
{"for i in range(X_test_raw.shape[0]): X_test_fuzzy[i] = [X_test_raw[i, 1], fuzzy_inference_numpy(X_test_raw[i, 1], X_test_raw[i, 0])]" if needs_fuzzy_in else ""}

X_train = {"X_train_fuzzy" if needs_fuzzy_in else "X_train_raw"}.reshape(({"X_train_fuzzy" if needs_fuzzy_in else "X_train_raw"}.shape[0], 1, {2 if needs_fuzzy_in else 1}))
X_test = {"X_test_fuzzy" if needs_fuzzy_in else "X_test_raw"}.reshape(({"X_test_fuzzy" if needs_fuzzy_in else "X_test_raw"}.shape[0], 1, {2 if needs_fuzzy_in else 1}))
"""
    return prep_train_test + f"""
all_rmse, all_mse, all_nmse = [], [], []
all_predictions = []
all_losses = []
all_c_means, all_c_stds, all_c_mins, all_c_maxs = [], [], [], []
all_o_means, all_o_stds, all_o_mins, all_o_maxs = [], [], [], []

for run in range(30):
    print(f'\\n===== RUN {{run+1}}/30 =====')
    np.random.seed(run)
    tf.random.set_seed(run)
    tf.keras.backend.clear_session()
    
    model = build_model(input_dim={2 if needs_fuzzy_in else 1}, units=8, batch_size=1)
    rnn_layer = model.layers[1]
    
    run_losses = []
    for epoch in range(100):
        history = model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=0, shuffle=False)
        run_losses.append(history.history['loss'][0])
        rnn_layer.reset_states()
    all_losses.append(run_losses)

    # Warmup
    for i in range(len(X_train)): model.predict(X_train[i:i+1], batch_size=1, verbose=0)

    predictions, c_gates, o_gates = [], [], []
    for i in range(len(X_test)):
        yhat, c_val, o_val = model.predict(X_test[i:i+1], batch_size=1, verbose=0)
        c_gates.append(c_val[0])
        o_gates.append(o_val[0])
        
        row = list(X_test_raw[i]) + [yhat[0, 0]]
        inv = scaler.inverse_transform([row])[0, -1] + raw_values[len(train) + i]
        predictions.append(inv)
        
    c_gates, o_gates = np.array(c_gates), np.array(o_gates)
    all_c_means.append(np.mean(c_gates)); all_c_stds.append(np.std(c_gates))
    all_c_mins.append(np.min(c_gates)); all_c_maxs.append(np.max(c_gates))
    all_o_means.append(np.mean(o_gates)); all_o_stds.append(np.std(o_gates))
    all_o_mins.append(np.min(o_gates)); all_o_maxs.append(np.max(o_gates))

    actual = raw_values[-{test_size}:]
    rmse = sqrt(mean_squared_error(actual, predictions))
    mse = mean_squared_error(actual, predictions)
    meanV = np.mean(actual)
    dominator = np.linalg.norm(np.array(predictions) - meanV, 2)
    nmse = mse / np.power(dominator, 2)

    all_rmse.append(rmse)
    all_mse.append(mse)
    all_nmse.append(nmse)
    all_predictions.append(predictions)
    print(f'Run {{run+1}} — RMSE: {{rmse:.6f}}, MSE: {{mse:.6f}}, NMSE: {{nmse:.10f}} (Gate Mean C: {{np.mean(c_gates):.4f}})')

print('\\n===== GATE STATISTICS =====')
print(f'C-Gate - Mean: {{np.mean(all_c_means):.4f}}, Min: {{np.min(all_c_mins):.4f}}, Max: {{np.max(all_c_maxs):.4f}}')
print(f'O-Gate - Mean: {{np.mean(all_o_means):.4f}}, Min: {{np.min(all_o_mins):.4f}}, Max: {{np.max(all_o_maxs):.4f}}')
print(f'\\nOverall RMSE: {{np.mean(all_rmse):.6f}}')
"""

def build_summary_stats_code(topology, mf_type, dataset_name):
    title = f"{topology} ({mf_type.upper()}) on {dataset_name}"
    return f"""\
# ============================================================
# Summary Statistics (30 runs)
# ============================================================

print('\\n===== FINAL RESULTS (30 runs) =====')
print(f'RMSE: {{np.mean(all_rmse):.6f}} ± {{np.std(all_rmse):.6f}}')
print(f'MSE:  {{np.mean(all_mse):.6f}} ± {{np.std(all_mse):.6f}}')
print(f'NMSE: {{np.mean(all_nmse):.10f}} ± {{np.std(all_nmse):.10f}}')

best_idx = np.argmin(all_rmse)
print(f'\\nBest run: {{best_idx+1}}')
print(f'  RMSE: {{all_rmse[best_idx]:.6f}}')
print(f'  MSE:  {{all_mse[best_idx]:.6f}}')
print(f'  NMSE: {{all_nmse[best_idx]:.10f}}')
"""

def build_graphs_code(topology, mf_type, dataset_name, test_size):
    title = f"{topology} ({mf_type.upper()}) — {dataset_name}"
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
plt.title('{title}\\nPredictions vs Actual (Best of 30 runs)')
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
plt.title('{title}\\nTraining Loss (Best Run)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# Average Loss Curve (All 30 Runs)
# ============================================================

avg_losses = np.mean(all_losses, axis=0)
std_losses = np.std(all_losses, axis=0)
epochs = np.arange(1, len(avg_losses) + 1)

plt.figure(figsize=(12, 4))
plt.plot(epochs, avg_losses, color='purple', linewidth=1.5, label='Mean Loss')
plt.fill_between(epochs, avg_losses - std_losses, avg_losses + std_losses,
                 alpha=0.2, color='purple', label='±1 Std Dev')
plt.title('{title}\\nAverage Training Loss (30 runs ± 1σ)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# RMSE Distribution (Boxplot + All Points)
# ============================================================

plt.figure(figsize=(8, 5))
plt.boxplot(all_rmse, vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='navy'),
            medianprops=dict(color='red', linewidth=2))
plt.scatter(np.ones(len(all_rmse)), all_rmse, alpha=0.5, color='navy', zorder=5)
plt.title('{title}\\nRMSE Distribution (30 runs)')
plt.ylabel('RMSE')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# Average Predictions vs Actual
# ============================================================

avg_predictions = np.mean(all_predictions, axis=0)
std_predictions = np.std(all_predictions, axis=0)

plt.figure(figsize=(12, 5))
plt.plot(actual, label='Actual', color='blue', linewidth=1.5)
plt.plot(avg_predictions, label='Mean Predicted (30 runs)', color='red',
         linewidth=1.5, linestyle='--')
plt.fill_between(range(len(actual)),
                 avg_predictions - std_predictions,
                 avg_predictions + std_predictions,
                 alpha=0.2, color='red', label='±1 Std Dev')
plt.title('{title}\\nAverage Predictions vs Actual (30 runs ± 1σ)')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# Gate Statistics Visualization
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# C-Gate statistics
axes[0].errorbar(range(len(all_c_means)), all_c_means, yerr=all_c_stds,
                 fmt='o-', color='teal', ecolor='lightcoral', capsize=3, markersize=4)
axes[0].set_title('C-Gate Mean ± Std per Run')
axes[0].set_xlabel('Run')
axes[0].set_ylabel('C-Gate Value')
axes[0].grid(True, alpha=0.3)

# O-Gate statistics
axes[1].errorbar(range(len(all_o_means)), all_o_means, yerr=all_o_stds,
                 fmt='o-', color='darkorange', ecolor='lightcoral', capsize=3, markersize=4)
axes[1].set_title('O-Gate Mean ± Std per Run')
axes[1].set_xlabel('Run')
axes[1].set_ylabel('O-Gate Value')
axes[1].grid(True, alpha=0.3)

plt.suptitle('{title} — Gate Statistics (30 runs)', fontsize=13)
plt.tight_layout()
plt.show()

# ============================================================
# Final Metrics Summary
# ============================================================

print('=== Best Run Metrics ===')
print(f'RMSE: {{all_rmse[best_idx]:.6f}}')
print(f'MSE:  {{all_mse[best_idx]:.6f}}')
print(f'NMSE: {{all_nmse[best_idx]:.10f}}')

print('\\n=== Average Metrics (30 runs) ===')
print(f'RMSE: {{np.mean(all_rmse):.6f}} ± {{np.std(all_rmse):.6f}}')
print(f'MSE:  {{np.mean(all_mse):.6f}} ± {{np.std(all_mse):.6f}}')
print(f'NMSE: {{np.mean(all_nmse):.10f}} ± {{np.std(all_nmse):.10f}}')
"""

def build_observations_md(topology, mf_type, dataset_name):
    title = f"{topology} ({mf_type.upper()}) on {dataset_name}"
    return f"""\
## Observations

### {title}

**Run the notebook to generate results and fill in observations:**

1. **Prediction Quality**: Compare RMSE/MSE/NMSE with other variants
2. **Training Stability**: Examine loss curves for convergence behavior
3. **Average Behaviour**: Compare average predictions vs actual
4. **Gate Dynamics**: Examine C-Gate and O-Gate statistics across runs
5. **Prediction Tracking**: Assess how well predictions track actual values
6. **Computational Cost**: Note training time per run

*After running all variant notebooks, perform cross-variant comparison to evaluate 
whether the membership function activation improves performance over standard activations.*
"""

def generate_notebook(topology, mf_type, dataset_key):
    ds = DATASETS[dataset_key]
    cells = []
    cells.append(md_cell(f"# {topology} using {mf_type.upper()}\\nDataset: {ds['name']}"))
    cells.append(code_cell(IMPORTS_CODE))
    if topology == "MF_Gates_Plus_FuzzyInput": cells.append(code_cell(FUZZY_INFERENCE_NUMPY_CODE))
    cells.append(code_cell(MEMBERSHIP_CELL_CODE))
    if topology == "MF_Gates_Plus_FuzzyOutput": cells.append(code_cell(FUZZY_OUTPUT_LAYER_CODE))
    cells.append(code_cell(build_model_code(topology, mf_type)))
    cells.append(code_cell(data_loading_code(ds['csv_path'])))
    cells.append(code_cell(PREPROCESSING_CODE))
    cells.append(code_cell(build_experiment_code(topology, ds['test_size'])))
    cells.append(md_cell("## Results"))
    cells.append(code_cell(build_summary_stats_code(topology, mf_type, ds['name'])))
    cells.append(code_cell(build_graphs_code(topology, mf_type, ds['name'], ds['test_size'])))
    cells.append(md_cell(build_observations_md(topology, mf_type, ds['name'])))
    return make_notebook(cells)

def main():
    TOPOLOGIES = ["MF_Gates_Only", "MF_Gates_Plus_FuzzyInput", "MF_Gates_Plus_FuzzyOutput"]
    MFS = ['dog', 'sg', 'gb']
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    for topo in TOPOLOGIES:
        for mf in MFS:
            for dkey in DATASETS.keys():
                nb = generate_notebook(topo, mf, dkey)
                filename = f"Modified_{topo}_{mf.upper()}_{dkey}.ipynb"
                with open(os.path.join(output_dir, filename), 'w') as f: json.dump(nb, f, indent=1)
                print(f"Generated {filename}")

if __name__ == "__main__":
    main()
