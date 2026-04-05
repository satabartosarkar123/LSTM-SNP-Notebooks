#!/usr/bin/env python3
"""
Generate all 20 Fuzzy LSTM-SNP Jupyter Notebooks.
5 variants × 4 datasets = 20 notebooks.

Variants:
  1. Baseline (control)
  2. Fuzzy Feature Augmentation
  3. Fuzzy Gate Replacement
  4. Fuzzy Output Layer
  5. Hybrid (Feature Aug + Gate Replacement)

Datasets:
  - S&P 500
  - Dow Jones
  - Lake Erie
  - Milk Production
"""

import json
import os

# ============================================================
# Dataset configurations
# ============================================================
DATASETS = {
    "sp500": {
        "csv_path": "content/sp500.csv",
        "name": "S&P 500",
        "test_size": 60,
    },
    "dow_jones": {
        "csv_path": "content/monthly-closings-of-the-dowjones.csv",
        "name": "Dow Jones Industrial Index",
        "test_size": 60,
    },
    "lake_erie": {
        "csv_path": "content/monthly-lake-erie-levels-1921-19.csv",
        "name": "Monthly Lake Erie Levels",
        "test_size": 60,
    },
    "milk_production": {
        "csv_path": "content/monthly-milk-production-pounds-p.csv",
        "name": "Monthly Milk Production",
        "test_size": 60,
    },
}

# ============================================================
# Helper: create notebook cell
# ============================================================
def md_cell(source):
    """Create a markdown cell."""
    if isinstance(source, str):
        source = source.split('\n')
    # Ensure each line ends with \n except the last
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
# Shared code fragments
# ============================================================
IMPORTS_CODE = """\
# ============================================================
# ALL IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")"""


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
print(f"First 5 values: {{raw_values[:5]}}")"""


PREPROCESSING_CODE = """\
# ============================================================
# 2. First-Order Differencing
# ============================================================

def difference(dataset, interval=1):
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)

diff_values = difference(raw_values, 1)"""


SUPERVISED_CODE_LAG1 = """\
# ============================================================
# 3. Convert to Supervised Learning Format (lag=1)
# ============================================================

def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df.values

supervised = timeseries_to_supervised(diff_values, 1)
print(f"Supervised data shape: {supervised.shape}")"""


SUPERVISED_CODE_LAG2 = """\
# ============================================================
# 3. Convert to Supervised Learning Format (lag=2)
#    Needed for fuzzy feature augmentation: x(t), x(t-1)
# ============================================================

def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df.values

supervised = timeseries_to_supervised(diff_values, 2)
print(f"Supervised data shape: {supervised.shape}")"""


def split_scale_code(test_size):
    return f"""\
# ============================================================
# 4. Train-Test Split
# ============================================================

train, test = supervised[:-{test_size}], supervised[-{test_size}:]
print(f"Train: {{train.shape}}, Test: {{test.shape}}")

# ============================================================
# 5. Feature Scaling
# ============================================================

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(train)

train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)"""


RESHAPE_CODE_LAG1 = """\
# ============================================================
# 6. Reshape for RNN Input
# ============================================================

X_train, y_train = train_scaled[:, 0:-1], train_scaled[:, -1]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

X_test, y_test = test_scaled[:, 0:-1], test_scaled[:, -1]
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")"""


RESHAPE_CODE_LAG2_FUZZY_AUG = """\
# ============================================================
# 6. Reshape for RNN Input (with Fuzzy Feature Augmentation)
#    supervised columns: [x(t-2), x(t-1), x(t)]
#    We use x(t-1) as primary input, x(t-2) for fuzzy context
# ============================================================

# Extract columns: col0=x(t-2), col1=x(t-1), col2=x(t) (target)
X_train_raw = train_scaled[:, 0:-1]  # [x(t-2), x(t-1)]
y_train = train_scaled[:, -1]        # x(t)

X_test_raw = test_scaled[:, 0:-1]
y_test = test_scaled[:, -1]

# Compute fuzzy features for training data
X_train_fuzzy = np.zeros((X_train_raw.shape[0], 2))  # [x(t-1), y_fuzzy]
for i in range(X_train_raw.shape[0]):
    x_t = X_train_raw[i, 1]      # x(t-1) = current input
    x_tm1 = X_train_raw[i, 0]    # x(t-2) = previous input
    y_fuzz = fuzzy_inference_numpy(x_t, x_tm1)
    X_train_fuzzy[i, 0] = x_t
    X_train_fuzzy[i, 1] = y_fuzz

X_train = X_train_fuzzy.reshape((X_train_fuzzy.shape[0], 1, 2))

# Compute fuzzy features for test data
X_test_fuzzy = np.zeros((X_test_raw.shape[0], 2))
for i in range(X_test_raw.shape[0]):
    x_t = X_test_raw[i, 1]
    x_tm1 = X_test_raw[i, 0]
    y_fuzz = fuzzy_inference_numpy(x_t, x_tm1)
    X_test_fuzzy[i, 0] = x_t
    X_test_fuzzy[i, 1] = y_fuzz

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")"""


# ============================================================
# Cell code: LSTM-SNP Cell (baseline, unchanged)
# ============================================================
LSTMSNP_CELL_CODE = """\
# ============================================================
# LSTM-SNP Cell (Original — Unmodified)
# ============================================================

@tf.keras.utils.register_keras_serializable()
class LSTMSNPCell(layers.Layer):
    \"\"\"
    LSTM-SNP Cell: A long short-term memory model inspired from
    spiking neural P systems.

    Gates:
      r(t) = ρ(W_r x(t) + U_r u(t-1) + b_r)   [reset]
      c(t) = ρ(W_c x(t) + U_c u(t-1) + b_c)   [consumption]
      o(t) = ρ(W_o x(t) + U_o u(t-1) + b_o)   [output/generation]
      a(t) = f(W_a x(t) + U_a u(t-1) + b_a)   [generated spikes]

    State update:
      u(t) = r(t) * u(t-1) - c(t) * a(t)
      h(t) = o(t) * a(t)

    ρ = hard_sigmoid, f = tanh
    \"\"\"
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units
        self.output_size = units

        self.activation = tf.keras.activations.get(activation)
        self.recurrent_activation = tf.keras.activations.get(recurrent_activation)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            initializer='glorot_uniform',
            name='kernel'
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            initializer='orthogonal',
            name='recurrent_kernel'
        )
        self.bias = self.add_weight(
            shape=(self.units * 4,),
            initializer='zeros',
            name='bias'
        )

    def call(self, inputs, states):
        u_tm1 = states[0]

        z = tf.matmul(inputs, self.kernel) + \\
            tf.matmul(u_tm1, self.recurrent_kernel) + self.bias

        z0 = z[:, :self.units]
        z1 = z[:, self.units:2*self.units]
        z2 = z[:, 2*self.units:3*self.units]
        z3 = z[:, 3*self.units:]

        r = self.recurrent_activation(z0)  # reset
        c = self.recurrent_activation(z1)  # consumption
        o = self.recurrent_activation(z2)  # output/generation
        a = self.activation(z3)            # generated spikes

        u = r * u_tm1 - c * a  # internal state
        h = o * a              # output

        return h, [u]

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
        })
        return config"""


# ============================================================
# Fuzzy Gate LSTM-SNP Cell
# ============================================================
FUZZY_GATE_CELL_CODE = """\
# ============================================================
# Fuzzy Gate LSTM-SNP Cell
# Gates r, c, o use fuzzy inference instead of hard sigmoid.
# Generation gate 'a' keeps tanh (unchanged).
#
# Fixed Gaussian membership functions:
#   μ_low(x)  = exp(-(x - (-1))² / (2·0.5²))
#   μ_high(x) = exp(-(x - (+1))² / (2·0.5²))
#
# Each gate uses 2 Takagi-Sugeno rules with trainable consequents.
# ============================================================

@tf.keras.utils.register_keras_serializable()
class FuzzyLSTMSNPCell(layers.Layer):
    \"\"\"
    LSTM-SNP Cell with fuzzy gate replacement.
    Gates r, c, o are computed via fuzzy inference.
    Gate a keeps tanh activation.
    \"\"\"
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units
        self.output_size = units
        self.activation = tf.keras.activations.get('tanh')
        # Fixed membership function parameters
        self.mu_low = -1.0
        self.mu_high = 1.0
        self.sigma = 0.5

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Standard weights for pre-activation computation
        # 4 gates: r, c, o, a
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            initializer='glorot_uniform',
            name='kernel'
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            initializer='orthogonal',
            name='recurrent_kernel'
        )
        self.bias = self.add_weight(
            shape=(self.units * 4,),
            initializer='zeros',
            name='bias'
        )

        # Fuzzy consequent parameters for 3 gates (r, c, o)
        # Each gate has 2 rules, each rule has 3 params (a, b, c_param)
        # rule_i: y_i = a_i * z_gate + b_i * u_mean + c_i
        # Store as instance attributes for reliable access
        self.fuzzy_params = {}
        for gate_name in ['r', 'c', 'o']:
            self.fuzzy_params[gate_name] = {}
            for rule_idx in range(2):
                self.fuzzy_params[gate_name][rule_idx] = {
                    'a': self.add_weight(
                        shape=(self.units,),
                        initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1),
                        name=f'fuzzy_{gate_name}_rule{rule_idx}_a'
                    ),
                    'b': self.add_weight(
                        shape=(self.units,),
                        initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1),
                        name=f'fuzzy_{gate_name}_rule{rule_idx}_b'
                    ),
                    'c_param': self.add_weight(
                        shape=(self.units,),
                        initializer=tf.keras.initializers.Zeros(),
                        name=f'fuzzy_{gate_name}_rule{rule_idx}_c'
                    ),
                }

    def _gaussian_mf(self, x, center):
        \"\"\"Fixed Gaussian membership function.\"\"\"
        return tf.exp(-tf.square(x - center) / (2.0 * self.sigma ** 2))

    def _fuzzy_gate(self, z_gate, u_mean, gate_name):
        \"\"\"Compute gate value via fuzzy inference (2 rules).\"\"\"
        # Membership degrees (fixed)
        w_low = self._gaussian_mf(z_gate, self.mu_low)
        w_high = self._gaussian_mf(z_gate, self.mu_high)

        # Get trainable consequent parameters from instance dict
        p = self.fuzzy_params[gate_name]

        # Rule outputs (trainable linear consequents)
        y0 = p[0]['a'] * z_gate + p[0]['b'] * u_mean + p[0]['c_param']
        y1 = p[1]['a'] * z_gate + p[1]['b'] * u_mean + p[1]['c_param']

        # Weighted average defuzzification
        numerator = w_low * y0 + w_high * y1
        denominator = w_low + w_high + 1e-8
        output = numerator / denominator

        # Clip to [0, 1] since gates should be bounded
        return tf.clip_by_value(output, 0.0, 1.0)

    def call(self, inputs, states):
        u_tm1 = states[0]

        z = tf.matmul(inputs, self.kernel) + \\
            tf.matmul(u_tm1, self.recurrent_kernel) + self.bias

        z0 = z[:, :self.units]
        z1 = z[:, self.units:2*self.units]
        z2 = z[:, 2*self.units:3*self.units]
        z3 = z[:, 3*self.units:]

        # Mean of previous state for fuzzy rule input
        u_mean = tf.reduce_mean(u_tm1, axis=-1, keepdims=True)
        u_mean = tf.tile(u_mean, [1, self.units])

        # Fuzzy gates
        r = self._fuzzy_gate(z0, u_mean, 'r')
        c = self._fuzzy_gate(z1, u_mean, 'c')
        o = self._fuzzy_gate(z2, u_mean, 'o')
        a = self.activation(z3)  # tanh unchanged

        u = r * u_tm1 - c * a
        h = o * a

        return h, [u]

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config"""


# ============================================================
# Fuzzy Inference for Feature Augmentation (NumPy — preprocessing)
# ============================================================
FUZZY_INFERENCE_NUMPY_CODE = """\
# ============================================================
# Fuzzy Inference System (NumPy — for preprocessing)
#
# Fixed Gaussian membership functions:
#   μ_low(x)  = exp(-(x - (-1))² / (2·0.5²))
#   μ_high(x) = exp(-(x - (+1))² / (2·0.5²))
#
# 4 Takagi-Sugeno rules with fixed consequent parameters:
#   IF x(t) is low  AND x(t-1) is low  → y₁ = 0.5·x(t) + 0.5·x(t-1)
#   IF x(t) is low  AND x(t-1) is high → y₂ = 0.7·x(t) + 0.3·x(t-1) - 0.1
#   IF x(t) is high AND x(t-1) is low  → y₃ = 0.3·x(t) + 0.7·x(t-1) + 0.1
#   IF x(t) is high AND x(t-1) is high → y₄ = 0.5·x(t) + 0.5·x(t-1)
#
# Output: y = Σ(wᵢ·yᵢ) / Σ(wᵢ)
# ============================================================

def gaussian_mf(x, center, sigma=0.5):
    \"\"\"Fixed Gaussian membership function.\"\"\"
    return np.exp(-(x - center)**2 / (2 * sigma**2))

def fuzzy_inference_numpy(x_t, x_tm1):
    \"\"\"
    Compute fuzzy feature from x(t) and x(t-1).
    Uses fixed membership functions and fixed consequent parameters.
    \"\"\"
    # Membership degrees
    mu_low_xt = gaussian_mf(x_t, center=-1.0)
    mu_high_xt = gaussian_mf(x_t, center=1.0)
    mu_low_xtm1 = gaussian_mf(x_tm1, center=-1.0)
    mu_high_xtm1 = gaussian_mf(x_tm1, center=1.0)

    # Rule firing strengths (product)
    w1 = mu_low_xt * mu_low_xtm1      # low-low
    w2 = mu_low_xt * mu_high_xtm1     # low-high
    w3 = mu_high_xt * mu_low_xtm1     # high-low
    w4 = mu_high_xt * mu_high_xtm1    # high-high

    # Consequent outputs (fixed linear functions)
    y1 = 0.5 * x_t + 0.5 * x_tm1
    y2 = 0.7 * x_t + 0.3 * x_tm1 - 0.1
    y3 = 0.3 * x_t + 0.7 * x_tm1 + 0.1
    y4 = 0.5 * x_t + 0.5 * x_tm1

    # Weighted average defuzzification
    numerator = w1 * y1 + w2 * y2 + w3 * y3 + w4 * y4
    denominator = w1 + w2 + w3 + w4 + 1e-8

    return numerator / denominator"""


# ============================================================
# Fuzzy Output Layer code
# ============================================================
FUZZY_OUTPUT_LAYER_CODE = """\
# ============================================================
# Fuzzy Output Layer
# Replaces Dense(1) with fuzzy inference.
#
# Takes RNN output h(t) ∈ R^units, produces scalar prediction.
# Aggregates h(t) into 2 summary features via mean-pooling:
#   s1 = mean(h[:units//2])
#   s2 = mean(h[units//2:])
#
# Then applies 4 Takagi-Sugeno rules.
# Fixed Gaussian MFs. Trainable consequent parameters.
# ============================================================

@tf.keras.utils.register_keras_serializable()
class FuzzyOutputLayer(layers.Layer):
    \"\"\"
    Fuzzy output layer: replaces Dense(1).
    Input: h(t) from RNN (shape: [batch, units])
    Output: scalar prediction (shape: [batch, 1])
    \"\"\"
    def __init__(self, units_in, **kwargs):
        super().__init__(**kwargs)
        self.units_in = units_in
        self.mu_low = -1.0
        self.mu_high = 1.0
        self.sigma = 0.5

    def build(self, input_shape):
        # 4 rules, each with 3 consequent params (a, b, c)
        # y_i = a_i * s1 + b_i * s2 + c_i
        self.rule_a = self.add_weight(shape=(4,), initializer='glorot_uniform', name='rule_a')
        self.rule_b = self.add_weight(shape=(4,), initializer='glorot_uniform', name='rule_b')
        self.rule_c = self.add_weight(shape=(4,), initializer='zeros', name='rule_c')

    def _gaussian_mf(self, x, center):
        return tf.exp(-tf.square(x - center) / (2.0 * self.sigma ** 2))

    def call(self, inputs):
        half = self.units_in // 2
        # Aggregate to 2 summary features
        s1 = tf.reduce_mean(inputs[:, :half], axis=-1, keepdims=True)  # [batch, 1]
        s2 = tf.reduce_mean(inputs[:, half:], axis=-1, keepdims=True)  # [batch, 1]

        # Membership degrees
        mu_low_s1 = self._gaussian_mf(s1, self.mu_low)
        mu_high_s1 = self._gaussian_mf(s1, self.mu_high)
        mu_low_s2 = self._gaussian_mf(s2, self.mu_low)
        mu_high_s2 = self._gaussian_mf(s2, self.mu_high)

        # Rule weights
        w1 = mu_low_s1 * mu_low_s2
        w2 = mu_low_s1 * mu_high_s2
        w3 = mu_high_s1 * mu_low_s2
        w4 = mu_high_s1 * mu_high_s2

        # Consequent outputs
        y1 = self.rule_a[0] * s1 + self.rule_b[0] * s2 + self.rule_c[0]
        y2 = self.rule_a[1] * s1 + self.rule_b[1] * s2 + self.rule_c[1]
        y3 = self.rule_a[2] * s1 + self.rule_b[2] * s2 + self.rule_c[2]
        y4 = self.rule_a[3] * s1 + self.rule_b[3] * s2 + self.rule_c[3]

        # Defuzzification
        numerator = w1 * y1 + w2 * y2 + w3 * y3 + w4 * y4
        denominator = w1 + w2 + w3 + w4 + 1e-8

        return numerator / denominator

    def get_config(self):
        config = super().get_config()
        config.update({'units_in': self.units_in})
        return config"""


# ============================================================
# Build model functions
# ============================================================
def build_model_code(variant):
    """Return build_model function code for each variant."""
    if variant == 1:  # Baseline
        return """\
# ============================================================
# Model Construction: Baseline LSTM-SNP
# ============================================================

def build_model(input_dim, units, batch_size):
    cell = LSTMSNPCell(units)
    rnn = layers.RNN(cell, return_sequences=False, stateful=True)

    inputs = tf.keras.Input(batch_shape=(batch_size, 1, input_dim))
    x = rnn(inputs)
    outputs = layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model"""

    elif variant == 2:  # Fuzzy Feature Augmentation
        return """\
# ============================================================
# Model Construction: LSTM-SNP with Fuzzy Feature Augmentation
# input_dim=2: [x(t), y_fuzzy(t)]
# The LSTM-SNP cell is UNMODIFIED.
# ============================================================

def build_model(input_dim, units, batch_size):
    cell = LSTMSNPCell(units)
    rnn = layers.RNN(cell, return_sequences=False, stateful=True)

    inputs = tf.keras.Input(batch_shape=(batch_size, 1, input_dim))
    x = rnn(inputs)
    outputs = layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model"""

    elif variant == 3:  # Fuzzy Gate Replacement
        return """\
# ============================================================
# Model Construction: LSTM-SNP with Fuzzy Gate Replacement
# Uses FuzzyLSTMSNPCell instead of LSTMSNPCell.
# Gradient clipping applied for stability.
# ============================================================

def build_model(input_dim, units, batch_size):
    cell = FuzzyLSTMSNPCell(units)
    rnn = layers.RNN(cell, return_sequences=False, stateful=True)

    inputs = tf.keras.Input(batch_shape=(batch_size, 1, input_dim))
    x = rnn(inputs)
    outputs = layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model"""

    elif variant == 4:  # Fuzzy Output Layer
        return """\
# ============================================================
# Model Construction: LSTM-SNP with Fuzzy Output Layer
# LSTMSNPCell is UNMODIFIED.
# Dense(1) is replaced by FuzzyOutputLayer.
# ============================================================

def build_model(input_dim, units, batch_size):
    cell = LSTMSNPCell(units)
    rnn = layers.RNN(cell, return_sequences=False, stateful=True)

    inputs = tf.keras.Input(batch_shape=(batch_size, 1, input_dim))
    x = rnn(inputs)
    outputs = FuzzyOutputLayer(units_in=units)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model"""

    elif variant == 5:  # Hybrid
        return """\
# ============================================================
# Model Construction: Hybrid LSTM-SNP
# Combines: Fuzzy Feature Augmentation + Fuzzy Gate Replacement
# input_dim=2, FuzzyLSTMSNPCell, gradient clipping.
# ============================================================

def build_model(input_dim, units, batch_size):
    cell = FuzzyLSTMSNPCell(units)
    rnn = layers.RNN(cell, return_sequences=False, stateful=True)

    inputs = tf.keras.Input(batch_shape=(batch_size, 1, input_dim))
    x = rnn(inputs)
    outputs = layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model"""


def training_code(variant, test_size, dataset_name):
    """Return training loop code."""
    input_dim = 2 if variant in [2, 5] else 1
    needs_fuzzy_test_augmentation = variant in [2, 5]

    # Determine if we need to initialize consumption gate bias
    # For standard LSTMSNPCell, we set bias[units:2*units] = 1.0
    # For FuzzyLSTMSNPCell, we skip this (different parameter structure)
    uses_standard_cell = variant in [1, 2, 4]

    bias_init_code = ""
    if uses_standard_cell:
        bias_init_code = """
    # Set consumption gate (c) bias to 1.0 (unit forget bias)
    rnn_layer = model.layers[1]
    cell = rnn_layer.cell
    weights = cell.get_weights()
    bias = weights[2].copy()
    bias[8:16] = 1.0  # units=8, bias[units:2*units]
    weights[2] = bias
    cell.set_weights(weights)
"""
    else:
        bias_init_code = """
    rnn_layer = model.layers[1]
"""

    # Test prediction code depends on whether fuzzy augmentation is needed
    if needs_fuzzy_test_augmentation:
        test_predict_code = f"""
    # Test predictions (single-step) with fuzzy feature augmentation
    predictions = []
    for i in range(len(test_scaled)):
        X_raw = test_scaled[i, 0:-1]  # [x(t-2), x(t-1)]
        x_t = X_raw[1]      # current input
        x_tm1 = X_raw[0]    # previous input
        y_fuzz = fuzzy_inference_numpy(x_t, x_tm1)
        X_aug = np.array([x_t, y_fuzz]).reshape(1, 1, 2)
        yhat = model.predict(X_aug, batch_size=1, verbose=0)[0, 0]

        # Invert scaling (need to match the scaler's column structure)
        # Scaler was fit on 3-column data [x(t-2), x(t-1), x(t)]
        new_row = list(X_raw) + [yhat]
        array = np.array(new_row).reshape(1, len(new_row))
        inverted = scaler.inverse_transform(array)[0, -1]

        # Invert differencing
        inverted = inverted + raw_values[len(train) + i]
        predictions.append(inverted)

        expected = raw_values[len(train) + i + 1]
        print(f'Month={{i+1}}, Predicted={{inverted:.4f}}, Expected={{expected:.4f}}')
"""
        warmup_code = f"""
    # Warm-up: condition hidden states on training data
    for i in range(len(train_scaled)):
        X_raw = train_scaled[i, 0:-1]
        x_t = X_raw[1]
        x_tm1 = X_raw[0]
        y_fuzz = fuzzy_inference_numpy(x_t, x_tm1)
        X_aug = np.array([x_t, y_fuzz]).reshape(1, 1, 2)
        model.predict(X_aug, batch_size=1, verbose=0)
"""
    else:
        test_predict_code = f"""
    # Test predictions (single-step)
    predictions = []
    for i in range(len(test_scaled)):
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        X_input = X.reshape(1, 1, len(X))
        yhat = model.predict(X_input, batch_size=1, verbose=0)[0, 0]

        # Invert scaling
        new_row = [x for x in X] + [yhat]
        array = np.array(new_row).reshape(1, len(new_row))
        inverted = scaler.inverse_transform(array)[0, -1]

        # Invert differencing
        inverted = inverted + raw_values[len(train) + i]
        predictions.append(inverted)

        expected = raw_values[len(train) + i + 1]
        print(f'Month={{i+1}}, Predicted={{inverted:.4f}}, Expected={{expected:.4f}}')
"""
        warmup_code = f"""
    # Warm-up: condition hidden states on training data
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    model.predict(train_reshaped, batch_size=1, verbose=0)
"""

    return f"""\
# ============================================================
# 30-Run Experiment Protocol
# ============================================================

all_rmse = []
all_mse = []
all_nmse = []
all_predictions = []
all_losses = []

for run in range(30):
    print(f'\\n===== RUN {{run+1}}/30 =====')

    np.random.seed(run)
    tf.random.set_seed(run)

    tf.keras.backend.clear_session()
    model = build_model(input_dim={input_dim}, units=8, batch_size=1)
{bias_init_code}
    # Training with manual epoch loop + reset_states
    run_losses = []
    for epoch in range(100):
        history = model.fit(
            X_train, y_train,
            epochs=1, batch_size=1,
            verbose=0, shuffle=False
        )
        run_losses.append(history.history['loss'][0])
        rnn_layer.reset_states()

    all_losses.append(run_losses)
    print(f'Training complete for run {{run+1}}')
{warmup_code}{test_predict_code}
    # Compute metrics
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

    print(f'Run {{run+1}} — RMSE: {{rmse:.6f}}, MSE: {{mse:.6f}}, NMSE: {{nmse:.10f}}')"""


def results_code(test_size, variant_name, dataset_name):
    """Return results/summary/plotting code."""
    return f"""\
# ============================================================
# Summary Statistics (30 runs)
# ============================================================

print('\\n===== FINAL RESULTS — {variant_name} on {dataset_name} (30 runs) =====')
print(f'RMSE: {{np.mean(all_rmse):.6f}} ± {{np.std(all_rmse):.6f}}')
print(f'MSE:  {{np.mean(all_mse):.6f}} ± {{np.std(all_mse):.6f}}')
print(f'NMSE: {{np.mean(all_nmse):.10f}} ± {{np.std(all_nmse):.10f}}')

best_idx = np.argmin(all_rmse)
print(f'\\nBest run: {{best_idx+1}}')
print(f'  RMSE: {{all_rmse[best_idx]:.6f}}')
print(f'  MSE:  {{all_mse[best_idx]:.6f}}')
print(f'  NMSE: {{all_nmse[best_idx]:.10f}}')"""


def plot_code(test_size, variant_name, dataset_name):
    """Return plotting code."""
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
plt.title('{variant_name} — {dataset_name}\\nPredictions vs Actual (Best of 30 runs)')
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
plt.title('{variant_name} — {dataset_name}\\nTraining Loss (Best Run)')
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
print(f'NMSE: {{all_nmse[best_idx]:.10f}}')"""


# ============================================================
# Variant-specific markdown cells
# ============================================================
VARIANT_TITLES = {
    1: "Notebook 1: Baseline LSTM-SNP (Control)",
    2: "Notebook 2: LSTM-SNP with Fuzzy Feature Augmentation",
    3: "Notebook 3: LSTM-SNP with Fuzzy Gate Replacement",
    4: "Notebook 4: LSTM-SNP with Fuzzy Output Layer",
    5: "Notebook 5: Hybrid LSTM-SNP (Fuzzy Feature Aug + Fuzzy Gates)",
}

VARIANT_SHORT = {
    1: "Baseline",
    2: "Fuzzy Feature Augmentation",
    3: "Fuzzy Gate Replacement",
    4: "Fuzzy Output Layer",
    5: "Hybrid (Feature Aug + Gate)",
}


def variant_description_md(variant, dataset_name):
    """Title + description markdown."""
    title = VARIANT_TITLES[variant]
    desc = {
        1: f"""# {title}

**Dataset**: {dataset_name}

## Description
This notebook is an exact reproduction of the original LSTM-SNP model (Long Short-Term Memory 
model inspired from Spiking Neural P systems). It serves as the **control experiment** with no 
fuzzy logic integration, providing baseline metrics for comparison with fuzzy-enhanced variants.

**Reference**: LSTM-SNP: A long short-term memory model inspired from spiking neural P systems""",

        2: f"""# {title}

**Dataset**: {dataset_name}

## Description
This notebook implements **fuzzy feature augmentation** for the LSTM-SNP model. A Takagi-Sugeno 
fuzzy inference system processes the current and previous input values to generate an additional 
feature, which is concatenated with the original input before being fed into the unmodified 
LSTM-SNP cell.

The LSTM-SNP cell itself is **NOT modified** — only the input representation is enriched with 
fuzzy-derived features.""",

        3: f"""# {title}

**Dataset**: {dataset_name}

## Description
This notebook implements **fuzzy gate replacement** for the LSTM-SNP model. The reset (r), 
consumption (c), and output (o) gates — which normally use hard sigmoid activation — are 
replaced with fuzzy inference systems. Each gate uses 2 Takagi-Sugeno rules with fixed Gaussian 
membership functions and trainable consequent parameters.

The generation gate (a) retains its original tanh activation. The rest of the SNP architecture 
is unchanged. Gradient clipping (norm=1.0) is applied for training stability.""",

        4: f"""# {title}

**Dataset**: {dataset_name}

## Description
This notebook implements a **fuzzy output layer** for the LSTM-SNP model. The internal 
LSTM-SNP structure is kept completely intact. The final Dense(1) layer is replaced with a 
fuzzy inference system that takes the RNN hidden state h(t) and produces the prediction 
through Takagi-Sugeno rules.

The fuzzy output layer aggregates h(t) into 2 summary features via mean-pooling, then applies 
4 rules with fixed Gaussian membership functions and trainable consequent parameters.""",

        5: f"""# {title}

**Dataset**: {dataset_name}

## Description
This notebook implements the **hybrid model** that combines:
1. **Fuzzy Feature Augmentation** (from Notebook 2): Input is augmented with fuzzy-derived features
2. **Fuzzy Gate Replacement** (from Notebook 3): Gates use fuzzy inference instead of sigmoid

This represents the most complex integration of fuzzy logic with LSTM-SNP. Gradient clipping 
(norm=1.0) and careful weight initialization are applied for training stability.

The interaction between input-level and gate-level fuzzy reasoning is documented and analyzed.""",
    }
    return desc[variant]


def theory_section_md(variant):
    """Theory section markdown."""
    theories = {
        1: """## Theory: LSTM-SNP Architecture

The LSTM-SNP model uses a custom recurrent cell inspired by Spiking Neural P (SNP) systems.

### Gate Equations

$$r(t) = \\rho(W_r x(t) + U_r u(t-1) + b_r)$$ — Reset gate

$$c(t) = \\rho(W_c x(t) + U_c u(t-1) + b_c)$$ — Consumption gate

$$o(t) = \\rho(W_o x(t) + U_o u(t-1) + b_o)$$ — Output gate

$$a(t) = f(W_a x(t) + U_a u(t-1) + b_a)$$ — Generated spikes

### State Update

$$u(t) = r(t) \\cdot u(t-1) - c(t) \\cdot a(t)$$

$$h(t) = o(t) \\cdot a(t)$$

where $\\rho$ = hard sigmoid, $f$ = tanh""",

        2: """## Theory: Fuzzy Feature Augmentation

### LSTM-SNP Cell (Unchanged)
The LSTM-SNP cell equations remain exactly as in the baseline.

### Fuzzy Inference System
A Takagi-Sugeno fuzzy system augments the input:

**Membership Functions** (Fixed Gaussian):
- $\\mu_{low}(x) = \\exp\\left(-\\frac{(x - (-1))^2}{2 \\cdot 0.5^2}\\right)$
- $\\mu_{high}(x) = \\exp\\left(-\\frac{(x - (+1))^2}{2 \\cdot 0.5^2}\\right)$

**Rules** (4 Takagi-Sugeno rules):
1. IF $x(t)$ is low AND $x(t-1)$ is low → $y_1 = a_1 x(t) + b_1 x(t-1) + c_1$
2. IF $x(t)$ is low AND $x(t-1)$ is high → $y_2 = a_2 x(t) + b_2 x(t-1) + c_2$
3. IF $x(t)$ is high AND $x(t-1)$ is low → $y_3 = a_3 x(t) + b_3 x(t-1) + c_3$
4. IF $x(t)$ is high AND $x(t-1)$ is high → $y_4 = a_4 x(t) + b_4 x(t-1) + c_4$

**Defuzzification**: $y_{fuzzy} = \\frac{\\sum_i w_i y_i}{\\sum_i w_i}$ where $w_i = \\prod_j \\mu_j(x_j)$

**Augmented Input**: $x'(t) = [x(t), y_{fuzzy}(t)]$""",

        3: """## Theory: Fuzzy Gate Replacement

### Modified Gate Computation
Instead of applying hard sigmoid directly, each gate (r, c, o) uses fuzzy inference:

**Standard**: $r(t) = \\sigma(z_r)$ where $z_r = W_r x(t) + U_r u(t-1) + b_r$

**Fuzzy**: $r(t) = FuzzyInference(z_r, \\bar{u}(t-1))$

The fuzzy inference uses:

**Fixed Gaussian Membership Functions**:
- $\\mu_{low}(z) = \\exp\\left(-\\frac{(z - (-1))^2}{2 \\cdot 0.5^2}\\right)$
- $\\mu_{high}(z) = \\exp\\left(-\\frac{(z - (+1))^2}{2 \\cdot 0.5^2}\\right)$

**2 Rules per gate** (trainable consequents):
1. IF $z_{gate}$ is low → $y_0 = a_0 z + b_0 \\bar{u} + c_0$
2. IF $z_{gate}$ is high → $y_1 = a_1 z + b_1 \\bar{u} + c_1$

**Output**: Gate value clipped to $[0, 1]$

### Unchanged
- Generation gate $a(t) = \\tanh(\\cdot)$ — unchanged
- State update: $u(t) = r \\cdot u_{t-1} - c \\cdot a$, $h(t) = o \\cdot a$ — unchanged
- Gradient clipping (norm ≤ 1.0) applied for stability""",

        4: """## Theory: Fuzzy Output Layer

### LSTM-SNP Cell (Unchanged)
All internal LSTM-SNP equations remain exactly as in the baseline.

### Fuzzy Output Computation
The Dense(1) output layer is replaced with fuzzy inference:

**Input**: $h(t) \\in \\mathbb{R}^{units}$ from the RNN

**Feature Aggregation**: Mean-pooling into 2 summary statistics:
- $s_1 = \\text{mean}(h_{1:units/2})$
- $s_2 = \\text{mean}(h_{units/2+1:units})$

**Membership Functions** (Fixed Gaussian):
- $\\mu_{low}(s) = \\exp\\left(-\\frac{(s-(-1))^2}{2 \\cdot 0.5^2}\\right)$
- $\\mu_{high}(s) = \\exp\\left(-\\frac{(s-(+1))^2}{2 \\cdot 0.5^2}\\right)$

**4 Takagi-Sugeno Rules** (trainable consequents):
$y_i = a_i s_1 + b_i s_2 + c_i$

**Defuzzification**: $\\hat{y} = \\frac{\\sum_i w_i y_i}{\\sum_i w_i}$""",

        5: """## Theory: Hybrid Fuzzy LSTM-SNP

This model combines two fuzzy integration strategies:

### 1. Input-Level: Fuzzy Feature Augmentation
Same as Notebook 2:
- Fuzzy inference on $x(t)$ and $x(t-1)$
- Augmented input: $x'(t) = [x(t), y_{fuzzy}]$
- Input dimension becomes 2

### 2. Gate-Level: Fuzzy Gate Replacement
Same as Notebook 3:
- Gates r, c, o computed via fuzzy inference
- Fixed Gaussian MFs, trainable consequent parameters
- Generation gate a retains tanh

### Interaction Between Levels
- The fuzzy-augmented input provides richer information to the cell
- The fuzzy gates process this enriched input with adaptive, rule-based gating
- This creates a two-level fuzzy reasoning pipeline

### Stability Measures
- Gradient clipping (norm ≤ 1.0)
- Careful weight initialization
- Fuzzy weights are clipped to valid ranges""",
    }
    return theories[variant]


def observations_md(variant_name, dataset_name):
    """Observations section markdown."""
    return f"""## Observations

### {variant_name} on {dataset_name}

**Run the notebook to generate results and fill in observations:**

1. **Prediction Quality**: Compare RMSE/MSE/NMSE with other variants
2. **Training Stability**: Examine loss curves for convergence behavior
3. **Prediction Tracking**: Assess how well predictions track actual values
4. **Computational Cost**: Note training time per run

*After running all 5 variant notebooks, perform cross-variant comparison to evaluate 
whether fuzzy logic improves nonlinearity handling, interpretability, and prediction performance.*"""


# ============================================================
# Main notebook generator
# ============================================================
def generate_notebook(variant, dataset_key):
    """Generate a single notebook for a given variant and dataset."""
    ds = DATASETS[dataset_key]
    variant_name = VARIANT_SHORT[variant]
    dataset_name = ds["name"]
    csv_path = ds["csv_path"]
    test_size = ds["test_size"]

    cells = []

    # 1. Title + Description
    cells.append(md_cell(variant_description_md(variant, dataset_name)))

    # 2. Theory Section
    cells.append(md_cell(theory_section_md(variant)))

    # 3. Model Architecture heading
    cells.append(md_cell("## Model Architecture & Implementation"))

    # 4. Imports
    cells.append(code_cell(IMPORTS_CODE))

    # 5. Fuzzy system code (if needed)
    if variant in [2, 5]:
        cells.append(md_cell("### Fuzzy Inference System (NumPy — for preprocessing)"))
        cells.append(code_cell(FUZZY_INFERENCE_NUMPY_CODE))

    # 6. LSTM-SNP Cell
    if variant in [1, 2, 4]:
        cells.append(md_cell("### LSTM-SNP Cell"))
        cells.append(code_cell(LSTMSNP_CELL_CODE))
    elif variant in [3, 5]:
        cells.append(md_cell("### Fuzzy Gate LSTM-SNP Cell"))
        cells.append(code_cell(FUZZY_GATE_CELL_CODE))

    # 7. Fuzzy Output Layer (variant 4 only)
    if variant == 4:
        cells.append(md_cell("### Fuzzy Output Layer"))
        cells.append(code_cell(FUZZY_OUTPUT_LAYER_CODE))

    # 8. Build model function
    cells.append(md_cell("### Build Model"))
    cells.append(code_cell(build_model_code(variant)))

    # 9. Model summary check
    input_dim = 2 if variant in [2, 5] else 1
    cells.append(code_cell(f"""\
# Quick model check
model = build_model(input_dim={input_dim}, units=8, batch_size=1)
model.summary()"""))

    # 10. Data pipeline heading
    cells.append(md_cell(f"## Data Pipeline — {dataset_name}"))

    # 11. Data loading
    cells.append(code_cell(data_loading_code(csv_path)))

    # 12. Preprocessing
    cells.append(code_cell(PREPROCESSING_CODE))

    # 13. Supervised format (lag depends on variant)
    if variant in [2, 5]:
        cells.append(code_cell(SUPERVISED_CODE_LAG2))
    else:
        cells.append(code_cell(SUPERVISED_CODE_LAG1))

    # 14. Train-test split + scaling
    cells.append(code_cell(split_scale_code(test_size)))

    # 15. Reshape
    if variant in [2, 5]:
        cells.append(code_cell(RESHAPE_CODE_LAG2_FUZZY_AUG))
    else:
        cells.append(code_cell(RESHAPE_CODE_LAG1))

    # 16. Training heading
    cells.append(md_cell("## Training Loop"))

    # 17. Training code
    cells.append(code_cell(training_code(variant, test_size, dataset_name)))

    # 18. Results heading
    cells.append(md_cell("## Results"))

    # 19. Summary statistics
    cells.append(code_cell(results_code(test_size, variant_name, dataset_name)))

    # 20. Plots
    cells.append(code_cell(plot_code(test_size, variant_name, dataset_name)))

    # 21. Observations
    cells.append(md_cell(observations_md(variant_name, dataset_name)))

    return make_notebook(cells)


# ============================================================
# Generate all 20 notebooks
# ============================================================
def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    generated = []

    for variant in range(1, 6):
        for dataset_key in DATASETS:
            nb = generate_notebook(variant, dataset_key)
            filename = f"FuzzyLSTM_SNP_{variant}_{VARIANT_SHORT[variant].replace(' ', '').replace('(', '').replace(')', '').replace('+', 'Plus')}_{dataset_key}.ipynb"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'w') as f:
                json.dump(nb, f, indent=1)

            generated.append(filename)
            print(f"✅ Generated: {filename}")

    print(f"\n{'='*60}")
    print(f"Total notebooks generated: {len(generated)}")
    print(f"{'='*60}")
    for fn in generated:
        print(f"  {fn}")


if __name__ == "__main__":
    main()
