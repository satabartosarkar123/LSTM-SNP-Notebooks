#!/usr/bin/env python3
"""
Mamba (Selective State Space Model) for Time Series Forecasting.

Uses the EXACT same preprocessing pipeline as the FLSTM notebooks:
  - First-order differencing
  - Supervised learning transformation (lag=1)
  - MinMaxScaler (-1, 1)
  - Train/Test split: last 60 observations for test
  - Epoch-wise training with state reset
  - Inverse-difference predictions for RMSE/MSE/NMSE

Mamba architecture based on:
  Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)

Implementation: Single MambaSSMCell (selective SSM with input-dependent B, C, Delta)
wrapped in layers.RNN — same pattern as MembershipFLSTMSNPCell in the FLSTM notebooks.

Datasets: Dow Jones, Lake Erie, Milk Production, S&P 500
Runs: 2 per dataset
"""

import os
import numpy as np
import pandas as pd
import sys
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import time

# ============================================================
# Force CPU execution (same as FLSTM notebooks)
# ============================================================
try:
    tf.config.set_visible_devices([], 'GPU')
    print('Forcing CPU execution (disabled GPU visibility).')
except RuntimeError as e:
    print(e)

# ============================================================
# Configuration — matches FLSTM notebooks exactly
# ============================================================
NUM_RUNS = 30
NUM_EPOCHS = 100
TEST_SPLIT = 60
LAG = 1

# Mamba-specific hyperparameters
D_MODEL = 8       # Inner model dimension (matches FLSTM units=8)
D_STATE = 16      # SSM state dimension N
EXPAND = 2        # Expansion factor E

DATASETS = {
    'dow_jones': {
        'csv_path': '../content/monthly-closings-of-the-dowjones.csv',
        'display_name': 'Dow Jones',
    },
    'lake_erie': {
        'csv_path': '../content/monthly-lake-erie-levels-1921-19.csv',
        'display_name': 'Lake Erie',
    },
    'milk_production': {
        'csv_path': '../content/monthly-milk-production-pounds-p.csv',
        'display_name': 'Milk Production',
    },
    'sp500': {
        'csv_path': '../content/sp500.csv',
        'display_name': 'S&P 500',
    },
}


# ============================================================
# Mamba Selective SSM Cell (single-step RNN cell for layers.RNN)
# ============================================================
#
# This cell implements the full Mamba block logic for ONE timestep:
#
#   x → in_proj → split(x_main, z)
#   x_main → Dense(conv surrogate) → SiLU → Selective SSM recurrence
#   z → SiLU
#   output = (SSM_out * z_gate) → out_proj
#
# The selective SSM uses input-dependent Delta, B, C (Algorithm 2 in paper).
# State carries: the SSM hidden state h of shape (D_inner * D_state,)
#
@tf.keras.utils.register_keras_serializable()
class MambaSSMCell(layers.Layer):
    """
    Combined Mamba block as a single RNN cell, compatible with layers.RNN.
    
    Implements:
      - Input projection (expand)
      - Conv1D surrogate (Dense)
      - SiLU activation  
      - Selective SSM (S6) with input-dependent Delta, B, C
      - Gated output (multiply with SiLU-activated gate branch)
      - Output projection (contract)
    
    State: SSM hidden state h, flattened to (d_inner * d_state,)
    """
    def __init__(self, d_model, d_state=16, expand=2, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = d_model * expand
        
        # RNN cell interface
        self.state_size = self.d_inner * self.d_state
        self.output_size = d_model

    def build(self, input_shape):
        D = self.d_inner
        N = self.d_state
        
        # --- Input projection: project to 2*d_inner (main + gate branches) ---
        self.W_in = self.add_weight(
            shape=(input_shape[-1], D * 2),
            initializer='glorot_uniform', name='W_in'
        )
        self.b_in = self.add_weight(
            shape=(D * 2,), initializer='zeros', name='b_in'
        )
        
        # --- Conv1D surrogate (Dense) for main branch ---
        self.W_conv = self.add_weight(
            shape=(D, D), initializer='glorot_uniform', name='W_conv'
        )
        self.b_conv = self.add_weight(
            shape=(D,), initializer='zeros', name='b_conv'
        )
        
        # --- Selective SSM parameters ---
        # A: diagonal state matrix, learned in log-space (S4D-Real init)
        init_A_1d = np.log(np.arange(1, N + 1, dtype=np.float32))
        init_A_2d = np.tile(-init_A_1d, (D, 1))  # (D, N)
        self.log_A = self.add_weight(
            shape=(D, N),
            initializer=tf.constant_initializer(init_A_2d),
            name='log_A'
        )
        
        # s_B(x) = Linear_N(x): input → B
        self.W_B = self.add_weight(
            shape=(D, N), initializer='glorot_uniform', name='W_B'
        )
        
        # s_C(x) = Linear_N(x): input → C  
        self.W_C = self.add_weight(
            shape=(D, N), initializer='glorot_uniform', name='W_C'
        )
        
        # s_Delta(x): input → Delta (controls gate-like focus/ignore)
        self.W_delta = self.add_weight(
            shape=(D, D), initializer='glorot_uniform', name='W_delta'
        )
        # Delta bias: initialized as softplus^{-1}(Uniform([0.001, 0.1]))
        delta_bias_vals = np.log(np.exp(
            np.random.RandomState(42).uniform(0.001, 0.1, size=(D,)).astype(np.float32)
        ) - 1.0)
        self.b_delta = self.add_weight(
            shape=(D,),
            initializer=tf.constant_initializer(delta_bias_vals),
            name='b_delta'
        )
        
        # --- Output projection: d_inner → d_model ---
        self.W_out = self.add_weight(
            shape=(D, self.d_model),
            initializer='glorot_uniform', name='W_out'
        )
        self.b_out = self.add_weight(
            shape=(self.d_model,), initializer='zeros', name='b_out'
        )

    def call(self, inputs, states):
        """
        inputs: (batch, input_dim)  — one timestep
        states: [(batch, d_inner * d_state)]  — flattened SSM hidden state
        """
        D = self.d_inner
        N = self.d_state
        
        h = tf.reshape(states[0], (-1, D, N))  # (B, D, N)
        
        # === Input projection: split into main branch + gate branch ===
        xz = tf.matmul(inputs, self.W_in) + self.b_in  # (B, 2*D)
        x_main = xz[:, :D]   # (B, D) — main branch
        z = xz[:, D:]         # (B, D) — gate branch
        
        # === Main branch: conv surrogate → SiLU ===
        x_main = tf.matmul(x_main, self.W_conv) + self.b_conv  # (B, D)
        x_main = tf.nn.silu(x_main)  # SiLU / Swish activation
        
        # === Selective SSM recurrence (Algorithm 2) ===
        # Compute input-dependent parameters (Selection Mechanism)
        
        # Delta: controls how much to focus on current input vs persist state
        delta = tf.nn.softplus(tf.matmul(x_main, self.W_delta) + self.b_delta)  # (B, D)
        
        # B: input-dependent, controls what enters the state
        B = tf.matmul(x_main, self.W_B)  # (B, N)
        
        # C: input-dependent, controls what exits the state
        C = tf.matmul(x_main, self.W_C)  # (B, N)
        
        # Discretize: A_bar = exp(Delta * A)
        A = -tf.exp(self.log_A)  # (D, N), negative for stability
        delta_exp = tf.expand_dims(delta, -1)  # (B, D, 1)
        A_bar = tf.exp(delta_exp * A)  # (B, D, N)
        
        # B_bar = Delta * B
        B_exp = tf.expand_dims(B, 1)    # (B, 1, N)
        B_bar = delta_exp * B_exp       # (B, D, N)
        
        # SSM recurrence: h_t = A_bar * h_{t-1} + B_bar * x_t
        x_exp = tf.expand_dims(x_main, -1)  # (B, D, 1)
        h_new = A_bar * h + B_bar * x_exp   # (B, D, N)
        
        # Output: y_t = sum(C * h_t, axis=-1) → (B, D)
        C_exp = tf.expand_dims(C, 1)      # (B, 1, N)
        y_ssm = tf.reduce_sum(h_new * C_exp, axis=-1)  # (B, D)
        
        # === Gate branch: SiLU ===
        z_gate = tf.nn.silu(z)  # (B, D)
        
        # === Multiply branches (gated output) ===
        y = y_ssm * z_gate  # (B, D)
        
        # === Output projection: d_inner → d_model ===
        output = tf.matmul(y, self.W_out) + self.b_out  # (B, d_model)
        
        # Flatten state for RNN interface
        h_flat = tf.reshape(h_new, (-1, D * N))
        
        return output, [h_flat]

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'd_state': self.d_state,
            'expand': self.expand,
        })
        return config


# ============================================================
# Model builder — same pattern as FLSTM notebooks
# ============================================================
def build_mamba_model(input_dim, d_model=D_MODEL, d_state=D_STATE,
                      expand=EXPAND, batch_size=1):
    """
    Build Mamba model for time series forecasting.
    
    Architecture (matches FLSTM notebook pattern):
      Input → RNN(MambaSSMCell) → Dense(1)
    
    The MambaSSMCell internally does:
      in_proj → conv → SiLU → Selective SSM → gate → out_proj
    """
    cell = MambaSSMCell(d_model, d_state=d_state, expand=expand)
    rnn = layers.RNN(cell, return_sequences=False, stateful=True)
    
    inputs = tf.keras.Input(batch_shape=(batch_size, 1, input_dim))
    x = rnn(inputs)           # (B, d_model)
    outputs = layers.Dense(1)(x)  # (B, 1)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss='mean_squared_error'
    )
    return model


# ============================================================
# Preprocessing — EXACT same as FLSTM notebooks
# ============================================================
def difference(dataset, interval=1):
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)


def timeseries_to_supervised(data, lag):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df.values


# ============================================================
# Run experiment for one dataset
# ============================================================
def run_experiment(ds_key, ds_config):
    print(f"\n{'='*80}")
    print(f"  MAMBA SSM — {ds_config['display_name']}")
    print(f"  Runs: {NUM_RUNS}, Epochs: {NUM_EPOCHS}, d_model: {D_MODEL}")
    print(f"  SSM State Dim: {D_STATE}, Expand: {EXPAND}")
    print(f"{'='*80}\n")
    
    # Load data
    series = pd.read_csv(ds_config['csv_path'], header=0, parse_dates=[0], index_col=0)
    raw_values = series.values.flatten()
    
    # Differencing
    diff_values = difference(raw_values, 1)
    
    # Supervised learning transformation
    supervised = timeseries_to_supervised(diff_values, LAG)
    train, test = supervised[:-TEST_SPLIT], supervised[-TEST_SPLIT:]
    
    # Scaling
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    
    X_train_raw, y_train = train_scaled[:, 0:-1], train_scaled[:, -1]
    X_test_raw, y_test = test_scaled[:, 0:-1], test_scaled[:, -1]
    
    # Reshape for RNN: (samples, 1, features)
    X_train = X_train_raw.reshape((X_train_raw.shape[0], 1, X_train_raw.shape[1]))
    X_test = X_test_raw.reshape((X_test_raw.shape[0], 1, X_test_raw.shape[1]))
    
    input_dim = X_train.shape[2]
    
    all_rmse, all_mse, all_nmse = [], [], []
    all_predictions = []
    all_losses = []
    
    for run in range(NUM_RUNS):
        print(f'\n===== RUN {run+1}/{NUM_RUNS} =====')
        run_start = time.time()
        
        np.random.seed(run)
        tf.random.set_seed(run)
        tf.keras.backend.clear_session()
        
        model = build_mamba_model(input_dim=input_dim, batch_size=1)
        
        if run == 0:
            model.summary()
        
        # Get RNN layer for state reset
        rnn_layer = model.layers[1]
        
        run_losses = []
        for epoch in range(NUM_EPOCHS):
            history = model.fit(X_train, y_train, epochs=1, batch_size=1,
                              verbose=0, shuffle=False)
            loss_val = history.history['loss'][0]
            run_losses.append(loss_val)
            # Reset states after each epoch (same as FLSTM)
            rnn_layer.reset_states()
            # Progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'  Epoch {epoch+1:3d}/{NUM_EPOCHS} — Loss: {loss_val:.6f}')
                sys.stdout.flush()
        all_losses.append(run_losses)
        
        # Warmup: pass training data through
        print('  Warmup...', end=' ')
        sys.stdout.flush()
        for i in range(len(X_train)):
            model.predict(X_train[i:i+1], batch_size=1, verbose=0)
        print('done.')
        sys.stdout.flush()
        
        # Predict on test set
        predictions = []
        for i in range(len(X_test)):
            yhat = model.predict(X_test[i:i+1], batch_size=1, verbose=0)
            
            row = list(X_test_raw[i]) + [yhat[0, 0]]
            inv = scaler.inverse_transform([row])[0, -1] + raw_values[len(train) + i]
            predictions.append(inv)
        
        actual = raw_values[-TEST_SPLIT:]
        rmse = sqrt(mean_squared_error(actual, predictions))
        mse = mean_squared_error(actual, predictions)
        meanV = np.mean(actual)
        dominator = np.linalg.norm(np.array(predictions) - meanV, 2)
        nmse = mse / np.power(dominator, 2)
        
        all_rmse.append(rmse)
        all_mse.append(mse)
        all_nmse.append(nmse)
        all_predictions.append(predictions)
        
        run_time = time.time() - run_start
        print(f'Run {run+1} — RMSE: {rmse:.6f}, MSE: {mse:.6f}, '
              f'NMSE: {nmse:.10f}, Time: {run_time:.1f}s')
    
    # Summary
    mean_rmse = np.mean(all_rmse)
    std_rmse = np.std(all_rmse)
    mean_mse = np.mean(all_mse)
    std_mse = np.std(all_mse)
    mean_nmse = np.mean(all_nmse)
    std_nmse = np.std(all_nmse)
    best_idx = np.argmin(all_rmse)
    
    print(f'\n===== FINAL RESULTS — Mamba SSM on {ds_config["display_name"]} ({NUM_RUNS} runs) =====')
    print(f'RMSE: {mean_rmse:.6f} ± {std_rmse:.6f}')
    print(f'MSE:  {mean_mse:.6f} ± {std_mse:.6f}')
    print(f'NMSE: {mean_nmse:.10f} ± {std_nmse:.10f}')
    print(f'\nBest run: {best_idx+1}')
    print(f'  RMSE: {all_rmse[best_idx]:.6f}')
    print(f'  MSE:  {all_mse[best_idx]:.6f}')
    print(f'  NMSE: {all_nmse[best_idx]:.10f}')
    
    return {
        'dataset': ds_config['display_name'],
        'mean_rmse': mean_rmse, 'std_rmse': std_rmse,
        'mean_mse': mean_mse, 'std_mse': std_mse,
        'mean_nmse': mean_nmse, 'std_nmse': std_nmse,
        'best_rmse': all_rmse[best_idx],
        'best_mse': all_mse[best_idx],
        'best_nmse': all_nmse[best_idx],
        'all_rmse': all_rmse,
        'all_mse': all_mse,
        'all_nmse': all_nmse,
    }


# ============================================================
# Main: Run all 4 datasets and print consolidated table
# ============================================================
if __name__ == '__main__':
    print("=" * 80)
    print("  MAMBA (Selective State Space Model) — Time Series Forecasting")
    print("  Same preprocessing pipeline as FLSTM notebooks")
    print(f"  Config: runs={NUM_RUNS}, epochs={NUM_EPOCHS}, d_model={D_MODEL}, "
          f"d_state={D_STATE}, expand={EXPAND}")
    print("=" * 80)
    
    total_start = time.time()
    results = []
    
    for ds_key, ds_config in DATASETS.items():
        res = run_experiment(ds_key, ds_config)
        results.append(res)
    
    total_time = time.time() - total_start
    
    # ============================================================
    # CONSOLIDATED RESULTS TABLE
    # ============================================================
    print("\n\n")
    print("=" * 120)
    print("  CONSOLIDATED RESULTS — MAMBA SSM (Selective State Space Model)")
    print(f"  Runs per dataset: {NUM_RUNS} | Epochs: {NUM_EPOCHS} | "
          f"d_model: {D_MODEL} | d_state: {D_STATE} | expand: {EXPAND}")
    print("=" * 120)
    
    # Header
    header = (f"{'Dataset':<20} | {'Mean RMSE':>14} | {'Std RMSE':>12} | "
              f"{'Mean MSE':>14} | {'Std MSE':>12} | "
              f"{'Mean NMSE':>16} | {'Std NMSE':>14} | "
              f"{'Best RMSE':>14} | {'Best MSE':>14} | {'Best NMSE':>16}")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for r in results:
        row = (f"{r['dataset']:<20} | {r['mean_rmse']:>14.6f} | {r['std_rmse']:>12.6f} | "
               f"{r['mean_mse']:>14.6f} | {r['std_mse']:>12.6f} | "
               f"{r['mean_nmse']:>16.10f} | {r['std_nmse']:>14.10f} | "
               f"{r['best_rmse']:>14.6f} | {r['best_mse']:>14.6f} | {r['best_nmse']:>16.10f}")
        print(row)
    
    print("-" * len(header))
    
    # Also print individual run details
    print(f"\n{'Dataset':<20} | {'Run':>4} | {'RMSE':>14} | {'MSE':>14} | {'NMSE':>16}")
    print("-" * 80)
    for r in results:
        for i in range(NUM_RUNS):
            print(f"{r['dataset']:<20} | {i+1:>4} | {r['all_rmse'][i]:>14.6f} | "
                  f"{r['all_mse'][i]:>14.6f} | {r['all_nmse'][i]:>16.10f}")
        print("-" * 80)
    
    print(f"\nTotal execution time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("Done.")
