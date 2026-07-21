#!/usr/bin/env python3
"""
Generator script for WM-FLSTM (Wang-Mendel Fuzzy Inference-based LSTM) notebooks.

Replaces the TSK interfacing from Modified_Fuzzy_MF notebooks with the
Wang-Mendel fuzzy prediction fusion from the FLSTM paper
(doi:10.1038/s41598-023-47812-3).

Creates 4 folders (one per dataset) with 9 notebooks each (3 MF types x 3 variants).
Each notebook runs at 4 Gaussian noise levels: 0%, 0.5%, 5%, 10%.
Combined results are shown at the end of each notebook.
"""

import json
import os

# ============================================================================
# Configuration
# ============================================================================

DATASETS = {
    'dow_jones': {
        'csv_path': '../content/monthly-closings-of-the-dowjones.csv',
        'display_name': 'Dow Jones',
        'folder_suffix': 'DowJones',
    },
    'lake_erie': {
        'csv_path': '../content/monthly-lake-erie-levels-1921-19.csv',
        'display_name': 'Lake Erie',
        'folder_suffix': 'LakeErie',
    },
    'milk_production': {
        'csv_path': '../content/monthly-milk-production-pounds-p.csv',
        'display_name': 'Milk Production',
        'folder_suffix': 'MilkProduction',
    },
    'sp500': {
        'csv_path': '../content/sp500.csv',
        'display_name': 'SP500',
        'folder_suffix': 'SP500',
    },
}

MF_TYPES = {
    'DOG': 'dog',
    'SG': 'sg',
    'GB': 'gb',
}

VARIANTS = ['Gates_Only', 'Gates_Plus_FuzzyInput', 'Gates_Plus_FuzzyOutput']

FOLDER_PREFIX = 'WM_FLSTM_Fuzzy_MF'
NB_PREFIX = 'WM_FLSTM_MF'

NUM_RUNS = 30
NUM_EPOCHS = 100
UNITS = 8
Q_REGIONS = 5
TEST_SPLIT = 60


# ============================================================================
# Cell builders
# ============================================================================

def make_markdown_cell(source_text):
    lines = source_text.split('\n')
    source = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def make_code_cell(source_text):
    lines = source_text.split('\n')
    source = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


# ============================================================================
# Cell content generators
# ============================================================================

def cell_title(variant_display, mf_display, dataset_display):
    return make_markdown_cell(
        f"# WM-FLSTM {variant_display} using {mf_display}\\nDataset: {dataset_display}\\n\\n"
        f"**Interfacing**: Wang-Mendel Fuzzy Prediction Fusion (doi:10.1038/s41598-023-47812-3)\\n"
        f"**Noise Levels**: 0% (clean), 0.5%, 5%, 10% Gaussian"
    )


def cell_pid():
    return make_code_cell(
        "# ============================================================\n"
        "# PROCESS IDENTIFICATION\n"
        "# ============================================================\n"
        "import os\n"
        'print(f"Process ID (PID): {os.getpid()}")\n'
    )


def cell_timer_start():
    return make_code_cell(
        "# ============================================================\n"
        "# NOTEBOOK TIMER — START\n"
        "# ============================================================\n"
        "import time as _timer_module\n"
        "_NOTEBOOK_START_TIME = _timer_module.time()\n"
        "print(f\"Notebook execution started at: {_timer_module.strftime('%Y-%m-%d %H:%M:%S')}\")\n"
    )


def cell_cpu_only():
    return make_code_cell(
        "# ============================================================\n"
        "# CPU ONLY Settings (Forced)\n"
        "# ============================================================\n"
        "import tensorflow as tf\n"
        "import platform\n"
        "\n"
        "try:\n"
        "    tf.config.set_visible_devices([], 'GPU')\n"
        "    print('Forcing CPU execution (disabled GPU visibility).')\n"
        "except RuntimeError as e:\n"
        "    print(e)\n"
    )


def cell_imports():
    return make_code_cell(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import tensorflow as tf\n"
        "from tensorflow.keras import layers\n"
        "from tensorflow.keras import Model\n"
        "from sklearn.preprocessing import MinMaxScaler\n"
        "from sklearn.metrics import mean_squared_error\n"
        "from math import sqrt\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "\n"
        "\n"
    )


def cell_wang_mendel_fuzzy():
    """Wang-Mendel fuzzy rule base with triangular MFs (NumPy, for preprocessing)."""
    return make_code_cell(
        "# ============================================================\n"
        "# Wang-Mendel Fuzzy Rule Base with Triangular MFs\n"
        "# (FLSTM paper: doi:10.1038/s41598-023-47812-3)\n"
        "#\n"
        "# Key equations implemented:\n"
        "#   Eq 7:  Rule extraction — fuzzify each feature to highest-MF set\n"
        "#   Eq 8:  Rule weight W_i = product of membership degrees\n"
        "#   Eq 9:  Center-average defuzzification: y_hat = sum(W_i * y_bar_i) / sum(W_i)\n"
        "#   Eq 10: Conflict rule resolution via center-average\n"
        "#   Eq 11: Complement rules — average central value of t-group\n"
        "#   Eq 12: Extrapolating rule generation\n"
        "#   Eq 13: Fuzzy rule-based prediction\n"
        "# ============================================================\n"
        f"Q_REGIONS = {Q_REGIONS}\n"
        "\n"
        "def triangular_mf_np(x, a, b, c):\n"
        '    """Triangular membership function: vertices at (a, 0), (b, 1), (c, 0)."""\n'
        "    return np.maximum(0.0, np.minimum((x - a) / (b - a + 1e-10),\n"
        "                                      (c - x) / (c - b + 1e-10)))\n"
        "\n"
        "def build_triangular_fuzzy_sets(data_min, data_max, q=Q_REGIONS):\n"
        '    """Build q equally-spaced triangular fuzzy sets over [data_min, data_max]."""\n'
        "    centers = np.linspace(data_min, data_max, q)\n"
        "    step = centers[1] - centers[0] if q > 1 else 1.0\n"
        "    fuzzy_sets = []\n"
        "    for center in centers:\n"
        "        fuzzy_sets.append((center - step, center, center + step))\n"
        "    return fuzzy_sets, centers\n"
        "\n"
        "def fuzzify_value(value, fuzzy_sets):\n"
        '    """Return (best_set_index, membership_degree) for a single value.\n'
        '    Assigns value to the fuzzy set with highest membership degree (Eq 7)."""\n'
        "    best_idx, best_mu = 0, 0.0\n"
        "    for i, (a, b, c) in enumerate(fuzzy_sets):\n"
        "        mu = triangular_mf_np(value, a, b, c)\n"
        "        if mu > best_mu:\n"
        "            best_mu = mu\n"
        "            best_idx = i\n"
        "    return best_idx, max(best_mu, 1e-10)\n"
        "\n"
        "def extract_wm_rules(X, y, fs_x_list, fs_y):\n"
        '    """Extract fuzzy rules using the Wang-Mendel method (Eq 7-8).\n'
        '    X: (N, n_features), y: (N,)\n'
        '    fs_x_list: list of fuzzy_sets (one per feature)\n'
        '    fs_y: fuzzy_sets for output\n'
        '    Returns dict: tuple(antecedent_indices) -> (consequent_index, weight)\n'
        '    \n'
        '    Handles redundant rules (same antecedent+consequent): keep one.\n'
        '    Handles conflict rules (same antecedent, different consequent): keep highest weight.\n'
        '    """\n'
        "    rules = {}\n"
        "    n_features = X.shape[1]\n"
        "    for i in range(len(X)):\n"
        "        antecedents = []\n"
        "        weight = 1.0\n"
        "        for j in range(n_features):\n"
        "            idx, mu = fuzzify_value(X[i, j], fs_x_list[j])\n"
        "            antecedents.append(idx)\n"
        "            weight *= mu  # Eq 8: W_i = product of membership degrees\n"
        "        cons_idx, cons_mu = fuzzify_value(y[i], fs_y)\n"
        "        weight *= cons_mu\n"
        "        key = tuple(antecedents)\n"
        "        if key not in rules or weight > rules[key][1]:\n"
        "            rules[key] = (cons_idx, weight)\n"
        "    return rules\n"
        "\n"
        "def resolve_conflict_rules(raw_rules, centers_y, fs_y):\n"
        '    """Resolve conflict rules using center-average defuzzification (Eq 9-10).\n'
        '    raw_rules: dict from extract step, maps antecedent -> list of (cons_idx, weight)\n'
        '    Returns cleaned dict: antecedent -> (cons_idx, weight)\n'
        '    """\n'
        "    # Already handled in extract_wm_rules by keeping highest weight\n"
        "    return raw_rules\n"
        "\n"
        "def complement_wm_rules(rules, q, n_features, centers_y):\n"
        '    """Complement fuzzy rule base for uncovered regions (Eq 11-12).\n'
        '    For each missing antecedent combination, find the t-group\n'
        '    (rules differing in t antecedents) and compute average central value.\n'
        '    """\n'
        "    import itertools\n"
        "    all_combos = list(itertools.product(range(q), repeat=n_features))\n"
        "    for combo in all_combos:\n"
        "        if combo not in rules:\n"
        "            # Find t-group: rules differing in exactly t antecedents\n"
        "            for t in range(1, n_features + 1):\n"
        "                t_group = []\n"
        "                for existing_key, (cons, w) in rules.items():\n"
        "                    diff_count = sum(1 for a, b in zip(combo, existing_key) if a != b)\n"
        "                    if diff_count == t:\n"
        "                        t_group.append((cons, w))\n"
        "                if t_group:  # Found the first non-empty t-group\n"
        "                    # Eq 11: average central value\n"
        "                    avg_y = np.mean([centers_y[cons] for cons, _ in t_group])\n"
        "                    # Find fuzzy set with max membership on avg_y\n"
        "                    best_cons = 0\n"
        "                    best_mu = 0.0\n"
        "                    for k in range(q):\n"
        "                        mu = triangular_mf_np(avg_y, *[(centers_y[k] - (centers_y[1]-centers_y[0]) if q > 1 else centers_y[k] - 1,\n"
        "                                                         centers_y[k],\n"
        "                                                         centers_y[k] + (centers_y[1]-centers_y[0]) if q > 1 else centers_y[k] + 1)][0])\n"
        "                        if mu > best_mu:\n"
        "                            best_mu = mu\n"
        "                            best_cons = k\n"
        "                    rules[combo] = (best_cons, 0.1)  # Eq 12\n"
        "                    break\n"
        "            else:\n"
        "                # Fallback: assign middle region\n"
        "                rules[combo] = (q // 2, 0.01)\n"
        "    return rules\n"
        "\n"
        "def wm_fuzzy_predict(x_features, rules, fs_x_list, centers_y, q=Q_REGIONS):\n"
        '    """Predict using fuzzy rule base with center-average defuzzification (Eq 9, 13).\n'
        '    x_features: 1D array of input features for one sample.\n'
        '    Returns: scalar prediction (center of matched fuzzy set).\n'
        '    """\n'
        "    n_features = len(x_features)\n"
        "    # Compute membership degrees for each feature across all q regions\n"
        "    memberships = []\n"
        "    for j in range(n_features):\n"
        "        mf_vals = []\n"
        "        for k in range(q):\n"
        "            a, b, c = fs_x_list[j][k]\n"
        "            mf_vals.append(triangular_mf_np(x_features[j], a, b, c))\n"
        "        memberships.append(mf_vals)\n"
        "    # Weighted average defuzzification over all rules (Eq 9)\n"
        "    total_weight = 0.0\n"
        "    weighted_sum = 0.0\n"
        "    for antecedents, (consequent, _) in rules.items():\n"
        "        strength = 1.0\n"
        "        for j, ant_idx in enumerate(antecedents):\n"
        "            strength *= memberships[j][ant_idx]\n"
        "        if strength > 1e-10:\n"
        "            weighted_sum += strength * centers_y[consequent]\n"
        "            total_weight += strength\n"
        "    if total_weight > 1e-10:\n"
        "        return weighted_sum / total_weight\n"
        "    return 0.0\n"
        "\n"
        "def build_wm_system(X_train, y_train, q=Q_REGIONS):\n"
        '    """Build complete Wang-Mendel fuzzy system from training data.\n'
        '    Returns: (rules, fs_x_list, centers_y)\n'
        '    """\n'
        "    n_features = X_train.shape[1]\n"
        "    fs_x_list = []\n"
        "    for j in range(n_features):\n"
        "        col = X_train[:, j]\n"
        "        fs, _ = build_triangular_fuzzy_sets(col.min(), col.max(), q)\n"
        "        fs_x_list.append(fs)\n"
        "    fs_y, centers_y = build_triangular_fuzzy_sets(y_train.min(), y_train.max(), q)\n"
        "    rules = extract_wm_rules(X_train, y_train, fs_x_list, fs_y)\n"
        "    rules = complement_wm_rules(rules, q, n_features, centers_y)\n"
        "    print(f'WM Fuzzy System: {len(rules)} rules (q={q}, features={n_features})')\n"
        "    return rules, fs_x_list, centers_y\n"
        "\n"
        "def compute_fuzzy_predictions(X, rules, fs_x_list, centers_y, q=Q_REGIONS):\n"
        '    """Compute Wang-Mendel fuzzy predictions for an array of samples.\n'
        '    Returns: 1D array of fuzzy prediction values r_t.\n'
        '    """\n'
        "    preds = np.zeros(len(X))\n"
        "    for i in range(len(X)):\n"
        "        preds[i] = wm_fuzzy_predict(X[i], rules, fs_x_list, centers_y, q)\n"
        "    return preds\n"
    )


def cell_mf_functions_and_flstm_cell(mf_type_code):
    """MF functions (DOG, SG, GB) + FLSTM Cell with fuzzy prediction fusion (Eq. 14-16)."""
    return make_code_cell(
        "# ============================================================\n"
        "# Membership Functions (DOG, SG, GB)\n"
        "# ============================================================\n"
        "def dog(x, mu1=-1.0, mu2=1.0, sigma1=0.5, sigma2=0.5):\n"
        "    return tf.exp(-tf.square(x - mu1)/ (2 * sigma1**2)) - tf.exp(-tf.square(x - mu2)/ (2 * sigma2**2))\n"
        "\n"
        "def signed_gaussian(x, sigma=0.5):\n"
        "    return x * tf.exp(-tf.square(x)/ (2 * sigma**2))\n"
        "\n"
        "def generalized_bell(x, a=1.0, b=2.0, c_val=0.0):\n"
        "    return 1.0 / (1.0 + tf.pow(tf.abs((x - c_val) / a), 2*b))\n"
        "\n"
        "# ============================================================\n"
        "# WM-FLSTM Cell: MF-based gates + Wang-Mendel Fuzzy Prediction Fusion\n"
        "#\n"
        "# The fuzzy prediction r_t (from WM system) is injected into\n"
        "# the gate computations via separate weight matrices (Eq. 14-16):\n"
        "#   f_t = sigma(W_fx * x_t + W_fh * h_{t-1} + W_ff * r_t + b_f)\n"
        "#   i_t = sigma(W_ix * x_t + W_ih * h_{t-1} + W_if * r_t + b_i)\n"
        "#   o_t = sigma(W_ox * x_t + W_oh * h_{t-1} + W_of * r_t + b_o)\n"
        "#\n"
        "# The c and o gates use the selected MF (DOG/SG/GB) as in the\n"
        "# MembershipLSTMSNPCell architecture.\n"
        "# ============================================================\n"
        "@tf.keras.utils.register_keras_serializable()\n"
        "class MembershipFLSTMSNPCell(layers.Layer):\n"
        f"    def __init__(self, units, mf_type='{mf_type_code}', **kwargs):\n"
        "        super().__init__(**kwargs)\n"
        "        self.units, self.mf_type = units, mf_type\n"
        "        self.state_size, self.output_size = (units, units, units), units\n"
        "        self.mf = dog if mf_type == 'dog' else signed_gaussian if mf_type == 'sg' else generalized_bell\n"
        "\n"
        "    def build(self, input_shape):\n"
        "        # input_shape[-1] includes the fuzzy prediction r_t as the LAST feature\n"
        "        total_input_dim = input_shape[-1]\n"
        "        actual_input_dim = total_input_dim - 1  # exclude r_t\n"
        "        self.kernel = self.add_weight(shape=(actual_input_dim, self.units * 4), initializer='glorot_uniform', name='kernel')\n"
        "        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 4), initializer='orthogonal', name='recurrent_kernel')\n"
        "        self.bias = self.add_weight(shape=(self.units * 4,), initializer='zeros', name='bias')\n"
        "        # Fuzzy prediction fusion weights: W_ff, W_if, W_of (Eq. 14-16)\n"
        "        # r_t is scalar -> separate weight vector for each of 3 gates\n"
        "        # (not applied to candidate/generation gate 'a')\n"
        "        self.fuzzy_gate_kernel = self.add_weight(shape=(1, self.units * 3), initializer='glorot_uniform', name='fuzzy_gate_kernel')\n"
        "\n"
        "    def call(self, inputs, states):\n"
        "        u_tm1 = states[0]\n"
        "        # Split: actual input features vs fuzzy prediction r_t\n"
        "        x_t = inputs[:, :-1]     # actual input features\n"
        "        r_t = inputs[:, -1:]     # WM fuzzy prediction (batch, 1)\n"
        "        z = tf.matmul(x_t, self.kernel) + tf.matmul(u_tm1, self.recurrent_kernel) + self.bias\n"
        "        z0, z1, z2, z3 = z[:, :self.units], z[:, self.units:2*self.units], z[:, 2*self.units:3*self.units], z[:, 3*self.units:]\n"
        "        # Add WM fuzzy prediction contribution to gates (Eq. 14-16)\n"
        "        fz = tf.matmul(r_t, self.fuzzy_gate_kernel)\n"
        "        z0 = z0 + fz[:, :self.units]              # r gate fusion\n"
        "        z1 = z1 + fz[:, self.units:2*self.units]   # c gate fusion\n"
        "        z2 = z2 + fz[:, 2*self.units:]             # o gate fusion\n"
        "\n"
        "        r = tf.tanh(z0)                                    # reset gate\n"
        "        c = tf.clip_by_value(self.mf(z1), -1.0, 1.0)      # consumption gate (MF)\n"
        "        o = tf.clip_by_value(self.mf(z2), -1.0, 1.0)      # output gate (MF)\n"
        "        a = tf.tanh(z3)                                    # generation gate\n"
        "\n"
        "        u = r * u_tm1 - c * a  # internal state update\n"
        "        h = o * a              # output\n"
        "        return h, [u, c, o]\n"
        "\n"
        "    def get_config(self):\n"
        "        config = super().get_config()\n"
        "        config.update({'units': self.units, 'mf_type': self.mf_type})\n"
        "        return config\n"
    )


def cell_strengthening_memory_layer():
    """Strengthening Memory Layer (Eq. 17-19 from FLSTM paper)."""
    return make_code_cell(
        "# ============================================================\n"
        "# Strengthening Memory Layer (Eq. 17-19)\n"
        "# From FLSTM paper: doi:10.1038/s41598-023-47812-3\n"
        "#\n"
        "#   Eq 17: ch_t = h_t + c_t     (combine hidden + cell state)\n"
        "#   Eq 18: s_t = tanh(Conv1d(ch_t))  (feature extraction)\n"
        "#   Eq 19: h_hat_t = ch_t + s_t  (strengthened memory state)\n"
        "#\n"
        "# Note: Conv1d on a single vector is equivalent to Dense.\n"
        "# ============================================================\n"
        "@tf.keras.utils.register_keras_serializable()\n"
        "class StrengtheningMemoryLayer(layers.Layer):\n"
        "    def __init__(self, units, **kwargs):\n"
        "        super().__init__(**kwargs)\n"
        "        self.units_out = units\n"
        "        self.dense = layers.Dense(units, activation='tanh')  # Eq 18\n"
        "\n"
        "    def call(self, inputs):\n"
        "        h_t, c_t = inputs\n"
        "        ch_t = h_t + c_t           # Eq 17: combine hidden + cell state\n"
        "        s_t = self.dense(ch_t)     # Eq 18: feature extraction\n"
        "        h_hat = ch_t + s_t         # Eq 19: strengthened memory state\n"
        "        return h_hat\n"
        "\n"
        "    def get_config(self):\n"
        "        config = super().get_config()\n"
        "        config.update({'units': self.units_out})\n"
        "        return config\n"
    )


def cell_fuzzy_output_layer():
    """WM-style fuzzy output layer with triangular MFs and center-average defuzzification."""
    return make_code_cell(
        "# ============================================================\n"
        "# WM Fuzzy Output Layer (Triangular MFs, center-average defuzz)\n"
        "#\n"
        "# Replaces Dense(1) with Wang-Mendel-style fuzzy inference.\n"
        "# Uses triangular MFs on aggregated hidden state features.\n"
        "# Learnable consequent parameters (a, b, c) per rule.\n"
        "# ============================================================\n"
        "@tf.keras.utils.register_keras_serializable()\n"
        "class WMFuzzyOutputLayer(layers.Layer):\n"
        f"    def __init__(self, units_in, q={Q_REGIONS}, **kwargs):\n"
        "        super().__init__(**kwargs)\n"
        "        self.units_in = units_in\n"
        "        self.q = q\n"
        "\n"
        "    def build(self, input_shape):\n"
        "        num_rules = self.q * self.q\n"
        "        self.rule_a = self.add_weight(shape=(num_rules,), initializer='glorot_uniform', name='rule_a')\n"
        "        self.rule_b = self.add_weight(shape=(num_rules,), initializer='glorot_uniform', name='rule_b')\n"
        "        self.rule_c = self.add_weight(shape=(num_rules,), initializer='zeros', name='rule_c')\n"
        "\n"
        "    def _triangular_mf(self, x, center, width):\n"
        "        left = center - width\n"
        "        right = center + width\n"
        "        return tf.maximum(0.0, tf.minimum((x - left) / (center - left + 1e-10),\n"
        "                                          (right - x) / (right - center + 1e-10)))\n"
        "\n"
        "    def call(self, inputs):\n"
        "        half = self.units_in // 2\n"
        "        s1 = tf.reduce_mean(inputs[:, :half], axis=-1, keepdims=True)\n"
        "        s2 = tf.reduce_mean(inputs[:, half:], axis=-1, keepdims=True)\n"
        "        centers = tf.linspace(-1.0, 1.0, self.q)\n"
        "        width = 2.0 / tf.cast(self.q - 1, tf.float32)\n"
        "        numerator = tf.zeros_like(s1)\n"
        "        denominator = tf.zeros_like(s1)\n"
        "        for i in range(self.q):\n"
        "            mu_s1 = self._triangular_mf(s1, centers[i], width)\n"
        "            for j in range(self.q):\n"
        "                mu_s2 = self._triangular_mf(s2, centers[j], width)\n"
        "                rule_idx = i * self.q + j\n"
        "                w = mu_s1 * mu_s2\n"
        "                y = self.rule_a[rule_idx] * s1 + self.rule_b[rule_idx] * s2 + self.rule_c[rule_idx]\n"
        "                numerator = numerator + w * y\n"
        "                denominator = denominator + w\n"
        "        return numerator / (denominator + 1e-8)\n"
        "\n"
        "    def get_config(self):\n"
        "        config = super().get_config()\n"
        "        config.update({'units_in': self.units_in, 'q': self.q})\n"
        "        return config\n"
    )


def cell_build_model(variant, mf_type_code):
    """build_model function — varies by variant."""
    if variant == 'Gates_Only':
        return make_code_cell(
            "def build_model(input_dim, units, batch_size):\n"
            f"    cell = MembershipFLSTMSNPCell(units, mf_type='{mf_type_code}')\n"
            "    rnn = layers.RNN(cell, return_sequences=False, return_state=True, stateful=True)\n"
            "\n"
            "    # input_dim includes r_t as last feature\n"
            "    inputs = tf.keras.Input(batch_shape=(batch_size, 1, input_dim))\n"
            "    x, u_out, c_out, o_out = rnn(inputs)\n"
            "    # Strengthening Memory Layer (Eq. 17-19)\n"
            "    h_strengthened = StrengtheningMemoryLayer(units)([x, u_out])\n"
            "    # Parameter Segment Sharing: Dense output (Eq. 20)\n"
            "    outputs = layers.Dense(1)(h_strengthened)\n"
            "\n"
            "    model = tf.keras.Model(inputs=inputs, outputs=[outputs, c_out, o_out])\n"
            "    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(clipnorm=1.0), loss=['mean_squared_error', None, None])\n"
            "    return model\n"
        )
    elif variant == 'Gates_Plus_FuzzyInput':
        return make_code_cell(
            "def build_model(input_dim, units, batch_size):\n"
            f"    cell = MembershipFLSTMSNPCell(units, mf_type='{mf_type_code}')\n"
            "    rnn = layers.RNN(cell, return_sequences=False, return_state=True, stateful=True)\n"
            "\n"
            "    # input_dim includes r_t as last feature\n"
            "    inputs = tf.keras.Input(batch_shape=(batch_size, 1, input_dim))\n"
            "    x, u_out, c_out, o_out = rnn(inputs)\n"
            "    # Strengthening Memory Layer (Eq. 17-19)\n"
            "    h_strengthened = StrengtheningMemoryLayer(units)([x, u_out])\n"
            "    # Parameter Segment Sharing: Dense output (Eq. 20)\n"
            "    outputs = layers.Dense(1)(h_strengthened)\n"
            "\n"
            "    model = tf.keras.Model(inputs=inputs, outputs=[outputs, c_out, o_out])\n"
            "    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(clipnorm=1.0), loss=['mean_squared_error', None, None])\n"
            "    return model\n"
        )
    else:  # Gates_Plus_FuzzyOutput
        return make_code_cell(
            "def build_model(input_dim, units, batch_size):\n"
            f"    cell = MembershipFLSTMSNPCell(units, mf_type='{mf_type_code}')\n"
            "    rnn = layers.RNN(cell, return_sequences=False, return_state=True, stateful=True)\n"
            "\n"
            "    # input_dim includes r_t as last feature\n"
            "    inputs = tf.keras.Input(batch_shape=(batch_size, 1, input_dim))\n"
            "    x, u_out, c_out, o_out = rnn(inputs)\n"
            "    # Strengthening Memory Layer (Eq. 17-19)\n"
            "    h_strengthened = StrengtheningMemoryLayer(units)([x, u_out])\n"
            "    # WM Fuzzy Output Layer (replaces Dense(1))\n"
            "    outputs = WMFuzzyOutputLayer(units)(h_strengthened)\n"
            "\n"
            "    model = tf.keras.Model(inputs=inputs, outputs=[outputs, c_out, o_out])\n"
            "    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(clipnorm=1.0), loss=['mean_squared_error', None, None])\n"
            "    return model\n"
        )


def cell_main_loop(variant, variant_display, mf_display, mf_type_code, csv_path, dataset_display):
    """Main training/evaluation loop with noise levels and combined results."""
    # Determine lag and input_dim based on variant
    if variant == 'Gates_Plus_FuzzyInput':
        lag = 2
        input_dim = 2  # [x_t, r_t]
    else:
        lag = 1
        input_dim = 2  # [x_t, r_t]

    # Build the data prep + noise loop + fuzzy preprocessing + training + eval
    code = (
        f"series = pd.read_csv('{csv_path}', header=0, parse_dates=[0], index_col=0)\n"
        "raw_values = series.values.flatten()\n"
        "import numpy as np\n"
        "original_raw_values = np.copy(raw_values)\n"
        "s_x = np.std(original_raw_values)\n"
        "noise_levels = [0.0, 0.005, 0.05, 0.1]\n"
        "\n"
        "# Storage for combined results across noise levels\n"
        "combined_results = {}\n"
        "\n"
        "for lam in noise_levels:\n"
        "    sigma = lam * s_x\n"
        '    print("\\n" + "="*80)\n'
        '    if lam == 0.0:\n'
        '        print(f"EVALUATING: NO NOISE (lambda=0)")\n'
        '    else:\n'
        '        print(f"EVALUATING NOISE LEVEL: {lam*100:.1f}% (lambda={lam}, sigma={sigma:.6f})")\n'
        '    print("="*80 + "\\n")\n'
        "    \n"
        "    np.random.seed(42)\n"
        "    tf.random.set_seed(42)\n"
        "    if lam > 0:\n"
        "        noise = np.random.normal(0, sigma, size=original_raw_values.shape)\n"
        "        raw_values = original_raw_values + noise\n"
        "    else:\n"
        "        raw_values = np.copy(original_raw_values)\n"
        "    \n"
        "    \n"
        "    def difference(dataset, interval=1):\n"
        "        diff = []\n"
        "        for i in range(interval, len(dataset)):\n"
        "            value = dataset[i] - dataset[i - interval]\n"
        "            diff.append(value)\n"
        "        return np.array(diff)\n"
        "    \n"
        "    diff_values = difference(raw_values, 1)\n"
        "    def timeseries_to_supervised(data, lag):\n"
        "        df = pd.DataFrame(data)\n"
        "        columns = [df.shift(i) for i in range(1, lag+1)]\n"
        "        columns.append(df)\n"
        "        df = pd.concat(columns, axis=1)\n"
        "        df.fillna(0, inplace=True)\n"
        "        return df.values\n"
        "    \n"
        "    \n"
        "    \n"
        f"    supervised = timeseries_to_supervised(diff_values, {lag})\n"
        f"    train, test = supervised[:-{TEST_SPLIT}], supervised[-{TEST_SPLIT}:]\n"
        "    scaler = MinMaxScaler(feature_range=(-1, 1))\n"
        "    train_scaled = scaler.fit_transform(train)\n"
        "    test_scaled = scaler.transform(test)\n"
        "    \n"
        "    X_train_raw, y_train = train_scaled[:, 0:-1], train_scaled[:, -1]\n"
        "    X_test_raw, y_test = test_scaled[:, 0:-1], test_scaled[:, -1]\n"
        "    \n"
    )

    # Add WM fuzzy preprocessing depending on variant
    if variant == 'Gates_Plus_FuzzyInput':
        code += (
            "    # Build WM fuzzy system from training data\n"
            "    wm_rules, wm_fs_x, wm_centers_y = build_wm_system(X_train_raw, y_train)\n"
            "    \n"
            "    # Compute WM fuzzy predictions as r_t (replaces TSK fuzzy_inference)\n"
            "    r_train = compute_fuzzy_predictions(X_train_raw, wm_rules, wm_fs_x, wm_centers_y)\n"
            "    r_test = compute_fuzzy_predictions(X_test_raw, wm_rules, wm_fs_x, wm_centers_y)\n"
            "    \n"
            "    # Input: [x_t, r_t] where x_t is the most recent lag feature\n"
            "    X_train_fuzzy = np.zeros((X_train_raw.shape[0], 2))\n"
            "    X_test_fuzzy = np.zeros((X_test_raw.shape[0], 2))\n"
            f"    for i in range(X_train_raw.shape[0]): X_train_fuzzy[i] = [X_train_raw[i, {lag - 1}], r_train[i]]\n"
            f"    for i in range(X_test_raw.shape[0]): X_test_fuzzy[i] = [X_test_raw[i, {lag - 1}], r_test[i]]\n"
            "    \n"
            "    X_train = X_train_fuzzy.reshape((X_train_fuzzy.shape[0], 1, X_train_fuzzy.shape[1]))\n"
            "    X_test = X_test_fuzzy.reshape((X_test_fuzzy.shape[0], 1, X_test_fuzzy.shape[1]))\n"
            "    \n"
        )
    else:
        # Gates_Only and FuzzyOutput: input is [x_t, r_t]
        code += (
            "    # Build WM fuzzy system from training data\n"
            "    wm_rules, wm_fs_x, wm_centers_y = build_wm_system(X_train_raw, y_train)\n"
            "    \n"
            "    # Compute WM fuzzy predictions r_t for gate fusion (Eq. 14-16)\n"
            "    r_train = compute_fuzzy_predictions(X_train_raw, wm_rules, wm_fs_x, wm_centers_y)\n"
            "    r_test = compute_fuzzy_predictions(X_test_raw, wm_rules, wm_fs_x, wm_centers_y)\n"
            "    \n"
            "    # Concatenate r_t as last feature: [x_t, r_t]\n"
            "    X_train_aug = np.column_stack([X_train_raw, r_train])\n"
            "    X_test_aug = np.column_stack([X_test_raw, r_test])\n"
            "    \n"
            "    X_train = X_train_aug.reshape((X_train_aug.shape[0], 1, X_train_aug.shape[1]))\n"
            "    X_test = X_test_aug.reshape((X_test_aug.shape[0], 1, X_test_aug.shape[1]))\n"
            "    \n"
        )

    # Training & evaluation loop
    code += (
        "    all_rmse, all_mse, all_nmse = [], [], []\n"
        "    all_predictions = []\n"
        "    all_losses = []\n"
        "    all_c_means, all_c_stds, all_c_mins, all_c_maxs = [], [], [], []\n"
        "    all_o_means, all_o_stds, all_o_mins, all_o_maxs = [], [], [], []\n"
        "    \n"
        f"    for run in range({NUM_RUNS}):\n"
        f"        print(f'\\n===== RUN {{run+1}}/{NUM_RUNS} =====')\n"
        "        np.random.seed(run)\n"
        "        tf.random.set_seed(run)\n"
        "        tf.keras.backend.clear_session()\n"
        "        \n"
        f"        model = build_model(input_dim={input_dim}, units={UNITS}, batch_size=1)\n"
        "        rnn_layer = model.layers[1]\n"
        "        \n"
        "        run_losses = []\n"
        f"        for epoch in range({NUM_EPOCHS}):\n"
        "            history = model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=0, shuffle=False)\n"
        "            run_losses.append(history.history['loss'][0])\n"
        "            rnn_layer.reset_states()\n"
        "        all_losses.append(run_losses)\n"
        "    \n"
        "        # Warmup: condition hidden states on training data\n"
        "        for i in range(len(X_train)): model.predict(X_train[i:i+1], batch_size=1, verbose=0)\n"
        "    \n"
        "        predictions, c_gates, o_gates = [], [], []\n"
        "        for i in range(len(X_test)):\n"
        "            yhat, c_val, o_val = model.predict(X_test[i:i+1], batch_size=1, verbose=0)\n"
        "            c_gates.append(c_val[0])\n"
        "            o_gates.append(o_val[0])\n"
        "            \n"
        "            row = list(X_test_raw[i]) + [yhat[0, 0]]\n"
        f"            inv = scaler.inverse_transform([row])[0, -1] + raw_values[len(train) + i]\n"
        "            predictions.append(inv)\n"
        "            \n"
        "        c_gates, o_gates = np.array(c_gates), np.array(o_gates)\n"
        "        all_c_means.append(np.mean(c_gates)); all_c_stds.append(np.std(c_gates))\n"
        "        all_c_mins.append(np.min(c_gates)); all_c_maxs.append(np.max(c_gates))\n"
        "        all_o_means.append(np.mean(o_gates)); all_o_stds.append(np.std(o_gates))\n"
        "        all_o_mins.append(np.min(o_gates)); all_o_maxs.append(np.max(o_gates))\n"
        "    \n"
        f"        actual = raw_values[-{TEST_SPLIT}:]\n"
        "        rmse = sqrt(mean_squared_error(actual, predictions))\n"
        "        mse = mean_squared_error(actual, predictions)\n"
        "        meanV = np.mean(actual)\n"
        "        dominator = np.linalg.norm(np.array(predictions) - meanV, 2)\n"
        "        nmse = mse / np.power(dominator, 2)\n"
        "    \n"
        "        all_rmse.append(rmse)\n"
        "        all_mse.append(mse)\n"
        "        all_nmse.append(nmse)\n"
        "        all_predictions.append(predictions)\n"
        "        print(f'Run {run+1} — RMSE: {rmse:.6f}, MSE: {mse:.6f}, NMSE: {nmse:.10f} (Gate Mean C: {np.mean(c_gates):.4f})')\n"
        "    \n"
        "    print('\\n===== GATE STATISTICS =====')\n"
        "    print(f'C-Gate - Mean: {np.mean(all_c_means):.4f}, Min: {np.min(all_c_mins):.4f}, Max: {np.max(all_c_maxs):.4f}')\n"
        "    print(f'O-Gate - Mean: {np.mean(all_o_means):.4f}, Min: {np.min(all_o_mins):.4f}, Max: {np.max(all_o_maxs):.4f}')\n"
        "    print(f'\\nOverall RMSE: {np.mean(all_rmse):.6f}')\n"
        "    \n"
        "    \n"
        f"    # Summary Statistics ({NUM_RUNS} runs)\n"
        "    \n"
        f"    noise_label = 'No Noise' if lam == 0.0 else f'{{lam*100:.1f}}% Noise'\n"
        f"    print(f'\\n===== FINAL RESULTS — WM-FLSTM {variant_display} ({mf_display}) on {dataset_display} — {{noise_label}} ({NUM_RUNS} runs) =====')\n"
        "    mean_rmse = np.mean(all_rmse)\n"
        "    std_rmse = np.std(all_rmse)\n"
        "    var_rmse = np.var(all_rmse)\n"
        "    \n"
        "    mean_mse = np.mean(all_mse)\n"
        "    std_mse = np.std(all_mse)\n"
        "    \n"
        "    mean_nmse = np.mean(all_nmse)\n"
        "    std_nmse = np.std(all_nmse)\n"
        "    \n"
        "    print(f'RMSE: {mean_rmse:.6f} ± {std_rmse:.6f} (Var: {var_rmse:.6f})')\n"
        "    print(f'MSE:  {mean_mse:.6f} ± {std_mse:.6f}')\n"
        "    print(f'NMSE: {mean_nmse:.10f} ± {std_nmse:.10f}')\n"
        "    \n"
        "    best_idx = np.argmin(all_rmse)\n"
        "    \n"
        "    print(f'\\nBest run: {best_idx+1}')\n"
        "    print(f'  RMSE: {all_rmse[best_idx]:.6f}')\n"
        "    print(f'  MSE:  {all_mse[best_idx]:.6f}')\n"
        "    print(f'  NMSE: {all_nmse[best_idx]:.10f}')\n"
        "    \n"
        "    # Store results for combined summary\n"
        "    combined_results[lam] = {\n"
        "        'mean_rmse': mean_rmse, 'std_rmse': std_rmse,\n"
        "        'mean_mse': mean_mse, 'std_mse': std_mse,\n"
        "        'mean_nmse': mean_nmse, 'std_nmse': std_nmse,\n"
        "        'best_rmse': all_rmse[best_idx],\n"
        "        'best_mse': all_mse[best_idx],\n"
        "        'best_nmse': all_nmse[best_idx],\n"
        "        'all_rmse': list(all_rmse),\n"
        "        'all_predictions': all_predictions,\n"
        "        'all_losses': all_losses,\n"
        "        'actual': actual,\n"
        "    }\n"
        "    \n"
        "    # Predictions vs Actual (Best Run)\n"
        "    \n"
        f"    actual = raw_values[-{TEST_SPLIT}:]\n"
        "    best_predictions = all_predictions[best_idx]\n"
        "    \n"
        "    plt.figure(figsize=(12, 5))\n"
        "    plt.plot(actual, label='Actual', color='blue', linewidth=1.5)\n"
        "    plt.plot(best_predictions, label='Predicted (Best Run)', color='red',\n"
        "             linewidth=1.5, linestyle='--')\n"
        f"    plt.title(f'WM-FLSTM {variant_display} ({mf_display}) — {dataset_display} — {{noise_label}}\\nPredictions vs Actual (Best of {NUM_RUNS} runs)')\n"
        "    plt.xlabel('Time Step')\n"
        "    plt.ylabel('Value')\n"
        "    plt.legend()\n"
        "    plt.grid(True, alpha=0.3)\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "    \n"
        "    # Loss Curve (Best Run)\n"
        "    \n"
        "    plt.figure(figsize=(12, 4))\n"
        "    plt.plot(all_losses[best_idx], color='green', linewidth=1.0)\n"
        f"    plt.title(f'WM-FLSTM {variant_display} ({mf_display}) — {dataset_display} — {{noise_label}}\\nTraining Loss (Best Run)')\n"
        "    plt.xlabel('Epoch')\n"
        "    plt.ylabel('MSE Loss')\n"
        "    plt.grid(True, alpha=0.3)\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "    \n"
        f"    # Average Loss Curve (All {NUM_RUNS} Runs)\n"
        "    \n"
        "    avg_losses = np.mean(all_losses, axis=0)\n"
        "    std_losses = np.std(all_losses, axis=0)\n"
        "    epochs = np.arange(1, len(avg_losses) + 1)\n"
        "    \n"
        "    plt.figure(figsize=(12, 4))\n"
        "    plt.plot(epochs, avg_losses, color='purple', linewidth=1.5, label='Mean Loss')\n"
        "    plt.fill_between(epochs, avg_losses - std_losses, avg_losses + std_losses,\n"
        "                     alpha=0.2, color='purple', label='±1 Std Dev')\n"
        f"    plt.title(f'WM-FLSTM {variant_display} ({mf_display}) — {dataset_display} — {{noise_label}}\\nAverage Training Loss ({NUM_RUNS} runs ± 1σ)')\n"
        "    plt.xlabel('Epoch')\n"
        "    plt.ylabel('MSE Loss')\n"
        "    plt.legend()\n"
        "    plt.grid(True, alpha=0.3)\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "    \n"
        "    # RMSE Distribution (Boxplot + All Points)\n"
        "    \n"
        "    plt.figure(figsize=(8, 5))\n"
        "    plt.boxplot(all_rmse, vert=True, patch_artist=True,\n"
        "                boxprops=dict(facecolor='lightblue', color='navy'),\n"
        "                medianprops=dict(color='red', linewidth=2))\n"
        "    plt.scatter(np.ones(len(all_rmse)), all_rmse, alpha=0.5, color='navy', zorder=5)\n"
        f"    plt.title(f'WM-FLSTM {variant_display} ({mf_display}) — {dataset_display} — {{noise_label}}\\nRMSE Distribution ({NUM_RUNS} runs)')\n"
        "    plt.ylabel('RMSE')\n"
        "    plt.grid(True, alpha=0.3)\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "    \n"
        "    # Average Predictions vs Actual\n"
        "    \n"
        "    avg_predictions = np.mean(all_predictions, axis=0)\n"
        "    std_predictions = np.std(all_predictions, axis=0)\n"
        "    \n"
        "    plt.figure(figsize=(12, 5))\n"
        "    plt.plot(actual, label='Actual', color='blue', linewidth=1.5)\n"
        f"    plt.plot(avg_predictions, label='Mean Predicted ({NUM_RUNS} runs)', color='red',\n"
        "             linewidth=1.5, linestyle='--')\n"
        "    plt.fill_between(range(len(actual)),\n"
        "                     avg_predictions - std_predictions,\n"
        "                     avg_predictions + std_predictions,\n"
        "                     alpha=0.2, color='red', label='±1 Std Dev')\n"
        f"    plt.title(f'WM-FLSTM {variant_display} ({mf_display}) — {dataset_display} — {{noise_label}}\\nAverage Predictions vs Actual ({NUM_RUNS} runs ± 1σ)')\n"
        "    plt.xlabel('Time Step')\n"
        "    plt.ylabel('Value')\n"
        "    plt.legend()\n"
        "    plt.grid(True, alpha=0.3)\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "    \n"
        "    # Gate Statistics Visualization\n"
        "    \n"
        "    fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
        "    \n"
        "    # C-Gate statistics\n"
        "    axes[0].errorbar(range(len(all_c_means)), all_c_means, yerr=all_c_stds,\n"
        "                     fmt='o-', color='teal', ecolor='lightcoral', capsize=3, markersize=4)\n"
        "    axes[0].set_title('C-Gate Mean ± Std per Run')\n"
        "    axes[0].set_xlabel('Run')\n"
        "    axes[0].set_ylabel('C-Gate Value')\n"
        "    axes[0].grid(True, alpha=0.3)\n"
        "    \n"
        "    # O-Gate statistics\n"
        "    axes[1].errorbar(range(len(all_o_means)), all_o_means, yerr=all_o_stds,\n"
        "                     fmt='o-', color='darkorange', ecolor='lightcoral', capsize=3, markersize=4)\n"
        "    axes[1].set_title('O-Gate Mean ± Std per Run')\n"
        "    axes[1].set_xlabel('Run')\n"
        "    axes[1].set_ylabel('O-Gate Value')\n"
        "    axes[1].grid(True, alpha=0.3)\n"
        "    \n"
        f"    plt.suptitle(f'WM-FLSTM {variant_display} ({mf_display}) — {dataset_display} — {{noise_label}} — Gate Statistics ({NUM_RUNS} runs)', fontsize=13)\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "    \n"
        "    # Final Metrics Summary\n"
        "    \n"
        "    print('=== Best Run Metrics ===')\n"
        "    print(f'RMSE: {all_rmse[best_idx]:.6f}')\n"
        "    print(f'MSE:  {all_mse[best_idx]:.6f}')\n"
        "    print(f'NMSE: {all_nmse[best_idx]:.10f}')\n"
        "    \n"
        f"    print(f'\\n=== Average Metrics ({NUM_RUNS} runs) ===')\n"
        "    print(f'RMSE: {np.mean(all_rmse):.6f} ± {np.std(all_rmse):.6f}')\n"
        "    print(f'MSE:  {np.mean(all_mse):.6f} ± {np.std(all_mse):.6f}')\n"
        "    print(f'NMSE: {np.mean(all_nmse):.10f} ± {np.std(all_nmse):.10f}')\n"
        "    "
    )

    return make_code_cell(code)


def cell_combined_results(variant_display, mf_display, dataset_display):
    """Combined results summary across all noise levels."""
    return make_code_cell(
        "# ============================================================\n"
        f"# COMBINED RESULTS: WM-FLSTM {variant_display} ({mf_display}) on {dataset_display}\n"
        "# All Noise Levels Comparison\n"
        "# ============================================================\n"
        "\n"
        "print('\\n' + '='*100)\n"
        f"print('COMBINED RESULTS — WM-FLSTM {variant_display} ({mf_display}) on {dataset_display}')\n"
        "print('='*100)\n"
        "\n"
        "# Print combined table\n"
        "print(f'{\"Noise Level\":<15} | {\"Mean RMSE\":<20} | {\"Mean MSE\":<20} | {\"Mean NMSE\":<25} | {\"Best RMSE\":<15}')\n"
        "print('-'*100)\n"
        "for lam in [0.0, 0.005, 0.05, 0.1]:\n"
        "    r = combined_results[lam]\n"
        "    label = 'No Noise' if lam == 0.0 else f'{lam*100:.1f}%'\n"
        "    print(f'{label:<15} | {r[\"mean_rmse\"]:.6f} ± {r[\"std_rmse\"]:.6f} | {r[\"mean_mse\"]:.6f} ± {r[\"std_mse\"]:.6f} | {r[\"mean_nmse\"]:.10f} ± {r[\"std_nmse\"]:.10f} | {r[\"best_rmse\"]:.6f}')\n"
        "\n"
        "print('\\n')\n"
        "\n"
        "# Comparative RMSE Bar Chart\n"
        "noise_labels = ['No Noise', '0.5%', '5%', '10%']\n"
        "mean_rmses = [combined_results[lam]['mean_rmse'] for lam in [0.0, 0.005, 0.05, 0.1]]\n"
        "std_rmses = [combined_results[lam]['std_rmse'] for lam in [0.0, 0.005, 0.05, 0.1]]\n"
        "\n"
        "plt.figure(figsize=(10, 6))\n"
        "bars = plt.bar(noise_labels, mean_rmses, yerr=std_rmses, capsize=8,\n"
        "               color=['#2ecc71', '#3498db', '#e67e22', '#e74c3c'],\n"
        "               edgecolor='black', linewidth=0.8, alpha=0.85)\n"
        "for bar, val in zip(bars, mean_rmses):\n"
        "    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),\n"
        "             f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')\n"
        f"plt.title('WM-FLSTM {variant_display} ({mf_display}) — {dataset_display}\\nMean RMSE Across Noise Levels ({NUM_RUNS} runs)', fontsize=13)\n"
        "plt.xlabel('Noise Level')\n"
        "plt.ylabel('Mean RMSE')\n"
        "plt.grid(True, alpha=0.3, axis='y')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "\n"
        "# Comparative RMSE Boxplot\n"
        "fig, ax = plt.subplots(figsize=(10, 6))\n"
        "rmse_data = [combined_results[lam]['all_rmse'] for lam in [0.0, 0.005, 0.05, 0.1]]\n"
        "bp = ax.boxplot(rmse_data, labels=noise_labels, patch_artist=True,\n"
        "                boxprops=dict(facecolor='lightblue', color='navy'),\n"
        "                medianprops=dict(color='red', linewidth=2))\n"
        "colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']\n"
        "for patch, color in zip(bp['boxes'], colors):\n"
        "    patch.set_facecolor(color)\n"
        "    patch.set_alpha(0.6)\n"
        f"ax.set_title('WM-FLSTM {variant_display} ({mf_display}) — {dataset_display}\\nRMSE Distribution Across Noise Levels ({NUM_RUNS} runs)', fontsize=13)\n"
        "ax.set_ylabel('RMSE')\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "\n"
        "# Degradation Analysis\n"
        "print('\\n=== Noise Degradation Analysis ===')\n"
        "base_rmse = combined_results[0.0]['mean_rmse']\n"
        "for lam in [0.005, 0.05, 0.1]:\n"
        "    degradation = ((combined_results[lam]['mean_rmse'] - base_rmse) / base_rmse) * 100\n"
        "    print(f'{lam*100:.1f}% noise: RMSE degradation = {degradation:+.2f}% (from {base_rmse:.6f} to {combined_results[lam][\"mean_rmse\"]:.6f})')\n"
    )


def cell_observations(variant_display, mf_display, dataset_display):
    return make_markdown_cell(
        f"## Observations\n"
        f"\n"
        f"### WM-FLSTM {variant_display} ({mf_display}) on {dataset_display}\n"
        f"\n"
        f"**Interfacing**: Wang-Mendel Fuzzy Prediction Fusion (replaces TSK)\n"
        f"\n"
        f"**Run the notebook to generate results and fill in observations:**\n"
        f"\n"
        f"1. **Prediction Quality**: Compare RMSE/MSE/NMSE with TSK-interfaced Modified_Fuzzy_MF results\n"
        f"2. **Training Stability**: Examine loss curves for convergence behavior\n"
        f"3. **Average Behaviour**: Compare average predictions vs actual\n"
        f"4. **Gate Dynamics**: Examine C-Gate and O-Gate statistics across runs\n"
        f"5. **Prediction Tracking**: Assess how well predictions track actual values\n"
        f"6. **Computational Cost**: Note training time per run\n"
        f"7. **Noise Robustness**: Compare metrics across noise levels (0%, 0.5%, 5%, 10%)\n"
        f"8. **WM-FLSTM Components**: Evaluate the impact of:\n"
        f"   - Wang-Mendel fuzzy prediction fusion (r_t in gates, Eq. 14-16)\n"
        f"   - Strengthening Memory Layer (Eq. 17-19)\n"
        f"   - Triangular MFs vs Gaussian MFs for fuzzy rules\n"
        f"   - Data-driven rule extraction vs fixed TSK rules\n"
        f"\n"
        f"*After running all variant notebooks, perform cross-variant comparison to evaluate \n"
        f"whether the WM-FLSTM interfacing improves performance over TSK interfacing.*\n"
    )


def cell_timer_end():
    return make_code_cell(
        "# ============================================================\n"
        "# NOTEBOOK TIMER — END\n"
        "# ============================================================\n"
        "import time as _timer_module\n"
        "_NOTEBOOK_END_TIME = _timer_module.time()\n"
        "_NOTEBOOK_ELAPSED = _NOTEBOOK_END_TIME - _NOTEBOOK_START_TIME\n"
        "_hours, _rem = divmod(_NOTEBOOK_ELAPSED, 3600)\n"
        "_minutes, _seconds = divmod(_rem, 60)\n"
        "print(f\"\\nTotal notebook execution time: {int(_hours)}h {int(_minutes)}m {_seconds:.2f}s\")\n"
        "print(f\"Total seconds: {_NOTEBOOK_ELAPSED:.2f}\")\n"
    )


# ============================================================================
# Notebook builder
# ============================================================================

def build_notebook(variant, mf_display, mf_type_code, dataset_key, dataset_config):
    """Build a complete notebook as a dict."""
    variant_display = variant.replace('_', ' ').replace('Plus', '+')
    dataset_display = dataset_config['display_name']
    csv_path = dataset_config['csv_path']

    cells = []

    # 1. Title
    cells.append(cell_title(variant_display, mf_display, dataset_display))

    # 2. PID
    cells.append(cell_pid())

    # 3. Timer start
    cells.append(cell_timer_start())

    # 4. CPU only
    cells.append(cell_cpu_only())

    # 5. Imports
    cells.append(cell_imports())

    # 6. Wang-Mendel fuzzy rule base
    cells.append(cell_wang_mendel_fuzzy())

    # 7. MF functions + FLSTM Cell with r_t fusion
    cells.append(cell_mf_functions_and_flstm_cell(mf_type_code))

    # 8. Strengthening Memory Layer
    cells.append(cell_strengthening_memory_layer())

    # 9. FuzzyOutput layer (only for FuzzyOutput variant)
    if variant == 'Gates_Plus_FuzzyOutput':
        cells.append(cell_fuzzy_output_layer())

    # 10. build_model
    cells.append(cell_build_model(variant, mf_type_code))

    # 11. Main training/evaluation loop (with noise loop)
    cells.append(cell_main_loop(variant, variant_display, mf_display, mf_type_code, csv_path, dataset_display))

    # 12. Combined results across noise levels
    cells.append(make_markdown_cell("## Combined Results Across Noise Levels"))
    cells.append(cell_combined_results(variant_display, mf_display, dataset_display))

    # 13. Observations markdown
    cells.append(cell_observations(variant_display, mf_display, dataset_display))

    # 14. Timer end
    cells.append(cell_timer_end())

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "SNN Transformer",
                "language": "python",
                "name": "snn_transformer",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.14",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    return notebook


# ============================================================================
# Main
# ============================================================================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    total_count = 0

    for dataset_key, dataset_config in DATASETS.items():
        folder_name = f"{FOLDER_PREFIX}_{dataset_config['folder_suffix']}"
        folder_path = os.path.join(base_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        for variant in VARIANTS:
            for mf_display, mf_type_code in MF_TYPES.items():
                nb = build_notebook(variant, mf_display, mf_type_code, dataset_key, dataset_config)

                filename = f"{NB_PREFIX}_{variant}_{mf_display}_{dataset_key}.ipynb"
                filepath = os.path.join(folder_path, filename)

                with open(filepath, 'w') as f:
                    json.dump(nb, f, indent=1)

                total_count += 1
                print(f"  [{total_count:2d}/36] Created: {folder_name}/{filename}")

    print(f"\nDone! Generated {total_count} notebooks across {len(DATASETS)} folders.")


if __name__ == '__main__':
    main()
