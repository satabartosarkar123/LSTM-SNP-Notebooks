#!/usr/bin/env python3
"""
Generate Lag=90 FLSTM Classification Notebooks

Creates new _Lag90 folders for all 11 classification directories.
Each notebook is a single self-contained cell matching the high-lag reference
from flstm-gb-lag-5-100.ipynb (no complement_wm_rules, dynamic input_dim, LAG=90).

Total: 11 folders × 9 notebooks = 99 notebooks
"""
import json, os

# ============================================================
# CONFIGURATION
# ============================================================
LAG_VALUE = 90
NUM_RUNS = 2
NUM_EPOCHS = 100

# Source → Destination folder mapping
FOLDERS = [
    ('FLSTM_Classification_EPL',                  'FLSTM_Classification_EPL_Lag90'),
    ('FLSTM_Classification_EPL_Median',           'FLSTM_Classification_EPL_Median_Lag90'),
    ('FLSTM_Classification_EPL_Mode',             'FLSTM_Classification_EPL_Mode_Lag90'),
    ('FLSTM_Classification_MackeyGlass',          'FLSTM_Classification_MackeyGlass_Lag90'),
    ('FLSTM_Classification_MackeyGlass_Median',   'FLSTM_Classification_MackeyGlass_Median_Lag90'),
    ('FLSTM_Classification_MackeyGlass_Midrange', 'FLSTM_Classification_MackeyGlass_Midrange_Lag90'),
    ('FLSTM_Classification_MackeyGlass_Mode',     'FLSTM_Classification_MackeyGlass_Mode_Lag90'),
    ('FLSTM_Classification_Sunspots',             'FLSTM_Classification_Sunspots_Lag90'),
    ('FLSTM_Classification_Sunspots_Huber',       'FLSTM_Classification_Sunspots_Huber_Lag90'),
    ('FLSTM_Classification_Sunspots_Median',      'FLSTM_Classification_Sunspots_Median_Lag90'),
    ('FLSTM_Classification_Sunspots_Mode',        'FLSTM_Classification_Sunspots_Mode_Lag90'),
]

# Determine dataset and threshold type from folder name
def get_dataset(folder_name):
    if 'EPL' in folder_name:
        return 'epl'
    elif 'MackeyGlass' in folder_name:
        return 'mackey_glass'
    elif 'Sunspots' in folder_name:
        return 'sunspots'

def get_threshold_type(folder_name):
    if '_Mode' in folder_name:
        return 'mode'
    elif '_Midrange' in folder_name:
        return 'midrange'
    elif '_Huber' in folder_name:
        return 'huber'
    else:
        return 'median'  # Default and _Median both use median

ARCHS = ['Gates_Only', 'Gates_Plus_FuzzyInput', 'Gates_Plus_FuzzyOutput']
MFS = ['DOG', 'SG', 'GB']

ARCH_LABELS = {
    'Gates_Only': 'Gates Only',
    'Gates_Plus_FuzzyInput': 'Gates+FuzzyInput',
    'Gates_Plus_FuzzyOutput': 'Gates+FuzzyOutput',
}

DATASET_LABELS = {
    'epl': 'EPL',
    'mackey_glass': 'Mackey-Glass',
    'sunspots': 'Sunspots',
}

DATASET_KEYS = {
    'epl': 'epl',
    'mackey_glass': 'mackey_glass',
    'sunspots': 'sunspots',
}

THRESHOLD_LABELS = {
    'median': 'Median',
    'mode': 'Mode',
    'midrange': 'Midrange',
    'huber': 'Huber',
}

KERNEL_SPEC = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.12.13"
    },
    "kaggle": {
        "accelerator": "nvidiaTeslaT4",
        "dataSources": [{"sourceType": "datasetVersion", "sourceId": 2705161}],
        "isInternetEnabled": True,
        "language": "python",
        "sourceType": "notebook",
        "isGpuEnabled": False
    }
}


# ============================================================
# DATA LOADING CODE GENERATORS
# ============================================================
def get_data_loading_code(dataset):
    if dataset == 'epl':
        return [
            "# 1. DATA LOADING & PREPROCESSING\n",
            "# ------------------------------------------------------------\n",
            "data_dir = next((d for d in [\n",
            "    '../content/English Premier League Dataset/data',\n",
            "    'content/English Premier League Dataset/data',\n",
            "    '/Users/satabarto/Research/content/English Premier League Dataset/data',\n",
            "    '/kaggle/input/datasets/saurabhshahane/english-premier-league-dataset'\n",
            "] if os.path.exists(d)), '../content/English Premier League Dataset/data')\n",
            "all_dfs = []\n",
            "\n",
            "csv_files = sorted(glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True))\n",
            "\n",
            "for filepath in csv_files:\n",
            "    try:\n",
            "        df = pd.read_csv(filepath)\n",
            "        cols_lower = {c.lower(): c for c in df.columns}\n",
            "        if 'fthg' in cols_lower and 'ftag' in cols_lower:\n",
            "            df['TotalGoals'] = (\n",
            "                pd.to_numeric(df[cols_lower['fthg']], errors='coerce') +\n",
            "                pd.to_numeric(df[cols_lower['ftag']], errors='coerce')\n",
            "            )\n",
            "            all_dfs.append(df[['TotalGoals']].dropna())\n",
            "    except Exception:\n",
            "        continue\n",
            "\n",
            "if not all_dfs:\n",
            "    raise FileNotFoundError(f'No valid match CSV files found under {data_dir}.')\n",
            "\n",
            "combined = pd.concat(all_dfs, ignore_index=True)\n",
            "raw_values = combined['TotalGoals'].values.astype(float)\n",
            "original_raw_values = np.copy(raw_values)\n",
        ]
    elif dataset == 'mackey_glass':
        return [
            "# 1. DATA LOADING & PREPROCESSING\n",
            "# ------------------------------------------------------------\n",
            "import openpyxl\n",
            "file_path = next((p for p in [\n",
            "    '../content/Mackey-Glass Time Series(taw17).xlsx',\n",
            "    'content/Mackey-Glass Time Series(taw17).xlsx',\n",
            "    '/Users/satabarto/Research/content/Mackey-Glass Time Series(taw17).xlsx',\n",
            "    '/kaggle/input/datasets/saurabhshahane/mackey-glass-time-series/Mackey-Glass Time Series(taw17).xlsx'\n",
            "] if os.path.exists(p)), '../content/Mackey-Glass Time Series(taw17).xlsx')\n",
            "series = pd.read_excel(file_path)\n",
            "raw_values = series['t+1'].values.flatten()\n",
            "original_raw_values = np.copy(raw_values)\n",
        ]
    elif dataset == 'sunspots':
        return [
            "# 1. DATA LOADING & PREPROCESSING\n",
            "# ------------------------------------------------------------\n",
            "file_path = next((p for p in [\n",
            "    '../content/Sunspots.csv',\n",
            "    'content/Sunspots.csv',\n",
            "    '/Users/satabarto/Research/content/Sunspots.csv',\n",
            "    '/kaggle/input/datasets/saurabhshahane/sunspots/Sunspots.csv'\n",
            "] if os.path.exists(p)), '../content/Sunspots.csv')\n",
            "series = pd.read_csv(file_path)\n",
            "raw_values = series['Monthly Mean Total Sunspot Number'].values.flatten()\n",
            "original_raw_values = np.copy(raw_values)\n",
        ]


# ============================================================
# THRESHOLD CODE GENERATORS
# ============================================================
def get_threshold_code(threshold_type):
    if threshold_type == 'median':
        return [
            "    # ---- Classification Metrics (Dynamic Threshold: Median) ----\n",
            "    train_raw_values = raw_values[:-60]\n",
            "    train_diffs = np.diff(train_raw_values)\n",
            "    tau = round(float(np.median(train_diffs)), 6)\n",
        ]
    elif threshold_type == 'mode':
        return [
            "    # ---- Classification Metrics (Dynamic Threshold: Mode) ----\n",
            "    import scipy.stats as stats\n",
            "    train_raw_values = raw_values[:-60]\n",
            "    train_diffs = np.diff(train_raw_values)\n",
            "    rounded_diffs = np.round(train_diffs, 3)\n",
            "    mode_res = stats.mode(rounded_diffs, keepdims=True)\n",
            "    tau = float(mode_res.mode[0]) if hasattr(mode_res.mode, '__len__') else float(mode_res.mode)\n",
        ]
    elif threshold_type == 'midrange':
        return [
            "    # ---- Classification Metrics (Dynamic Threshold: Midrange) ----\n",
            "    train_raw_values = raw_values[:-60]\n",
            "    train_diffs = np.diff(train_raw_values)\n",
            "    tau = round(float(0.5 * (np.min(train_diffs) + np.max(train_diffs))), 6)\n",
        ]
    elif threshold_type == 'huber':
        return [
            "    # ---- Classification Metrics (Dynamic Threshold: Huber) ----\n",
            "    tau = -0.035150\n",
        ]


# ============================================================
# MF FUNCTION CODE
# ============================================================
def get_mf_code(mf_type):
    if mf_type == 'DOG':
        return [
            "def dog(x, mu1=-1.0, mu2=1.0, sigma1=0.5, sigma2=0.5):\n",
            "    \"\"\"Difference of Gaussians (DOG) Membership Function\"\"\"\n",
            "    return tf.exp(-tf.square(x - mu1) / (2 * sigma1**2)) - tf.exp(-tf.square(x - mu2) / (2 * sigma2**2))\n",
            "\n",
        ]
    elif mf_type == 'SG':
        return [
            "def signed_gaussian(x, sigma=0.5):\n",
            "    \"\"\"Signed Gaussian (SG) Membership Function\"\"\"\n",
            "    return x * tf.exp(-tf.square(x) / (2 * sigma**2))\n",
            "\n",
        ]
    elif mf_type == 'GB':
        return [
            "def generalized_bell(x, a=1.0, b=2.0, c_val=0.0):\n",
            "    \"\"\"Generalized Bell (GB) Membership Function\"\"\"\n",
            "    return 1.0 / (1.0 + tf.pow(tf.abs((x - c_val) / a), 2*b))\n",
            "\n",
        ]


def get_mf_varname(mf_type):
    return {'DOG': 'dog', 'SG': 'signed_gaussian', 'GB': 'generalized_bell'}[mf_type]


# ============================================================
# FLSTM CELL CODE
# ============================================================
def get_cell_code(mf_type):
    mf_var = get_mf_varname(mf_type)
    return [
        "@tf.keras.utils.register_keras_serializable()\n",
        "class MembershipFLSTMSNPCell(layers.Layer):\n",
        f"    def __init__(self, units, mf_type='{mf_type.lower()}', **kwargs):\n",
        "        super().__init__(**kwargs)\n",
        "        self.units, self.mf_type = units, mf_type\n",
        "        self.state_size, self.output_size = (units, units, units), units\n",
        f"        self.mf = {mf_var}\n",
        "\n",
        "    def build(self, input_shape):\n",
        "        total_input_dim = input_shape[-1]\n",
        "        actual_input_dim = total_input_dim - 1  # Exclude r_t\n",
        "        self.kernel = self.add_weight(shape=(actual_input_dim, self.units * 4), initializer='glorot_uniform', name='kernel')\n",
        "        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 4), initializer='orthogonal', name='recurrent_kernel')\n",
        "        self.bias = self.add_weight(shape=(self.units * 4,), initializer='zeros', name='bias')\n",
        "        self.fuzzy_gate_kernel = self.add_weight(shape=(1, self.units * 3), initializer='glorot_uniform', name='fuzzy_gate_kernel')\n",
        "\n",
        "    def call(self, inputs, states):\n",
        "        u_tm1 = states[0]\n",
        "        x_t = inputs[:, :-1]\n",
        "        r_t = inputs[:, -1:]\n",
        "        \n",
        "        z = tf.matmul(x_t, self.kernel) + tf.matmul(u_tm1, self.recurrent_kernel) + self.bias\n",
        "        z0, z1, z2, z3 = z[:, :self.units], z[:, self.units:2*self.units], z[:, 2*self.units:3*self.units], z[:, 3*self.units:]\n",
        "        \n",
        "        fz = tf.matmul(r_t, self.fuzzy_gate_kernel)\n",
        "        z0 = z0 + fz[:, :self.units]\n",
        "        z1 = z1 + fz[:, self.units:2*self.units]\n",
        "        z2 = z2 + fz[:, 2*self.units:]\n",
        "\n",
        "        r = tf.tanh(z0)\n",
        "        c = tf.clip_by_value(self.mf(z1), -1.0, 1.0)\n",
        "        o = tf.clip_by_value(self.mf(z2), -1.0, 1.0)\n",
        "        a = tf.tanh(z3)\n",
        "\n",
        "        u = r * u_tm1 - c * a\n",
        "        h = o * a\n",
        "        return h, [u, c, o]\n",
        "\n",
        "    def get_config(self):\n",
        "        config = super().get_config()\n",
        "        config.update({'units': self.units, 'mf_type': self.mf_type})\n",
        "        return config\n",
        "\n",
    ]


# ============================================================
# STRENGTHENING MEMORY LAYER
# ============================================================
MEMORY_LAYER_CODE = [
    "@tf.keras.utils.register_keras_serializable()\n",
    "class StrengtheningMemoryLayer(layers.Layer):\n",
    "    def __init__(self, units, **kwargs):\n",
    "        super().__init__(**kwargs)\n",
    "        self.units_out = units\n",
    "        self.dense = layers.Dense(units, activation='tanh')\n",
    "\n",
    "    def call(self, inputs):\n",
    "        h_t, c_t = inputs\n",
    "        ch_t = h_t + c_t\n",
    "        s_t = self.dense(ch_t)\n",
    "        h_hat = ch_t + s_t\n",
    "        return h_hat\n",
    "\n",
    "    def get_config(self):\n",
    "        config = super().get_config()\n",
    "        config.update({'units': self.units_out})\n",
    "        return config\n",
    "\n",
]


# ============================================================
# WM FUZZY OUTPUT LAYER (only for Gates_Plus_FuzzyOutput)
# ============================================================
FUZZY_OUTPUT_LAYER_CODE = [
    "@tf.keras.utils.register_keras_serializable()\n",
    "class WMFuzzyOutputLayer(layers.Layer):\n",
    "    def __init__(self, units_in, q=5, **kwargs):\n",
    "        super().__init__(**kwargs)\n",
    "        self.units_in = units_in\n",
    "        self.q = q\n",
    "\n",
    "    def build(self, input_shape):\n",
    "        num_rules = self.q * self.q\n",
    "        self.rule_a = self.add_weight(shape=(num_rules,), initializer='glorot_uniform', name='rule_a')\n",
    "        self.rule_b = self.add_weight(shape=(num_rules,), initializer='glorot_uniform', name='rule_b')\n",
    "        self.rule_c = self.add_weight(shape=(num_rules,), initializer='zeros', name='rule_c')\n",
    "\n",
    "    def _triangular_mf(self, x, center, width):\n",
    "        left = center - width\n",
    "        right = center + width\n",
    "        return tf.maximum(0.0, tf.minimum((x - left) / (center - left + 1e-10),\n",
    "                                          (right - x) / (right - center + 1e-10)))\n",
    "\n",
    "    def call(self, inputs):\n",
    "        half = self.units_in // 2\n",
    "        s1 = tf.reduce_mean(inputs[:, :half], axis=-1, keepdims=True)\n",
    "        s2 = tf.reduce_mean(inputs[:, half:], axis=-1, keepdims=True)\n",
    "        centers = tf.linspace(-1.0, 1.0, self.q)\n",
    "        width = 2.0 / tf.cast(self.q - 1, tf.float32)\n",
    "        numerator = tf.zeros_like(s1)\n",
    "        denominator = tf.zeros_like(s1)\n",
    "        for i in range(self.q):\n",
    "            mu_s1 = self._triangular_mf(s1, centers[i], width)\n",
    "            for j in range(self.q):\n",
    "                mu_s2 = self._triangular_mf(s2, centers[j], width)\n",
    "                rule_idx = i * self.q + j\n",
    "                w = mu_s1 * mu_s2\n",
    "                y = self.rule_a[rule_idx] * s1 + self.rule_b[rule_idx] * s2 + self.rule_c[rule_idx]\n",
    "                numerator = numerator + w * y\n",
    "                denominator = denominator + w\n",
    "        return numerator / (denominator + 1e-8)\n",
    "\n",
    "    def get_config(self):\n",
    "        config = super().get_config()\n",
    "        config.update({'units_in': self.units_in, 'q': self.q})\n",
    "        return config\n",
    "\n",
]


# ============================================================
# BUILD MODEL CODE
# ============================================================
def get_build_model_code(arch, mf_type):
    mf_key = mf_type.lower()
    if arch == 'Gates_Plus_FuzzyOutput':
        return [
            "def build_model(input_dim, units=8, batch_size=1):\n",
            f"    cell = MembershipFLSTMSNPCell(units, mf_type='{mf_key}')\n",
            "    rnn = layers.RNN(cell, return_sequences=False, return_state=True, stateful=True)\n",
            "\n",
            "    inputs = tf.keras.Input(batch_shape=(batch_size, 1, input_dim))\n",
            "    x, u_out, c_out, o_out = rnn(inputs)\n",
            "    h_strengthened = StrengtheningMemoryLayer(units)([x, u_out])\n",
            "    outputs = WMFuzzyOutputLayer(units)(h_strengthened)\n",
            "\n",
            "    model = tf.keras.Model(inputs=inputs, outputs=[outputs, c_out, o_out])\n",
            "    model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0), loss=['mean_squared_error', None, None])\n",
            "    return model\n",
            "\n",
        ]
    else:
        return [
            "def build_model(input_dim, units=8, batch_size=1):\n",
            f"    cell = MembershipFLSTMSNPCell(units, mf_type='{mf_key}')\n",
            "    rnn = layers.RNN(cell, return_sequences=False, return_state=True, stateful=True)\n",
            "\n",
            "    inputs = tf.keras.Input(batch_shape=(batch_size, 1, input_dim))\n",
            "    x, u_out, c_out, o_out = rnn(inputs)\n",
            "    h_strengthened = StrengtheningMemoryLayer(units)([x, u_out])\n",
            "    outputs = layers.Dense(1)(h_strengthened)\n",
            "\n",
            "    model = tf.keras.Model(inputs=inputs, outputs=[outputs, c_out, o_out])\n",
            "    model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0), loss=['mean_squared_error', None, None])\n",
            "    return model\n",
            "\n",
        ]


# ============================================================
# WANG-MENDEL FUZZY SYSTEM (NO complement_wm_rules)
# ============================================================
WM_SYSTEM_CODE = [
    "# ------------------------------------------------------------\n",
    "# 3. WANG-MENDEL FUZZY RULE BASE (No complement rules)\n",
    "# ------------------------------------------------------------\n",
    "Q_REGIONS = 5\n",
    "\n",
    "def triangular_mf_np(x, a, b, c):\n",
    "    return np.maximum(0.0, np.minimum((x - a) / (b - a + 1e-10), (c - x) / (c - b + 1e-10)))\n",
    "\n",
    "def build_triangular_fuzzy_sets(data_min, data_max, q=Q_REGIONS):\n",
    "    centers = np.linspace(data_min, data_max, q)\n",
    "    step = centers[1] - centers[0] if q > 1 else 1.0\n",
    "    fuzzy_sets = [(center - step, center, center + step) for center in centers]\n",
    "    return fuzzy_sets, centers\n",
    "\n",
    "def fuzzify_value(value, fuzzy_sets):\n",
    "    best_idx, best_mu = 0, 0.0\n",
    "    for i, (a, b, c) in enumerate(fuzzy_sets):\n",
    "        mu = triangular_mf_np(value, a, b, c)\n",
    "        if mu > best_mu:\n",
    "            best_mu = mu\n",
    "            best_idx = i\n",
    "    return best_idx, max(best_mu, 1e-10)\n",
    "\n",
    "def extract_wm_rules(X, y, fs_x_list, fs_y):\n",
    "    rules = {}\n",
    "    n_features = X.shape[1]\n",
    "    for i in range(len(X)):\n",
    "        antecedents = []\n",
    "        weight = 1.0\n",
    "        for j in range(n_features):\n",
    "            idx, mu = fuzzify_value(X[i, j], fs_x_list[j])\n",
    "            antecedents.append(idx)\n",
    "            weight *= mu\n",
    "        cons_idx, cons_mu = fuzzify_value(y[i], fs_y)\n",
    "        weight *= cons_mu\n",
    "        key = tuple(antecedents)\n",
    "        if key not in rules or weight > rules[key][1]:\n",
    "            rules[key] = (cons_idx, weight)\n",
    "    return rules\n",
    "\n",
    "def wm_fuzzy_predict(x_features, rules, fs_x_list, centers_y, q=Q_REGIONS):\n",
    "    n_features = len(x_features)\n",
    "    memberships = []\n",
    "    for j in range(n_features):\n",
    "        mf_vals = [triangular_mf_np(x_features[j], fs_x_list[j][k][0], fs_x_list[j][k][1], fs_x_list[j][k][2]) for k in range(q)]\n",
    "        memberships.append(mf_vals)\n",
    "    \n",
    "    total_weight = 0.0\n",
    "    weighted_sum = 0.0\n",
    "    for antecedents, (consequent, _) in rules.items():\n",
    "        strength = 1.0\n",
    "        for j, ant_idx in enumerate(antecedents):\n",
    "            strength *= memberships[j][ant_idx]\n",
    "        if strength > 1e-10:\n",
    "            weighted_sum += strength * centers_y[consequent]\n",
    "            total_weight += strength\n",
    "            \n",
    "    if total_weight > 1e-10:\n",
    "        return weighted_sum / total_weight\n",
    "    # Fallback default value if no rules trigger\n",
    "    return centers_y[q // 2]\n",
    "\n",
    "def build_wm_system(X_train, y_train, q=Q_REGIONS):\n",
    "    n_features = X_train.shape[1]\n",
    "    fs_x_list = []\n",
    "    for j in range(n_features):\n",
    "        col = X_train[:, j]\n",
    "        fs, _ = build_triangular_fuzzy_sets(col.min(), col.max(), q)\n",
    "        fs_x_list.append(fs)\n",
    "    fs_y, centers_y = build_triangular_fuzzy_sets(y_train.min(), y_train.max(), q)\n",
    "    \n",
    "    # Extract rules without creating full combinatorial grid\n",
    "    rules = extract_wm_rules(X_train, y_train, fs_x_list, fs_y)\n",
    "    return rules, fs_x_list, centers_y\n",
    "\n",
    "def compute_fuzzy_predictions(X, rules, fs_x_list, centers_y, q=Q_REGIONS):\n",
    "    preds = np.zeros(len(X))\n",
    "    for i in range(len(X)):\n",
    "        preds[i] = wm_fuzzy_predict(X[i], rules, fs_x_list, centers_y, q)\n",
    "    return preds\n",
    "\n",
]


# ============================================================
# BUILD FULL NOTEBOOK
# ============================================================
def build_notebook(arch, mf_type, dataset, threshold_type, folder_label):
    arch_label = ARCH_LABELS[arch]
    ds_label = DATASET_LABELS[dataset]
    ds_key = DATASET_KEYS[dataset]
    thresh_label = THRESHOLD_LABELS[threshold_type]

    # Title markdown cell
    title_cell = {
        "cell_type": "markdown",
        "source": f"# FLSTM {arch_label} using {mf_type}\nDataset: {ds_label} — Lag={LAG_VALUE}\n\n**Classification Metrics Evaluation ({thresh_label} Threshold, No Noise)**",
        "metadata": {}
    }

    # Build the single self-contained code cell
    src = []

    # --- Imports ---
    src.extend([
        "import os\n",
        "import glob\n",
        "import warnings\n",
        "import gc\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from math import sqrt\n",
        "import matplotlib.pyplot as plt\n",
        "import tensorflow as tf\n",
        "from tensorflow.keras import layers\n",
        "from sklearn.preprocessing import MinMaxScaler\n",
        "from sklearn.metrics import mean_squared_error, roc_auc_score\n",
        "\n",
        "# Suppress expected UserWarnings\n",
        "warnings.filterwarnings('ignore', category=UserWarning, module='keras')\n",
        "\n",
        "# ------------------------------------------------------------\n",
    ])

    # --- Data Loading ---
    src.extend(get_data_loading_code(dataset))

    # --- Lag Setup ---
    src.extend([
        "\n",
        "# ------------------------------------------------------------\n",
        "# 2. TIME SERIES & LAG SETUP\n",
        "# ------------------------------------------------------------\n",
        f"LAG_STEPS = {LAG_VALUE}  # Configurable lag step parameter\n",
        "\n",
        "def difference(dataset, interval=1):\n",
        "    diff = []\n",
        "    for i in range(interval, len(dataset)):\n",
        "        diff.append(dataset[i] - dataset[i - interval])\n",
        "    return np.array(diff)\n",
        "\n",
        "def timeseries_to_supervised(data, lag):\n",
        "    df = pd.DataFrame(data)\n",
        "    columns = [df.shift(i) for i in range(1, lag+1)]\n",
        "    columns.append(df)\n",
        "    df = pd.concat(columns, axis=1)\n",
        "    df.fillna(0, inplace=True)\n",
        "    return df.values\n",
        "\n",
        "diff_values = difference(raw_values, 1)\n",
        "supervised = timeseries_to_supervised(diff_values, LAG_STEPS)\n",
        "\n",
        "train, test = supervised[:-60], supervised[-60:]\n",
        "scaler = MinMaxScaler(feature_range=(-1, 1))\n",
        "train_scaled = scaler.fit_transform(train)\n",
        "test_scaled = scaler.transform(test)\n",
        "\n",
        "X_train_raw, y_train = train_scaled[:, 0:-1], train_scaled[:, -1]\n",
        "X_test_raw, y_test = test_scaled[:, 0:-1], test_scaled[:, -1]\n",
        "\n",
    ])

    # --- WM Fuzzy System ---
    src.extend(WM_SYSTEM_CODE)

    src.extend([
        "# Build WM fuzzy system from dynamic-lag training data\n",
        "wm_rules, wm_fs_x, wm_centers_y = build_wm_system(X_train_raw, y_train)\n",
        "\n",
        "# Compute fuzzy predictions r_t\n",
        "r_train = compute_fuzzy_predictions(X_train_raw, wm_rules, wm_fs_x, wm_centers_y)\n",
        "r_test = compute_fuzzy_predictions(X_test_raw, wm_rules, wm_fs_x, wm_centers_y)\n",
        "\n",
        "# Concatenate r_t as last feature: [x_t, r_t]\n",
        "X_train_aug = np.column_stack([X_train_raw, r_train])\n",
        "X_test_aug = np.column_stack([X_test_raw, r_test])\n",
        "\n",
        "num_features = X_train_aug.shape[1]  # Equal to LAG_STEPS + 1\n",
        "\n",
        "X_train = X_train_aug.reshape((X_train_aug.shape[0], 1, num_features))\n",
        "X_test = X_test_aug.reshape((X_test_aug.shape[0], 1, num_features))\n",
        "\n",
        "# ------------------------------------------------------------\n",
        f"# 4. FLSTM CELL WITH {mf_type} & MEMORY LAYER\n",
        "# ------------------------------------------------------------\n",
    ])

    # --- MF Function ---
    src.extend(get_mf_code(mf_type))

    # --- FLSTM Cell ---
    src.extend(get_cell_code(mf_type))

    # --- Strengthening Memory Layer ---
    src.extend(MEMORY_LAYER_CODE)

    # --- WM Fuzzy Output Layer (only for FuzzyOutput arch) ---
    if arch == 'Gates_Plus_FuzzyOutput':
        src.extend(FUZZY_OUTPUT_LAYER_CODE)

    # --- Build Model ---
    src.extend(get_build_model_code(arch, mf_type))

    # --- Training & Evaluation ---
    src.extend([
        "# ------------------------------------------------------------\n",
        f"# 5. TRAINING & EVALUATION ({NUM_RUNS} Runs)\n",
        "# ------------------------------------------------------------\n",
        "print('\\n' + '='*80)\n",
        f"print(f'EVALUATING: {arch_label} ({mf_type}) on {ds_label} — Lag Steps: {{LAG_STEPS}}')\n",
        "print('='*80 + '\\n')\n",
        "\n",
        "all_rmse, all_mae, all_predictions, all_losses = [], [], [], []\n",
        "all_accuracy, all_specificity, all_precision, all_recall, all_auc = [], [], [], [], []\n",
        "\n",
        f"for run in range({NUM_RUNS}):\n",
        f"    print(f'\\\\n===== RUN {{run+1}}/{NUM_RUNS} =====')\n",
        "    np.random.seed(run)\n",
        "    tf.random.set_seed(run)\n",
        "    \n",
        "    tf.keras.backend.clear_session()\n",
        "    gc.collect()\n",
        "\n",
        "    model = build_model(input_dim=num_features, units=8, batch_size=1)\n",
        "    rnn_layer = model.layers[1]\n",
        "\n",
        "    run_losses = []\n",
        f"    for epoch in range({NUM_EPOCHS}):\n",
        "        history = model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=0, shuffle=False)\n",
        "        run_losses.append(history.history['loss'][0])\n",
        "        rnn_layer.reset_states()\n",
        "    all_losses.append(run_losses)\n",
        "\n",
        "    # Warmup RNN states (Direct model call avoids memory build-up)\n",
        "    for i in range(len(X_train)): \n",
        "        _ = model(X_train[i:i+1], training=False)\n",
        "\n",
        "    predictions = []\n",
        "    for i in range(len(X_test)):\n",
        "        # Direct functional call instead of model.predict inside loops\n",
        "        yhat, c_val, o_val = model(X_test[i:i+1], training=False)\n",
        "\n",
        "        row = list(X_test_raw[i]) + [yhat.numpy()[0, 0]]\n",
        "        diff_pred = scaler.inverse_transform([row])[0, -1]\n",
        "        \n",
        "        prev_actual_idx = len(train) + i\n",
        "        inv = diff_pred + raw_values[prev_actual_idx]\n",
        "        predictions.append(inv)\n",
        "\n",
        "    actual = raw_values[-60:]\n",
        "    rmse = sqrt(mean_squared_error(actual, predictions))\n",
        "    mae = np.mean(np.abs(np.array(actual) - np.array(predictions)))\n",
        "\n",
        "    all_rmse.append(rmse)\n",
        "    all_mae.append(mae)\n",
        "    all_predictions.append(predictions)\n",
        "\n",
    ])

    # --- Classification Metrics (threshold-specific) ---
    src.extend(get_threshold_code(threshold_type))

    src.extend([
        "    y_prev = raw_values[-(len(actual)+1):-1]\n",
        "    true_dir = ((np.array(actual) - y_prev) > tau).astype(int)\n",
        "    pred_dir = ((np.array(predictions) - y_prev) > tau).astype(int)\n",
        "    pred_scores = (np.array(predictions) - y_prev) - tau\n",
        "\n",
        "    TP = np.sum((true_dir == 1) & (pred_dir == 1))\n",
        "    TN = np.sum((true_dir == 0) & (pred_dir == 0))\n",
        "    FP = np.sum((true_dir == 0) & (pred_dir == 1))\n",
        "    FN = np.sum((true_dir == 1) & (pred_dir == 0))\n",
        "\n",
        "    accuracy = (TP + TN) / len(true_dir) if len(true_dir) > 0 else 0.0\n",
        "    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0\n",
        "    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0\n",
        "    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0\n",
        "    try:\n",
        "        auc = roc_auc_score(true_dir, pred_scores)\n",
        "    except ValueError:\n",
        "        auc = 0.5\n",
        "\n",
        "    all_accuracy.append(accuracy)\n",
        "    all_specificity.append(specificity)\n",
        "    all_precision.append(precision)\n",
        "    all_recall.append(recall)\n",
        "    all_auc.append(auc)\n",
        "\n",
        "    print(f'Run {run+1} \\u2014 Acc: {accuracy*100:.1f}%, Spec: {specificity*100:.1f}%, Prec: {precision*100:.1f}%, Recall: {recall*100:.1f}%, AUC: {auc*100:.1f}%')\n",
        "\n",
    ])

    # --- Summary & Plots ---
    src.extend([
        "# ------------------------------------------------------------\n",
        "# 6. SUMMARY & PLOTS\n",
        "# ------------------------------------------------------------\n",
        "mean_accuracy = np.mean(all_accuracy) * 100\n",
        "mean_specificity = np.mean(all_specificity) * 100\n",
        "mean_precision = np.mean(all_precision) * 100\n",
        "mean_recall = np.mean(all_recall) * 100\n",
        "mean_auc = np.mean(all_auc) * 100\n",
        "\n",
        f"print(f'\\\\n===== FINAL RESULTS \\u2014 {arch_label} ({mf_type}) on {ds_label} ({{NUM_RUNS}} runs, Lag={{LAG_STEPS}}) =====')\n".replace('{NUM_RUNS}', str(NUM_RUNS)),
        "print(f'Accuracy:    {mean_accuracy:.1f}%')\n",
        "print(f'Specificity: {mean_specificity:.1f}%')\n",
        "print(f'Precision:   {mean_precision:.1f}%')\n",
        "print(f'Recall:      {mean_recall:.1f}%')\n",
        "print(f'AUC:         {mean_auc:.1f}%')\n",
        "\n",
        "best_idx = np.argmin(all_rmse)\n",
        "actual = raw_values[-60:]\n",
        "best_predictions = all_predictions[best_idx]\n",
        "\n",
        "plt.figure(figsize=(12, 5))\n",
        "plt.plot(actual, label='Actual', color='blue', linewidth=1.5)\n",
        "plt.plot(best_predictions, label='Predicted (Best Run)', color='purple', linewidth=1.5, linestyle='--')\n",
        f"plt.title(f'{arch_label} ({mf_type}) \\u2014 {ds_label} (Lag={{LAG_STEPS}})\\\\nPredictions vs Actual (Best Run)')\n",
        "plt.xlabel('Time Step')\n",
        "plt.ylabel('Value')\n",
        "plt.legend()\n",
        "plt.grid(True, alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# ============================================================\n",
        "# COMBINED CLASSIFICATION SUMMARY TABLE\n",
        "# ============================================================\n",
        "print('\\n' + '\\u2554' + '\\u2550'*85 + '\\u2557')\n",
        f"print('\\u2551 CLASSIFICATION RESULTS \\u2014 {arch_label} ({mf_type}) on {ds_label} (Lag={{LAG_STEPS}}) '.ljust(86) + '\\u2551')\n".replace('{LAG_STEPS}', '{LAG_STEPS}'),
        "print('\\u2560' + '\\u2550'*15 + '\\u2566' + '\\u2550'*15 + '\\u2566' + '\\u2550'*15 + '\\u2566' + '\\u2550'*15 + '\\u2566' + '\\u2550'*18 + '\\u2563')\n",
        "print('\\u2551 Accuracy      \\u2551 Specificity   \\u2551 Precision     \\u2551 Recall        \\u2551 AUC              \\u2551')\n",
        "print('\\u2560' + '\\u2550'*15 + '\\u256c' + '\\u2550'*15 + '\\u256c' + '\\u2550'*15 + '\\u256c' + '\\u2550'*15 + '\\u256c' + '\\u2550'*18 + '\\u2563')\n",
        "acc_str = f'{mean_accuracy:.1f}%'\n",
        "spec_str = f'{mean_specificity:.1f}%'\n",
        "prec_str = f'{mean_precision:.1f}%'\n",
        "rec_str = f'{mean_recall:.1f}%'\n",
        "auc_str = f'{mean_auc:.1f}%'\n",
        "print(f'\\u2551 {acc_str:<13} \\u2551 {spec_str:<13} \\u2551 {prec_str:<13} \\u2551 {rec_str:<13} \\u2551 {auc_str:<16} \\u2551')\n",
        "print('\\u255a' + '\\u2550'*15 + '\\u2569' + '\\u2550'*15 + '\\u2569' + '\\u2550'*15 + '\\u2569' + '\\u2550'*15 + '\\u2569' + '\\u2550'*18 + '\\u255d')\n",
    ])

    # Build the code cell
    main_cell = {
        "cell_type": "code",
        "source": src,
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None
    }

    # Timer start cell
    timer_start_cell = {
        "cell_type": "code",
        "source": "# ============================================================\n# NOTEBOOK TIMER \u2014 START\n# ============================================================\nimport time as _timer_module\n_NOTEBOOK_START_TIME = _timer_module.time()\nprint(f\"Notebook execution started at: {_timer_module.strftime('%Y-%m-%d %H:%M:%S')}\")\n",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None
    }

    # Timer end cell
    timer_end_cell = {
        "cell_type": "code",
        "source": "# ============================================================\n# NOTEBOOK TIMER \u2014 END\n# ============================================================\nimport time as _timer_module\n_NOTEBOOK_END_TIME = _timer_module.time()\n_NOTEBOOK_ELAPSED = _NOTEBOOK_END_TIME - _NOTEBOOK_START_TIME\n_hours, _rem = divmod(_NOTEBOOK_ELAPSED, 3600)\n_minutes, _seconds = divmod(_rem, 60)\nprint(f\"\\nTotal notebook execution time: {int(_hours)}h {int(_minutes)}m {_seconds:.2f}s\")\nprint(f\"Total seconds: {_NOTEBOOK_ELAPSED:.2f}\")\n",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None
    }

    # PID cell
    pid_cell = {
        "cell_type": "code",
        "source": "# ============================================================\n# PROCESS IDENTIFICATION\n# ============================================================\nimport os\nprint(f\"Process ID (PID): {os.getpid()}\")\n",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None
    }

    # Observations markdown
    obs_cell = {
        "cell_type": "markdown",
        "source": f"## Observations\n\n### {arch_label} ({mf_type}) on {ds_label} — Lag={LAG_VALUE}\n\n**Run the notebook to generate results.**\n",
        "metadata": {}
    }

    cells = [
        title_cell,
        pid_cell,
        timer_start_cell,
        main_cell,
        obs_cell,
        timer_end_cell,
    ]

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": KERNEL_SPEC,
        "cells": cells
    }

    return notebook


# ============================================================
# MAIN: Generate all notebooks
# ============================================================
total = 0
for src_folder, dst_folder in FOLDERS:
    os.makedirs(dst_folder, exist_ok=True)
    dataset = get_dataset(src_folder)
    threshold_type = get_threshold_type(src_folder)

    for arch in ARCHS:
        for mf in MFS:
            nb_name = f"FLSTM_Classif_{arch}_{mf}_{DATASET_KEYS[dataset]}.ipynb"

            nb = build_notebook(arch, mf, dataset, threshold_type, dst_folder)

            path = os.path.join(dst_folder, nb_name)
            with open(path, 'w') as f:
                json.dump(nb, f, indent=1)

            total += 1
            print(f"Created: {path}")

print(f"\nTotal notebooks generated: {total}")
print(f"Folders created: {len(FOLDERS)}")
