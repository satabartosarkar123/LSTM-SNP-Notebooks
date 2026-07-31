#!/usr/bin/env python3
"""
Master Classification Notebook Generator.
Generates all 99 notebooks across 11 Classification folders with dynamic lag (LAG_STEPS) and CUDA GPU acceleration.

Folders generated:
  1. FLSTM_Classification_Sunspots_Mode
  2. FLSTM_Classification_Sunspots_Median
  3. FLSTM_Classification_Sunspots_Huber
  4. FLSTM_Classification_Sunspots
  5. FLSTM_Classification_MackeyGlass_Mode
  6. FLSTM_Classification_MackeyGlass_Midrange
  7. FLSTM_Classification_MackeyGlass_Median
  8. FLSTM_Classification_MackeyGlass
  9. FLSTM_Classification_EPL_Mode
 10. FLSTM_Classification_EPL_Median
 11. FLSTM_Classification_EPL
"""

import json
import os
import copy

# Reference notebook reader
def read_ref(path):
    with open(path) as f:
        return json.load(f)

archs = {
    'Gates_Only': 'Gates_Only',
    'Gates_Plus_FuzzyInput': 'Gates_Plus_FuzzyInput',
    'Gates_Plus_FuzzyOutput': 'Gates_Plus_FuzzyOutput',
}

mf_map = {'dog': 'DOG', 'sg': 'SG', 'gb': 'GB'}

def get_ref_path(arch_key, mf):
    return f'FLSTM_Fuzzy_MF_DowJones/FLSTM_MF_{arch_key}_{mf_map[mf]}_dow_jones.ipynb'

def extract_cell_by_content(nb, marker):
    for cell in nb['cells']:
        src = ''.join(cell['source'])
        if marker in src:
            return copy.deepcopy(cell)
    return None

print("Extracting reference cells from FLSTM_Fuzzy_MF_DowJones...")
ref_cells = {}
for arch_key in archs:
    ref_cells[arch_key] = {}
    for mf in ['dog', 'sg', 'gb']:
        path = get_ref_path(arch_key, mf)
        nb = read_ref(path)
        ref_cells[arch_key][mf] = {
            'pid': extract_cell_by_content(nb, 'Process ID (PID)'),
            'timer_start': extract_cell_by_content(nb, 'NOTEBOOK TIMER — START'),
            'imports': extract_cell_by_content(nb, 'from sklearn.preprocessing import MinMaxScaler'),
            'wm_fuzzy': extract_cell_by_content(nb, 'build_wm_system'),
            'mf_cell': extract_cell_by_content(nb, 'class MembershipFLSTMSNPCell'),
            'build_model': extract_cell_by_content(nb, 'def build_model'),
            'timer_end': extract_cell_by_content(nb, 'NOTEBOOK TIMER — END'),
        }
        if arch_key == 'Gates_Plus_FuzzyOutput':
            ref_cells[arch_key][mf]['fuzzy_output_layer'] = extract_cell_by_content(nb, 'class WMFuzzyOutputLayer')

# Device setup cell (CUDA GPU Acceleration)
device_setup_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================\n",
        "# Device Setup (CUDA GPU Acceleration Enabled)\n",
        "# ============================================================\n",
        "import tensorflow as tf\n",
        "\n",
        "gpus = tf.config.list_physical_devices('GPU')\n",
        "if gpus:\n",
        "    try:\n",
        "        for gpu in gpus:\n",
        "            tf.config.experimental.set_memory_growth(gpu, True)\n",
        "        print(f\"CUDA Acceleration Enabled: {len(gpus)} GPU(s) detected.\")\n",
        "    except RuntimeError as e:\n",
        "        print(f\"GPU setup error: {e}\")\n",
        "else:\n",
        "    print(\"Running on CPU.\")\n"
    ]
}

# Data loading cell generators
def get_data_loader_source(dataset_key):
    if dataset_key == 'sunspots':
        return [
            "import os as _os\n",
            "file_path = next((p for p in ['../content/Sunspots.csv', 'content/Sunspots.csv', '/Users/satabarto/Research/content/Sunspots.csv'] if _os.path.exists(p)), '../content/Sunspots.csv')\n",
            "series = pd.read_csv(file_path)\n",
            "raw_values = series['Monthly Mean Total Sunspot Number'].values.flatten()\n"
        ]
    elif dataset_key == 'mackey_glass':
        return [
            "import openpyxl\n",
            "import os as _os\n",
            "file_path = next((p for p in ['../content/Mackey-Glass Time Series(taw17).xlsx', 'content/Mackey-Glass Time Series(taw17).xlsx', '/Users/satabarto/Research/content/Mackey-Glass Time Series(taw17).xlsx'] if _os.path.exists(p)), '../content/Mackey-Glass Time Series(taw17).xlsx')\n",
            "series = pd.read_excel(file_path)\n",
            "raw_values = series['t+1'].values.flatten()\n"
        ]
    elif dataset_key == 'epl':
        return [
            "import os as _os\n",
            "data_dirs = ['../content/English Premier League Dataset/data', 'content/English Premier League Dataset/data', '/Users/satabarto/Research/content/English Premier League Dataset/data', '/kaggle/input/datasets/saurabhshahane/english-premier-league-dataset']\n",
            "data_dir = next((d for d in data_dirs if _os.path.exists(d)), '../content/English Premier League Dataset/data')\n",
            "all_dfs = []\n",
            "for _f in sorted(_os.listdir(data_dir)):\n",
            "    if _f.endswith('_csv.csv'):\n",
            "        df = pd.read_csv(_os.path.join(data_dir, _f))\n",
            "        df['TotalGoals'] = df['FTHG'] + df['FTAG']\n",
            "        all_dfs.append(df)\n",
            "combined = pd.concat(all_dfs, ignore_index=True)\n",
            "raw_values = combined['TotalGoals'].values.astype(float)\n"
        ]

def get_threshold_code(threshold_type):
    if threshold_type == 'mode':
        return [
            "    # ---- Classification Metrics (Dynamic Threshold: Mode) ----\n",
            "    import scipy.stats as stats\n",
            "    train_raw_values = raw_values[:-60]\n",
            "    train_diffs = np.diff(train_raw_values)\n",
            "    rounded_diffs = np.round(train_diffs, 3)\n",
            "    mode_res = stats.mode(rounded_diffs, keepdims=True)\n",
            "    tau = float(mode_res.mode[0]) if hasattr(mode_res.mode, '__len__') else float(mode_res.mode)\n",
        ]
    elif threshold_type == 'median':
        return [
            "    # ---- Classification Metrics (Dynamic Threshold: Median) ----\n",
            "    train_raw_values = raw_values[:-60]\n",
            "    train_diffs = np.diff(train_raw_values)\n",
            "    tau = round(float(np.median(train_diffs)), 6)\n",
        ]
    elif threshold_type == 'huber':
        return [
            "    # ---- Classification Metrics (Dynamic Threshold: Huber) ----\n",
            "    tau = -0.035150\n",
        ]
    elif threshold_type == 'midrange':
        return [
            "    # ---- Classification Metrics (Dynamic Threshold: Midrange) ----\n",
            "    train_raw_values = raw_values[:-60]\n",
            "    train_diffs = np.diff(train_raw_values)\n",
            "    tau = round(float(0.5 * (np.min(train_diffs) + np.max(train_diffs))), 6)\n",
        ]
    elif threshold_type == 'mean':
        return [
            "    # ---- Classification Metrics (Dynamic Threshold: Mean) ----\n",
            "    train_raw_values = raw_values[:-60]\n",
            "    train_diffs = np.diff(train_raw_values)\n",
            "    tau = round(float(np.mean(train_diffs)), 6)\n",
        ]
    else: # default per dataset
        return [
            "    # ---- Classification Metrics (Dynamic Threshold: Default) ----\n",
            "    train_raw_values = raw_values[:-60]\n",
            "    train_diffs = np.diff(train_raw_values)\n",
            "    tau = round(float(np.median(train_diffs)), 6)\n",
        ]

def build_main_cell(arch_label, mf_label, dataset_label, dataset_key, threshold_type):
    dl = get_data_loader_source(dataset_key)
    thresh_code = get_threshold_code(threshold_type)
    
    src = dl + [
        "import numpy as np\n",
        "original_raw_values = np.copy(raw_values)\n",
        "LAG_STEPS = 5  # Dynamic lag parameter\n",
        "\n",
        "# Classification metrics imports\n",
        "from sklearn.metrics import precision_score, recall_score, roc_auc_score\n",
        "\n",
        "def difference(dataset, interval=1):\n",
        "    diff = []\n",
        "    for i in range(interval, len(dataset)):\n",
        "        value = dataset[i] - dataset[i - interval]\n",
        "        diff.append(value)\n",
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
        "train, test = supervised[:-60], supervised[-60:]\n",
        "scaler = MinMaxScaler(feature_range=(-1, 1))\n",
        "train_scaled = scaler.fit_transform(train)\n",
        "test_scaled = scaler.transform(test)\n",
        "\n",
        "X_train_raw, y_train = train_scaled[:, 0:-1], train_scaled[:, -1]\n",
        "X_test_raw, y_test = test_scaled[:, 0:-1], test_scaled[:, -1]\n",
        "\n",
        "# Build WM fuzzy system from training data\n",
        "wm_rules, wm_fs_x, wm_centers_y = build_wm_system(X_train_raw, y_train)\n",
        "\n",
        "# Compute WM fuzzy predictions r_t for gate fusion\n",
        "r_train = compute_fuzzy_predictions(X_train_raw, wm_rules, wm_fs_x, wm_centers_y)\n",
        "r_test = compute_fuzzy_predictions(X_test_raw, wm_rules, wm_fs_x, wm_centers_y)\n",
        "\n",
        "# Concatenate r_t as last feature: [x_t, r_t]\n",
        "X_train_aug = np.column_stack([X_train_raw, r_train])\n",
        "X_test_aug = np.column_stack([X_test_raw, r_test])\n",
        "num_features = X_train_aug.shape[1]\n",
        "\n",
        "X_train = X_train_aug.reshape((X_train_aug.shape[0], 1, num_features))\n",
        "X_test = X_test_aug.reshape((X_test_aug.shape[0], 1, num_features))\n",
        "\n",
        "# ============================================================\n",
        "# TRAINING & EVALUATION (2 runs, No Noise)\n",
        "# ============================================================\n",
        f"print('\\n' + '='*80)\n",
        f"print('EVALUATING: {arch_label} ({mf_label.upper()}) on {dataset_label} — Lag Steps: {{LAG_STEPS}}')\n",
        f"print('='*80 + '\\n')\n",
        "\n",
        "all_rmse, all_mae, all_predictions, all_losses = [], [], [], []\n",
        "all_c_means, all_c_stds, all_c_mins, all_c_maxs = [], [], [], []\n",
        "all_o_means, all_o_stds, all_o_mins, all_o_maxs = [], [], [], []\n",
        "all_accuracy, all_specificity, all_precision, all_recall, all_auc = [], [], [], [], []\n",
        "\n",
        "for run in range(2):\n",
        "    print(f'\\n===== RUN {run+1}/2 =====')\n",
        "    np.random.seed(run)\n",
        "    tf.random.set_seed(run)\n",
        "    tf.keras.backend.clear_session()\n",
        "\n",
        "    model = build_model(input_dim=num_features, units=8, batch_size=1)\n",
        "    rnn_layer = model.layers[1]\n",
        "\n",
        "    run_losses = []\n",
        "    for epoch in range(100):\n",
        "        history = model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=0, shuffle=False)\n",
        "        run_losses.append(history.history['loss'][0])\n",
        "        rnn_layer.reset_states()\n",
        "    all_losses.append(run_losses)\n",
        "\n",
        "    # Warmup\n",
        "    for i in range(len(X_train)): model.predict(X_train[i:i+1], batch_size=1, verbose=0)\n",
        "\n",
        "    predictions, c_gates, o_gates = [], [], []\n",
        "    for i in range(len(X_test)):\n",
        "        yhat, c_val, o_val = model.predict(X_test[i:i+1], batch_size=1, verbose=0)\n",
        "        c_gates.append(c_val[0])\n",
        "        o_gates.append(o_val[0])\n",
        "        pred_val = yhat[0, 0] if hasattr(yhat, 'shape') else yhat\n",
        "        row = list(X_test_raw[i]) + [pred_val]\n",
        "        inv = scaler.inverse_transform([row])[0, -1] + raw_values[len(train) + i]\n",
        "        predictions.append(inv)\n",
        "\n",
        "    c_gates, o_gates = np.array(c_gates), np.array(o_gates)\n",
        "    all_c_means.append(np.mean(c_gates)); all_c_stds.append(np.std(c_gates))\n",
        "    all_c_mins.append(np.min(c_gates)); all_c_maxs.append(np.max(c_gates))\n",
        "    all_o_means.append(np.mean(o_gates)); all_o_stds.append(np.std(o_gates))\n",
        "    all_o_mins.append(np.min(o_gates)); all_o_maxs.append(np.max(o_gates))\n",
        "\n",
        "    actual = raw_values[-60:]\n",
        "    rmse = sqrt(mean_squared_error(actual, predictions))\n",
        "    mae = np.mean(np.abs(np.array(actual) - np.array(predictions)))\n",
        "\n",
        "    all_rmse.append(rmse)\n",
        "    all_mae.append(mae)\n",
        "    all_predictions.append(predictions)\n",
        "\n"
    ] + thresh_code + [
        "    y_prev = raw_values[-(len(actual)+1):-1]\n",
        "    true_dir = ((np.array(actual) - y_prev) > tau).astype(int)\n",
        "    pred_dir = ((np.array(predictions) - y_prev) > tau).astype(int)\n",
        "    pred_scores = (np.array(predictions) - y_prev) - tau  # confidence score for AUC\n",
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
        "    print(f'Run {run+1} — Acc: {accuracy*100:.1f}%, Spec: {specificity*100:.1f}%, Prec: {precision*100:.1f}%, Recall: {recall*100:.1f}%, AUC: {auc*100:.1f}%')\n",
        "\n",
        "# ============================================================\n",
        "# FINAL SUMMARY\n",
        "# ============================================================\n",
        "mean_accuracy = np.mean(all_accuracy) * 100\n",
        "mean_specificity = np.mean(all_specificity) * 100\n",
        "mean_precision = np.mean(all_precision) * 100\n",
        "mean_recall = np.mean(all_recall) * 100\n",
        "mean_auc = np.mean(all_auc) * 100\n",
        "\n",
        f"print('\\n===== FINAL RESULTS — {arch_label} ({mf_label.upper()}) on {dataset_label} (2 runs, Lag={{LAG_STEPS}}) =====')\n",
        "print(f'Accuracy:    {mean_accuracy:.1f}%')\n",
        "print(f'Specificity: {mean_specificity:.1f}%')\n",
        "print(f'Precision:   {mean_precision:.1f}%')\n",
        "print(f'Recall:      {mean_recall:.1f}%')\n",
        "print(f'AUC:         {mean_auc:.1f}%')\n",
        "# ---- Plots ----\n",
        "best_idx = np.argmin(all_rmse)\n",
        "actual = raw_values[-60:]\n",
        "best_predictions = all_predictions[best_idx]\n",
        "\n",
        "plt.figure(figsize=(12, 5))\n",
        "plt.plot(actual, label='Actual', color='blue', linewidth=1.5)\n",
        "plt.plot(best_predictions, label='Predicted (Best Run)', color='red', linewidth=1.5, linestyle='--')\n",
        f"plt.title('{arch_label} ({mf_label.upper()}) — {dataset_label}\\\\nPredictions vs Actual (Best of 2 runs)')\n",
        "plt.xlabel('Time Step')\n",
        "plt.ylabel('Value')\n",
        "plt.legend()\n",
        "plt.grid(True, alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "plt.figure(figsize=(12, 4))\n",
        "plt.plot(all_losses[best_idx], color='green', linewidth=1.0)\n",
        f"plt.title('{arch_label} ({mf_label.upper()}) — {dataset_label}\\\\nTraining Loss (Best Run)')\n",
        "plt.xlabel('Epoch')\n",
        "plt.ylabel('MSE Loss')\n",
        "plt.grid(True, alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "avg_predictions = np.mean(all_predictions, axis=0)\n",
        "std_predictions = np.std(all_predictions, axis=0)\n",
        "\n",
        "plt.figure(figsize=(12, 5))\n",
        "plt.plot(actual, label='Actual', color='blue', linewidth=1.5)\n",
        "plt.plot(avg_predictions, label='Mean Predicted (2 runs)', color='red', linewidth=1.5, linestyle='--')\n",
        "plt.fill_between(range(len(actual)), avg_predictions - std_predictions, avg_predictions + std_predictions,\n",
        "                 alpha=0.2, color='red', label='±1 Std Dev')\n",
        f"plt.title('{arch_label} ({mf_label.upper()}) — {dataset_label}\\\\nAverage Predictions vs Actual (2 runs ± 1σ)')\n",
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
        "print('\\n' + '╔' + '═'*85 + '╗')\n",
        f"print('║ CLASSIFICATION RESULTS — {arch_label} ({mf_label.upper()}) on {dataset_label} (2 runs) '.ljust(86) + '║')\n",
        "print('╠' + '═'*15 + '╦' + '═'*15 + '╦' + '═'*15 + '╦' + '═'*15 + '╦' + '═'*18 + '╣')\n",
        "print('║ Accuracy      ║ Specificity   ║ Precision     ║ Recall        ║ AUC              ║')\n",
        "print('╠' + '═'*15 + '╬' + '═'*15 + '╬' + '═'*15 + '╬' + '═'*15 + '╬' + '═'*18 + '╣')\n",
        "acc_str = f'{mean_accuracy:.1f}%'\n",
        "spec_str = f'{mean_specificity:.1f}%'\n",
        "prec_str = f'{mean_precision:.1f}%'\n",
        "rec_str = f'{mean_recall:.1f}%'\n",
        "auc_str = f'{mean_auc:.1f}%'\n",
        "print(f'║ {acc_str:<13} ║ {spec_str:<13} ║ {prec_str:<13} ║ {rec_str:<13} ║ {auc_str:<16} ║')\n",
        "print('╚' + '═'*15 + '╩' + '═'*15 + '╩' + '═'*15 + '╩' + '═'*15 + '╩' + '═'*18 + '╝')\n",
    ]
    return {
        "cell_type": "code",
        "metadata": {},
        "source": src,
        "outputs": [],
        "execution_count": None
    }

CLASSIFICATION_TARGETS = [
    {'dataset_key': 'sunspots', 'label': 'Sunspots', 'folder': 'FLSTM_Classification_Sunspots_Mode', 'threshold': 'mode'},
    {'dataset_key': 'sunspots', 'label': 'Sunspots', 'folder': 'FLSTM_Classification_Sunspots_Median', 'threshold': 'median'},
    {'dataset_key': 'sunspots', 'label': 'Sunspots', 'folder': 'FLSTM_Classification_Sunspots_Huber', 'threshold': 'huber'},
    {'dataset_key': 'sunspots', 'label': 'Sunspots', 'folder': 'FLSTM_Classification_Sunspots', 'threshold': 'default'},
    {'dataset_key': 'mackey_glass', 'label': 'Mackey-Glass', 'folder': 'FLSTM_Classification_MackeyGlass_Mode', 'threshold': 'mode'},
    {'dataset_key': 'mackey_glass', 'label': 'Mackey-Glass', 'folder': 'FLSTM_Classification_MackeyGlass_Midrange', 'threshold': 'midrange'},
    {'dataset_key': 'mackey_glass', 'label': 'Mackey-Glass', 'folder': 'FLSTM_Classification_MackeyGlass_Median', 'threshold': 'median'},
    {'dataset_key': 'mackey_glass', 'label': 'Mackey-Glass', 'folder': 'FLSTM_Classification_MackeyGlass', 'threshold': 'default'},
    {'dataset_key': 'epl', 'label': 'EPL', 'folder': 'FLSTM_Classification_EPL_Mode', 'threshold': 'mode'},
    {'dataset_key': 'epl', 'label': 'EPL', 'folder': 'FLSTM_Classification_EPL_Median', 'threshold': 'median'},
    {'dataset_key': 'epl', 'label': 'EPL', 'folder': 'FLSTM_Classification_EPL', 'threshold': 'default'},
]

arch_labels = {
    'Gates_Only': 'Gates Only',
    'Gates_Plus_FuzzyInput': 'Gates+FuzzyInput',
    'Gates_Plus_FuzzyOutput': 'Gates+FuzzyOutput',
}

KERNEL_SPEC = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.12.0"
    }
}

def main():
    total_count = 0
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for target in CLASSIFICATION_TARGETS:
        folder = os.path.join(base_dir, target['folder'])
        os.makedirs(folder, exist_ok=True)
        ds_key = target['dataset_key']
        ds_label = target['label']
        thresh_type = target['threshold']

        for arch_key, arch_label in arch_labels.items():
            for mf in ['dog', 'sg', 'gb']:
                nb_name = f"FLSTM_Classif_{arch_key}_{mf_map[mf]}_{ds_key}.ipynb"
                filepath = os.path.join(folder, nb_name)

                rc = ref_cells[arch_key][mf]

                title_cell = {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"# FLSTM {arch_label} using {mf_map[mf]}\n",
                        f"Dataset: {ds_label}\n",
                        "\n",
                        "**Classification Metrics Evaluation (No Noise)**"
                    ]
                }

                cells = [
                    title_cell,
                    rc['pid'],
                    rc['timer_start'],
                    device_setup_cell,
                    rc['imports'],
                    rc['wm_fuzzy'],
                    rc['mf_cell'],
                ]

                if arch_key == 'Gates_Plus_FuzzyOutput':
                    cells.append(rc['fuzzy_output_layer'])

                cells.append(rc['build_model'])
                cells.append(build_main_cell(arch_label, mf, ds_label, ds_key, thresh_type))

                # Observations & Timer End
                obs_cell = {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Observations\n",
                        "\n",
                        f"### {arch_label} ({mf_map[mf]}) on {ds_label}\n",
                        "\n",
                        "**Run the notebook to generate results.**\n"
                    ]
                }
                cells.append(obs_cell)
                cells.append(rc['timer_end'])

                nb_dict = {
                    "cells": cells,
                    "metadata": KERNEL_SPEC,
                    "nbformat": 4,
                    "nbformat_minor": 5
                }

                with open(filepath, 'w') as f:
                    json.dump(nb_dict, f, indent=1)

                total_count += 1
                print(f"  [{total_count:2d}/99] Created: {target['folder']}/{nb_name}")

    print(f"\nDone! Generated {total_count} classification notebooks across 11 folders.")

if __name__ == '__main__':
    main()
