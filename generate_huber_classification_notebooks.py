#!/usr/bin/env python3
"""
Generate 27 FLSTM Classification Metric Notebooks (v2 — Fixed)

Copies exact cell code from the working DowJones reference notebooks,
only changing the data loading and adding classification metrics.
"""
import json, os, copy

# ============================================================
# HELPER: Read reference notebook
# ============================================================
def read_ref(path):
    with open(path) as f:
        return json.load(f)

# ============================================================
# EXTRACT EXACT CELLS FROM WORKING REFERENCE NOTEBOOKS
# ============================================================
archs = {
    'Gates_Only': 'Gates_Only',
    'Gates_Plus_FuzzyInput': 'Gates_Plus_FuzzyInput',
    'Gates_Plus_FuzzyOutput': 'Gates_Plus_FuzzyOutput',
}

mf_map = {'dog': 'DOG', 'sg': 'SG', 'gb': 'GB'}

def get_ref_path(arch_key, mf):
    return f'FLSTM_Fuzzy_MF_DowJones/FLSTM_MF_{arch_key}_{mf_map[mf]}_dow_jones.ipynb'

def extract_cell_by_content(nb, marker):
    """Find cell whose source contains the marker string."""
    for cell in nb['cells']:
        src = ''.join(cell['source'])
        if marker in src:
            return copy.deepcopy(cell)
    return None

# Pre-load all reference cells
ref_cells = {}
for arch_key in archs:
    ref_cells[arch_key] = {}
    for mf in ['dog', 'sg', 'gb']:
        path = get_ref_path(arch_key, mf)
        nb = read_ref(path)
        ref_cells[arch_key][mf] = {
            'pid': extract_cell_by_content(nb, 'Process ID (PID)'),
            'timer_start': extract_cell_by_content(nb, 'NOTEBOOK TIMER — START'),
            'cpu': extract_cell_by_content(nb, 'CPU ONLY Settings'),
            'imports': extract_cell_by_content(nb, 'from sklearn.preprocessing import MinMaxScaler'),
            'wm_fuzzy': extract_cell_by_content(nb, 'build_wm_system'),
            'mf_cell': extract_cell_by_content(nb, 'class MembershipFLSTMSNPCell'),
            'build_model': extract_cell_by_content(nb, 'def build_model'),
            'timer_end': extract_cell_by_content(nb, 'NOTEBOOK TIMER — END'),
        }
        # FuzzyOutput arch has an extra WMFuzzyOutputLayer cell
        if arch_key == 'Gates_Plus_FuzzyOutput':
            ref_cells[arch_key][mf]['fuzzy_output_layer'] = extract_cell_by_content(nb, 'class WMFuzzyOutputLayer')

print("All reference cells extracted successfully.")

# ============================================================
# DATA LOADING CODE PER DATASET
# ============================================================
data_loading = {
    'mackey_glass': [
        "import openpyxl\n",
        "series = pd.read_excel('../content/Mackey-Glass Time Series(taw17).xlsx')\n",
        "raw_values = series['t+1'].values.flatten()\n",
    ],
    'sunspots': [
        "series = pd.read_csv('../content/Sunspots.csv')\n",
        "raw_values = series['Monthly Mean Total Sunspot Number'].values.flatten()\n",
    ],
    'epl': [
        "import os as _os\n",
        "data_dir = '../content/English Premier League Dataset/data'\n",
        "all_dfs = []\n",
        "for _f in sorted(_os.listdir(data_dir)):\n",
        "    if _f.endswith('_csv.csv'):\n",
        "        df = pd.read_csv(_os.path.join(data_dir, _f))\n",
        "        df['TotalGoals'] = df['FTHG'] + df['FTAG']\n",
        "        all_dfs.append(df)\n",
        "combined = pd.concat(all_dfs, ignore_index=True)\n",
        "raw_values = combined['TotalGoals'].values.astype(float)\n",
    ],
}

# ============================================================
# BUILD MAIN EXECUTION CELL (No noise, with classification metrics)
# ============================================================
def build_main_cell(arch_label, mf_label, dataset_label, dataset_key):
    """Build the main data-loading + training + evaluation cell."""
    
    dl = data_loading[dataset_key]
    
    src = dl + [
        "import numpy as np\n",
        "original_raw_values = np.copy(raw_values)\n",
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
        "supervised = timeseries_to_supervised(diff_values, 1)\n",
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
        "\n",
        "X_train = X_train_aug.reshape((X_train_aug.shape[0], 1, X_train_aug.shape[1]))\n",
        "X_test = X_test_aug.reshape((X_test_aug.shape[0], 1, X_test_aug.shape[1]))\n",
        "\n",
        "# ============================================================\n",
        "# TRAINING & EVALUATION (2 runs, No Noise)\n",
        "# ============================================================\n",
        f"print('\\n' + '='*80)\n",
        f"print('EVALUATING: {arch_label} ({mf_label.upper()}) on {dataset_label} — Normal Condition (No Noise)')\n",
        f"print('='*80 + '\\n')\n",
        "\n",
        "all_rmse, all_mae, all_predictions, all_losses = [], [], [], []\n",
        "all_c_means, all_c_stds, all_c_mins, all_c_maxs = [], [], [], []\n",
        "all_o_means, all_o_stds, all_o_mins, all_o_maxs = [], [], [], []\n",
        "all_accuracy, all_specificity, all_precision, all_recall, all_auc = [], [], [], [], []\n",
        "\n",
        "for run in range(30):\n",
        "    print(f'\\n===== RUN {run+1}/30 =====')\n",
        "    np.random.seed(run)\n",
        "    tf.random.set_seed(run)\n",
        "    tf.keras.backend.clear_session()\n",
        "\n",
        "    model = build_model(input_dim=2, units=8, batch_size=1)\n",
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
        "\n",
        "        row = list(X_test_raw[i]) + [yhat[0, 0]]\n",
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
        "\n",
        "    # ---- Classification Metrics (Dynamic Threshold: Huber) ----\n",
        "    train_raw_values = raw_values[:-60]\n",
        "    train_diffs = np.diff(train_raw_values)\n",
        "    tau = -0.035150\n",
        "    \n",
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
        f"print('\\n===== FINAL RESULTS — {arch_label} ({mf_label.upper()}) on {dataset_label} (30 runs) =====')\n",
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
        f"plt.title('{arch_label} ({mf_label.upper()}) — {dataset_label}\\nPredictions vs Actual (Best of 30 runs)')\n",
        "plt.xlabel('Time Step')\n",
        "plt.ylabel('Value')\n",
        "plt.legend()\n",
        "plt.grid(True, alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "plt.figure(figsize=(12, 4))\n",
        "plt.plot(all_losses[best_idx], color='green', linewidth=1.0)\n",
        f"plt.title('{arch_label} ({mf_label.upper()}) — {dataset_label}\\nTraining Loss (Best Run)')\n",
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
        "plt.plot(avg_predictions, label='Mean Predicted (30 runs)', color='red', linewidth=1.5, linestyle='--')\n",
        "plt.fill_between(range(len(actual)), avg_predictions - std_predictions, avg_predictions + std_predictions,\n",
        "                 alpha=0.2, color='red', label='±1 Std Dev')\n",
        f"plt.title('{arch_label} ({mf_label.upper()}) — {dataset_label}\\nAverage Predictions vs Actual (30 runs ± 1σ)')\n",
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
        f"print('║ CLASSIFICATION RESULTS — {arch_label} ({mf_label.upper()}) on {dataset_label} (30 runs) '.ljust(86) + '║')\n",
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

# ============================================================
# GENERATION
# ============================================================
datasets = {
    'sunspots': {'label': 'Sunspots', 'folder': 'FLSTM_Classification_Sunspots_Huber'},
}

arch_labels = {
    'Gates_Only': 'Gates Only',
    'Gates_Plus_FuzzyInput': 'Gates+FuzzyInput',
    'Gates_Plus_FuzzyOutput': 'Gates+FuzzyOutput',
}

KERNEL_SPEC = {
    "kernelspec": {
        "display_name": "SNN Transformer",
        "language": "python",
        "name": "snn_transformer"
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.11.14"
    }
}

total = 0
for ds_key, ds_info in datasets.items():
    folder = ds_info['folder']
    os.makedirs(folder, exist_ok=True)
    
    for arch_key, arch_label in arch_labels.items():
        for mf in ['dog', 'sg', 'gb']:
            nb_name = f"FLSTM_Classif_{arch_key}_{mf_map[mf]}_{ds_key}.ipynb"
            
            rc = ref_cells[arch_key][mf]
            
            # Title cell
            title_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# FLSTM {arch_label} using {mf_map[mf]}\n",
                    f"Dataset: {ds_info['label']}\n",
                    "\n",
                    "**Classification Metrics Evaluation (No Noise)**"
                ]
            }
            
            # Assemble cells in the correct order (matching reference)
            cells = [
                title_cell,
                copy.deepcopy(rc['pid']),
                copy.deepcopy(rc['timer_start']),
                copy.deepcopy(rc['cpu']),
                copy.deepcopy(rc['imports']),
                copy.deepcopy(rc['wm_fuzzy']),
                copy.deepcopy(rc['mf_cell']),
            ]
            
            # FuzzyOutput has extra WMFuzzyOutputLayer cell
            if arch_key == 'Gates_Plus_FuzzyOutput':
                cells.append(copy.deepcopy(rc['fuzzy_output_layer']))
            
            cells.append(copy.deepcopy(rc['build_model']))
            
            # Main execution cell (dataset-specific, with classification metrics)
            cells.append(build_main_cell(arch_label, mf, ds_info['label'], ds_key))
            
            # Results and observations markdown
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Results"]
            })
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"## Observations\n",
                    "\n",
                    f"### {arch_label} ({mf_map[mf]}) on {ds_info['label']}\n",
                    "\n",
                    "**Run the notebook to generate results.**\n"
                ]
            })
            cells.append(copy.deepcopy(rc['timer_end']))
            
            # Clear all outputs from reference cells
            for cell in cells:
                if cell['cell_type'] == 'code':
                    cell['outputs'] = []
                    cell['execution_count'] = None
            
            notebook = {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": KERNEL_SPEC,
                "cells": cells
            }
            
            path = os.path.join(folder, nb_name)
            with open(path, 'w') as f:
                json.dump(notebook, f, indent=1)
            
            total += 1
            print(f"Created: {path}")

print(f"\nTotal notebooks generated: {total}")
