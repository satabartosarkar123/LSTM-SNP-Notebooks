import nbformat
import os
import sys

def update_notebook(path):
    print(f"Updating {path}...")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return

    # 1. Add smape function to imports cell
    found_imports = False
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'from sklearn.metrics import mean_squared_error' in cell.source:
            if 'def smape' not in cell.source:
                cell.source += "\n\ndef smape(actual, predicted):\n    actual, predicted = np.array(actual), np.array(predicted)\n    return 100/len(actual) * np.sum(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted) + 1e-8))"
            found_imports = True
            break
    
    if not found_imports:
        print(f"Warning: Could not find imports cell in {path}")

    # 2. Add all_smape initialization and calculation in experiment loop
    found_loop = False
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'all_rmse = []' in cell.source:
            if 'all_smape = []' not in cell.source:
                cell.source = cell.source.replace('all_rmse = []', 'all_rmse = []\nall_smape = []')
            
            # Add smape calculation
            if 's_mape = smape(actual, predictions)' not in cell.source:
                if 'rmse = sqrt(mean_squared_error(actual, predictions))' in cell.source:
                    cell.source = cell.source.replace('rmse = sqrt(mean_squared_error(actual, predictions))', 
                                                      'rmse = sqrt(mean_squared_error(actual, predictions))\n    s_mape = smape(actual, predictions)')
                
                if 'all_rmse.append(rmse)' in cell.source:
                    cell.source = cell.source.replace('all_rmse.append(rmse)', 'all_rmse.append(rmse)\n    all_smape.append(s_mape)')
                
                # Update print - handle various dashes and spaces
                import re
                # Match "Run {run+1} ... NMSE: {nmse:.10f}')"
                pattern = r"print\(f'Run \{run\+1\}.*NMSE: \{nmse:\.10f\}'\)"
                match = re.search(pattern, cell.source)
                if match:
                    original_print = match.group(0)
                    new_print = original_print[:-2] + f", sMAPE: {{s_mape:.6f}}')"
                    cell.source = cell.source.replace(original_print, new_print)
            found_loop = True
            break
    
    if not found_loop:
        print(f"Warning: Could not find training loop cell in {path}")

    # 3. Update Summary Statistics
    found_summary = False
    for cell in nb.cells:
        if cell.cell_type == 'code' and "print('\\n===== FINAL RESULTS" in cell.source:
            if "print(f'sMAPE: {np.mean(all_smape):.6f}" not in cell.source:
                # Add to mean/std section
                if "print(f'NMSE: {np.mean(all_nmse):.10f}" in cell.source:
                    import re
                    pattern = r"print\(f'NMSE: \{np\.mean\(all_nmse\):\.10f\}.*'\)"
                    match = re.search(pattern, cell.source)
                    if match:
                        original_line = match.group(0)
                        cell.source = cell.source.replace(original_line, original_line + "\nprint(f'sMAPE: {np.mean(all_smape):.6f} \u00b1 {np.std(all_smape):.6f}')")
                
                # Add to best run section
                if "print(f'  NMSE: {all_nmse[best_idx]:.10f}')" in cell.source:
                    cell.source = cell.source.replace("print(f'  NMSE: {all_nmse[best_idx]:.10f}')",
                                                      "print(f'  NMSE: {all_nmse[best_idx]:.10f}')\nprint(f'  sMAPE: {all_smape[best_idx]:.6f}')")
            found_summary = True
            break
    
    if not found_summary:
        print(f"Warning: Could not find summary statistics cell in {path}")

    # 4. Update Best Run Metrics (Final Summary at bottom)
    for cell in nb.cells:
        if cell.cell_type == 'code' and "print('=== Best Run Metrics ===')" in cell.source:
            if "print(f'sMAPE: {all_smape[best_idx]:.6f}')" not in cell.source:
                if "print(f'NMSE: {all_nmse[best_idx]:.10f}')" in cell.source:
                    cell.source = cell.source.replace("print(f'NMSE: {all_nmse[best_idx]:.10f}')",
                                                      "print(f'NMSE: {all_nmse[best_idx]:.10f}')\nprint(f'sMAPE: {all_smape[best_idx]:.6f}')")
            break

    try:
        with open(path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
    except Exception as e:
        print(f"Error writing {path}: {e}")

if __name__ == "__main__":
    paths = sys.argv[1:]
    for p in paths:
        if os.path.exists(p):
            update_notebook(p)
        else:
            print(f"File not found: {p}")
