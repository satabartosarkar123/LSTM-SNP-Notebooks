import os
import json
import re

datasets = [
    {
        'name': 'sunspots',
        'folder': 'FLSTM_Fuzzy_MF_Sunspots',
        'title': 'Sunspots',
        'load_code': "series = pd.read_csv('../content/Sunspots.csv')\nraw_values = series['Monthly Mean Total Sunspot Number'].values.flatten()"
    },
    {
        'name': 'mackey_glass',
        'folder': 'FLSTM_Fuzzy_MF_MackeyGlass',
        'title': 'Mackey-Glass',
        'load_code': "import openpyxl\nseries = pd.read_excel('../content/Mackey-Glass Time Series(taw17).xlsx')\nraw_values = series['t+1'].values.flatten()"
    },
    {
        'name': 'epl',
        'folder': 'FLSTM_Fuzzy_MF_EPL',
        'title': 'EPL',
        'load_code': "import os\ndata_dir = '../content/English Premier League Dataset/data'\nall_dfs = []\nfor f in sorted(os.listdir(data_dir)):\n    if f.endswith('_csv.csv'):\n        df = pd.read_csv(os.path.join(data_dir, f))\n        df['TotalGoals'] = df['FTHG'] + df['FTAG']\n        all_dfs.append(df)\ncombined = pd.concat(all_dfs, ignore_index=True)\nraw_values = combined['TotalGoals'].values.astype(float)"
    }
]

ref_dir = 'FLSTM_Fuzzy_MF_DowJones'
ref_notebooks = [f for f in os.listdir(ref_dir) if f.endswith('.ipynb')]

def process_cell_source(source, ds_load_code, ds_title):
    src = "".join(source)
    
    # 1. Replace Title in Markdown
    if "Dataset: Dow Jones" in src:
        src = src.replace("Dataset: Dow Jones", f"Dataset: {ds_title}")
    
    if "Dow Jones" in src:
        src = src.replace("Dow Jones", ds_title)
    if "dow_jones" in src:
        src = src.replace("dow_jones", ds_title.lower().replace("-", "_").replace(" ", "_"))

    # 2. Modify training cell
    if "raw_values =" in src and "noise_levels" in src:
        # Replace data loading
        load_pattern = r"series = pd\.read_csv\('\.\./content/monthly-closings-of-the-dowjones\.csv', header=0, parse_dates=\[0\], index_col=0\)\nraw_values = series\.values\.flatten\(\)"
        src = re.sub(load_pattern, ds_load_code, src)
        
        # Runs 30 -> 2
        src = src.replace("for run in range(30):", "for run in range(30):")
        src = src.replace("===== RUN {run+1}/30 =====", "===== RUN {run+1}/30 =====")
        src = src.replace("(30 runs)", "(30 runs)")
        src = src.replace("Average Metrics (30 runs)", "Average Metrics (30 runs)")

        # Initialize summary_results
        src = src.replace("for lam in noise_levels:", "summary_results = {}\n\nfor lam in noise_levels:")

        # Metrics lists
        src = src.replace("all_rmse, all_mse, all_nmse = [], [], []", "all_rmse, all_mae = [], []")

        # Compute MAE instead of MSE/NMSE
        mse_nmse_comp = """        mse = mean_squared_error(actual, predictions)
        meanV = np.mean(actual)
        dominator = np.linalg.norm(np.array(predictions) - meanV, 2)
        nmse = mse / np.power(dominator, 2)
    
        all_rmse.append(rmse)
        all_mse.append(mse)
        all_nmse.append(nmse)"""
        mae_comp = """        mae = np.mean(np.abs(np.array(actual) - np.array(predictions)))
        
        all_rmse.append(rmse)
        all_mae.append(mae)"""
        src = src.replace(mse_nmse_comp, mae_comp)

        # Print run metrics
        src = src.replace("RMSE: {rmse:.6f}, MSE: {mse:.6f}, NMSE: {nmse:.10f}", "RMSE: {rmse:.6f}, MAE: {mae:.6f}")

        # Summary metrics definitions
        mean_metrics = """    mean_mse = np.mean(all_mse)
    std_mse = np.std(all_mse)
    
    mean_nmse = np.mean(all_nmse)
    std_nmse = np.std(all_nmse)"""
        mean_metrics_new = """    mean_mae = np.mean(all_mae)
    std_mae = np.std(all_mae)"""
        src = src.replace(mean_metrics, mean_metrics_new)

        # Print overall metrics
        print_overall = """    print(f'MSE:  {mean_mse:.6f} ± {std_mse:.6f}')
    print(f'NMSE: {mean_nmse:.10f} ± {std_nmse:.10f}')"""
        print_overall_new = """    print(f'MAE:  {mean_mae:.6f} ± {std_mae:.6f}')"""
        src = src.replace(print_overall, print_overall_new)

        # Best run prints
        best_run_prints = """    print(f'  MSE:  {all_mse[best_idx]:.6f}')
    print(f'  NMSE: {all_nmse[best_idx]:.10f}')"""
        best_run_prints_new = """    print(f'  MAE:  {all_mae[best_idx]:.6f}')"""
        src = src.replace(best_run_prints, best_run_prints_new)

        # Best run metrics at end
        best_metrics = """    print(f'MSE:  {all_mse[best_idx]:.6f}')
    print(f'NMSE: {all_nmse[best_idx]:.10f}')"""
        best_metrics_new = """    print(f'MAE:  {all_mae[best_idx]:.6f}')"""
        src = src.replace(best_metrics, best_metrics_new)

        # Average metrics at end
        avg_metrics = """    print(f'MSE:  {np.mean(all_mse):.6f} ± {np.std(all_mse):.6f}')
    print(f'NMSE: {np.mean(all_nmse):.10f} ± {np.std(all_nmse):.10f}')"""
        avg_metrics_new = """    print(f'MAE:  {np.mean(all_mae):.6f} ± {np.std(all_mae):.6f}')"""
        src = src.replace(avg_metrics, avg_metrics_new)

        # Store summary at the end of loop
        end_of_loop = "print(f'MAE:  {np.mean(all_mae):.6f} ± {np.std(all_mae):.6f}')"
        if end_of_loop in src:
            src = src.replace(end_of_loop, end_of_loop + "\n    summary_results[lam] = {'rmse_mean': mean_rmse, 'rmse_std': std_rmse, 'mae_mean': mean_mae, 'mae_std': std_mae}")

        # Append summary table at the end of the cell
        summary_code = """

# ==========================================
# COMBINED SUMMARY FOR ALL NOISE LEVELS
# ==========================================
print("\\n" + "╔" + "═"*78 + "╗")
print(f"║ COMBINED RESULTS (30 runs) ".ljust(79) + "║")
print("╠" + "═"*15 + "╦" + "═"*30 + "╦" + "═"*31 + "╣")
print("║ Noise Level   ║ RMSE ± std                   ║ MAE ± std                     ║")
print("╠" + "═"*15 + "╬" + "═"*30 + "╬" + "═"*31 + "╣")
for lam in noise_levels:
    res = summary_results[lam]
    noise_str = "0% (No Noise)" if lam == 0.0 else f"{lam*100:.1f}%"
    rmse_str = f"{res['rmse_mean']:.6f} ± {res['rmse_std']:.6f}"
    mae_str = f"{res['mae_mean']:.6f} ± {res['mae_std']:.6f}"
    print(f"║ {noise_str:<13} ║ {rmse_str:<28} ║ {mae_str:<29} ║")
print("╚" + "═"*15 + "╩" + "═"*30 + "╩" + "═"*31 + "╝")
"""
        src = src + summary_code

    return [src]

def generate():
    for ds in datasets:
        os.makedirs(ds['folder'], exist_ok=True)
        print(f"Processing dataset: {ds['name']}")
        
        for ref_nb in ref_notebooks:
            with open(os.path.join(ref_dir, ref_nb), 'r') as f:
                nb = json.load(f)
            
            # Change filename: dow_jones -> ds['name']
            new_nb_name = ref_nb.replace("dow_jones", ds['name'])
            
            for cell in nb['cells']:
                cell['source'] = process_cell_source(cell['source'], ds['load_code'], ds['title'])
                
            out_path = os.path.join(ds['folder'], new_nb_name)
            with open(out_path, 'w') as f:
                json.dump(nb, f, indent=1)
            print(f"  Created {new_nb_name}")

if __name__ == '__main__':
    generate()
