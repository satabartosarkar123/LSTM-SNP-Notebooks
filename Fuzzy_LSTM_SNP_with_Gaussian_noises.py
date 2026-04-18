import os
import glob
import json
import shutil

source_dirs = [
    "Fuzzy_LSTM_SNP/Type_2",
    "Fuzzy_LSTM_SNP/Type_3",
    "Fuzzy_LSTM_SNP/Type_4",
    "Fuzzy_LSTM_SNP/Type_5"
]

target_base = "Fuzzy_LSTM_SNP_With_Gaussian_Noises"

def process_notebook(nb_path, target_path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Find the first cell containing pd.read_csv to start combining logic
    start_cell_idx = -1
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') == "code":
            source_content = "".join(cell.get('source', []))
            if "pd.read_csv" in source_content:
                start_cell_idx = i
                break
            
    if start_cell_idx == -1:
        print(f"Skipping {nb_path} (could not find pd.read_csv)")
        return
        
    # Combine all code from the start_cell onwards
    all_code_cells = [cell for cell in nb['cells'][start_cell_idx:] if cell.get('cell_type') == "code"]
    combined_source = ""
    for cell in all_code_cells:
        combined_source += "".join(cell.get('source', [])) + "\n\n"
            
    lines = combined_source.split('\n')
    
    # Identify the loading block (from start to raw_values assignment)
    end_loading_idx = -1
    for i, line in enumerate(lines):
        if "raw_values =" in line and "series.values.flatten()" in line:
            end_loading_idx = i
            break
            
    if end_loading_idx == -1:
        print(f"Skipping {nb_path} (could not find raw_values assignment)")
        return
        
    loading_lines = lines[:end_loading_idx+1]
    logic_lines = lines[end_loading_idx+1:]
    
    # FIX PATHS: Because notebooks are in sub-sub-folders, they need ../../content/
    fixed_loading_lines = []
    for line in loading_lines:
        line = line.replace("'../content/", "'../../content/")
        line = line.replace('"../content/', '"../../content/')
        line = line.replace("'content/", "'../../content/")
        line = line.replace('"content/', '"../../content/')
        fixed_loading_lines.append(line)
    loading_lines = fixed_loading_lines

    # Construct new source
    new_source_lines = []
    
    # 1. Loading block (outside loop)
    while loading_lines and not loading_lines[0].strip(): loading_lines.pop(0)
    new_source_lines.extend(loading_lines)
    
    # 2. Setup noise variables
    new_source_lines.append("")
    new_source_lines.append("import numpy as np")
    new_source_lines.append("original_raw_values = np.copy(raw_values)")
    new_source_lines.append("s_x = np.std(original_raw_values)")
    new_source_lines.append("noise_levels = [0.005]")
    new_source_lines.append("")
    
    # 3. Start Loop
    new_source_lines.append("for lam in noise_levels:")
    new_source_lines.append('    sigma = lam * s_x')
    new_source_lines.append('    print("\\n" + "="*80)')
    new_source_lines.append(f'    print(f"EVALUATING NOISE LEVEL: {{lam*100:.1f}}% (lambda={{lam}}, sigma={{sigma:.6f}})")')
    new_source_lines.append('    print("="*80 + "\\n")')
    new_source_lines.append("    ")
    new_source_lines.append("    np.random.seed(42)")
    new_source_lines.append("    tf.random.set_seed(42)")
    new_source_lines.append("    noise = np.random.normal(0, sigma, size=original_raw_values.shape)")
    new_source_lines.append("    raw_values = original_raw_values + noise")
    new_source_lines.append("    ")
    
    # 4. Indented Logic
    for line in logic_lines:
        if line.strip() or line == "":
            new_source_lines.append("    " + line)
        else:
            new_source_lines.append("    ")
            
    # Final cleanup of the cells
    new_cells = nb['cells'][:start_cell_idx]
    
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in new_source_lines]
    }
    
    if new_cell["source"]:
        new_cell["source"][-1] = new_cell["source"][-1].rstrip()
        
    new_cells.append(new_cell)
    
    trailing_markdowns = [cell for cell in nb['cells'][start_cell_idx:] if cell.get('cell_type') == "markdown"]
    new_cells.extend(trailing_markdowns)
    
    nb['cells'] = new_cells
    
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    
    print(f"Fixed paths and noise in: {nb_path} -> {target_path}")

# Run
for src_dir in source_dirs:
    for nb_path in glob.glob(f"{src_dir}/*.ipynb"):
        filename = os.path.basename(nb_path)
        type_folder = os.path.basename(src_dir)
        target_path = os.path.join(target_base, type_folder, filename)
        process_notebook(nb_path, target_path)

print("\nDone recreating all notebooks with fixed paths and Gaussian noise.")
