#!/usr/bin/env python3
import os
import glob
import json

folders = [
    "Modified_Fuzzy_MF_With_Noise_DowJones",
    "Modified_Fuzzy_MF_With_Noise_LakeErie",
    "Modified_Fuzzy_MF_With_Noise_MilkProduction",
    "Modified_Fuzzy_MF_With_Noise_SP500"
]

def process_notebook(nb_path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    start_idx = -1
    for i, c in enumerate(nb['cells']):
        if c.get('cell_type') == "code":
            source_content = "".join(c.get('source', []))
            if "series = pd.read_csv" in source_content:
                start_idx = i
                break
            
    if start_idx == -1:
        print(f"Skipping {nb_path} (could not find data load cell)")
        return
        
    combined_code = []
    
    for c in nb['cells'][start_idx:]:
        if c.get('cell_type') == "code":
            combined_code.append("".join(c.get('source', [])))
            
    source = "\n\n".join(combined_code)
    
    lines = source.split('\n')
    filtered_lines = []
    data_loading_lines = []
    for line in lines:
        if "pd.read_csv" in line and "series =" in line:
            data_loading_lines.append(line)
        elif "raw_values =" in line and "series.values.flatten()" in line:
            data_loading_lines.append(line)
        else:
            filtered_lines.append(line)
            
    new_code = []
    new_code.extend(data_loading_lines)
    new_code.append("import numpy as np")
    new_code.append("original_raw_values = np.copy(raw_values)")
    new_code.append("s_x = np.std(original_raw_values)")
    new_code.append("noise_levels = [0.001, 0.005, 0.01]")
    new_code.append("")
    new_code.append("for lam in noise_levels:")
    new_code.append('    sigma = lam * s_x')
    new_code.append('    print("\\n" + "="*80)')
    new_code.append('    print(f"EVALUATING NOISE LEVEL: {lam*100:.1f}% (lambda={lam}, sigma={sigma:.6f})")')
    new_code.append('    print("="*80 + "\\n")')
    new_code.append("    ")
    new_code.append("    np.random.seed(42)")
    new_code.append("    tf.random.set_seed(42)")
    new_code.append("    noise = np.random.normal(0, sigma, size=original_raw_values.shape)")
    new_code.append("    raw_values = original_raw_values + noise")
    new_code.append("    ")
    
    for line in filtered_lines:
        new_code.append("    " + line)
        
    final_source_lines = [line + "\\n" for line in new_code]
    if final_source_lines:
        final_source_lines[-1] = final_source_lines[-1][:-2] # strip last newline
        
    new_cells = nb['cells'][:start_idx]
    
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in new_code]
    }
    
    if new_cell["source"]:
        new_cell["source"][-1] = new_cell["source"][-1][:-1]
        
    new_cells.append(new_cell)
    
    trailing_markdowns = [c for c in nb['cells'][start_idx:] if c.get('cell_type') == "markdown"]
    new_cells.extend(trailing_markdowns)
    
    nb['cells'] = new_cells
    
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    
    print(f"Successfully processed {nb_path}")
    
    print(f"Successfully processed {nb_path}")

for folder in folders:
    for nb_path in glob.glob(f"{folder}/*.ipynb"):
        process_notebook(nb_path)
