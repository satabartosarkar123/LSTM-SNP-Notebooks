import os
import re
import json

def process_file(filepath):
    # Determine file type
    if filepath.endswith('.py'):
        process_py(filepath)
    elif filepath.endswith('.ipynb'):
        process_ipynb(filepath)

def replace_text(text):
    # Runs
    text = re.sub(r'N_RUNS\s*=\s*\d+', 'N_RUNS = 30', text)
    text = re.sub(r'NUM_RUNS\s*=\s*\d+', 'NUM_RUNS = 30', text)
    text = re.sub(r'n_runs\s*=\s*\d+', 'n_runs=30', text)  # kwargs format
    text = re.sub(r'n_runs=\d+', 'n_runs=30', text)
    # Epochs
    text = re.sub(r'N_EPOCHS\s*=\s*\d+', 'N_EPOCHS = 100', text)
    text = re.sub(r'NUM_EPOCHS\s*=\s*\d+', 'NUM_EPOCHS = 100', text)
    text = re.sub(r'n_epochs\s*=\s*\d+', 'n_epochs=100', text)
    text = re.sub(r'n_epochs=\d+', 'n_epochs=100', text)
    
    # epochs=X but NOT epochs=1 (because epochs=1 is used inside manual epoch loops with model.fit)
    def replace_epochs(match):
        val = match.group(2)
        if val == '1':
            return match.group(0)
        else:
            return f"{match.group(1)}100"
            
    text = re.sub(r'(epochs\s*=\s*)(\d+)', replace_epochs, text)
    return text

def process_py(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    new_content = replace_text(content)

    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {filepath}")

def process_ipynb(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    changed = False
    for cell in data.get('cells', []):
        if cell.get('cell_type') == 'code':
            new_source = []
            for line in cell.get('source', []):
                new_line = replace_text(line)
                if new_line != line:
                    changed = True
                new_source.append(new_line)
            cell['source'] = new_source

    if changed:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
            f.write('\n')
        print(f"Updated {filepath}")

if __name__ == '__main__':
    search_dirs = [
        "/Users/satabarto/Research/32_Sized_LSTM",
        "/Users/satabarto/Research/autocorrelation_analysis",
        "/Users/satabarto/Research/basic_statistical_analysis",
        "/Users/satabarto/Research/complexity_analysis",
        "/Users/satabarto/Research/content",
        "/Users/satabarto/Research/FLSTM_Classification_EPL",
        "/Users/satabarto/Research/FLSTM_Classification_EPL_Median",
        "/Users/satabarto/Research/FLSTM_Classification_MackeyGlass",
        "/Users/satabarto/Research/FLSTM_Classification_MackeyGlass_Median",
        "/Users/satabarto/Research/FLSTM_Classification_MackeyGlass_Midrange",
        "/Users/satabarto/Research/FLSTM_Classification_Sunspots",
        "/Users/satabarto/Research/FLSTM_Classification_Sunspots_Huber",
        "/Users/satabarto/Research/FLSTM_Classification_Sunspots_Median",
        "/Users/satabarto/Research/FLSTM_Fuzzy_MF_DowJones",
        "/Users/satabarto/Research/FLSTM_Fuzzy_MF_EPL",
        "/Users/satabarto/Research/FLSTM_Fuzzy_MF_LakeErie",
        "/Users/satabarto/Research/FLSTM_Fuzzy_MF_MackeyGlass",
        "/Users/satabarto/Research/FLSTM_Fuzzy_MF_MilkProduction",
        "/Users/satabarto/Research/FLSTM_Fuzzy_MF_SP500",
        "/Users/satabarto/Research/FLSTM_Fuzzy_MF_Sunspots",
        "/Users/satabarto/Research/frequency_domain_analysis",
        "/Users/satabarto/Research/Fuzzy_LSTM_SNP",
        "/Users/satabarto/Research/Fuzzy_LSTM_SNP_With_Gaussian_Noises",
        "/Users/satabarto/Research/it2_tsk_plus",
        "/Users/satabarto/Research/LSTM_SNP",
        "/Users/satabarto/Research/Mamba_All_Datasets",
        "/Users/satabarto/Research/Median_Lag5_Notebooks",
        "/Users/satabarto/Research/Modified_Fuzzy_MF_DowJones",
        "/Users/satabarto/Research/Modified_Fuzzy_MF_LakeErie",
        "/Users/satabarto/Research/Modified_Fuzzy_MF_MilkProduction",
        "/Users/satabarto/Research/Modified_Fuzzy_MF_SP500",
        "/Users/satabarto/Research/Modified_Fuzzy_MF_With_Noise_DowJones",
        "/Users/satabarto/Research/Modified_Fuzzy_MF_With_Noise_LakeErie",
        "/Users/satabarto/Research/Modified_Fuzzy_MF_With_Noise_MilkProduction",
        "/Users/satabarto/Research/Modified_Fuzzy_MF_With_Noise_SP500",
        "/Users/satabarto/Research/noise_analysis",
        "/Users/satabarto/Research/Pure_GRU",
        "/Users/satabarto/Research/Pure_LSTM",
        "/Users/satabarto/Research/SNN_LSTM",
        "/Users/satabarto/Research/SNN_Transformer",
        "/Users/satabarto/Research/snn_venv",
        "/Users/satabarto/Research/Transformer_Models",
        "/Users/satabarto/Research/transformer_venv",
        "/Users/satabarto/Research/type2_huarng_model",
        "/Users/satabarto/Research/type2_huarng_model_with_noise",
        "/Users/satabarto/Research/venv",
        "/Users/satabarto/Research/venv_files",
        "/Users/satabarto/Research/WM_FLSTM_Fuzzy_MF_DowJones",
        "/Users/satabarto/Research/WM_FLSTM_Fuzzy_MF_LakeErie",
        "/Users/satabarto/Research/WM_FLSTM_Fuzzy_MF_MilkProduction",
        "/Users/satabarto/Research/WM_FLSTM_Fuzzy_MF_SP500"
    ]
    
    # Process files in search directories
    for d in search_dirs:
        for root, _, files in os.walk(d):
            if 'venv' in root or '__pycache__' in root or '.git' in root:
                continue
            for f in files:
                if f.endswith('.py') or f.endswith('.ipynb'):
                    filepath = os.path.join(root, f)
                    process_file(filepath)

    # Process .py files in the root Research dir as well
    for f in os.listdir("/Users/satabarto/Research"):
        filepath = os.path.join("/Users/satabarto/Research", f)
        if os.path.isfile(filepath) and filepath.endswith('.py'):
            process_file(filepath)
