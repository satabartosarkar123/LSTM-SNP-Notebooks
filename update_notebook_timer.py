import os
import glob
import json

directories = [
    "/Users/satabarto/Research/LSTM_SNP",
    "/Users/satabarto/Research/Pure_GRU",
    "/Users/satabarto/Research/Pure_LSTM",
    "/Users/satabarto/Research/SNN_LSTM",
    "/Users/satabarto/Research/SNN_Transformer"
]

def process_notebook(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        modified = False
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                if not source:
                    continue
                is_timer_cell = any("NOTEBOOK TIMER — END" in line for line in source)
                if is_timer_cell:
                    # check if lines are already added
                    if not any("Modified time (s * 0.5 * 0.25)" in line for line in source):
                        # Ensure trailing newline on the last string
                        if not source[-1].endswith('\n'):
                            source[-1] = source[-1] + '\n'
                        
                        source.append('print(f"Modified time (s * 0.5 * 0.25): {_NOTEBOOK_ELAPSED * 0.5 * 0.25:.2f} mins")\n')
                        source.append('import os\n')
                        source.append('print(f"Process ID: {os.getpid()}")\n')
                        modified = True
        
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1)
            print(f"Updated {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    for d in directories:
        print(f"Processing directory: {d}")
        for filepath in glob.glob(os.path.join(d, "**", "*.ipynb"), recursive=True):
            if ".ipynb_checkpoints" not in filepath:
                process_notebook(filepath)
