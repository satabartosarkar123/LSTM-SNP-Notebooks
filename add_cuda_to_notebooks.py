import nbformat as nbf
import os
import glob

# CUDA Snippet to add
CUDA_SNIPPET = """# ============================================================
# CUDA ACCELERATION (GPU CONFIGURATION)
# ============================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Detected {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("No GPU detected. Running on CPU.")
"""

def add_cuda_to_notebook(filepath):
    print(f"[*] Processing {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    modified = False
    for cell in nb.cells:
        if cell.cell_type == 'code':
            # Check if this is the import cell
            if 'import tensorflow as tf' in cell.source:
                # To avoid double insertion
                if 'CUDA ACCELERATION' in cell.source:
                    print(f"[!] CUDA block already exists in {os.path.basename(filepath)}")
                    return False
                
                # Find a good place to insert: after imports or at the end
                lines = cell.source.splitlines()
                # Find the line with tensorflow import
                for i, line in enumerate(lines):
                    if 'import tensorflow as tf' in line:
                        # Insert after this line + any subsequent print/alias
                        insert_idx = i + 1
                        # If the next line is a print of tf version, skip it too
                        if insert_idx < len(lines) and 'tf.__version__' in lines[insert_idx]:
                            insert_idx += 1
                        
                        lines.insert(insert_idx, "\n" + CUDA_SNIPPET)
                        cell.source = "\n".join(lines)
                        modified = True
                        break
                
                if modified:
                    break # Only do it for the first import cell

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
        print(f"[OK] Updated {os.path.basename(filepath)}")
        return True
    else:
        print(f"[?] Could not find tensorflow import in {os.path.basename(filepath)}")
        return False

def main():
    base_dir = r"c:\Users\USER\Research\SNP\LSTM-SNP-Notebooks\Fuzzy_LSTM_SNP_With_Gaussian_Noises"
    types = ["Type_2", "Type_3", "Type_4", "Type_5"]
    
    total_updated = 0
    for t in types:
        pattern = os.path.join(base_dir, t, "*.ipynb")
        notebooks = glob.glob(pattern)
        print(f"--- Processing {t} ({len(notebooks)} files) ---")
        for nb_path in notebooks:
            if add_cuda_to_notebook(nb_path):
                total_updated += 1
    
    print(f"\nFinished! Total notebooks updated: {total_updated}")

if __name__ == "__main__":
    main()
