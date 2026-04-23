#!/usr/bin/env python3
"""
Update ALL notebooks to detect and use Apple M2 Metal GPU (MPS) in addition to CUDA.

For TensorFlow notebooks:
  - Replace/add GPU detection that checks for both CUDA and Apple Metal GPU
  - tensorflow-metal plugin makes Metal GPUs show up via tf.config.list_physical_devices('GPU')
  - So the existing code mostly works, but we update the messaging and add explicit Metal detection

For PyTorch notebooks:
  - Replace `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
    with: cuda -> mps -> cpu priority chain
  - Update seed control to handle MPS

For analysis notebooks (no framework):
  - Skip GPU detection (not needed)
"""

import json
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DIRS_TO_PROCESS = [
    '32_Sized_LSTM',
    'autocorrelation_analysis',
    'basic_statistical_analysis',
    'complexity_analysis',
    'frequency_domain_analysis',
    'Fuzzy_LSTM_SNP',
    'Fuzzy_LSTM_SNP_With_Gaussian_Noises',
    'LSTM_SNP',
    'Modified_Fuzzy_MF_DowJones',
    'Modified_Fuzzy_MF_LakeErie',
    'Modified_Fuzzy_MF_MilkProduction',
    'Modified_Fuzzy_MF_SP500',
    'Modified_Fuzzy_MF_With_Noise_DowJones',
    'Modified_Fuzzy_MF_With_Noise_LakeErie',
    'Modified_Fuzzy_MF_With_Noise_MilkProduction',
    'Modified_Fuzzy_MF_With_Noise_SP500',
    'noise_analysis',
    'SNN_Transformer',
    'Transformer_Models',
]

# ── New GPU detection cells ────────────────────────────────────────────────────

def make_tf_gpu_cell():
    """TensorFlow GPU detection cell that handles CUDA + Apple Metal."""
    source = [
        "# ============================================================\n",
        "# GPU Acceleration Settings (CUDA + Apple Metal Support)\n",
        "# ============================================================\n",
        "import tensorflow as tf\n",
        "import platform\n",
        "\n",
        "physical_devices = tf.config.list_physical_devices('GPU')\n",
        "if physical_devices:\n",
        "    try:\n",
        "        for device in physical_devices:\n",
        "            tf.config.experimental.set_memory_growth(device, True)\n",
        "        # Detect GPU type\n",
        "        if platform.system() == 'Darwin' and platform.processor() == 'arm':\n",
        "            print(f\"Apple Metal GPU Enabled: Found {len(physical_devices)} GPU(s) via tensorflow-metal\")\n",
        "            print(f\"  Chipset: Apple {platform.machine()} (M-series)\")\n",
        "        else:\n",
        "            print(f\"CUDA GPU Enabled: Found {len(physical_devices)} GPU(s) - Memory Growth Set\")\n",
        "        for d in physical_devices:\n",
        "            print(f\"  Device: {d.name}\")\n",
        "    except RuntimeError as e:\n",
        "        print(f\"GPU configuration error: {e}\")\n",
        "else:\n",
        "    if platform.system() == 'Darwin' and platform.processor() == 'arm':\n",
        "        print(\"Apple Silicon detected but no GPU found.\")\n",
        "        print(\"  Install tensorflow-metal: pip install tensorflow-metal\")\n",
        "    else:\n",
        "        print(\"No GPU found. Falling back to CPU.\")\n",
        "        print(\"  For CUDA: ensure NVIDIA drivers + CUDA toolkit are installed.\")\n",
    ]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


# ── PyTorch replacement patterns ──────────────────────────────────────────────

OLD_PYTORCH_DEVICE_BLOCK = """device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")"""

NEW_PYTORCH_DEVICE_BLOCK = """# Device priority: CUDA > Apple MPS (Metal) > CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using device: CUDA GPU")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
    import platform
    print(f"Using device: Apple Metal (MPS)")
    print(f"  Chipset: Apple {platform.machine()} (M-series)")
else:
    device = torch.device('cpu')
    print(f"Using device: CPU")"""

OLD_PYTORCH_SEED_BLOCK = """def set_seed(seed):
    \"\"\"Set all random seeds for reproducibility.\"\"\"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False"""

NEW_PYTORCH_SEED_BLOCK = """def set_seed(seed):
    \"\"\"Set all random seeds for reproducibility.\"\"\"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS doesn't need explicit seed setting beyond torch.manual_seed
        pass"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def collect_notebooks():
    notebooks = []
    for d in DIRS_TO_PROCESS:
        dirpath = os.path.join(BASE_DIR, d)
        if not os.path.isdir(dirpath):
            continue
        for root, subdirs, files in os.walk(dirpath):
            subdirs[:] = [s for s in subdirs if s != '.ipynb_checkpoints']
            for f in sorted(files):
                if f.endswith('.ipynb'):
                    notebooks.append(os.path.join(root, f))
    return notebooks


def detect_framework(nb):
    """Detect which ML framework the notebook uses."""
    full_src = ''
    for cell in nb.get('cells', []):
        full_src += ''.join(cell.get('source', []))
    
    has_tf = 'import tensorflow' in full_src or 'from tensorflow' in full_src
    has_torch = 'import torch' in full_src
    
    if has_torch:
        return 'pytorch'
    elif has_tf:
        return 'tensorflow'
    else:
        return 'none'


def has_gpu_cell_already(cells):
    """Check if notebook already has our updated GPU detection cell."""
    for cell in cells:
        src = ''.join(cell.get('source', []))
        if 'Apple Metal GPU Enabled' in src or 'Apple Metal (MPS)' in src:
            return True
    return False


def find_cuda_cell_index(cells):
    """Find the index of existing CUDA acceleration settings cell in TF notebooks."""
    for i, cell in enumerate(cells):
        src = ''.join(cell.get('source', []))
        if cell.get('cell_type') == 'code' and 'CUDA Acceleration Settings' in src:
            return i
    return -1


def find_imports_cell_index(cells, framework):
    """Find the index of the main imports cell."""
    for i, cell in enumerate(cells):
        src = ''.join(cell.get('source', []))
        if cell.get('cell_type') != 'code':
            continue
        if framework == 'tensorflow' and ('import tensorflow' in src or 'from tensorflow' in src):
            return i
        elif framework == 'pytorch' and 'import torch' in src:
            return i
    return -1


def process_tensorflow_notebook(nb_path, nb, dry_run=False):
    """Update TF notebook to detect both CUDA and Apple Metal GPU."""
    cells = nb['cells']
    modified = False
    
    if has_gpu_cell_already(cells):
        return False
    
    # Find existing CUDA cell
    cuda_idx = find_cuda_cell_index(cells)
    
    if cuda_idx >= 0:
        # Replace existing CUDA cell with our new GPU detection cell
        cells[cuda_idx] = make_tf_gpu_cell()
        modified = True
    else:
        # No CUDA cell exists — insert GPU detection cell after the timer start cell
        # Find the right place: after PID + timer cells, before imports
        imports_idx = find_imports_cell_index(cells, 'tensorflow')
        if imports_idx >= 0:
            cells.insert(imports_idx, make_tf_gpu_cell())
            modified = True
        else:
            # Fallback: insert after the timer start cell
            for i, cell in enumerate(cells):
                src = ''.join(cell.get('source', []))
                if '_NOTEBOOK_START_TIME' in src:
                    cells.insert(i + 1, make_tf_gpu_cell())
                    modified = True
                    break
    
    return modified


def process_pytorch_notebook(nb_path, nb, dry_run=False):
    """Update PyTorch notebook to detect CUDA, MPS, or CPU."""
    cells = nb['cells']
    modified = False
    
    if has_gpu_cell_already(cells):
        return False
    
    for i, cell in enumerate(cells):
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell.get('source', []))
        new_lines = list(cell['source'])
        cell_changed = False
        
        # --- Replace device selection block ---
        # Pattern 1: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # This is the most common pattern in Transformer notebooks
        full_src = ''.join(new_lines)
        
        # Replace the full device selection + print block
        if "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')" in full_src:
            # Find start and end of the block to replace
            block_start = None
            block_end = None
            for j, line in enumerate(new_lines):
                if "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')" in line:
                    block_start = j
                if block_start is not None and 'torch.cuda.get_device_properties' in line:
                    block_end = j
                    break
            
            if block_start is not None:
                if block_end is None:
                    # Just the single line device selection
                    block_end = block_start
                    # Check next few lines for related prints
                    for j in range(block_start + 1, min(block_start + 5, len(new_lines))):
                        line = new_lines[j].strip()
                        if line.startswith('print(f"Using device') or line.startswith("print(f'Using device"):
                            block_end = j
                        elif line.startswith('if torch.cuda.is_available()'):
                            block_end = j
                        elif 'torch.cuda.get_device_name' in line:
                            block_end = j
                        elif 'torch.cuda.get_device_properties' in line or 'total_mem' in line:
                            block_end = j
                
                # Build replacement
                indent = ''
                for ch in new_lines[block_start]:
                    if ch in ' \t':
                        indent += ch
                    else:
                        break
                
                replacement = []
                for rline in NEW_PYTORCH_DEVICE_BLOCK.split('\n'):
                    replacement.append(indent + rline + '\n')
                
                new_lines = new_lines[:block_start] + replacement + new_lines[block_end + 1:]
                cell_changed = True
        
        # Pattern 2: DEVICE = "mps" if ... (SNN_Transformer already has this)
        # Just verify it's good and update if needed
        if 'DEVICE = "mps"' in full_src or "DEVICE = 'mps'" in full_src:
            # Already has MPS, check if it's the right priority
            pass
        
        # Pattern 3: Simple device = "cuda" or device = 'cuda'  
        for j, line in enumerate(new_lines):
            stripped = line.strip()
            if re.match(r'^device\s*=\s*["\']cuda["\']', stripped):
                indent = line[:len(line) - len(line.lstrip())]
                new_lines[j] = indent + "device = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')\n"
                cell_changed = True
        
        if cell_changed:
            cell['source'] = new_lines
            modified = True
        
        # --- Replace seed block ---
        full_src = ''.join(cell['source'])
        if 'def set_seed(seed):' in full_src and 'torch.cuda.manual_seed' in full_src:
            if 'mps' not in full_src:
                # Need to add MPS handling to seed function
                new_lines = list(cell['source'])
                cell_changed = False
                
                # Find the seed function and add MPS handling
                in_seed_fn = False
                cudnn_benchmark_idx = None
                for j, line in enumerate(new_lines):
                    if 'def set_seed(seed):' in line:
                        in_seed_fn = True
                    if in_seed_fn:
                        if 'torch.backends.cudnn.benchmark = False' in line:
                            cudnn_benchmark_idx = j
                            break
                
                if cudnn_benchmark_idx is not None:
                    # Get indentation
                    indent = '    '
                    for ch in new_lines[cudnn_benchmark_idx]:
                        if ch in ' \t':
                            indent = ch
                        else:
                            break
                    base_indent = new_lines[cudnn_benchmark_idx][:len(new_lines[cudnn_benchmark_idx]) - len(new_lines[cudnn_benchmark_idx].lstrip())]
                    
                    # Move cudnn lines inside the cuda block
                    # Check if they're already inside
                    cuda_check_line = None
                    for j in range(cudnn_benchmark_idx - 1, -1, -1):
                        if 'torch.cuda.is_available()' in new_lines[j]:
                            cuda_check_line = j
                            break
                    
                    # Add MPS seed handling after cudnn.benchmark
                    mps_lines = [
                        base_indent + "if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():\n",
                        base_indent + "    # MPS seed is set via torch.manual_seed\n",
                        base_indent + "    pass\n",
                    ]
                    insert_at = cudnn_benchmark_idx + 1
                    for ml in reversed(mps_lines):
                        new_lines.insert(insert_at, ml)
                    cell_changed = True
                
                if cell_changed:
                    cell['source'] = new_lines
                    modified = True
        
        # --- Update CUDA availability prints to also show MPS ---
        new_lines = list(cell['source'])
        cell_changed = False
        for j, line in enumerate(new_lines):
            if "print(f\"CUDA available: {torch.cuda.is_available()}\")" in line:
                indent = line[:len(line) - len(line.lstrip())]
                new_lines[j] = line  # Keep the CUDA line
                # Add MPS line after if not already there
                if j + 1 < len(new_lines) and 'mps' in new_lines[j + 1].lower():
                    pass  # Already has MPS check
                elif j + 1 < len(new_lines) and 'MPS available' in new_lines[j + 1]:
                    pass  # Already has MPS check
                else:
                    mps_check = indent + "if hasattr(torch.backends, 'mps'):\n"
                    mps_print = indent + "    print(f\"Apple MPS available: {torch.backends.mps.is_available()}\")\n"
                    new_lines.insert(j + 1, mps_check)
                    new_lines.insert(j + 2, mps_print)
                    cell_changed = True
                break
        
        if cell_changed:
            cell['source'] = new_lines
            modified = True
    
    return modified


def process_notebook(nb_path, dry_run=False):
    """Process a single notebook."""
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  SKIP (JSON error): {os.path.relpath(nb_path, BASE_DIR)} — {e}")
        return False
    
    framework = detect_framework(nb)
    
    if framework == 'tensorflow':
        modified = process_tensorflow_notebook(nb_path, nb, dry_run)
    elif framework == 'pytorch':
        modified = process_pytorch_notebook(nb_path, nb, dry_run)
    else:
        # No framework (analysis notebooks) — skip GPU detection
        return False
    
    if modified and not dry_run:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write('\n')
    
    return modified


def main():
    dry_run = '--dry-run' in sys.argv
    if dry_run:
        print("DRY RUN MODE — no files will be modified\n")
    
    notebooks = collect_notebooks()
    print(f"Found {len(notebooks)} notebooks to process\n")
    
    modified_count = 0
    skipped_count = 0
    error_count = 0
    
    for nb_path in notebooks:
        rel_path = os.path.relpath(nb_path, BASE_DIR)
        try:
            result = process_notebook(nb_path, dry_run=dry_run)
            if result:
                print(f"  ✓ Modified: {rel_path}")
                modified_count += 1
            else:
                print(f"  — Skipped:  {rel_path}")
                skipped_count += 1
        except Exception as e:
            import traceback
            print(f"  ✗ Error:    {rel_path} — {e}")
            traceback.print_exc()
            error_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total notebooks:  {len(notebooks)}")
    print(f"  Modified:         {modified_count}")
    print(f"  Skipped:          {skipped_count}")
    print(f"  Errors:           {error_count}")
    if dry_run:
        print(f"\n  (DRY RUN — no files were actually modified)")


if __name__ == '__main__':
    main()
