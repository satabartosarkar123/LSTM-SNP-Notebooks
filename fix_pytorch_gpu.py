#!/usr/bin/env python3
"""
Update ALL PyTorch notebooks with a proper standalone Apple Metal GPU
detection cell — matching the style used in TensorFlow notebooks.

Changes:
  1. Removes any inline device-selection code from import cells
  2. Inserts a dedicated "GPU Acceleration Settings" cell (like the TF notebooks)
  3. Makes device variable available globally for all subsequent cells
  4. Updates seed function to properly handle Metal/MPS
  5. Updates any DEVICE = "mps" if ... patterns (SNN_Transformer style)
"""

import json
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DIRS_TO_PROCESS = [
    'SNN_Transformer',
    'Transformer_Models',
]

# ── The new GPU detection cell (matches TF notebook style) ────────────────────

GPU_CELL_SOURCE = [
    "# ============================================================\n",
    "# GPU Acceleration Settings (CUDA + Apple Metal Support)\n",
    "# ============================================================\n",
    "import torch\n",
    "import platform\n",
    "\n",
    "# Device priority: CUDA > Apple MPS (Metal) > CPU\n",
    "if torch.cuda.is_available():\n",
    "    device = torch.device('cuda')\n",
    "    print(f\"CUDA GPU Enabled: {torch.cuda.get_device_name(0)}\")\n",
    "    print(f\"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB\")\n",
    "elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():\n",
    "    device = torch.device('mps')\n",
    "    print(f\"Apple Metal GPU Enabled via MPS backend\")\n",
    "    print(f\"  Chipset: Apple {platform.machine()} (M-series)\")\n",
    "    print(f\"  MPS built: {torch.backends.mps.is_built()}\")\n",
    "else:\n",
    "    device = torch.device('cpu')\n",
    "    if platform.system() == 'Darwin' and platform.processor() == 'arm':\n",
    "        print(\"Apple Silicon detected but MPS not available.\")\n",
    "        print(\"  Upgrade PyTorch: pip install --upgrade torch\")\n",
    "    else:\n",
    "        print(\"No GPU found. Falling back to CPU.\")\n",
    "        print(\"  For CUDA: ensure NVIDIA drivers + CUDA toolkit are installed.\")\n",
    "\n",
    "print(f\"\\nUsing device: {device}\")\n",
    "print(f\"PyTorch version: {torch.__version__}\")\n",
]

SEED_CELL_SOURCE = [
    "# ============================================================\n",
    "# Seed Control — Reproducibility\n",
    "# ============================================================\n",
    "import random\n",
    "\n",
    "def set_seed(seed):\n",
    "    \"\"\"Set all random seeds for reproducibility.\"\"\"\n",
    "    random.seed(seed)\n",
    "    np.random.seed(seed)\n",
    "    torch.manual_seed(seed)\n",
    "    if torch.cuda.is_available():\n",
    "        torch.cuda.manual_seed(seed)\n",
    "        torch.cuda.manual_seed_all(seed)\n",
    "        torch.backends.cudnn.deterministic = True\n",
    "        torch.backends.cudnn.benchmark = False\n",
    "    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():\n",
    "        # MPS uses torch.manual_seed for seeding\n",
    "        pass\n",
]


def make_gpu_cell():
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": list(GPU_CELL_SOURCE),
    }


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


def has_new_gpu_cell(cells):
    """Check if notebook already has our new-style GPU cell."""
    for cell in cells:
        src = ''.join(cell.get('source', []))
        if ('Apple Metal GPU Enabled via MPS backend' in src and
            'GPU Acceleration Settings' in src):
            return True
    return False


def remove_device_from_imports(cell):
    """
    Remove inline device-selection code from an imports cell.
    Returns (new_source, was_changed).
    
    This handles the Transformer_Models pattern where device selection
    is embedded in the imports cell.
    """
    src = ''.join(cell.get('source', []))
    
    # Only process code cells that have imports AND device selection together
    if 'Device Selection' not in src and "device = torch.device(" not in src:
        return cell['source'], False
    if 'import numpy' not in src and 'import torch' not in src:
        return cell['source'], False
    
    lines = list(cell['source'])
    new_lines = []
    skip_until_blank = False
    in_device_block = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Detect start of device selection block
        if '# Device Selection' in line or '# =====' in line and 'Device' in line:
            in_device_block = True
            continue
        
        if in_device_block:
            # Skip the device priority comment
            if stripped.startswith('# Device priority'):
                continue
            # Skip the if/elif/else device block
            if (stripped.startswith('if torch.cuda.is_available()') or
                stripped.startswith('elif hasattr(torch.backends') or
                stripped.startswith('else:') or
                stripped.startswith('device = torch.device(') or
                stripped.startswith("print(f\"Using device:") or
                stripped.startswith("print(f\"  GPU:") or
                stripped.startswith("print(f\"  Memory:") or
                stripped.startswith("print(f\"  Chipset:") or
                stripped.startswith("import platform")):
                continue
            # Skip the closing ===== line
            if '# =====' in line and in_device_block:
                continue
            # Blank line after device block — end of block
            if stripped == '':
                in_device_block = False
                continue
            # If we hit non-device code, we're done
            in_device_block = False
        
        new_lines.append(line)
    
    # Remove trailing blank lines from the cell
    while new_lines and new_lines[-1].strip() == '':
        new_lines.pop()
    if new_lines:
        # Ensure last line has newline
        if not new_lines[-1].endswith('\n'):
            new_lines[-1] += '\n'
    
    changed = len(new_lines) != len(lines)
    return new_lines, changed


def fix_snn_device_pattern(cells):
    """
    Fix the SNN_Transformer pattern:
      DEVICE = "mps" if torch.backends.mps.is_available() else (...)
    Replace with just using the global `device` variable from the GPU cell.
    """
    modified = False
    for cell in cells:
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell.get('source', []))
        
        # Replace DEVICE = "mps" if ... pattern with DEVICE = str(device)
        if 'DEVICE = "mps"' in src or "DEVICE = 'mps'" in src:
            new_lines = []
            for line in cell['source']:
                if ('DEVICE = "mps"' in line or "DEVICE = 'mps'" in line):
                    # Replace with reference to the global device
                    indent = line[:len(line) - len(line.lstrip())]
                    new_lines.append(indent + "DEVICE = str(device)  # Set by GPU Acceleration cell\n")
                    modified = True
                else:
                    new_lines.append(line)
            cell['source'] = new_lines
    
    return modified


def fix_seed_function(cells):
    """
    Ensure the seed function properly handles MPS.
    """
    modified = False
    for cell in cells:
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell.get('source', []))
        
        if 'def set_seed(seed):' not in src:
            continue
        if 'torch.cuda.manual_seed' not in src:
            continue
        
        # Check if MPS handling is already there
        if 'mps' in src:
            # Check if the cudnn lines are properly indented inside cuda block
            lines = list(cell['source'])
            new_lines = []
            i = 0
            changed = False
            while i < len(lines):
                line = lines[i]
                stripped = line.strip()
                
                # Fix: cudnn lines should be inside the cuda if-block
                if 'torch.backends.cudnn.deterministic = True' == stripped:
                    indent = line[:len(line) - len(line.lstrip())]
                    # Check if previous line is 'if torch.cuda.is_available():' at same indent
                    # If so, they need to be indented more
                    for k in range(i-1, max(i-5, -1), -1):
                        if 'torch.cuda.is_available()' in lines[k]:
                            if_indent = lines[k][:len(lines[k]) - len(lines[k].lstrip())]
                            if len(indent) <= len(if_indent):
                                # Need more indentation
                                extra = if_indent + '    '
                                new_lines.append(extra + stripped + '\n')
                                changed = True
                                i += 1
                                # Also fix next line if it's benchmark
                                if i < len(lines) and 'torch.backends.cudnn.benchmark' in lines[i]:
                                    new_lines.append(extra + lines[i].strip() + '\n')
                                    changed = True
                                    i += 1
                                break
                            else:
                                new_lines.append(line)
                                i += 1
                                break
                    else:
                        new_lines.append(line)
                        i += 1
                else:
                    new_lines.append(line)
                    i += 1
            
            if changed:
                cell['source'] = new_lines
                modified = True
            continue
        
        # MPS not present — add it
        lines = list(cell['source'])
        new_lines = []
        for j, line in enumerate(lines):
            new_lines.append(line)
            if 'torch.backends.cudnn.benchmark = False' in line:
                base_indent = line[:len(line) - len(line.lstrip())]
                # Go up to find the if cuda indent level
                cuda_indent = base_indent
                for k in range(j, -1, -1):
                    if 'if torch.cuda.is_available()' in lines[k]:
                        cuda_indent = lines[k][:len(lines[k]) - len(lines[k].lstrip())]
                        break
                new_lines.append(cuda_indent + "if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():\n")
                new_lines.append(cuda_indent + "    # MPS uses torch.manual_seed for seeding\n")
                new_lines.append(cuda_indent + "    pass\n")
                modified = True
        
        cell['source'] = new_lines
    
    return modified


def process_notebook(nb_path, dry_run=False):
    """Process a single PyTorch notebook."""
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  SKIP (broken JSON): {os.path.relpath(nb_path, BASE_DIR)}")
        return False
    
    cells = nb.get('cells', [])
    if not cells:
        return False
    
    # Check if it's actually a PyTorch notebook
    full_src = ''
    for cell in cells:
        full_src += ''.join(cell.get('source', []))
    if 'import torch' not in full_src:
        return False
    
    modified = False
    
    # Skip if already has our new GPU cell
    if has_new_gpu_cell(cells):
        # Still check seed function and DEVICE pattern
        if fix_snn_device_pattern(cells):
            modified = True
        if fix_seed_function(cells):
            modified = True
        if modified:
            nb['cells'] = cells
            if not dry_run:
                with open(nb_path, 'w', encoding='utf-8') as f:
                    json.dump(nb, f, indent=1, ensure_ascii=False)
                    f.write('\n')
        return modified
    
    # 1. Remove device selection from import cells
    for cell in cells:
        if cell.get('cell_type') != 'code':
            continue
        new_source, changed = remove_device_from_imports(cell)
        if changed:
            cell['source'] = new_source
            modified = True
    
    # 2. Find the right place to insert GPU cell
    # Strategy: insert right after the timer start cell (matching TF pattern),
    # or right before the first imports cell if no timer
    insert_idx = None
    
    # Look for timer start cell
    for i, cell in enumerate(cells):
        src = ''.join(cell.get('source', []))
        if '_NOTEBOOK_START_TIME' in src:
            insert_idx = i + 1
            break
    
    # Fallback: look for PID cell
    if insert_idx is None:
        for i, cell in enumerate(cells):
            src = ''.join(cell.get('source', []))
            if 'os.getpid()' in src:
                insert_idx = i + 1
                break
    
    # Fallback: insert before first code cell with imports
    if insert_idx is None:
        for i, cell in enumerate(cells):
            if cell.get('cell_type') != 'code':
                continue
            src = ''.join(cell.get('source', []))
            if 'import' in src:
                insert_idx = i
                break
    
    if insert_idx is None:
        insert_idx = 0
    
    # Insert GPU detection cell
    cells.insert(insert_idx, make_gpu_cell())
    modified = True
    
    # 3. Fix DEVICE = "mps" if ... pattern (SNN_Transformer)
    fix_snn_device_pattern(cells)
    
    # 4. Fix seed function
    fix_seed_function(cells)
    
    nb['cells'] = cells
    
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
    print(f"Found {len(notebooks)} PyTorch notebooks to process\n")
    
    modified_count = 0
    skipped_count = 0
    
    for nb_path in notebooks:
        rel = os.path.relpath(nb_path, BASE_DIR)
        result = process_notebook(nb_path, dry_run)
        if result:
            print(f"  ✓ Modified: {rel}")
            modified_count += 1
        else:
            print(f"  — Skipped:  {rel}")
            skipped_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total:    {len(notebooks)}")
    print(f"  Modified: {modified_count}")
    print(f"  Skipped:  {skipped_count}")
    if dry_run:
        print(f"\n  (DRY RUN)")


if __name__ == '__main__':
    main()
