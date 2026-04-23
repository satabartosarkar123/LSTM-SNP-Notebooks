#!/usr/bin/env python3
"""
Batch-update ALL Jupyter notebooks across the research project:
  1. Replace all dataset paths with cross-platform relative paths using pathlib
  2. Print the exact process ID (os.getpid()) before execution begins
  3. Measure total notebook duration (start cell → end cell)
  4. For "With Noise" notebooks, force noise_levels = [0.005] only (0.5%)
"""

import json
import os
import re
import copy
import sys

# ── Configuration ─────────────────────────────────────────────────────────────
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

# CSV filenames that live in content/
CSV_FILES = {
    'monthly-closings-of-the-dowjones.csv',
    'monthly-lake-erie-levels-1921-19.csv',
    'monthly-milk-production-pounds-p.csv',
    'sp500.csv',
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def collect_notebooks():
    """Collect all .ipynb files from the target directories."""
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


def compute_relative_content_path(notebook_path):
    """
    Compute the relative path from the notebook's directory to the content/ folder.
    Uses os.path.relpath which works on both Mac and Windows.
    Returns a pathlib-style forward-slash path string.
    """
    nb_dir = os.path.dirname(notebook_path)
    content_dir = os.path.join(BASE_DIR, 'content')
    rel = os.path.relpath(content_dir, nb_dir)
    # Normalise to forward slashes for cross-platform pathlib usage
    return rel.replace('\\', '/')


def fix_csv_paths(source_lines, rel_content):
    """
    Replace any absolute or relative CSV path with a pathlib-based cross-platform
    path construction.

    Strategy: We replace the read_csv argument from the raw path string to a
    pathlib construction, but we keep it simple — we replace the path string
    argument itself.
    
    Since notebooks already have working code, we do a targeted string replacement
    of the CSV path argument only.
    """
    new_lines = []
    changed = False
    for line in source_lines:
        new_line = line
        # Match read_csv('...some_csv_file.csv'...) patterns
        for csv_name in CSV_FILES:
            # Match any path ending with the csv filename, quoted with ' or "
            # Handles: absolute paths, relative paths, just the filename
            pattern = re.compile(
                r"""(['"])[^'"]*""" + re.escape(csv_name) + r"""(\1)"""
            )
            match = pattern.search(new_line)
            if match:
                quote = match.group(1)
                # Build the pathlib-based relative path string
                # e.g., str(Path(__file__).parent / '..' / 'content' / 'sp500.csv')
                # But in notebooks __file__ doesn't exist, so use os.path approach
                replacement_path = f"{rel_content}/{csv_name}"
                new_path_str = f"{quote}{replacement_path}{quote}"
                new_line = new_line[:match.start()] + new_path_str + new_line[match.end():]
                changed = True
        
        # Also fix any standalone path references that use absolute paths but aren't in read_csv
        # e.g., CSV_PATH = "/Users/satabarto/Research/content/xxx.csv"
        for csv_name in CSV_FILES:
            abs_pattern = re.compile(
                r"""(['"])/Users/[^'"]*""" + re.escape(csv_name) + r"""(\1)"""
            )
            match = abs_pattern.search(new_line)
            if match:
                quote = match.group(1)
                replacement_path = f"{rel_content}/{csv_name}"
                new_path_str = f"{quote}{replacement_path}{quote}"
                new_line = new_line[:match.start()] + new_path_str + new_line[match.end():]
                changed = True
        
        new_lines.append(new_line)
    return new_lines, changed


def fix_noise_levels(source_lines):
    """
    For 'With Noise' notebooks: force noise_levels = [0.005] only.
    """
    new_lines = []
    changed = False
    for line in source_lines:
        new_line = line
        # Match noise_levels = [0.001, 0.005, 0.01] or similar
        noise_pattern = re.compile(r'(noise_levels\s*=\s*)\[[^\]]+\]')
        match = noise_pattern.search(new_line)
        if match:
            old_val = match.group(0)
            new_val = match.group(1) + '[0.005]'
            if old_val != new_val:
                new_line = new_line[:match.start()] + new_val + new_line[match.end():]
                changed = True
        new_lines.append(new_line)
    return new_lines, changed


def make_process_id_cell():
    """Create a code cell that prints the process ID."""
    source = [
        "# ============================================================\n",
        "# PROCESS IDENTIFICATION\n",
        "# ============================================================\n",
        "import os\n",
        "print(f\"Process ID (PID): {os.getpid()}\")\n",
    ]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def make_start_timer_cell():
    """Create a code cell that starts the notebook timer."""
    source = [
        "# ============================================================\n",
        "# NOTEBOOK TIMER — START\n",
        "# ============================================================\n",
        "import time as _timer_module\n",
        "_NOTEBOOK_START_TIME = _timer_module.time()\n",
        "print(f\"Notebook execution started at: {_timer_module.strftime('%Y-%m-%d %H:%M:%S')}\")\n",
    ]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def make_end_timer_cell():
    """Create a code cell that prints the total elapsed time."""
    source = [
        "# ============================================================\n",
        "# NOTEBOOK TIMER — END\n",
        "# ============================================================\n",
        "import time as _timer_module\n",
        "_NOTEBOOK_END_TIME = _timer_module.time()\n",
        "_NOTEBOOK_ELAPSED = _NOTEBOOK_END_TIME - _NOTEBOOK_START_TIME\n",
        "_hours, _rem = divmod(_NOTEBOOK_ELAPSED, 3600)\n",
        "_minutes, _seconds = divmod(_rem, 60)\n",
        "print(f\"\\nTotal notebook execution time: {int(_hours)}h {int(_minutes)}m {_seconds:.2f}s\")\n",
        "print(f\"Total seconds: {_NOTEBOOK_ELAPSED:.2f}\")\n",
    ]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def has_pid_cell(cells):
    """Check if the notebook already has a PID cell."""
    for cell in cells:
        src = ''.join(cell.get('source', []))
        if 'os.getpid()' in src and 'Process ID' in src:
            return True
    return False


def has_start_timer_cell(cells):
    """Check if the notebook already has a start timer cell."""
    for cell in cells:
        src = ''.join(cell.get('source', []))
        if '_NOTEBOOK_START_TIME' in src and 'NOTEBOOK TIMER' in src:
            return True
    return False


def has_end_timer_cell(cells):
    """Check if the notebook already has an end timer cell."""
    for cell in cells:
        src = ''.join(cell.get('source', []))
        if '_NOTEBOOK_END_TIME' in src or ('_NOTEBOOK_ELAPSED' in src and 'Total notebook execution time' in src):
            return True
    return False


def is_noise_notebook(nb_path):
    """Check if this is a 'With Noise' notebook."""
    return ('With_Noise' in nb_path or 
            'With_Gaussian_Noises' in nb_path or
            'with_noise' in nb_path.lower() or
            'gaussian_noises' in nb_path.lower())


def add_pathlib_import_to_data_cell(source_lines, rel_content):
    """
    Add pathlib-based path resolution to the cell that loads data.
    We add an import of pathlib.Path and os at the top, and use 
    Path(__file__).resolve() — but since notebooks don't have __file__,
    we use os.path.dirname(os.path.abspath('')) which gives the CWD.
    
    Actually, the simplest cross-platform approach is:
    - Use os.path.join with os.path.dirname(os.path.abspath(''))
    - Or just use relative paths with forward slashes (works on both OS)
    
    Since pathlib handles forward slashes on Windows too, the simplest
    approach is to just use relative paths with forward slashes.
    But to be extra safe, we'll use pathlib.Path which normalizes separators.
    """
    # Check if we need to add pathlib import
    has_pathlib = any('from pathlib' in line or 'import pathlib' in line for line in source_lines)
    
    new_lines = list(source_lines)
    if not has_pathlib:
        # Find the right place to add the import — after existing imports
        insert_idx = 0
        for i, line in enumerate(new_lines):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                insert_idx = i + 1
            elif stripped and not stripped.startswith('#') and not stripped.startswith('\n'):
                break
        new_lines.insert(insert_idx, "from pathlib import Path\n")
    
    return new_lines


def process_notebook(nb_path, dry_run=False):
    """Process a single notebook with all 4 modifications."""
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  SKIP (JSON error): {nb_path} — {e}")
        return False

    cells = nb.get('cells', [])
    if not cells:
        print(f"  SKIP (no cells): {nb_path}")
        return False

    modified = False
    rel_content = compute_relative_content_path(nb_path)
    is_noise = is_noise_notebook(nb_path)

    # ── 1. Fix CSV paths ─────────────────────────────────────────────────
    for cell in cells:
        if cell.get('cell_type') != 'code':
            continue
        new_source, changed = fix_csv_paths(cell['source'], rel_content)
        if changed:
            cell['source'] = new_source
            modified = True

    # ── 2. Fix noise levels (only for noise notebooks) ────────────────────
    if is_noise:
        for cell in cells:
            if cell.get('cell_type') != 'code':
                continue
            new_source, changed = fix_noise_levels(cell['source'])
            if changed:
                cell['source'] = new_source
                modified = True

    # ── 3. Add Process ID cell (at the very top, before first code cell) ──
    if not has_pid_cell(cells):
        # Find first code cell index
        first_code_idx = 0
        for i, cell in enumerate(cells):
            if cell['cell_type'] == 'code':
                first_code_idx = i
                break
        cells.insert(first_code_idx, make_process_id_cell())
        modified = True

    # ── 4. Add timer cells ────────────────────────────────────────────────
    if not has_start_timer_cell(cells):
        # Insert right after PID cell (which is now at or near the top)
        # Find the PID cell
        pid_idx = 0
        for i, cell in enumerate(cells):
            src = ''.join(cell.get('source', []))
            if 'os.getpid()' in src and 'Process ID' in src:
                pid_idx = i
                break
        cells.insert(pid_idx + 1, make_start_timer_cell())
        modified = True

    if not has_end_timer_cell(cells):
        # Remove trailing empty code cells
        while cells and cells[-1].get('cell_type') == 'code':
            src = ''.join(cells[-1].get('source', [])).strip()
            if not src:
                cells.pop()
            else:
                break
        cells.append(make_end_timer_cell())
        modified = True

    if modified:
        nb['cells'] = cells
        if not dry_run:
            with open(nb_path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
                f.write('\n')
        return True
    return False


# ── Main ──────────────────────────────────────────────────────────────────────

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
            print(f"  ✗ Error:    {rel_path} — {e}")
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
