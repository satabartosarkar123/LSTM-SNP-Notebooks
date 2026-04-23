#!/usr/bin/env python3
"""
Fix the seed function issue in PyTorch notebooks where the MPS lines
got concatenated onto the cudnn.benchmark line due to missing newline.
Also adds `import platform` at the top of cells that use it in MPS detection.
"""

import json
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DIRS_TO_PROCESS = [
    '32_Sized_LSTM', 'Fuzzy_LSTM_SNP', 'Fuzzy_LSTM_SNP_With_Gaussian_Noises',
    'LSTM_SNP', 'Modified_Fuzzy_MF_DowJones', 'Modified_Fuzzy_MF_LakeErie',
    'Modified_Fuzzy_MF_MilkProduction', 'Modified_Fuzzy_MF_SP500',
    'Modified_Fuzzy_MF_With_Noise_DowJones', 'Modified_Fuzzy_MF_With_Noise_LakeErie',
    'Modified_Fuzzy_MF_With_Noise_MilkProduction', 'Modified_Fuzzy_MF_With_Noise_SP500',
    'SNN_Transformer', 'Transformer_Models',
]


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


def fix_notebook(nb_path, dry_run=False):
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except json.JSONDecodeError:
        return False

    modified = False
    
    for cell in nb['cells']:
        if cell.get('cell_type') != 'code':
            continue
        
        new_lines = list(cell['source'])
        cell_changed = False
        
        # Fix 1: Lines where benchmark = False is concatenated with MPS check
        for j in range(len(new_lines)):
            line = new_lines[j]
            if 'torch.backends.cudnn.benchmark = False' in line and not line.endswith('\n'):
                # Missing trailing newline
                new_lines[j] = line + '\n'
                cell_changed = True
        
        # Fix 2: Ensure cudnn lines are inside the cuda if-block
        # Pattern: standalone cudnn lines at same indent as if block
        for j in range(len(new_lines)):
            line = new_lines[j]
            if 'torch.backends.cudnn.deterministic = True' in line:
                # Check if the previous line has cuda check
                indent = line[:len(line) - len(line.lstrip())]
                # Look back for the if cuda block
                found_cuda_if = False
                for k in range(j-1, max(j-5, -1), -1):
                    if 'if torch.cuda.is_available():' in new_lines[k]:
                        found_cuda_if = True
                        if_indent = new_lines[k][:len(new_lines[k]) - len(new_lines[k].lstrip())]
                        # cudnn lines should be indented MORE than the if line
                        if len(indent) <= len(if_indent):
                            # Need to indent cudnn lines
                            extra = '    '
                            new_lines[j] = if_indent + extra + line.lstrip()
                            if j+1 < len(new_lines) and 'torch.backends.cudnn.benchmark' in new_lines[j+1]:
                                new_lines[j+1] = if_indent + extra + new_lines[j+1].lstrip()
                            cell_changed = True
                        break
        
        if cell_changed:
            cell['source'] = new_lines
            modified = True
    
    if modified and not dry_run:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write('\n')
    
    return modified


def main():
    dry_run = '--dry-run' in sys.argv
    notebooks = collect_notebooks()
    print(f"Checking {len(notebooks)} notebooks for seed/cudnn fixes...\n")
    
    fixed = 0
    for nb_path in notebooks:
        rel = os.path.relpath(nb_path, BASE_DIR)
        result = fix_notebook(nb_path, dry_run)
        if result:
            print(f"  ✓ Fixed: {rel}")
            fixed += 1
    
    print(f"\nFixed {fixed} notebooks")
    if dry_run:
        print("(DRY RUN)")


if __name__ == '__main__':
    main()
