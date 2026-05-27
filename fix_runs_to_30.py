#!/usr/bin/env python3
"""
Brute-force find-and-replace across ALL .ipynb and .py files.
Uses raw text replacement (no JSON parsing) so it works even on
malformed notebooks. Replaces any run count < 30 with 30.
"""
import os
import re

BASE_DIR = "/Users/satabarto/Research"

TARGET_DIRS = [
    "32_Sized_LSTM", "autocorrelation_analysis", "basic_statistical_analysis",
    "complexity_analysis", "content",
    "FLSTM_Fuzzy_MF_DowJones", "FLSTM_Fuzzy_MF_LakeErie",
    "FLSTM_Fuzzy_MF_MilkProduction", "FLSTM_Fuzzy_MF_SP500",
    "frequency_domain_analysis", "Fuzzy_LSTM_SNP",
    "Fuzzy_LSTM_SNP_With_Gaussian_Noises", "it2_tsk_plus", "LSTM_SNP",
    "Modified_Fuzzy_MF_DowJones", "Modified_Fuzzy_MF_LakeErie",
    "Modified_Fuzzy_MF_MilkProduction", "Modified_Fuzzy_MF_SP500",
    "Modified_Fuzzy_MF_With_Noise_DowJones",
    "Modified_Fuzzy_MF_With_Noise_LakeErie",
    "Modified_Fuzzy_MF_With_Noise_MilkProduction",
    "Modified_Fuzzy_MF_With_Noise_SP500",
    "noise_analysis", "Pure_GRU", "Pure_LSTM",
    "SNN_LSTM", "SNN_Transformer",
    "Transformer_Models",
    "type2_huarng_model", "type2_huarng_model_with_noise",
]

# Skip venv dirs entirely
SKIP_DIRS = {"snn_venv", "transformer_venv", "__pycache__", ".git", "node_modules"}


def apply_replacements(text):
    """Apply all run-count replacements using regex on raw file text."""

    # 1. Variable assignments: NUM_RUNS = <N>, N_RUNS = <N>
    text = re.sub(r'\bNUM_RUNS\s*=\s*\d+', 'NUM_RUNS = 30', text)
    text = re.sub(r'\bN_RUNS\s*=\s*\d+', 'N_RUNS = 30', text)

    # 2. for run in range(<N>):  — only change if N < 30
    def fix_run_range(m):
        n = int(m.group(1))
        if n < 30:
            return f'for run in range(30):'
        return m.group(0)
    text = re.sub(r'for run in range\((\d+)\):', fix_run_range, text)

    # 3. RUN {run+1}/<N>  (in print strings, f-strings)
    def fix_run_slash(m):
        n = int(m.group(1))
        if n < 30:
            return 'RUN {run+1}/30'
        return m.group(0)
    text = re.sub(r'RUN \{run\+1\}/(\d+)', fix_run_slash, text)

    # 4. (<N> runs)  in text/print statements
    def fix_n_runs_paren(m):
        n = int(m.group(1))
        if n < 30:
            return '(30 runs)'
        return m.group(0)
    text = re.sub(r'\((\d+) runs\)', fix_n_runs_paren, text)

    # 5. <N> runs each
    def fix_n_runs_each(m):
        n = int(m.group(1))
        if n < 30:
            return '30 runs each'
        return m.group(0)
    text = re.sub(r'(\d+) runs each', fix_n_runs_each, text)

    # 6. <N>-Run Protocol
    def fix_run_protocol(m):
        n = int(m.group(1))
        if n < 30:
            return '30-Run Protocol'
        return m.group(0)
    text = re.sub(r'(\d+)-Run Protocol', fix_run_protocol, text)

    # 7. "over <N> runs"
    def fix_over_runs(m):
        n = int(m.group(1))
        if n < 30:
            return 'over 30 runs'
        return m.group(0)
    text = re.sub(r'over (\d+) runs', fix_over_runs, text)

    # 8. FINAL RESULTS ... (<N> runs)  — broader pattern
    def fix_final_results(m):
        n = int(m.group(1))
        if n < 30:
            return m.group(0).replace(f'({n} runs)', '(30 runs)')
        return m.group(0)
    text = re.sub(r'FINAL RESULTS[^(]*\((\d+) runs\)', fix_final_results, text)

    # 9. Summary Statistics (<N> runs)
    def fix_summary_stats(m):
        n = int(m.group(1))
        if n < 30:
            return m.group(0).replace(f'({n} runs)', '(30 runs)')
        return m.group(0)
    text = re.sub(r'Summary Statistics[^(]*\((\d+) runs\)', fix_summary_stats, text)

    # 10. Average ... (<N> runs ...
    def fix_avg_runs(m):
        n = int(m.group(1))
        if n < 30:
            return m.group(0).replace(f'{n} runs', '30 runs')
        return m.group(0)
    text = re.sub(r'(?:Average|Mean)[^(]*\((\d+) runs', fix_avg_runs, text)

    # 11. Best of <N> runs
    def fix_best_of(m):
        n = int(m.group(1))
        if n < 30:
            return f'Best of 30 runs'
        return m.group(0)
    text = re.sub(r'Best of (\d+) runs', fix_best_of, text)

    return text


def process_file(filepath):
    """Read file, apply replacements, write back if changed."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            original = f.read()
    except Exception as e:
        print(f"  SKIP (read error): {filepath}: {e}")
        return False

    updated = apply_replacements(original)

    if updated != original:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(updated)
            print(f"  UPDATED: {filepath}")
            return True
        except Exception as e:
            print(f"  SKIP (write error): {filepath}: {e}")
            return False
    return False


def main():
    total_updated = 0

    # Process each target directory recursively
    for dirname in TARGET_DIRS:
        dirpath = os.path.join(BASE_DIR, dirname)
        if not os.path.isdir(dirpath):
            print(f"  SKIP (not a directory): {dirpath}")
            continue

        for root, dirs, files in os.walk(dirpath):
            # Prune skip dirs
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            for fname in files:
                if fname.endswith('.ipynb') or fname.endswith('.py'):
                    fpath = os.path.join(root, fname)
                    if process_file(fpath):
                        total_updated += 1

    # Process top-level .py files
    print("\n--- Top-level .py files ---")
    for fname in sorted(os.listdir(BASE_DIR)):
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.isfile(fpath) and fname.endswith('.py'):
            if process_file(fpath):
                total_updated += 1

    print(f"\n=== Done. Updated {total_updated} files. ===")


if __name__ == '__main__':
    main()
