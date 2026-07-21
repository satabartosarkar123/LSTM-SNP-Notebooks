import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

folders = [
    'FLSTM_Fuzzy_MF_MackeyGlass',
    'FLSTM_Fuzzy_MF_Sunspots',
    'FLSTM_Fuzzy_MF_EPL'
]

def run_notebook(nb_path):
    print(f"Starting {nb_path} ...")
    cmd = [
        'jupyter', 'nbconvert',
        '--to', 'notebook',
        '--execute',
        '--inplace',
        '--ExecutePreprocessor.timeout=-1',
        nb_path
    ]
    nb_start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    nb_elapsed = time.time() - nb_start
    if result.returncode == 0:
        return f"[SUCCESS] {nb_path} ({nb_elapsed/60:.1f} min)"
    else:
        err = "\n".join(result.stderr.splitlines()[-10:])
        return f"[FAILED] {nb_path} ({nb_elapsed/60:.1f} min)\nError: {err}"

start_time = time.time()
tasks = []
for folder in folders:
    notebooks = sorted([f for f in os.listdir(folder) if f.endswith('.ipynb')])
    for nb in notebooks:
        tasks.append(os.path.join(folder, nb))

print(f"Total notebooks to run: {len(tasks)}")
with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(run_notebook, tasks))

print("\n" + "="*50)
print("EXECUTION SUMMARY")
print("="*50)
for res in results:
    print(res)

total_elapsed = time.time() - start_time
print(f"\nAll notebooks processed in {total_elapsed/60:.1f} minutes.")
