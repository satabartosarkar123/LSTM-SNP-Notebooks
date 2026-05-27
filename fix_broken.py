from update_runs_count import replace_text
import os

files = [
    "/Users/satabarto/Research/Transformer_Models/Pyraformer_DowJones.ipynb",
    "/Users/satabarto/Research/Transformer_Models/Pyraformer_LakeErie.ipynb",
    "/Users/satabarto/Research/Transformer_Models/Pyraformer_SP500.ipynb"
]

for filepath in files:
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = replace_text(content)
        
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated {filepath} via raw text replacement")
        else:
            print(f"No changes needed in {filepath}")
