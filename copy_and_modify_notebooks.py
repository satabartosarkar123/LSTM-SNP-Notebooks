import os
import json
import glob

notebook_files = glob.glob('LSTM_SNP_*.ipynb')
processed_count = 0

for filepath in notebook_files:
    if '32sizedlstm' in filepath:
        continue
        
    base_name = os.path.splitext(filepath)[0]
    new_filepath = f"32sizedlstm_{base_name}.ipynb"
    
    with open(filepath, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
        
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            new_source = []
            for line in cell.get('source', []):
                # Replace units=8 with units=32
                line = line.replace('units=8', 'units=32')
                # Replace bias[8:16] with bias[32:64]
                line = line.replace('bias[8:16]', 'bias[32:64]')
                # Replace bias[units:2*units] = bias[8:16] to 32:64 (some comments)
                new_source.append(line)
            cell['source'] = new_source
            
    with open(new_filepath, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    processed_count += 1
    print(f"Created: {new_filepath}")

print(f"Processed {processed_count} notebooks.")
