import json
import sys

def search_notebook(filepath):
    with open(filepath, 'r') as f:
        nb = json.load(f)
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if 'range(' in source or 'runs' in source.lower():
                print(f"--- Cell {i} ---")
                print(source)

search_notebook('/Users/satabarto/Research/32_Sized_LSTM/32sizedlstm_LSTM_SNP_dow_jones_closing.ipynb')
