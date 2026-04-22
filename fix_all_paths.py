import glob
import os
import io

directory = r"C:\Users\USER\Research\SNP\LSTM-SNP-Notebooks"
subdirs = [
    "Modified_Fuzzy_MF_With_Noise_LakeErie",
    "Modified_Fuzzy_MF_With_Noise_DowJones", 
    "Modified_Fuzzy_MF_With_Noise_SP500",
    "Modified_Fuzzy_MF_With_Noise_Milk"
]

for subdir in subdirs:
    full_dir = os.path.join(directory, subdir)
    if not os.path.exists(full_dir):
        continue
        
    for filepath in glob.glob(os.path.join(full_dir, "*.ipynb")):
        with io.open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Make sure we don't accidentally replace ones that already have ../
        new_content = content.replace("'content/monthly-lake-erie-levels-1921-19.csv'", "'../content/monthly-lake-erie-levels-1921-19.csv'")
        new_content = new_content.replace("'content/sp500.csv'", "'../content/sp500.csv'")
        new_content = new_content.replace("'content/monthly-milk-production-pounds-p.csv'", "'../content/monthly-milk-production-pounds-p.csv'")
        new_content = new_content.replace("'content/monthly-closings-of-the-dowjones.csv'", "'../content/monthly-closings-of-the-dowjones.csv'")
        
        if new_content != content:
            with io.open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Updated paths in: {subdir}/{os.path.basename(filepath)}")
