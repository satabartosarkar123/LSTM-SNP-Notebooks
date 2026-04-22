import glob
import os
import io

directory = r"C:\Users\USER\Research\SNP\LSTM-SNP-Notebooks"
count = 0

for filepath in glob.glob(os.path.join(directory, "**", "*.ipynb"), recursive=True):
    with io.open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Replace all dataset paths safely
    new_content = content.replace("'content/monthly-lake-erie-levels-1921-19.csv'", "'../content/monthly-lake-erie-levels-1921-19.csv'")
    new_content = new_content.replace("'content/sp500.csv'", "'../content/sp500.csv'")
    new_content = new_content.replace("'content/monthly-milk-production-pounds-p.csv'", "'../content/monthly-milk-production-pounds-p.csv'")
    new_content = new_content.replace("'content/monthly-closings-of-the-dowjones.csv'", "'../content/monthly-closings-of-the-dowjones.csv'")
    
    # Just in case there are double quotes used
    new_content = new_content.replace('"content/monthly-lake-erie-levels-1921-19.csv"', '"../content/monthly-lake-erie-levels-1921-19.csv"')
    new_content = new_content.replace('"content/sp500.csv"', '"../content/sp500.csv"')
    new_content = new_content.replace('"content/monthly-milk-production-pounds-p.csv"', '"../content/monthly-milk-production-pounds-p.csv"')
    new_content = new_content.replace('"content/monthly-closings-of-the-dowjones.csv"', '"../content/monthly-closings-of-the-dowjones.csv"')

    if new_content != content:
        with io.open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        count += 1
        print(f"Updated paths in: {os.path.basename(os.path.dirname(filepath))}/{os.path.basename(filepath)}")

print(f"Total notebooks updated: {count}")
