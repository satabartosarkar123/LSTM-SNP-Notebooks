import glob
import os
import io

directory = r"C:\Users\USER\Research\SNP\LSTM-SNP-Notebooks"
count = 0

for filepath in glob.glob(os.path.join(directory, "**", "*.ipynb"), recursive=True):
    with io.open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Remove the problematic decorator completely to prevent re-registration issues
    if '"@tf.keras.utils.register_keras_serializable()\\n",' in content:
        new_content = content.replace('"@tf.keras.utils.register_keras_serializable()\\n",', '')
        
        # In case it's missing the trailing comma
        new_content = new_content.replace('"@tf.keras.utils.register_keras_serializable()\\n"', '')

        if new_content != content:
            with io.open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)
            count += 1
            print(f"Fixed: {os.path.basename(filepath)}")

print(f"Total notebooks fixed: {count}")
