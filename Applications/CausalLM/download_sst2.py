import os
import urllib.request
import zipfile
import csv

def download_and_extract_sst2(data_dir="sst2_data"):
    os.makedirs(data_dir, exist_ok=True)
    url = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
    zip_path = os.path.join(data_dir, "SST-2.zip")
    
    if not os.path.exists(zip_path):
        print(f"Downloading SST-2 from {url}...")
        urllib.request.urlretrieve(url, zip_path)
    
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Process train and val
    sst2_dir = os.path.join(data_dir, "SST-2")
    
    for split in ["train", "dev"]:
        in_file = os.path.join(sst2_dir, f"{split}.tsv")
        out_file = os.path.join(data_dir, f"{'val' if split == 'dev' else 'train'}.txt")
        print(f"Formatting {split} to {out_file}...")
        
        with open(in_file, 'r', encoding='utf-8') as f_in, open(out_file, 'w', encoding='utf-8') as f_out:
            reader = csv.reader(f_in, delimiter='\t')
            next(reader) # skip header
            for row in reader:
                if len(row) < 2: continue
                sentence = row[0].strip()
                label = "Positive" if row[1].strip() == "1" else "Negative"
                # Formatted for Causal LM Next-Token Prediction
                f_out.write(f"Sentence: {sentence} Sentiment: {label}\n")
                
    print("Done! You can use train.txt and val.txt with the lora_train pipeline.")

if __name__ == "__main__":
    download_and_extract_sst2()