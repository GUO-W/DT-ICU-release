#!/usr/bin/env python
# encoding: utf-8

# Read subject_id and stay_id from test_ids.csv, train .. val ..
# Count the number of rows in that file and subtract 1 (to exclude the header) in corresponding dynamic.csv
# Write the stay_id and the resulting row count to data_seqlen_analy/test_seqlen.txt, val .. train .. 


import os
import pandas as pd
from tqdm import tqdm

def generate_seqlen_file(folder_path: str, split_name: str) -> None:
    ids_path = os.path.join(folder_path, f"../../data/mimiciii1.4/train_test_val_split/{split_name}_ids.csv")
    output_path = os.path.join(folder_path, f"data_seqlen_analy/{split_name}_seqlen.txt")

    ids = pd.read_csv(ids_path)
    with open(output_path, 'w') as f_out:
        for _, row in tqdm(ids.iterrows(), total=len(ids), desc="Writing IDs"):
        #for _, row in ids.iterrows():
            subject_id = str(row['subject_id'])
            stay_id = str(row['stay_id'])

            subfolder = f"{subject_id}_{stay_id}"
            dynamic_file = os.path.join(folder_path, '../../data/mimiciii1.4/processed_icu/', subfolder, 'dynamic.csv')

            if os.path.exists(dynamic_file):
                # Count number of lines (excluding header)
                with open(dynamic_file, 'r') as f:
                    num_lines = sum(1 for _ in f) - 1
                f_out.write(f"{stay_id} {num_lines}\n")
            else:
                print(f"[Warning] File not found: {dynamic_file}")
                exit()



if __name__ == "__main__":
    base_folder = './' 
    for split in ['train', 'val', 'test']:
        print(f"Processing {split}_ids.csv...")
        generate_seqlen_file(base_folder, split)

    print("done.")