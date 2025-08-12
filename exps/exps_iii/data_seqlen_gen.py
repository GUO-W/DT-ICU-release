#!/usr/bin/env python
# encoding: utf-8

# Read subject_id and stay_id from test_ids.csv, train .. val ..
# Count the number of rows in that file and subtract 1 (to exclude the header) in corresponding dynamic.csv
# Write the stay_id and the resulting row count to data_seqlen_analy/test_seqlen.txt, val .. train .. 


import os
import pandas as pd
from tqdm import tqdm

import json
from pathlib import Path


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

def generate_train_balance_json(seqlen_path: str, labels_path: str, output_path: str) -> None:
    seqlen_dict = {}
    with open(seqlen_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                stay_id, seqlen = parts
                seqlen_dict[int(stay_id)] = int(seqlen)

    labels_df = pd.read_csv(labels_path)
    output_dict = {}

    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Building length dict"):
        subject_id = str(row['subject_id'])
        stay_id = int(row['stay_id'])
        icu_death = int(row['icu_death'])

        if stay_id in seqlen_dict:
            output_dict[stay_id] = [str(seqlen_dict[stay_id]), icu_death]

    with open(output_path, 'w') as f:
        json.dump(output_dict, f)
    print("done")

if __name__ == "__main__":
    ## gen ./data_seqlen_analy/{train/val/test}_seqlen.txt
    # base_folder = './' 
    # for split in ['train', 'val', 'test']:
    #     print(f"Processing {split}_ids.csv...")
    #     generate_seqlen_file(base_folder, split)
    # print("done.")

    ## gen ./data_seqlen_analy/train_length_dict.json
    seqlen_path = Path("./data_seqlen_analy/train_seqlen.txt")
    labels_path = Path("../../data/mimiciii1.4/icu_death_labels.csv")
    output_path = Path("./data_seqlen_analy/train_length_dict.json")
    generate_train_balance_json(seqlen_path, labels_path, output_path)


