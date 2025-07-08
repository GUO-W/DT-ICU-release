import random
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle as pkl
from imblearn.over_sampling import RandomOverSampler
from multiprocessing import Pool, cpu_count
import os

import torch
from torch.utils.data import Dataset, Sampler, DataLoader

from config import cfg
from utils import init_read


# Function to load the data using multiprocessing
def load_data(mode, ids):
    file_paths = [cfg.root_dir + "/data/mimiciv3.1/processed_icu/" + str(i[0]) + "_" + str(i[1]) for i in ids][:10000]
    print("Loading all data using multiprocessing...")

    race_vocab, gender_vocab, insurance_vocab, admission_vocab, icu_vocab = init_read(cfg.root_dir)
    num_subsets = cpu_count()
    subset_length = len(file_paths) // num_subsets

    subfiles = []
    for i in range(num_subsets):
        if i == num_subsets - 1:
            subfiles.append(file_paths[subset_length * i:])
        else:
            subfiles.append(file_paths[subset_length * i: subset_length * (i + 1)])

    args_list = [(mode, str(i), f, gender_vocab, race_vocab, insurance_vocab, admission_vocab, icu_vocab) for i, f in enumerate(subfiles)]
    with Pool(processes=num_subsets) as pool:
        output_files = pool.map(process_files, args_list)
    # Collect the results into the info dictionary, same as in the original function
    return output_files


def process_files(args):  # Function to process a single file
    mode, rank, file_paths, gender_vocab, race_vocab, insurance_vocab, admission_vocab, icu_vocab = args
    infos = {}
    # w = open("data/" + mode + "/"  + rank + "_seqlen.txt", "w")
    for file_path in tqdm(file_paths):
        """Function to process a single file."""
        file_id = int(file_path.split("/")[-1].split("_")[-1])
        info = {}

        exist_all = os.path.exists(f"{file_path}/dynamic.csv") and os.path.exists(f"{file_path}/diagnoses.csv") and os.path.exists(f"{file_path}/demo.csv")
        if not exist_all:
            print(file_path)
            return -1, {}

        # Read the CSV files
        dyn = pd.read_csv(f"{file_path}/dynamic.csv", header=0) # inputevents, procedureevents, outputevents, chartevents; datetimeevents, ingredientevents
        stat = pd.read_csv(f"{file_path}/diagnoses.csv", header=0)
        demo = pd.read_csv(f"{file_path}/demo.csv", header=0) # gender, anchor_age, insurance, race; icu_type, admission_type, patientweight; language, marital_status

        matching_columns = [col for col in dyn.columns if "inputevents" in col]
        meds = dyn[matching_columns]

        # w.write(str(file_id) + " " + str(meds.shape[0]) + "\n")
        
        #if meds.shape[0] < 72:
        #    print("skip data with shorter than 6h", file_path)
        #    continue

        # # Replace demographic values based on vocab_data
        demo["gender"].replace(gender_vocab, inplace=True)
        demo["race"].replace(race_vocab, inplace=True)
        demo["insurance"].replace(insurance_vocab, inplace=True)
        demo["admission_type"].replace(admission_vocab, inplace=True)
        demo["icu_type"].replace(icu_vocab, inplace=True)

        # Store the processed data in a dictionary for this file
        info["dynamic"] = dyn 
        info["static"] = stat.to_numpy().reshape(-1) 
        info["demo"] = demo[["gender", "race", "insurance", "anchor_age", "admission_type", "icu_type"]].values.reshape(-1)
        info["demo"] = np.nan_to_num(info["demo"], nan=0)
        infos[file_id] = info

    with open("data/" + mode + "/"  + rank + ".pkl", "wb") as w:
        pkl.dump(infos, w)
    print("Finished with samples:", len(infos))
    return infos


class MimicDataset(Dataset):
    def __init__(self, mode, ids, labels, data_icu, balanced_sampling=False):
        self.mode = mode
        print("Building dataset for ", self.mode)
        self.ids = ids
        self.labels = labels
        self.data_icu = data_icu
        self.balanced_sampling = balanced_sampling

        t0 = time.time()
        self.data = load_data(mode, self.ids)
        t1 = time.time()
        print(f"data loaded by multi-processing, used {t1-t0} s.")


class data_config:
    def __init__(self, data_icu, modalities, min_length=5, max_length=20, train=True):
        self.data_icu = data_icu
        self.modalities = modalities
        self.min_length = min_length
        self.max_length = max_length


def build_datasets(train_hids, val_hids, test_hids, labels, data_config):
    train_dataset = MimicDataset(
        "train",
        train_hids,
        labels,
        data_config.data_icu,
    ) if train_hids else None
    val_dataset = MimicDataset(
        "val",
        val_hids,
        labels,
        data_config.data_icu,
    ) if val_hids else None
    test_dataset = MimicDataset(
        "test",
        test_hids,
        labels,
        data_config.data_icu,
    ) if test_hids else None
    return train_dataset, val_dataset, test_dataset


def load_trainval_data():
    labels = pd.read_csv(cfg.root_dir +  "/data/mimiciv3.1/processed_icu/icu_death_labels.csv", header=0)
    stay_id = labels.iloc[:, 2]
    death_label = labels.iloc[:, 3]
    print("Total Samples", len(stay_id))
    print("Positive Samples", death_label.sum())
    print("Negative Samples", len(death_label) - death_label.sum())

    train_ids = pd.read_csv(cfg.root_dir +  "/data/mimiciv3.1/train_test_val_split/train_ids.csv", header=0)
    train_ids = train_ids.iloc[:, :2].values.tolist()
    test_ids = pd.read_csv(cfg.root_dir +  "/data/mimiciv3.1/train_test_val_split/test_ids.csv", header=0)
    test_ids = test_ids.iloc[:, :2].values.tolist()
    val_ids = pd.read_csv(cfg.root_dir +  "/data/mimiciv3.1/train_test_val_split/val_ids.csv", header=0)
    val_ids = val_ids.iloc[:, :2].values.tolist()
    return train_ids, val_ids, test_ids, labels

if __name__ == "__main__":
    from utils import fix_seed, get_data_mean_std

    fix_seed(cfg.seed)
    print("seed:", cfg.seed)
    # Init dataset cfg
    modalities = (
        cfg.med_flag
        + cfg.chart_flag
        + cfg.out_flag
        + cfg.proc_flag
        + cfg.date_flag
        + cfg.ing_flag
    )
    print("total modalities:", modalities)
    datacfg = data_config(cfg.data_icu, modalities, cfg.train_min_length, cfg.train_max_length, train=True)
    train_ids, val_ids, test_ids, labels = load_trainval_data()

    train_dataset, val_dataset, eval_dataset = build_datasets(train_ids, val_ids, test_ids, labels, datacfg)
    
    # calculate mean/std
    #train_dataset, val_dataset, eval_dataset = build_datasets(train_ids, None, None, labels, datacfg)
    # meds, chart, out, proc, date, ing, stat, _, _, _ = train_dataset[0]
    # datacfg.stat_vocab_size = stat.shape[-1]
    # datacfg.proc_vocab_size = proc.shape[-1]
    # datacfg.med_vocab_size = meds.shape[-1]
    # datacfg.out_vocab_size = out.shape[-1]
    # datacfg.chart_vocab_size = chart.shape[-1]
    # datacfg.date_vocab_size = date.shape[-1]
    # datacfg.ing_vocab_size = ing.shape[-1]
    # means, stds = get_data_mean_std(train_dataset, val_dataset)


