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
    file_paths = [cfg.root_dir + "/data/mimiciv3.1/processed_icu/" + str(i[0]) + "_" + str(i[1]) for i in ids][:100]
    print("Loading all data using multiprocessing...")
    info = {}

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
        if meds.shape[0] <= 1:
            print("skip data with shorter than 1h", file_path)
            continue

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

        if self.balanced_sampling:
            pos_samples = []
            neg_samples = []
            for sample in self.ids:
                y = int(self.labels[self.labels["stay_id"] == sample[1]]["icu_death"].iloc[0])
                if y == 0:
                    neg_samples.append(sample[1])
                else:
                    pos_samples.append(sample[1])
            self.pos_samples = pos_samples
            self.neg_samples = neg_samples

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if not self.balanced_sampling:
            sample = self.ids[idx][1]
        else:
            if np.random.rand() > 0.5:
                # sample positive samples
                # choice 0: pos from pos samples(can randomly sample every frame)
                # choice 1: pos from neg samples(last frame must NOT be included)
                if np.random.rand() > 0.5:
                    sample = np.random.choice(self.pos_samples)
                    choice = 0
                else:
                    sample = np.random.choice(self.neg_samples)
                    choice = 1
            else:
                # sample negative samples
                # choice 2: neg from neg sample(last frame must be included)
                sample = np.random.choice(self.neg_samples)
                choice = 2

        # Load dynamic data and handle missing keys
        meds, chart, out, proc, date, ing = [
            torch.zeros(size=(0, 0))
        ] * 6  # Initialize placeholder tensors for the data keys we expect
        #######
        if sample not in self.data:
            keys = list(self.data.keys())
            sample = np.random.choice(keys)
        #######
        dyn = self.data[sample]["dynamic"]
        #inputevents, procedureevents, outputevents, chartevents; datetimeevents, ingredientevents
        keys = ["inputevents", "procedureevents", "outputevents", "chartevents", "datetimeevents", "ingredientevents"] 
        for key in keys:
            if key == "inputevents":
                matching_columns = [col for col in dyn.columns if "inputevents" in col]
                meds = torch.tensor(dyn[matching_columns].to_numpy()).float()
            elif key == "procedureevents":
                matching_columns = [col for col in dyn.columns if "procedureevents" in col]
                chart = torch.tensor(dyn[matching_columns].to_numpy()).float()
            elif key == "outputevents":
                matching_columns = [col for col in dyn.columns if "outputevents" in col]
                out = torch.tensor(dyn[matching_columns].to_numpy()).float()
            elif key == "chartevents":
                matching_columns = [col for col in dyn.columns if "chartevents" in col]
                proc = torch.tensor(dyn[matching_columns].to_numpy()).float()
            elif key == "datetimeevents":
                matching_columns = [col for col in dyn.columns if "datetimeevents" in col]
                date = torch.tensor(dyn[matching_columns].to_numpy()).float()
            elif key == "ingredientevents":
                matching_columns = [col for col in dyn.columns if "ingredientevents" in col]
                ing = torch.tensor(dyn[matching_columns].to_numpy()).float()

        # Load static data
        stat = self.data[sample]["static"]
        stat = torch.tensor(stat).float()

        # Load demographic data (and handle missing values)
        demo = self.data[sample]["demo"] # gender, anchor_age, insurance, race; icu_type, admission_type, patientweight; language, marital_status
        demo = torch.tensor(demo).float()

        # Load label y
        # if self.data_icu:
        #     y = self.labels[self.labels["stay_id"] == sample]["icu_death"].iloc[0]
        # else:
        #     y = self.labels[self.labels["hadm_id"] == sample]["icu_death"].iloc[0]
        if not self.balanced_sampling:
            y = self.labels[self.labels["stay_id"] == sample]["icu_death"].iloc[0]
            choice = 0 # unused
        else:
            if choice == 2:
                y = 0
            else:
                y = 1

        y = torch.tensor(int(y)).reshape(-1).long()
        choice = torch.tensor(int(choice)).reshape(-1).long()

        assert meds.shape[0] == chart.shape[0] == out.shape[0] == proc.shape[0] == date.shape[0] == ing.shape[0], "number of frames must be equal!! if you see this error, that means your data preprocessing has some issues.."
        outputs = [meds, chart, out, proc, date, ing, stat, demo, y, choice]
        outputs = [i.cuda() for i in outputs]
        return outputs


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
    )
    val_dataset = MimicDataset(
        "val",
        val_hids,
        labels,
        data_config.data_icu,
    )
    test_dataset = MimicDataset(
        "test",
        test_hids,
        labels,
        data_config.data_icu,
    )
    return train_dataset, val_dataset, test_dataset


def build_dataloaders(train_dataset, val_dataset, test_dataset, data_config):
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        collate_fn=lambda b: custom_collate(b, min_seq_len=data_config.min_length + 1, max_seq_len=data_config.max_length + 1),
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        collate_fn=lambda b: custom_collate(b, min_seq_len=data_config.min_length + 1, max_seq_len=data_config.max_length + 1),
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test_batch_size,
        collate_fn=lambda b: custom_collate(b, min_seq_len=data_config.min_length + 1, max_seq_len=data_config.max_length + 1),
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader


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

def main():
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

if __name__ == "__main__":
    main()
