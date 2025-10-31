import random
import time
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from multiprocessing import Pool, cpu_count
import os

import torch
from torch.utils.data import Dataset, Sampler, DataLoader

from config import cfg
from utils import init_read


# Function to load the data using multiprocessing
def load_data(ids):
    file_paths = [cfg.root_dir + "/data/mimiciv3.1/processed_icu/" + str(i[0]) + "_" + str(i[1]) for i in ids]#[:10000]
    print("Loading all data using multiprocessing...")
    info = {}

    race_vocab, gender_vocab, insurance_vocab, admission_vocab, icu_vocab = init_read(cfg.root_dir)
    # Create a pool of workers for multiprocessing
    nb_preprocess = 20 #cpu_count()
    with Pool(nb_preprocess) as pool:
        results = list(
            pool.starmap(
                process_file,
                [(f, gender_vocab, race_vocab, insurance_vocab, admission_vocab, icu_vocab) for f in file_paths],
            ),
        )
    # Collect the results into the info dictionary, same as in the original function
    for file_id, file_info in results:
        if file_id == -1:
            continue
        info[file_id] = file_info
    return info


def process_file(file_path, gender_vocab, race_vocab, insurance_vocab, admission_vocab, icu_vocab):  # Function to process a single file
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
    return file_id, info


def custom_collate(batch, min_seq_len=5, max_seq_len=None):
    """
    Custom collate function for samples that are lists of tensors.
    
    For the first 5 tensors (sequential data):
      - Determines a random subsequence length L (common to the entire batch) that is at least min_seq_len.
      - Uses the minimum available temporal length among samples (or max_seq_len, if provided) as an upper bound.
      - Crops a contiguous segment of length L from each tensor.
      
    For the remaining tensors (non-sequential data):
      - Simply stacks them along the batch dimension.
    
    Args:
        batch: list of samples; each sample is a list of tensors.
        min_seq_len: minimum allowed length for the temporal crop.
        max_seq_len: if provided, maximum allowed temporal length.
        
    Returns:
        A list of tensors where the first 5 elements are batched sequential data with shape (batch_size, L, ...)
        and the remaining elements are batched non-sequential data.
    """
    # Number of sequential tensors assumed to be 5.
    num_seq = 6
    batch_size = len(batch)

    # For the sequential parts, assume all 5 tensors in a sample have the same temporal length.
    # Get the temporal lengths from the first sequential tensor of each sample.
    seq_lengths = [sample[0].shape[0] for sample in batch]
    # in the case of choice 1, we must NOT include the last frame as the last frame is death
    seq_lengths_choice_one = []
    for sample in batch:
        if sample[-1] == 1:
            seq_lengths_choice_one.append(sample[0].shape[0] - 1)
    batch_min_length = min(seq_lengths)
    if len(seq_lengths_choice_one) == 0:
        batch_min_length_choice_one = batch_min_length
    else:
        batch_min_length_choice_one = min(seq_lengths_choice_one)

    batch_min_length = min(batch_min_length, batch_min_length_choice_one)
    
    # If a maximum allowed sequence length is provided, use the smaller of batch_min_length and max_seq_len.
    allowed_max = batch_min_length if max_seq_len is None else min(batch_min_length, max_seq_len)
    
    # Ensure min_seq_len does not exceed allowed_max.
    if min_seq_len > allowed_max:
        raise ValueError(f"{min_seq_len} exceeds the available sequence length {allowed_max} in this batch.")
    
    # Choose a random subsequence length L for this batch.
    L = random.randint(min_seq_len, allowed_max)
    
    # Prepare lists to hold cropped sequential data for each of the first 5 tensors.
    batched_seqs = [[] for _ in range(num_seq)]
    
    # Also prepare a dictionary for the non-sequential parts.
    non_seq_indices = range(num_seq, len(batch[0]) - 1)
    batched_non_seq = {i: [] for i in non_seq_indices}
    
    # Process each sample.
    for sample in batch:
        # All sequential tensors in the sample should have the same temporal length.
        seq_len = sample[0].shape[0]
        # Randomly choose a starting index so that the segment of length L fits.
        start_idx = random.randint(0, seq_len - L)

        choice = sample[-1]
        
        # Crop each of the first 5 sequential tensors.
        for i in range(num_seq):
            if choice == 0 or choice == 1:
                cropped = sample[i][start_idx : start_idx + L]
            elif choice == 2:
                cropped = sample[i][-L:]
            else:
                raise NotImplementedError(f"{choice} mode has not been implemented!!!")
            batched_seqs[i].append(cropped)
        
        # For non-sequential tensors, just collect them.
        for i in non_seq_indices:
            batched_non_seq[i].append(sample[i])
    
    # Stack the sequential parts so that each becomes a tensor of shape (batch_size, L, ...).
    batched_seqs = [torch.stack(seq_list) for seq_list in batched_seqs]
    
    # Stack non-sequential tensors normally.
    batched_non_seq = [torch.stack(batched_non_seq[i]) for i in non_seq_indices]
    
    # Return a list combining the sequential and non-sequential outputs.
    # The first 5 elements are the processed sequential tensors.
    return batched_seqs + batched_non_seq


class MimicDataset(Dataset):
    def __init__(self, mode, ids, labels, data_icu, balanced_sampling=False, min_length=None):
        self.mode = mode
        print("Building dataset for ", self.mode)
        self.ids = ids
        self.labels = labels
        self.data_icu = data_icu
        self.balanced_sampling = balanced_sampling
        self.min_length = min_length

        t0 = time.time()
        self.data = load_data(self.ids)
        t1 = time.time()
        print(f"data loaded by multi-processing, used {t1-t0} s.")
        # assert len(set(self.ids)) == len(self.data)

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

    def get_dynamic_data(self, dyn):
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
        return meds, chart, out, proc, date, ing

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
        meds, chart, out, proc, date, ing = self.get_dynamic_data(dyn)

        if self.min_length:
            if choice == 1:
                while meds.shape[0] < self.min_length + 1:
                    keys = list(self.data.keys())
                    sample = np.random.choice(keys)
                    dyn = self.data[sample]["dynamic"]
                    meds, chart, out, proc, date, ing = self.get_dynamic_data(dyn)          
            else:
                while meds.shape[0] < self.min_length:
                    keys = list(self.data.keys())
                    sample = np.random.choice(keys)
                    dyn = self.data[sample]["dynamic"]
                    meds, chart, out, proc, date, ing = self.get_dynamic_data(dyn)

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
        balanced_sampling=True,
        min_length=data_config.min_length + 1
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
