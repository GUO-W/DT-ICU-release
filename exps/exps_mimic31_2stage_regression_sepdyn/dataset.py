import random
import time
import json
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from multiprocessing import Pool, cpu_count
import os
import pickle as pkl
from tqdm import tqdm
import bisect, random, math
from collections import defaultdict

import torch
from torch.utils.data import Dataset, Sampler, DataLoader, BatchSampler
import torch.nn.functional as F

from config import cfg
from utils import init_read


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



class LengthAwareBalancedBatchSampler(BatchSampler):
    """
    Bucket‑by‑length + class‑balanced sampler for variable‑length data.

    Args
    ----
    seq_lengths : Sequence[int]        # len(dataset) elements
    labels      : Sequence[int]        # 0 = negative, 1 = positive
    bucket_boundaries : list[int]      # e.g. [128, 256, 512, 1024]
    batch_size  : int
    epoch_size  : int  (optional)      # #batches per epoch; default ≈ len(dataset)/batch_size
    replacement : bool                 # usually True so rare buckets never “run out”
    rng         : random.Random or None
    """
    def __init__(self,
                 seq_lengths,
                 labels,
                 bucket_boundaries,
                 batch_size,
                 mode,
                 epoch_size=None,
                 replacement=True,
                 rng=None):

        self.batch_size   = batch_size
        self.replacement  = replacement
        self.rng          = rng or random.Random()
        self.mode         = mode

        if mode != "train":
            # nothing fancy: just keep indices in natural order
            self.indices = list(range(len(seq_lengths)))
            return

        assert len(seq_lengths) == len(labels)
        # 1. assign every item to a length bucket
        bucket_of = [bisect.bisect_right(bucket_boundaries, L) for L in seq_lengths]
        self.n_buckets = n_b = len(bucket_boundaries) + 1      # last bucket = “> last boundary”

        # 2. gather indices => buckets[b][c]  (c = 0/1)
        self.buckets = [defaultdict(list) for _ in range(n_b)]
        for idx, (b, c) in enumerate(zip(bucket_of, labels)):
            self.buckets[b][c].append(idx)

        for i in range(len(self.buckets)):
            for j in range(len(self.buckets[i])):
                print("total:", ["short", "long"][i], ["neg", "pos"][j], len(self.buckets[i][j]))
        # 3. pre‑compute inverse‑freq probabilities
        #self.bucket_p = self._inv_freq_probs(
        #    [len(self.buckets[b][0]) + len(self.buckets[b][1]) for b in range(n_b)])

        #self.class_p  = {b: self._inv_freq_probs(
        #                    [len(self.buckets[b][c]) for c in (0,1)])
        #                 for b in range(n_b)}
        #print("Sample prob: neg ", self.class_p[0], "pos ", self.class_p[1])
        # 4. how many batches constitute “one epoch”?
        N = len(seq_lengths)
        self.epoch_size = epoch_size or math.ceil(N / batch_size)

    #@staticmethod
    #def _inv_freq_probs(counts):
    #    inv = [1/x if x else 0.0 for x in counts]
    #    s   = sum(inv)
    #    return [x/s for x in inv] if s else [0]*len(inv)

    # ------------------------------------------------------------ #

    def __len__(self):
        if self.mode != "train":
            return math.ceil(len(self.indices) / self.batch_size)
        return self.epoch_size

    def __iter__(self):
        if self.mode == "train":
            draw = self.rng.choices   # alias
            combs = [[0, 0], [0, 0], [0, 1], [1, 0], [1, 0], [1, 1]] # 1:2
            for idx in range(self.epoch_size):
                # 1) bucket
                #b = draw(range(self.n_buckets), weights=self.bucket_p, k=1)[0]

                # 2) label inside bucket (0 = neg, 1 = pos)
                #c = draw((0,1), weights=self.class_p[b], k=1)[0]

                # 3) indices
                #pool = self.buckets[b][c]
                #print("sampling from: ", ["short", "long"][b], ["neg", "pos"][c])
                #if not pool:                       # empty slice → fall back to any class
                #    pool = self.buckets[b][0] + self.buckets[b][1]

                choice = idx % len(combs)
                comb = combs[choice]
                pool = self.buckets[comb[0]][comb[1]]

                batch = draw(pool, k=self.batch_size) if self.replacement \
                        else self.rng.sample(pool, k=self.batch_size)

                yield batch

        else:
            for i in range(0, len(self.indices), self.batch_size):
                yield self.indices[i : i + self.batch_size]
            return

class MimicDataset(Dataset):
    def __init__(self, mode, ids, labels, data_icu, root_dir, gender_vocab, race_vocab, insurance_vocab, admission_vocab, icu_vocab, dyn_mode=1):
        self.mode = mode
        print("Building dataset for ", self.mode)
        self.ids = ids
        self.labels = labels
        self.data_icu = data_icu

        self.root_path = root_dir + "/data/mimiciv3.1/processed_icu/"
        self.gender_vocab = gender_vocab
        self.race_vocab = race_vocab
        self.insurance_vocab = insurance_vocab
        self.admission_vocab = admission_vocab
        self.icu_vocab = icu_vocab
        self.dyn_mode = dyn_mode

    def process_file(self, file_path):  # Function to process a single file
        """Function to process a single file."""
        file_id = int(file_path.split("/")[-1].split("_")[-1])
        info = {}

        exist_all = os.path.exists(f"{file_path}/dynamic.csv") and os.path.exists(f"{file_path}/diagnoses.csv") and os.path.exists(f"{file_path}/demo.csv")
        if not exist_all:
            raise ValueError

        # Read the CSV files
        dyn = pd.read_csv(f"{file_path}/dynamic.csv", header=0) # inputevents, procedureevents, outputevents, chartevents; datetimeevents, ingredientevents
        stat = pd.read_csv(f"{file_path}/diagnoses.csv", header=0)
        demo = pd.read_csv(f"{file_path}/demo.csv", header=0) # gender, anchor_age, insurance, race; icu_type, admission_type, patientweight; language, marital_status

        # # Replace demographic values based on vocab_data
        demo["gender"].replace(self.gender_vocab, inplace=True)
        demo["race"].replace(self.race_vocab, inplace=True)
        demo["insurance"].replace(self.insurance_vocab, inplace=True)
        demo["admission_type"].replace(self.admission_vocab, inplace=True)
        demo["icu_type"].replace(self.icu_vocab, inplace=True)

        # Store the processed data in a dictionary for this file
        stat = stat.to_numpy().reshape(-1)
        demo = demo[["gender", "race", "insurance", "anchor_age", "admission_type", "icu_type"]].values.reshape(-1)
        demo = np.nan_to_num(demo, nan=0)
        return dyn, stat, demo


    def __len__(self):
        return len(self.ids) # 1000

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

    def get_sample(self, dyn, stat, demo, sample):
        # Load dynamic data and handle missing keys
        meds, chart, out, proc, date, ing = [
            torch.zeros(size=(0, 0))
        ] * 6  # Initialize placeholder tensors for the data keys we expect
        #inputevents, procedureevents, outputevents, chartevents; datetimeevents, ingredientevents
        meds, chart, out, proc, date, ing = self.get_dynamic_data(dyn)

        # Load static data
        stat = torch.tensor(stat).float()

        # Load demographic data (and handle missing values)
        demo = torch.tensor(demo).float()

        # Load label y
        # y = label
        y = self.labels[self.labels["stay_id"] == sample]["icu_death"].iloc[0]

        y = torch.tensor(int(y)).reshape(-1).long()

        assert meds.shape[0] == chart.shape[0] == out.shape[0] == proc.shape[0] == date.shape[0] == ing.shape[0], "number of frames must be equal!! if you see this error, that means your data preprocessing has some issues.."
        outputs = [meds, chart, out, proc, date, ing, stat, demo, y]
        outputs = [i for i in outputs]
        return outputs

    def __getitem__(self, idx):
        id = self.ids[idx]
        sample = id[1]
        path = self.root_path + str(id[0]) + "_" + str(id[1])
        dyn, stat, demo = self.process_file(path)
        outputs = self.get_sample(dyn, stat, demo, sample)
        return outputs

class data_config:
    def __init__(self, data_icu, modalities, min_length=5, max_length=20, train=True):
        self.data_icu = data_icu
        self.modalities = modalities
        self.min_length = min_length
        self.max_length = max_length


def build_datasets(train_hids, val_hids, test_hids, labels, data_config, root_dir):
    race_vocab, gender_vocab, insurance_vocab, admission_vocab, icu_vocab = init_read(root_dir)
    train_dataset_length_info = json.load(open("./data_seqlen_analy/train_length_dict.json", "r"))
    val_dataset_length_info, test_dataset_length_info = build_val_test_length_info_dict(data_config)

    train_hids_new = []
    for id in train_hids:
        newid = str(id[1])
        length = int(train_dataset_length_info[newid][0])
        if data_config.max_length > length >= data_config.min_length:
            train_hids_new.append(id)
    val_hids_new = []
    for id in val_hids:
        newid = str(id[1])
        if newid in val_dataset_length_info:
            val_hids_new.append(id)
    test_hids_new = []
    for id in test_hids:
        newid = str(id[1])
        if newid in test_dataset_length_info:
            test_hids_new.append(id)

    train_dataset = MimicDataset(
        "train",
        train_hids_new,
        labels,
        data_config.data_icu,
        root_dir=root_dir,
        gender_vocab=gender_vocab,
        race_vocab=race_vocab,
        insurance_vocab=insurance_vocab,
        admission_vocab=admission_vocab,
        icu_vocab=icu_vocab,
    )
    val_dataset = MimicDataset(
        "val",
        val_hids_new,
        labels,
        data_config.data_icu,
        root_dir=root_dir,
        gender_vocab=gender_vocab,
        race_vocab=race_vocab,
        insurance_vocab=insurance_vocab,
        admission_vocab=admission_vocab,
        icu_vocab=icu_vocab,
    )
    test_dataset = MimicDataset(
        "test",
        test_hids_new,
        labels,
        data_config.data_icu,
        root_dir=root_dir,
        gender_vocab=gender_vocab,
        race_vocab=race_vocab,
        insurance_vocab=insurance_vocab,
        admission_vocab=admission_vocab,
        icu_vocab=icu_vocab,
    )
    return train_dataset, val_dataset, test_dataset


def pad_to_bucket_max(batch, boundaries=[0], pad_value=0.0, n_temporal=6):
    # ------------------------------------------------------------
    B = len(batch)
    # transpose list‑of‑tuples -> tuple‑of‑lists
    cols = list(zip(*batch))        # len == 9

    # ---------- STEP 1: pad the 6 temporal tensors --------------
    ts_list = cols[:n_temporal]
    T_lengths = [ts.shape[0] for ts in ts_list[0]]      # len per sample via ts₁
    buckets = [bisect.bisect_right(boundaries, L) for L in T_lengths]
    buckets  = torch.tensor(buckets,  dtype=torch.int64)

    max_T = max(T_lengths)
    padded_tensors = []

    for k, ts_col in enumerate(ts_list):
        seqs = [torch.as_tensor(t) for t in ts_col]     # each (T, Dₖ)
        D_k  = seqs[0].shape[1]

        padded = torch.stack([
            F.pad(
                s,                               # tensor
                (0, 0, 0, max_T - s.shape[0]),   # pad on time axis
                value=pad_value
            )
            for s in seqs
        ])                                       # (B, max_T, Dₖ)
        padded_tensors.append(padded)

    # ---------- STEP 2: stack the non‑temporal fields -----------
    def stack_maybe(col):
        first = col[0]
        # scalars / lists → tensor; existing tensors → stack
        if torch.is_tensor(first):
            return torch.stack([torch.as_tensor(x) for x in col])
        else:
            return torch.as_tensor(col)

    rest = [stack_maybe(c) for c in cols[n_temporal:]]

    # ---------- STEP 3: build the padding / attention mask ------
    attention_mask = torch.zeros(B, max_T, dtype=torch.int64)
    for i, L in enumerate(T_lengths):
        attention_mask[i, L:] = 1   # 1 = padding

    attention_mask = attention_mask.to(torch.bool)
    # ---------- RETURN in original order + mask -----------------
    return (*padded_tensors, *rest, attention_mask, buckets)


def build_val_test_length_info_dict(data_config):
    val_dataset_length_info = open("./data_seqlen_analy/val_seqlen.txt", "r").readlines()
    tmp = {}
    for line in val_dataset_length_info:
        line = line.strip()
        id = line.split(" ")[0]
        length = int(line.split(" ")[1])
        if data_config.max_length > length >= data_config.min_length:
            seqlen = int(line.split(" ")[1])
            tmp[id] = seqlen
    val_dataset_length_info = tmp
    test_dataset_length_info = open("./data_seqlen_analy/test_seqlen.txt", "r").readlines()
    tmp = {}
    for line in test_dataset_length_info:
        line = line.strip()
        id = line.split(" ")[0]
        length = int(line.split(" ")[1])
        if data_config.max_length >= length > data_config.min_length:
            seqlen = int(line.split(" ")[1])
            tmp[id] = seqlen
    test_dataset_length_info = tmp
    return val_dataset_length_info, test_dataset_length_info

def build_dataloaders(train_dataset, val_dataset, test_dataset, data_config, train_ids, val_ids, test_ids):
    train_dataset_length_info = json.load(open("./data_seqlen_analy/train_length_dict.json", "r"))
    val_dataset_length_info, test_dataset_length_info = build_val_test_length_info_dict(data_config)

    train_seq_lengths = []
    val_seq_lengths = []
    test_seq_lengths = []
    train_labels = []
    for id in train_ids:
        id = str(id[1])
        length = int(train_dataset_length_info[id][0])
        if data_config.max_length > length >= data_config.min_length:
            train_seq_lengths.append(int(train_dataset_length_info[id][0]))
            train_labels.append(int(train_dataset_length_info[id][1]))

    for id in val_ids:
        id = str(id[1])
        if id in val_dataset_length_info:
            val_seq_lengths.append(int(val_dataset_length_info[id]))

    for id in test_ids:
        id = str(id[1])
        if id in test_dataset_length_info:
            test_seq_lengths.append(int(test_dataset_length_info[id]))

    train_batch_sampler = LengthAwareBalancedBatchSampler(train_seq_lengths, train_labels, [cfg.long_short_threshold], cfg.batch_size, mode="train")
    val_batch_sampler = LengthAwareBalancedBatchSampler(val_seq_lengths, [], [], cfg.test_batch_size, mode="val")
    test_batch_sampler = LengthAwareBalancedBatchSampler(test_seq_lengths, [], [], cfg.test_batch_size, mode="test")

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=lambda b: pad_to_bucket_max(b, boundaries=[cfg.long_short_threshold]),
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_batch_sampler,
        collate_fn=pad_to_bucket_max,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_batch_sampler,
        collate_fn=pad_to_bucket_max,
        num_workers=cfg.num_workers,
    )
    return train_loader, val_loader, test_loader


def load_trainval_data():
    labels = pd.read_csv(cfg.root_dir +  "/data/mimiciv3.1/processed_icu/icu_death_labels.csv", header=0)
    stay_id = labels.iloc[:, 2]
    death_label = labels.iloc[:, 3]
    print("Total Samples", len(stay_id))
    print("Positive Samples", death_label.sum())
    print("Negative Samples", len(death_label) - death_label.sum())

    # oversampling for train set
    #if cfg.oversampling:
    #    train_ids_df = pd.read_csv(cfg.root_dir + "/data/mimiciv3.1/train_test_val_split/train_ids.csv", header=0)
    #    sampling_ids = train_ids_df.iloc[:, 1].values
    #    print("============ oversampling for train data ==============")
    #    from imblearn.over_sampling import RandomOverSampler
    #    oversample = RandomOverSampler(sampling_strategy="minority", random_state=getattr(cfg, "random_state", None)) # TODO set random_state to ensure reproducibility if specified in cfg

    #    # Filter the data in stay_id that belongs to the training set based on sampling_ids
    #    train_mask = stay_id.isin(sampling_ids)
    #    train_stay_ids = stay_id[train_mask]
    #    if set(train_stay_ids.values) != set(sampling_ids):
    #        raise ValueError("train id error during oversampling")
    #    death_label_train = death_label[train_mask]

    #    x_train = train_stay_ids.values.reshape(-1, 1) # reshaping into a 2D array (n_samples, 1)
    #    y_train = death_label_train.values
    #    x_resampled, y_resampled = oversample.fit_resample(X_train, y_train)
    #    resampled_ids = X_resampled.flatten()
    #    train_ids_indexed = train_ids_df.set_index(train_ids_df.columns[1])
    #    train_ids_resampled_full = []
    #    for x in resampled_ids:
    #        record = train_ids_indexed.loc[x]
    #        if isinstance(record, pd.DataFrame):
    #            record = record.iloc[0]
    #        train_ids_resampled_full.append([record.tolist()[0], x])
    #    print("Total Samples in train set", len(train_ids_resampled_full))
    #    print("Positive Samples in train set", np.sum(y_resampled))
    #    print("Negative Samples in train set", len(y_resampled) - np.sum(y_resampled))
    #    train_ids = train_ids_resampled_full

    #else:
    #    train_ids = pd.read_csv(cfg.root_dir +  "/data/mimiciv3.1/train_test_val_split/train_ids.csv", header=0)
    #    train_ids = train_ids.iloc[:, :2].values.tolist()

    train_ids = pd.read_csv(cfg.root_dir +  "/data/mimiciv3.1/train_test_val_split/train_ids.csv", header=0)
    train_ids = train_ids.iloc[:, :2].values.tolist()
    test_ids = pd.read_csv(cfg.root_dir +  "/data/mimiciv3.1/train_test_val_split/test_ids.csv", header=0)
    test_ids = test_ids.iloc[:, :2].values.tolist()
    val_ids = pd.read_csv(cfg.root_dir +  "/data/mimiciv3.1/train_test_val_split/val_ids.csv", header=0)
    val_ids = val_ids.iloc[:, :2].values.tolist()
    return train_ids, val_ids, test_ids, labels
