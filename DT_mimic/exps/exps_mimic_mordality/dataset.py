import random
import time
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from multiprocessing import Pool, cpu_count

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from config import cfg
from utils import init_vocab


# Function to load the data using multiprocessing
def load_data(vocab_data, ids):
    ids = list(set(ids))
    file_paths = [(cfg.root_dir + f"/data/csv/{f}", int(f)) for f in ids]
    print("Loading all data using multiprocessing...")
    info = {}
    # Create a pool of workers for multiprocessing
    nb_preprocess = cpu_count()
    with Pool(nb_preprocess) as pool:
        results = list(
            pool.starmap(
                process_file,
                [(f, file_id, vocab_data) for f, file_id in file_paths],
            ),
        )
    # Collect the results into the info dictionary, same as in the original function
    for file_id, file_info in results:
        info[file_id] = file_info
    return info


def process_file(file_path, file_id, vocab_data):  # Function to process a single file
    """Function to process a single file."""
    info = {}

    # Read the CSV files
    dyn = pd.read_csv(f"{file_path}/dynamic.csv", header=[0, 1])
    stat = pd.read_csv(f"{file_path}/static.csv", header=[0, 1])["COND"]
    demo = pd.read_csv(f"{file_path}/demo.csv", header=0)

    # Replace demographic values based on vocab_data
    demo["gender"].replace(vocab_data["gender"], inplace=True)
    demo["ethnicity"].replace(vocab_data["ethnicity"], inplace=True)
    demo["insurance"].replace(vocab_data["insurance"], inplace=True)
    demo["Age"].replace(vocab_data["age"], inplace=True)

    # Store the processed data in a dictionary for this file
    info["dynamic"] = dyn
    info["static"] = stat.to_numpy().reshape(-1)
    info["demo"] = demo[["gender", "ethnicity", "insurance", "Age"]].values.reshape(-1)

    return file_id, info


class MimicDataset(Dataset):
    def __init__(self, mode, ids, labels, vocab_data, data_icu):
        self.mode = mode
        print("Building dataset for ", self.mode)
        self.ids = ids
        self.labels = labels
        self.vocab_data = vocab_data
        self.data_icu = data_icu

        t0 = time.time()
        self.data = load_data(self.vocab_data, self.ids)
        t1 = time.time()
        print(f"data loaded by multi-processing, used {t1-t0} s.")
        assert len(set(self.ids)) == len(self.data)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample = self.ids[idx]

        # Load dynamic data and handle missing keys
        meds, chart, out, proc, lab = [
            torch.zeros(size=(0, 0))
        ] * 5  # Initialize placeholder tensors for the data keys we expect
        dyn = self.data[sample]["dynamic"]
        keys = dyn.columns.levels[0]
        for key in keys:
            if key == "MEDS":
                meds = (
                    torch.tensor(dyn[key].to_numpy()).float()
                    if key in dyn
                    else torch.zeros(size=(1, 0))
                )
            if key == "CHART":
                chart = (
                    torch.tensor(dyn[key].to_numpy()).float()
                    if key in dyn
                    else torch.zeros(size=(1, 0))
                )
            if key == "OUT":
                out = (
                    torch.tensor(dyn[key].to_numpy()).float()
                    if key in dyn
                    else torch.zeros(size=(1, 0))
                )
            if key == "PROC":
                proc = (
                    torch.tensor(dyn[key].to_numpy()).float()
                    if key in dyn
                    else torch.zeros(size=(1, 0))
                )
            if key == "LAB":
                lab = (
                    torch.tensor(dyn[key].to_numpy()).float()
                    if key in dyn
                    else torch.zeros(size=(1, 0))
                )

        # Load static data
        stat = self.data[sample]["static"]
        stat = torch.tensor(stat).float()

        # Load demographic data (and handle missing values)
        demo = self.data[sample]["demo"]
        demo = torch.tensor(demo).float()

        # Load label y
        if self.data_icu:
            y = self.labels[self.labels["stay_id"] == sample]["label"]
        else:
            y = self.labels[self.labels["hadm_id"] == sample]["label"]
        y = torch.tensor(int(y)).reshape(-1).long()

        outputs = [meds, chart, out, proc, lab, stat, demo, y]
        outputs = [i.cuda() for i in outputs]
        return outputs


class data_config:
    def __init__(self, data_icu, modalities, train=True):
        self.data_icu = data_icu
        self.modalities = modalities
        (
            self.cond_vocab_size,
            self.proc_vocab_size,
            self.med_vocab_size,
            self.out_vocab_size,
            self.chart_vocab_size,
            self.lab_vocab_size,
            self.eth_vocab,
            self.gender_vocab,
            self.age_vocab,
            self.ins_vocab,
        ) = init_vocab(
            cfg.diag_flag,
            cfg.proc_flag,
            cfg.out_flag,
            cfg.chart_flag,
            cfg.med_flag,
            cfg.lab_flag,
            cfg.root_dir,
            is_train=train,
        )
        (
            self.eth_vocab_size,
            self.gender_vocab_size,
            self.age_vocab_size,
            self.ins_vocab_size,
        ) = (
            len(self.eth_vocab),
            len(self.gender_vocab),
            len(self.age_vocab),
            len(self.ins_vocab),
        )


def build_dataloaders(train_hids, val_hids, test_hids, labels, data_config):
    train_dataset = MimicDataset(
        "train",
        train_hids,
        labels,
        {
            "gender": data_config.gender_vocab,
            "ethnicity": data_config.eth_vocab,
            "insurance": data_config.ins_vocab,
            "age": data_config.age_vocab,
        },
        data_config.data_icu,
    )
    val_dataset = MimicDataset(
        "val",
        val_hids,
        labels,
        {
            "gender": data_config.gender_vocab,
            "ethnicity": data_config.eth_vocab,
            "insurance": data_config.ins_vocab,
            "age": data_config.age_vocab,
        },
        data_config.data_icu,
    )
    test_dataset = MimicDataset(
        "test",
        test_hids,
        labels,
        {
            "gender": data_config.gender_vocab,
            "ethnicity": data_config.eth_vocab,
            "insurance": data_config.ins_vocab,
            "age": data_config.age_vocab,
        },
        data_config.data_icu,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader


def create_kfolds():
    labels = pd.read_csv(cfg.root_dir +  "/data/csv/labels.csv", header=0)
    if cfg.k_fold == 0:
        k_fold = 5
        cfg.k_fold = 1
    else:
        k_fold = cfg.k_fold
    hids = labels.iloc[:, 0]
    y = labels.iloc[:, 1]
    print("Total Samples", len(hids))
    print("Positive Samples", y.sum())
    if cfg.oversampling:
        print("=============OVERSAMPLING===============")
        oversample = RandomOverSampler(sampling_strategy="minority")
        hids = np.asarray(hids).reshape(-1, 1)
        hids, y = oversample.fit_resample(hids, y)
        hids = hids[:, 0]
        print("Total Samples", len(hids))
        print("Positive Samples", y.sum())

    ids = range(0, len(hids))
    batch_size = int(len(ids) / k_fold)
    k_hids = []
    for i in range(0, k_fold):
        rids = random.sample(ids, batch_size)
        ids = list(set(ids) - set(rids))
        if i == 0:
            k_hids.append(hids[rids])
        else:
            k_hids.append(hids[rids])
    return k_hids, labels


def split_train_val_test(k_hids, index):
    test_hids = list(k_hids[index])
    train_ids = list(set(range(cfg.k_fold)) - set([index]))
    train_hids = [hid for j in train_ids for hid in k_hids[j]]

    np.random.shuffle(train_hids)
    val_num = int(len(train_hids) * 0.1)
    val_hids = train_hids[-val_num:]
    train_hids = train_hids[:-val_num]
    return train_hids, val_hids, test_hids
