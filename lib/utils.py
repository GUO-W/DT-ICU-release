import random
import numpy as np
import torch
import pickle
from typing import Optional
from tqdm import tqdm


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return



def create_vocab(root_dir, file):
    with open(root_dir + "/data/dict/" + file, "rb") as fp:
        condVocab = pickle.load(fp)
    condVocabDict = {}
    condVocabDict[0] = 0
    for val in range(len(condVocab)):
        condVocabDict[condVocab[val]] = val + 1
    return condVocabDict


def gender_vocab():
    genderVocabDict = {}
    genderVocabDict["<PAD>"] = 0
    genderVocabDict["M"] = 1
    genderVocabDict["F"] = 2
    return genderVocabDict


def init_vocab(
    diag_flag, proc_flag, out_flag, chart_flag, med_flag, lab_flag, root_dir, is_train=True
):
    condVocabDict = {}
    procVocabDict = {}
    medVocabDict = {}
    outVocabDict = {}
    chartVocabDict = {}
    labVocabDict = {}
    ethVocabDict = {}
    ageVocabDict = {}
    genderVocabDict = {}
    insVocabDict = {}

    if is_train:
        mode = "wb"
    else:
        mode = "rb"

    ethVocabDict = create_vocab(root_dir, "ethVocab")
    with open(root_dir + "/data/dict/" + "ethVocabDict", mode) as fp:
        pickle.dump(ethVocabDict, fp)

    ageVocabDict = create_vocab(root_dir, "ageVocab")
    with open(root_dir + "/data/dict/" + "ageVocabDict", mode) as fp:
        pickle.dump(ageVocabDict, fp)

    genderVocabDict = gender_vocab()
    with open(root_dir + "/data/dict/" + "genderVocabDict", mode) as fp:
        pickle.dump(genderVocabDict, fp)

    insVocabDict = create_vocab(root_dir, "insVocab")
    with open(root_dir + "/data/dict/" + "insVocabDict", mode) as fp:
        pickle.dump(insVocabDict, fp)

    if diag_flag:
        file = "condVocab"
        with open(root_dir + "/data/dict/" + file, "rb") as fp:
            condVocabDict = pickle.load(fp)
    if proc_flag:
        file = "procVocab"
        with open(root_dir + "/data/dict/" + file, "rb") as fp:
            procVocabDict = pickle.load(fp)
    if med_flag:
        file = "medVocab"
        with open(root_dir + "/data/dict/" + file, "rb") as fp:
            medVocabDict = pickle.load(fp)
    if out_flag:
        file = "outVocab"
        with open(root_dir + "/data/dict/" + file, "rb") as fp:
            outVocabDict = pickle.load(fp)
    if chart_flag:
        file = "chartVocab"
        with open(root_dir + "/data/dict/" + file, "rb") as fp:
            chartVocabDict = pickle.load(fp)
    if lab_flag:
        file = "labsVocab"
        with open(root_dir + "/data/dict/" + file, "rb") as fp:
            labVocabDict = pickle.load(fp)

    return (
        len(condVocabDict),
        len(procVocabDict),
        len(medVocabDict),
        len(outVocabDict),
        len(chartVocabDict),
        len(labVocabDict),
        ethVocabDict,
        genderVocabDict,
        ageVocabDict,
        insVocabDict,
    )


class OnlineChannelStats:
    """
    Efficient online computation of per-channel mean and variance for time series data
    using a batch update approach with PyTorch tensors on CUDA.

    Each time series sample is expected to be a 2D tensor with shape (frames, channels)
    and already on the CUDA device.

    Attributes:
        n (int): Total number of frames processed.
        mean (torch.Tensor): Running mean for each channel.
        M2 (torch.Tensor): Running sum of squared differences for each channel.
        sample_variance (bool): Whether to compute sample variance (n-1) or population variance (n).
    """
    def __init__(self, channels: int, sample_variance: bool = True, device: torch.device = torch.device("cuda")) -> None:
        self.n: int = 0
        self.mean: torch.Tensor = torch.zeros(channels, dtype=torch.float32, device=device)
        self.M2: torch.Tensor = torch.zeros(channels, dtype=torch.float32, device=device)
        self.sample_variance: bool = sample_variance
        self.device = device

    def update(self, x: torch.Tensor) -> None:
        """
        Update the statistics using a batch of frames from a time series sample.

        Args:
            x (torch.Tensor): A 2D tensor with shape (frames, channels) on the CUDA device.
        """
        # Get number of frames in the batch.
        n_batch = x.shape[0]
        if n_batch == 0:
            return

        # Compute the batch mean and M2 in a vectorized manner.
        batch_mean = torch.mean(x, dim=0)
        batch_M2 = torch.sum((x - batch_mean) ** 2, dim=0)

        new_n = self.n + n_batch
        if self.n == 0:
            # First batch: initialize aggregates.
            self.mean = batch_mean
            self.M2 = batch_M2
            self.n = n_batch
        else:
            delta = batch_mean - self.mean
            new_mean = (self.n * self.mean + n_batch * batch_mean) / new_n
            new_M2 = self.M2 + batch_M2 + (delta ** 2) * self.n * n_batch / new_n
            self.mean = new_mean
            self.M2 = new_M2
            self.n = new_n

    def get_mean(self) -> torch.Tensor:
        """
        Get the per-channel mean.

        Returns:
            torch.Tensor: The running mean for each channel, or None if no frames have been processed.
        """
        return self.mean if self.n > 0 else None

    def get_variance(self) -> torch.Tensor:
        """
        Compute and return the per-channel variance.

        Returns:
            torch.Tensor: The per-channel variance (sample or population based on initialization),
                          or None if less than two frames have been processed.
        """
        if self.n < 2:
            return None
        if self.sample_variance:
            return self.M2 / (self.n - 1)
        else:
            return self.M2 / self.n

    def __str__(self) -> str:
        mean = self.get_mean()
        variance = self.get_variance()
        return f"Frames processed: {self.n}\nMean: {mean}\nVariance: {variance}"

def get_data_mean_std_new(
    train_dataset,
    val_dataset,
    keys=["meds", "chart", "out", "proc", "date", "ing", "stat"],
):
    outputs = [[] for i in range(len(keys))]

    num_frames = 0
    example = train_dataset[0]
    channels = [example[i].shape[-1] for i in range(len(keys))]
    statics = [OnlineChannelStats(channels=channels[i], sample_variance=True, device=torch.device("cuda")) for i in range(len(keys))]

    for i in tqdm(range(len(train_dataset))):
        batch = train_dataset[i]
        for j in range(len(keys)):
            statics[j].update(batch[j])

    if val_dataset:
        for i in tqdm(range(len(val_dataset))):
            batch = val_dataset[i]
            for j in range(len(keys)):
                statics[j].update(batch[j])

    means = {keys[i]: statics[i].get_mean().unsqueeze(0).unsqueeze(0) for i in range(len(statics))} # shape: 1, 1, channels
    stds = {keys[i]: statics[i].get_variance().unsqueeze(0).unsqueeze(0) for i in range(len(statics))} # shape: 1, 1, channels
    del statics
    torch.cuda.empty_cache()
    print("Calculated mean and variance..")
    #print(means)
    #print(stds)
    return means, stds


def get_data_mean_std(
    train_dataset,
    val_dataset,
    keys=["meds", "chart", "out", "proc", "date", "ing", "stat"],
):
    outputs = [[] for i in range(len(keys))]

    num_frames = 0
    example = train_dataset[0]
    channels = [example[i].shape[-1] for i in range(len(keys))]
    statics = [OnlineChannelStats(channels=channels[i], sample_variance=True, device=torch.device("cuda")) for i in range(len(keys))]

    sample_keys = list(train_dataset.data.keys())
    for sample_key in sample_keys:
        batch = train_dataset.get_sample(sample_key)
        for j in range(len(keys)):
            statics[j].update(batch[j])

    if val_dataset:
        sample_keys = list(val_dataset.data.keys())
        for sample_key in sample_keys:
            batch = val_dataset.get_sample(sample_key)
            for j in range(len(keys)):
                statics[j].update(batch[j])

    means = {keys[i]: statics[i].get_mean().unsqueeze(0).unsqueeze(0) for i in range(len(statics))} # shape: 1, 1, channels
    stds = {keys[i]: statics[i].get_variance().unsqueeze(0).unsqueeze(0) for i in range(len(statics))} # shape: 1, 1, channels
    del statics
    torch.cuda.empty_cache()
    print("Calculated mean and variance..")
    #print(means)
    #print(stds)
    return means, stds


def init_read(root_dir):

    with open(root_dir + "/data/mimiciv3.1/dict/" + "raceDict", "rb") as fp:
        ethVocabDict = pickle.load(fp)

    with open(root_dir + "/data/mimiciv3.1/dict/" + "genderDict", "rb") as fp:
        genderVocabDict = pickle.load(fp)

    with open(root_dir + "/data/mimiciv3.1/dict/" + "insuranceDict", "rb") as fp:
        insVocabDict = pickle.load(fp)

    with open(root_dir + "/data/mimiciv3.1/dict/" + "admission_typeDict", "rb") as fp:
        admVocabDict = pickle.load(fp)

    with open(root_dir + "/data/mimiciv3.1/dict/" + "icu_typeDict", "rb") as fp:
        icuVocabDict = pickle.load(fp)

    return (
        ethVocabDict,
        genderVocabDict,
        insVocabDict,
        admVocabDict,
        icuVocabDict,
    )
