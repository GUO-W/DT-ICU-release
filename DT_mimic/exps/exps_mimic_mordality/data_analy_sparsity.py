#!/usr/bin/env python
# encoding: utf-8


import pandas as pd
import numpy as np

def sparsity(file_path):

    df = pd.read_csv(file_path, low_memory=False)
    numeric = df.apply(pd.to_numeric, errors="coerce")

    is_zero = np.isclose(numeric.to_numpy(), 0, atol=1e-12)

    zero_count  = is_zero.sum()                             
    total_count = (~np.isnan(numeric.to_numpy())).sum()       

    sparsity = zero_count / total_count if total_count else np.nan


    print(f"nb ow: {df.shape[0]}")
    print(f"nb col: {df.shape[1]}")
    print(f"nb all cells: {total_count}")
    print(f'nb 0 cells: {zero_count}')
    print(f"sparsity (zero rate) = {sparsity:.4%}")


if __name__ == "__main__":

    file_path1 = "/cluster/work/scaimed/users/wguo/datasets/mimiciv/csv/38443909/dynamic.csv" #32587108
    file_path2 = "~/projs/DT_mimic/data/mimiciv3.1/processed_icu/15175429_38443909/dynamic.csv" #12476282_35564830
    sparsity(file_path1)
    sparsity(file_path2)