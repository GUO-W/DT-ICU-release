#Create stratified train/val/test splits 

from __future__ import annotations
import sys, math
from pathlib import Path
from datetime import timedelta

import pandas as pd
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────

def load_static(root: Path) -> pd.DataFrame:
    """Assemble one row per ICU stay with the static fields we need."""
    icu = pd.read_csv("/cluster/work/scaimed/users/wguo/datasets/mimiciii/mimic_iii_icu.csv", parse_dates=["INTIME", "OUTTIME"])

    # add los (days) and icu_death flag if OUTTIME missing
    icu["los"] = (icu["OUTTIME"] - icu["INTIME"]).dt.total_seconds() / 86400.0

    # ICU death: death occurred before or within 2 h after ICU discharge
    if 'DOD' in icu.columns:
        icu['DOD'] = pd.to_datetime(icu['DOD'], errors='coerce')
        delta_sec = (icu['DOD'] - icu['OUTTIME']).dt.total_seconds()
        icu['icu_death'] = ((~icu['DOD'].isna()) & (delta_sec <= 7200)).astype(int)
    else:
        icu['icu_death'] = 0
    
    # drop duplicate column names & reset index
    icu = icu.loc[:, ~icu.columns.duplicated()]
    icu = icu.reset_index(drop=True)
    
    # Check available columns and rename accordingly
    print("Available columns:", icu.columns.tolist())
    
    # Basic renaming for common columns
    rename_dict = {
        "SUBJECT_ID": "subject_id",
        "ICUSTAY_ID": "stay_id",
        "GENDER": "gender",
        "AGE": "age",
        "ETHNICITY": "race",
    }
    
    # Check if FIRST_CAREUNIT exists, otherwise use a default or skip
    if "FIRST_CAREUNIT" in icu.columns:
        rename_dict["FIRST_CAREUNIT"] = "icu_type"
    else:
        # Create a default icu_type if the column doesn't exist
        icu["icu_type"] = "Unknown"
    
    icu.rename(columns=rename_dict, inplace=True)
    
    # Ensure required columns exist
    required_cols = ['subject_id', 'stay_id', 'gender', 'age', 'race', 'icu_type', 'los', 'icu_death']
    missing_cols = [col for col in required_cols if col not in icu.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        # Fill missing columns with defaults
        for col in missing_cols:
            if col == 'gender':
                icu[col] = 'Unknown'
            elif col == 'race':
                icu[col] = 'Unknown'
            elif col == 'icu_type':
                icu[col] = 'Unknown'
            elif col in ['age', 'los']:
                icu[col] = 0
    
    # Save labels separately (only the required columns for labels)
    labels_df = icu[['subject_id', 'stay_id', 'icu_death']].copy()
    labels_df.to_csv('/cluster/home/guowen/projs/DT_mimic/data/mimiciii1.4/icu_death_labels.csv', index=False)
    
    return icu

# ──────────────────────────────────────────────────────────────────────────

def stratified_split(static: pd.DataFrame):
    # guarantee unique col labels + clean index
    static = static.loc[:, ~static.columns.duplicated()].copy()
    static.reset_index(drop=True, inplace=True)
    
    # Check if required columns exist
    required_cols = ['age', 'los', 'gender', 'race', 'icu_type', 'icu_death']
    for col in required_cols:
        if col not in static.columns:
            raise ValueError(f"Required column '{col}' not found in static DataFrame")
    
    # Handle missing values
    static = static.dropna(subset=['age', 'los', 'gender', 'race', 'icu_type'])
    
    if len(static) == 0:
        raise ValueError("No valid data remaining after dropping NaN values")
    
    # Create bins for continuous variables
    try:
        static["age_bin"] = pd.qcut(static["age"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates='drop')
        static["los_bin"] = pd.qcut(static["los"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates='drop')
    except Exception as e:
        print(f"Warning: Error in creating quantile bins: {e}")
        # Fallback to simple binning
        static["age_bin"] = pd.cut(static["age"], bins=4, labels=["Q1", "Q2", "Q3", "Q4"])
        static["los_bin"] = pd.cut(static["los"], bins=4, labels=["Q1", "Q2", "Q3", "Q4"])

    # Create stratification column
    static["stratum"] = (
        static["age_bin"].astype(str) + "_" +
        static["gender"].astype(str) + "_" +
        static["race"].astype(str) + "_" +
        static["icu_type"].astype(str) + "_" +
        static["los_bin"].astype(str) + "_" +
        static["icu_death"].astype(str)
    )

    # drop strata with <2 rows
    counts = static["stratum"].value_counts()
    common = static[~static["stratum"].isin(counts[counts < 2].index)]
    rare   = static.loc[static.index.difference(common.index)]  # backup

    print(f"Total samples: {len(static)}")
    print(f"Common strata samples: {len(common)}")
    print(f"Rare strata samples: {len(rare)}")

    if len(common) < 10:
        raise ValueError("Not enough samples for stratified split")

    # 80‑20 common split
    train_val, test = train_test_split(common, test_size=0.2, stratify=common["stratum"], random_state=42)

    # further split train_val 87.5‑12.5 → overall 70‑10
    train_part, val = train_test_split(train_val, test_size=0.125, stratify=train_val["stratum"], random_state=42)

    # put rare cases into train
    train = pd.concat([train_part, rare])

    return train, val, test

# ──────────────────────────────────────────────────────────────────────────

def write_ids(df: pd.DataFrame, fname: Path):
    df[["subject_id", "stay_id"]].to_csv(fname, index=False)

# ──────────────────────────────────────────────────────────────────────────

def summary(df: pd.DataFrame, fname: Path):
    with open(fname, "w") as f:
        f.write(f"N_stays: {len(df)}\n")
        f.write(f"Unique patients: {df['subject_id'].nunique()}\n")
        f.write(f"ICU death rate: {df['icu_death'].mean():.3f}\n")
        
        # Check if columns exist before creating summaries
        if 'gender' in df.columns:
            f.write("Gender:\n" + df["gender"].value_counts(normalize=True).to_string() + "\n\n")
        if 'race' in df.columns:
            f.write("Race:\n" + df["race"].value_counts(normalize=True).to_string() + "\n\n")
        if 'icu_type' in df.columns:
            f.write("ICU type:\n" + df["icu_type"].value_counts(normalize=True).to_string() + "\n\n")

# ──────────────────────────────────────────────────────────────────────────

def main(root_dir: str):
    root = Path(root_dir)
    static = load_static(root)

    train, val, test = stratified_split(static)

    out_dir = root / "../train_test_val_split"
    out_dir.mkdir(exist_ok=True)

    write_ids(train, out_dir / "train_ids.csv")
    write_ids(val,   out_dir / "val_ids.csv")
    write_ids(test,  out_dir / "test_ids.csv")

    summary(train, out_dir / "train_summary.txt")
    summary(val,   out_dir / "val_summary.txt")
    summary(test,  out_dir / "test_summary.txt")

    print("Splits + summaries written to", out_dir)

# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mimic_data_dir = '/cluster/work/scaimed/users/wguo/datasets/mimiciii/1.4/'
    main(mimic_data_dir)



# #Create stratified train/val/test splits 

# from __future__ import annotations
# import sys, math
# from pathlib import Path
# from datetime import timedelta

# import pandas as pd
# from sklearn.model_selection import train_test_split

# # ──────────────────────────────────────────────────────────────────────────

# def load_static(root: Path) -> pd.DataFrame:
#     """Assemble one row per ICU stay with the static fields we need."""
#     icu = pd.read_csv(root / "../mimic_iii_icu.csv", parse_dates=["INTIME", "OUTTIME"])

#     # add los (days) and icu_death flag if OUTTIME missing
#     icu["los"] = (icu["OUTTIME"] - icu["INTIME"]).dt.total_seconds() / 86400.0

#     # ICU death: death occurred before or within 2 h after ICU discharge
#     if 'DOD' in icu.columns:
#         icu['DOD'] = pd.to_datetime(icu['DOD'], errors='coerce')
#         delta_sec = (icu['DOD'] - icu['OUTTIME']).dt.total_seconds()
#         icu['icu_death'] = ((~icu['DOD'].isna()) & (delta_sec <= 7200)).astype(int)
#     else:
#         icu['icu_death'] = 0
    
#     # drop duplicate column names & reset index
#     icu = icu.loc[:, ~icu.columns.duplicated()]
#     icu = icu.reset_index(drop=True)
    
#     # rename for convenience
#     icu.rename(columns={
#         "SUBJECT_ID": "subject_id",
#         "ICUSTAY_ID": "stay_id",
#         "GENDER": "gender",
#         "AGE": "age",  # already set
#         "ETHNICITY": "race",
#         "FIRST_CAREUNIT": "icu_type",
#     }, inplace=True)
    
#     # Select only the desired columns
#     selected_columns = icu[['subject_id', 'stay_id', 'icu_death']]

#     # Save to a new CSV file
#     selected_columns.to_csv('/cluster/home/guowen/projs/DT_mimic/data/mimiciii1.4/icu_death_labels.csv', index=False)
#     return icu

# # ──────────────────────────────────────────────────────────────────────────

# def stratified_split(static: pd.DataFrame):
#     # guarantee unique col labels + clean index
#     static = static.loc[:, ~static.columns.duplicated()].copy()
#     static.reset_index(drop=True, inplace=True)
    
#     static["age_bin"] = pd.qcut(static["age"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
#     static["los_bin"] = pd.qcut(static["los"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

#     static["stratum"] = (
#         static["age_bin"].astype(str) + "_" +
#         static["gender"].astype(str) + "_" +
#         static["race"].astype(str) + "_" +
#         static["icu_type"].astype(str) + "_" +
#         static["los_bin"].astype(str) + "_" +
#         static["icu_death"].astype(str)
#     )

#     # drop strata with <2 rows
#     counts = static["stratum"].value_counts()
#     common = static[~static["stratum"].isin(counts[counts < 2].index)]
#     rare   = static.loc[static.index.difference(common.index)]  # backup

#     # 80‑20 common split
#     train_val, test = train_test_split(common, test_size=0.2, stratify=common["stratum"], random_state=42)

#     # further split train_val 87.5‑12.5 → overall 70‑10
#     train_part, val = train_test_split(train_val, test_size=0.125, stratify=train_val["stratum"], random_state=42)

#     # put rare cases into train
#     train = pd.concat([train_part, rare])

#     return train, val, test

# # ──────────────────────────────────────────────────────────────────────────

# def write_ids(df: pd.DataFrame, fname: Path):
#     df[["subject_id", "stay_id"]].to_csv(fname, index=False)

# # ──────────────────────────────────────────────────────────────────────────

# def summary(df: pd.DataFrame, fname: Path):
#     with open(fname, "w") as f:
#         f.write(f"N_stays: {len(df)}\n")
#         f.write(f"Unique patients: {df['subject_id'].nunique()}\n")
#         f.write(f"ICU death rate: {df['icu_death'].mean():.3f}\n")
#         f.write("Gender:\n" + df["gender"].value_counts(normalize=True).to_string() + "\n\n")
#         f.write("Race:\n"   + df["race"].value_counts(normalize=True).to_string()   + "\n\n")
#         f.write("ICU type:\n"+ df["icu_type"].value_counts(normalize=True).to_string()+ "\n\n")

# # ──────────────────────────────────────────────────────────────────────────

# def main(root_dir: str):
#     root = Path(root_dir)
#     static = load_static(root)

#     train, val, test = stratified_split(static)

#     out_dir = root / "train_test_val_split"
#     out_dir.mkdir(exist_ok=True)

#     write_ids(train, out_dir / "train_ids.csv")
#     write_ids(val,   out_dir / "val_ids.csv")
#     write_ids(test,  out_dir / "test_ids.csv")

#     summary(train, out_dir / "train_summary.txt")
#     summary(val,   out_dir / "val_summary.txt")
#     summary(test,  out_dir / "test_summary.txt")

#     print("Splits + summaries written to", out_dir)

# # ──────────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     mimic_data_dir = '/cluster/work/scaimed/users/wguo/datasets/mimiciii/1.4/'
#     main(mimic_data_dir)

