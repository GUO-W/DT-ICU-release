# import sys
# import pickle
# from pathlib import Path
# import pandas as pd

# # ───────────────────────────────────────────────
# # Paths
# mimic_data_dir = '/cluster/work/scaimed/users/wguo/datasets/mimiciii/1.4/'
# root = Path(mimic_data_dir)
# icu_csv = root / "../mimic_iii_icu.csv"
# voc_dir = root / "../dict"
# voc_dir.mkdir(exist_ok=True)

# # ───────────────────────────────────────────────
# # Columns to extract vocab from
# cols = {
#     "GENDER":         "genderDict.pkl",
#     "ETHNICITY":      "raceDict.pkl",
#     "INSURANCE":      "insuranceDict.pkl",
#     "ADMISSION_TYPE": "admission_typeDict.pkl",
#     "FIRST_CAREUNIT": "icu_typeDict.pkl",
# }

# # ───────────────────────────────────────────────
# # Load ICU data
# try:
#     icu = pd.read_csv(icu_csv)
# except Exception as e:
#     print(f"[ERROR] Failed to load ICU CSV: {icu_csv}")
#     sys.exit(1)

# # If FIRST_CAREUNIT missing, merge from ICUSTAYS
# if "FIRST_CAREUNIT" not in icu.columns:
#     print("[INFO] 'FIRST_CAREUNIT' not found in mimic_iii_icu.csv, attempting to load from ICUSTAYS.csv.gz...")
#     icustays_path = root / "ICUSTAYS.csv.gz"
#     try:
#         care_df = pd.read_csv(icustays_path, usecols=["ICUSTAY_ID", "FIRST_CAREUNIT"])
#         icu = pd.merge(icu, care_df, on="ICUSTAY_ID", how="left")
#         print("[INFO] Successfully merged FIRST_CAREUNIT into ICU dataframe.")
#     except Exception as e:
#         print(f"[ERROR] Failed to load or merge FIRST_CAREUNIT: {e}")
#         sys.exit(1)

# # ───────────────────────────────────────────────
# # Filter available columns
# available_cols = {k: v for k, v in cols.items() if k in icu.columns}
# missing_cols = [k for k in cols if k not in icu.columns]

# if missing_cols:
#     print(f"[WARN] These columns are missing and will be skipped: {missing_cols}")

# # ───────────────────────────────────────────────
# # Generate and save vocab dicts as .pkl
# for col, out_fname in available_cols.items():
#     print(f"Processing column: {col}")
#     tokens = (
#         icu[col].astype(str)
#            .str.strip()
#            .str.upper()
#            .replace({"": pd.NA})
#            .dropna()
#            .unique()
#            .tolist()
#     )
#     tokens.sort()
#     token_dict = {token: idx for idx, token in enumerate(tokens)}

#     out_path = voc_dir / out_fname
#     with open(out_path, 'wb') as f:
#         pickle.dump(token_dict, f)

#     print(f"✓ Saved {len(tokens)} entries to {out_fname}")

# # ───────────────────────────────────────────────
# print("\nAll vocabularies saved to:", voc_dir.resolve())


import sys
from pathlib import Path
import pandas as pd

# ────────────── Path setup ──────────────
mimic_data_dir = '/cluster/work/scaimed/users/wguo/datasets/mimiciii/1.4/'
root = Path(mimic_data_dir)
icu_csv = root / "../mimic_iii_icu.csv"
voc_dir = root / "../dict"
voc_dir.mkdir(exist_ok=True)

# Columns to extract vocab from
cols = {
    "GENDER":         "gender.csv",
    "ETHNICITY":      "race.csv",
    "INSURANCE":      "insurance.csv",
    "ADMISSION_TYPE": "admission_type.csv",
    "FIRST_CAREUNIT": "icu_type.csv",
}

# ────────────── Load ICU file ──────────────
try:
    icu = pd.read_csv(icu_csv)
except Exception as e:
    print(f"[ERROR] Failed to load ICU CSV: {icu_csv}")
    sys.exit(1)

# Merge FIRST_CAREUNIT if missing
if "FIRST_CAREUNIT" not in icu.columns:
    print("[INFO] 'FIRST_CAREUNIT' not found in mimic_iii_icu.csv, attempting to load from ICUSTAYS.csv.gz...")
    try:
        icustays_path = root / "ICUSTAYS.csv.gz"
        care_df = pd.read_csv(icustays_path, usecols=["ICUSTAY_ID", "FIRST_CAREUNIT"])
        icu = pd.merge(icu, care_df, on="ICUSTAY_ID", how="left")
        print("[INFO] Successfully merged FIRST_CAREUNIT into ICU dataframe.")
    except Exception as e:
        print(f"[ERROR] Failed to load or merge FIRST_CAREUNIT: {e}")
        sys.exit(1)

# ────────────── Filter and generate vocab ──────────────
available_cols = {k: v for k, v in cols.items() if k in icu.columns}
missing_cols = [k for k in cols if k not in icu.columns]

if missing_cols:
    print(f"[WARN] These columns are missing and will be skipped: {missing_cols}")

for col, out_fname in available_cols.items():
    print(f"Processing column: {col}")
    tokens = sorted(set(
        str(x).strip().upper()
        for x in icu[col].dropna()
        if str(x).strip() != ""
    ))
    out_path = voc_dir / out_fname
    pd.DataFrame({"value": tokens, "index": range(len(tokens))}).to_csv(out_path, index=False)
    print(f"Saved {len(tokens)} entries to {out_fname}")

print("All vocab CSVs saved to:", voc_dir.resolve())

