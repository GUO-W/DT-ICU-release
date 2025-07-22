import sys
from pathlib import Path
import pandas as pd

# Path setup
mimic_data_dir = '/cluster/work/scaimed/users/wguo/datasets/mimiciii/1.4/'
root = Path(mimic_data_dir)
icu_csv = root / "../mimic_iii_icu.csv"
voc_dir = root / "../vocabs"
voc_dir.mkdir(exist_ok=True)

# Columns to extract vocab from
cols = {
    "GENDER":         "gender.csv",
    "ETHNICITY":      "race.csv",
    "INSURANCE":      "insurance.csv",
    "ADMISSION_TYPE": "admission_type.csv",
    "FIRST_CAREUNIT": "icu_type.csv",
}

# Read ICU file (full, then filter)
try:
    icu = pd.read_csv(icu_csv)
except Exception as e:
    print(f"[ERROR] Failed to load ICU CSV: {icu_csv}")
    sys.exit(1)

# Check and keep only existing columns
available_cols = {k: v for k, v in cols.items() if k in icu.columns}
missing_cols = [k for k in cols if k not in icu.columns]

if missing_cols:
    print(f"[WARN] These columns are missing from {icu_csv.name} and will be skipped: {missing_cols}")

# Generate vocab files
for col, out_fname in available_cols.items():
    print(f"Processing column: {col}")
    tokens = (
        icu[col].astype(str)
           .str.strip()
           .str.upper()
           .replace({"": pd.NA})
           .dropna()
           .unique()
           .tolist()
    )
    tokens.sort()
    out_path = voc_dir / out_fname
    pd.DataFrame({"value": tokens, "index": range(len(tokens))}).to_csv(out_path, index=False)
    print(f"Saved {len(tokens)} entries to {out_fname}")

print("\nVocabulary files saved to:", voc_dir.resolve())


# import sys
# from pathlib import Path
# import pandas as pd

# mimic_data_dir = '/cluster/work/scaimed/users/wguo/datasets/mimiciii/1.4/'
# root = Path(mimic_data_dir)

# icu_csv = root / "../mimic_iii_icu.csv"
# voc_dir = root / "../vocabs"                   # write next to the ICU file
# voc_dir.mkdir(exist_ok=True)

# cols = {
#     "GENDER":         "gender.csv",
#     "ETHNICITY":      "race.csv",
#     "INSURANCE":      "insurance.csv",
#     "ADMISSION_TYPE": "admission_type.csv",
#     "FIRST_CAREUNIT": "icu_type.csv",
# }
# icu = pd.read_csv(icu_csv, usecols=cols)

# for col, out in cols.items():
#     tokens = (
#         icu[col].astype(str).str.strip().str.upper()
#            .replace({"": pd.NA}).dropna().unique().tolist()
#     )
#     tokens.sort()
#     pd.DataFrame({"value": tokens, "index": range(len(tokens))}) \
#         .to_csv(voc_dir / out, index=False)
#     print(f"✓ {out:<18} {len(tokens)} entries")

# print("\nVocabularies saved to", voc_dir)