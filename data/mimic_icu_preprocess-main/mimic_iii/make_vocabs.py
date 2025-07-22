import sys
from pathlib import Path
import pandas as pd

if len(sys.argv) != 2:
    sys.exit("Usage: python make_vocabs.py <mimic_iii_dir>")

root = Path(sys.argv[1]).resolve()          # e.g. C:\…\mimiciii\1.4
icu_csv = root / "mimic_iii_icu.csv"
voc_dir = root / "vocabs"                   # write next to the ICU file
voc_dir.mkdir(exist_ok=True)

cols = {
    "GENDER":         "gender.csv",
    "ETHNICITY":      "race.csv",
    "INSURANCE":      "insurance.csv",
    "ADMISSION_TYPE": "admission_type.csv",
    "FIRST_CAREUNIT": "icu_type.csv",
}
icu = pd.read_csv(icu_csv, usecols=cols)

for col, out in cols.items():
    tokens = (
        icu[col].astype(str).str.strip().str.upper()
           .replace({"": pd.NA}).dropna().unique().tolist()
    )
    tokens.sort()
    pd.DataFrame({"value": tokens, "index": range(len(tokens))}) \
        .to_csv(voc_dir / out, index=False)
    print(f"✓ {out:<18} {len(tokens)} entries")

print("\nVocabularies saved to", voc_dir)