# End‑to‑End MIMIC‑III → Digital‑Twin Pipeline

This guide walks you through **three self‑contained steps** that transform raw MIMIC‑III v1.4 CSVs into the folder structure and train/val/test splits expected by the Digital‑Twin (DT) code‑base.

```
raw MIMIC‑III ↘  mimic_iii_prepare.py
                 ① mimic_iii_events.csv / mimic_iii_icu.csv
                        ↘  mimic_iii_process.py
                           ② processed_icu/<HADM_ID>_<ICUSTAY_ID>/{dynamic,diagnoses,demo}.csv
                                 ↘  split_dataset.py
                                    ③ train_test_val_split/{train,val,test}_ids.csv
```

---

## 0  Prerequisites

| requirement      | tested version(s) | notes                                            |
| ---------------- | ----------------- | ------------------------------------------------ |
| Python           |  ≥ 3.9            |                                                  |
| **polars**       |  0.20 – 1.22      | keep < 1.23 until the new streaming engine lands |
| **pandas**       |  ≥ 2.0            |                                                  |
| **tqdm‑rich**    |  ≥ 0.2            | pretty progress bars                             |
| **scikit‑learn** |  ≥ 1.3            | only needed for the split step                   |

```bash
# conda example
conda create -n mimic-dt python=3.10 -y
conda activate mimic-dt
pip install polars==0.22.12 pandas tqdm rich scikit-learn
```

> **Tip:** set `POLARS_MAX_THREADS=$(nproc)` (Linux/Mac) or `%set POLARS_MAX_THREADS=NUMBER%` (Windows) to saturate all cores.

---

## 1  Prepare long‑format event & ICU files

```bash
python mimic_iii_prepare.py   /path/to/mimiciii/1.4/
```

*Input:* full set of original `*.csv.gz` tables (CHARTEVENTS, LABEVENTS, …).
*Output:* two CSVs next to the raw data

```
/…/mimiciii/1.4/
├─ mimic_iii_events.csv   # ~7 GB long table (ICUSTAY_ID,NAME,CHARTTIME,…)
└─ mimic_iii_icu.csv      # stay‑level demographics & timestamps
```

Run time on a 12‑core machine: **≈ 7 minutes** (streaming, < 5 GB RAM).

---

## 2  Explode to per‑ICU‑stay folders

```bash
python mimic_iii_process.py  /path/to/mimiciii/1.4/
```

*Reads* the two CSVs above and writes one folder per stay:

```
processed_icu/
  211127_502298/
    dynamic.csv      # hourly matrix, fixed 122 columns (global header)
    diagnoses.csv    # 1×N ICD multi‑hot vector (N ≈ 6,500)
    demo.csv         # gender,race,insurance,anchor_age,admission_type,icu_type
  … (≈ 61k folders)
```

Progress is shown with a rich progress bar.  The first run also stores
`dynamic_header.json` (column vocabulary) and `icd_vocab.csv` (ICD vector
mapping) in the root so future re‑runs are deterministic.

---

## 3  Stratified Train / Val / Test split

```bash
python split_dataset.py     /path/to/mimiciii/1.4/
```

*Loads* `mimic_iii_icu.csv` + every folder’s `demo.csv`, derives

* length‑of‑stay (`los`)
* quartile bins for age & LOS
* **ICU‑death flag**: patient’s `DOD` ≤ `OUTTIME + 2 h`

and builds a composite `stratum`:

`ageQ × gender × race × icu_type × losQ × icu_death`

The script drops strata that appear fewer than twice to avoid singleton issues.
then produces a **70 / 10 / 20** split with stratification.

Outputs:

```
train_test_val_split/
  train_ids.csv           # SUBJECT_ID,STAY_ID
  val_ids.csv
  test_ids.csv
  train_summary.txt       # quick descriptive stats
  val_summary.txt
  test_summary.txt
```
---
## 4  Generate categorical **vocab** CSVs

`dataset.py` and `gendata.py` expect five lookup tables. Create them once:

```bash
# from the mimiciii folder (or anywhere)
python make_vocabs.py  ./1.4/
```
---

## 5  Next steps: Digital‑Twin training

Point the DT repo’s `config.py` paths to
`…/mimiciii/1.4/processed_icu` and `…/train_test_val_split/` and run:

```bash
python gendata.py   # builds pickled tensors (optional cache)
python train.py     # starts model training
```
