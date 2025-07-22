#!/usr/bin/env python
"""mimic_iii_process.py – v7
Convert long‑format CSVs (mimic_iii_events/icu) to the folder layout required
by the Digital‑Twin repo.

processed_icu/<HADM_ID>_<ICUSTAY_ID>/
  dynamic.csv     # hourly matrix with fixed header across stays
  diagnoses.csv   # 1×V multi‑hot ICD‑9 vector (V ≈ 6k)
  demo.csv        # demographics / socio‑economic static vars
"""
from __future__ import annotations
import sys, json, re
from pathlib import Path
from typing import List
import shutil

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

pl.enable_string_cache()

# ═════════════════════ helper expressions ═══════════════════════════════════
SLUG_RE = re.compile(r"[^a-z0-9]+")
slug = lambda s: SLUG_RE.sub("_", s.lower()).strip("_")

_as_int = lambda e: (
    e.cast(pl.Float64, strict=False)  # "2345.0" → 2345.0
     .round(0)
     .cast(pl.Int64, strict=False)
)

def mod_slug_expr() -> pl.Expr:
    """Polars Expr → '<modality>_<slug(NAME)>'"""
    return (
        pl.when(pl.col("TABLE").str.contains("input")).then(pl.lit("inputevents"))
          .when(pl.col("TABLE") == "output").then(pl.lit("outputevents"))
          .when(pl.col("TABLE") == "lab").then(pl.lit("chartevents"))
          .otherwise(pl.lit("chartevents"))
        + pl.lit("_")
        + pl.col("NAME").str.replace_all(r"[^a-zA-Z0-9]+", "_").str.to_lowercase()
    )

# ═════════════════════ header builder ═══════════════════════════════════════

def build_header(events_csv: Path, header_file: Path) -> List[str]:
    print("[header] discovering dynamic columns …")
    cols = (
        pl.scan_csv(events_csv, infer_schema_length=0)
          .with_columns(mod_slug_expr().alias("col"))
          .select("col").unique()
          .collect(streaming=True)["col"].to_list()
    )
    cols = sorted(cols)
    header_file.write_text(json.dumps(cols))
    print(f"[header] {len(cols)} columns written → {header_file.name}")
    return cols

# ═════════════════════ ICD helpers ═════════════════════════════════════════=

def build_icd_lookup(diag_csv: Path) -> tuple[pl.DataFrame, int]:
    diag = pl.read_csv(diag_csv, columns=["HADM_ID", "ICD9_CODE"], infer_schema_length=0)
    diag = diag.rename({"ICD9_CODE": "ICD_CODE"}).with_columns(_as_int(pl.col("HADM_ID")).alias("HADM_ID"))
    vc   = diag.to_pandas()["ICD_CODE"].value_counts()
    vocab = {c: i for i, c in enumerate(vc.index)}
    vocab_df = pl.DataFrame({"ICD_CODE": list(vocab), "index": list(vocab.values())})
    diag = diag.join(vocab_df, on="ICD_CODE", how="left")
    return diag, len(vocab)

def one_hot_icd(diag_tbl: pl.DataFrame, hadm_id: int, vocab_size: int) -> np.ndarray:
    idx = (
        diag_tbl.filter(pl.col("HADM_ID") == hadm_id)["index"]
                .drop_nulls()                       # ← removes None
                .cast(pl.UInt32, strict=False)      # ← ensures ints
                .unique()                           # ← remove duplicates
                .to_list()
    )
    vec = np.zeros(vocab_size, dtype="int8")
    if idx:                                         # idx is now pure int list
        vec[idx] = 1
    return vec


# ═════════════════════ pivot helper ═════════════════════════════════════════

def pivot_stay(events: pl.DataFrame, intime: pd.Timestamp, header: List[str]) -> pd.DataFrame:
    if events.is_empty():
        return pd.DataFrame(columns=header)
    df = events.with_columns(
        ((pl.col("CHARTTIME") - pl.lit(intime)).dt.total_seconds() // 3600).cast(pl.Int32).alias("hour")
    )
    wide = (
        df.group_by(["hour", "col"]).agg(pl.col("VALUENUM").last())
          .pivot(values="VALUENUM", index="hour", on="col")
          .to_pandas()
          .reindex(columns=header)
    )
    return wide

# ═════════════════════ main =================================================

def main(data_dir: str):
    root = Path(data_dir)
    events_csv = root / "mimic_iii_events.csv"
    icu_csv    = root / "mimic_iii_icu.csv"
    if not (events_csv.exists() and icu_csv.exists()):
        sys.exit("[err] run mimic_iii_prepare.py first – missing long CSVs")

    # header
    header_file = root / "dynamic_header.json"
    header = json.loads(header_file.read_text()) if header_file.exists() else build_header(events_csv, header_file)

    # ICD lookup
    diag_tbl, vocab_size = build_icd_lookup(root / "DIAGNOSES_ICD.csv.gz")

    # ICU + ADMISSIONS merge
    icu_df = pl.read_csv(icu_csv, infer_schema_length=0).with_columns(
        _as_int(pl.col("ICUSTAY_ID")).alias("ICUSTAY_ID"),
        _as_int(pl.col("HADM_ID")).alias("HADM_ID")
    )
    
    # --- ADMISSIONS extras --------------------------------
    adm_cols = ["HADM_ID", "ETHNICITY", "LANGUAGE", "MARITAL_STATUS", "ADMISSION_TYPE", "INSURANCE"]
    icu_df = icu_df.drop(adm_cols[1:])

    admissions_df = (
        pl.read_csv(root / "ADMISSIONS.csv.gz",
                    columns=adm_cols, infer_schema_length=0).with_columns(_as_int(pl.col("HADM_ID")).alias("HADM_ID"))
    )
    
    icu_df = icu_df.join(admissions_df, on="HADM_ID", how="left")

    # ─ pull care-unit if missing ─
    if "FIRST_CAREUNIT" not in icu_df.columns:
        care_cols = ["ICUSTAY_ID", "FIRST_CAREUNIT", "LAST_CAREUNIT"]
        icu_care = pl.read_csv(root / "ICUSTAYS.csv.gz", columns=care_cols, infer_schema_length=0)
        icu_care = icu_care.with_columns(
            _as_int(pl.col("ICUSTAY_ID")).alias("ICUSTAY_ID")
        )
        icu_df = icu_df.join(icu_care, on="ICUSTAY_ID", how="left")
    
    # cast IDs & parse times
    icu_df = icu_df.with_columns(
        _as_int(pl.col("ICUSTAY_ID")).alias("ICUSTAY_ID"),
        _as_int(pl.col("HADM_ID")).alias("HADM_ID"),
        pl.col("INTIME").str.strptime(pl.Datetime),
    )

    # load all events once
    events_mem = (
        pl.scan_csv(events_csv, infer_schema_length=0)
          .with_columns(
              mod_slug_expr().alias("col"),
              _as_int(pl.col("ICUSTAY_ID")).alias("ICUSTAY_ID"),
              _as_int(pl.col("HADM_ID")).alias("HADM_ID"),
              pl.col("CHARTTIME").str.strptime(pl.Datetime).alias("CHARTTIME"),
              pl.col("VALUENUM").fill_null(0)
          )
          .collect(streaming=True)
    )

    empty_value = events_mem.filter(pl.col("VALUENUM").is_null())
    print(f"[debug] {len(empty_value)} rows with null VALUENUM")

    # overwrite the ICU metadata CSV with the new columns
    icu_df.write_csv(icu_csv)
    
    out_root = root / "processed_icu"
    out_root.mkdir(exist_ok=True)
    required = {"dynamic.csv", "diagnoses.csv", "demo.csv"}
    
    written = 0
    skipped_num = 0
    for row in tqdm(icu_df.iter_rows(named=True), total=len(icu_df), desc="ICU stays"):
        icu_id, hadm_id = int(row["ICUSTAY_ID"]), int(row["HADM_ID"])
        folder = out_root / f"{hadm_id}_{icu_id}"
        #folder = out_root / f"{icu_id}"

        dyn = pivot_stay(events_mem.filter(pl.col("ICUSTAY_ID") == icu_id), row["INTIME"], header)
        if dyn.empty:
            print(f"[warn] Empty dynamic for ICUSTAY_ID={icu_id}, HADM_ID={hadm_id}")

        # Remove folder if it exists
        if folder.exists():
            shutil.rmtree(folder)
        
        # Skip if empty
        if dyn.empty:
            skipped_num += 1
            print("current skipped number:", skipped_num)
            continue

        # Recreate the folder
        folder.mkdir(parents=True, exist_ok=True)

        # dynamic
        dyn.to_csv(folder / "dynamic.csv", index=False)

        # diagnoses
        vec = one_hot_icd(diag_tbl, hadm_id, vocab_size)
        pd.DataFrame(vec.reshape(1, -1)).to_csv(folder / "diagnoses.csv", index=False)

        # demo
        demo = pd.DataFrame({
            "gender":         [row.get("GENDER", "")],
            "race":           [row.get("ETHNICITY", "")],
            "language":       [row.get("LANGUAGE", "")],
            "marital_status": [row.get("MARITAL_STATUS", "")],
            "insurance":      [row.get("INSURANCE", "")],
            "anchor_age":     [row.get("AGE", np.nan)],
            "admission_type": [row.get("ADMISSION_TYPE", "")],
            "icu_type":       [row.get("FIRST_CAREUNIT", "")],
        })
        demo.to_csv(folder / "demo.csv", index=False)
        written += 1
    
    print(f"[✓] wrote/updated {written} stay folders → {out_root}")
    print(f"Skipped {skipped_num} folders")

# ═════════════════════ entry point ══════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python mimic_iii_process.py <mimic_iii_dir>")
    main(sys.argv[1])
