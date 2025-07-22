import os
import gzip
import csv
import tempfile
import shutil
import hashlib
from io import StringIO
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import polars as pl
import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# ---------------------------
# Logging Configuration
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------
# Configuration (using Path)
# ---------------------------
BASE_DIR = Path(os.getcwd()) / "mimic-iv-3.1"
HOSP_DIR = BASE_DIR / "hosp"
ICU_DIR = BASE_DIR / "icu"
PARQUET_DIR = Path(os.getcwd()) / "icu_parquet"
OUTPUT_DIR = Path(os.getcwd()) / "processed_icu"

PARQUET_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------
# Utility: Load CSV.gz using Polars with UTF-8 error ignoring
# ---------------------------
def load_csv_gz(filepath: Path, infer_length: int = 10000) -> Optional[pd.DataFrame]:
    try:
        with gzip.open(filepath, "rb") as f:
            text = f.read().decode("utf-8", errors="ignore")
        buffer = StringIO(text)
        df = pl.read_csv(buffer, infer_schema_length=infer_length, ignore_errors=True)
        logging.info(f"Loaded {filepath.name} with shape {df.shape}")
        return df.to_pandas()
    except Exception as e:
        logging.error(f"Error loading {filepath}: {e}")
        return None
    
# =============================================================================
# Step A: Build Dictionaries from Reference Tables
# =============================================================================
# 1. Build the d_items dictionary (from ICU module) using composite keys "itemid_linksto".
ditems_path = ICU_DIR / "d_items.csv.gz"
ditems_df = pd.read_csv(ditems_path)
# We assume ditems_df contains columns: 'itemid', 'label', 'abbreviation', 'linksto', 'category',
# 'unitname', 'param_type', 'lownormalvalue', 'highnormalvalue'
# Build a dictionary mapping composite key -> {"label": ..., "param_type": ...}
items_dict: Dict[str, Dict[str, str]] = {
    f"{str(row['itemid']).strip()}_{str(row['linksto']).strip()}": {
        "label": str(row["label"]).strip(), 
        "param_type": str(row["param_type"]).strip()}
    for _, row in ditems_df.iterrows()}

logging.info(f"Built items dictionary with {len(items_dict)} entries from d_items.")

# Create an auxiliary mapping from itemid to composite key.
itemid_map: Dict[str, str] = {
    str(row["itemid"]).strip(): f"{str(row['itemid']).strip()}_{str(row['linksto']).strip()}"
    for _, row in ditems_df.iterrows()}

'''
grouped_counts = ditems_df.groupby("linksto")["param_type"].value_counts()
print(grouped_counts)

linksto           param_type      
chartevents       Text                1908
                  Numeric              732
                  Checkbox             372
                  Numeric with tag      43
datetimeevents    Date and time        190
ingredientevents  Ingredient           124
inputevents       Solution             476
outputevents      Numeric               72
                  Text                   3
                  Date and time          1
                  Ingredient             1
procedureevents   Processes            173
Name: count, dtype: int64
'''

# 2. Build the d_icd_diagnoses dictionary (from Hosp module) for diagnoses.
# --- Helper: Convert ICD codes using a mapping file ---
def convert_diagnoses(df: pd.DataFrame, mapping_path: Path) -> pd.DataFrame:
    """
    Convert ICD-9 codes in the diagnoses DataFrame to ICD-10 codes using a mapping file.
    For rows where df['icd_version']==9, the ICD-9 code (df['icd_code']) is matched against
    the mapping file's 'icd9cm' column. The first matching ICD-10 code is stored in a new column,
    'converted'. A 'root' column is also created with the first three characters of the converted code.
    """
    # Read the mapping file; note that the file may have no extension but is in the 'dict' folder.
    mapping = pd.read_csv(mapping_path, delimiter='\t', header=0)
    #mapping = pd.read_csv(mapping_path, delimiter=',', header=0)
    # Clean up the mapping columns
    mapping['icd9cm'] = mapping['icd9cm'].astype(str).str.strip()
    mapping['icd10cm'] = mapping['icd10cm'].astype(str).str.strip()
    
    # Default: use original ICD code
    df['converted'] = df['icd_code']
    
    # For ICD-9 codes, convert them using the mapping.
    mask = df['icd_version'] == 9
    for code, group in df.loc[mask].groupby('icd_code'):
        code = str(code).strip()
        try:
            new_code = mapping.loc[mapping['icd9cm'] == code, 'icd10cm'].iloc[0]
        except Exception as e:
            logging.warning(f"Mapping not found for ICD-9 code {code}: {e}")
            new_code = code #np.nan
        df.loc[group.index, 'converted'] = new_code

    # Create a 'root' column from the converted ICD-10 code (first three characters)
    df['root'] = df['converted'].apply(lambda x: x[:3] if isinstance(x, str) else np.nan)
    return df

# --- Load diagnoses data and apply ICD conversion ---
diagnoses_icd_path = HOSP_DIR / "diagnoses_icd.csv.gz"
diagnoses_icd_df = pd.read_csv(diagnoses_icd_path, compression="gzip")
logging.info(f"Loaded diagnoses_icd.csv.gz with shape: {diagnoses_icd_df.shape}")

# Convert ICD codes using the mapping file in the 'dict' folder.
icd_mapping_path = Path("ICD9_to_ICD10_mapping.txt") # icd9toicd10cmgem.csv, ICD9_to_ICD10_mapping.txt
diagnoses_icd_df = convert_diagnoses(diagnoses_icd_df, icd_mapping_path)
logging.info("ICD conversion complete. Sample converted codes:")
print(diagnoses_icd_df[['icd_code', 'icd_version', 'converted', 'root']].head())
# After running the conversion (using convert_diagnoses), build a dictionary:
unique_converted = diagnoses_icd_df['converted'].dropna().unique()
unique_roots = diagnoses_icd_df['root'].dropna().unique()


# =============================================================================
# Utility Functions for Parallel Processing (CSV-to-Parquet conversion)
# =============================================================================
NUM_BUCKETS = 1024

def compute_bucket_key_full(subject_id: Any, hadm_id: Any, stay_id: Any, num_buckets: int = NUM_BUCKETS) -> str:
    key_str = f"{subject_id}_{hadm_id}_{stay_id}"
    h = hashlib.md5(key_str.encode("utf-8")).hexdigest()
    return str(int(h, 16) % num_buckets)

def add_bucket_column(table: pa.Table, num_buckets: int = NUM_BUCKETS) -> pa.Table:
    for col in ["subject_id", "hadm_id", "stay_id"]:
        if col not in table.column_names:
            raise ValueError(f"{col} not found in table")
    subj_list = table.column("subject_id").to_pylist()
    hadm_list = table.column("hadm_id").to_pylist()
    stay_list = table.column("stay_id").to_pylist()
    buckets = [compute_bucket_key_full(s, h, t, num_buckets) for s, h, t in zip(subj_list, hadm_list, stay_list)]
    bucket_array = pa.array(buckets, type=pa.string())
    return table.append_column("bucket", bucket_array)

def get_csv_header(file_path: Path) -> list:
    """
    Read the header (first row) from a gzipped CSV file.
    """
    with gzip.open(file_path, "rt") as f:  # text mode
        reader = csv.reader(f)
        header = next(reader)
    return header

def build_convert_options_from_header(file_path: Path, event_filename: str) -> pacsv.ConvertOptions:
    """
    Build ConvertOptions for a CSV file by inspecting its header.
    - For columns "subject_id", "hadm_id", "stay_id", use pa.large_string.
    - For any column whose name contains 'time' or 'date', use pa.timestamp("ms").
    - Additionally, if event_filename is "datetimeevents.csv.gz" and "value" is present, force it to pa.timestamp("ms").
    """
    header = get_csv_header(file_path)
    column_types = {}
    for col in header:
        col_lower = col.lower()
        if col in ["subject_id", "hadm_id", "stay_id"]:
            column_types[col] = pa.large_string()
        elif "time" in col_lower:
            column_types[col] = pa.timestamp("ms")
    if event_filename == "datetimeevents.csv.gz" and "value" in header:
        column_types["value"] = pa.timestamp("ms")
    return pacsv.ConvertOptions(column_types=column_types)

def convert_csv_gz_to_parquet_stream(event_filename: str, icu_dir: Path, parquet_dir: Path, blocksize: int = 10**9) -> Path:
    """
    Convert a large CSV.gz file into partitioned Parquet files by reading it in streaming chunks.
    ConvertOptions are built dynamically from the file header so that all columns that look like dates/times
    are parsed as timestamp[ms]. For "datetimeevents.csv.gz", the 'value' column is also forced to timestamp[ms].
    """
    csv_path = icu_dir / event_filename
    base_name = event_filename.replace(".csv.gz", "")
    out_dir = parquet_dir / base_name
    out_dir.mkdir(exist_ok=True)
    
    # Skip conversion if files already exist.
    if any(fname.endswith((".parquet", ".parq", ".pq")) for fname in os.listdir(out_dir)):
        logging.info(f"Parquet for {event_filename} already exists at {out_dir}.")
        return out_dir

    logging.info(f"Converting {event_filename} to partitioned Parquet at {out_dir}...")

    # Decompress CSV.gz to a temporary file.
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp:
        tmp_filename = Path(tmp.name)
        with gzip.open(csv_path, "rb") as f_in:
            shutil.copyfileobj(f_in, tmp)
    logging.info(f"Uncompression complete for {event_filename}. Starting streaming CSV read...")

    # Build the convert options dynamically based on the file header.
    convert_options = build_convert_options_from_header(csv_path, event_filename)
    read_options = pacsv.ReadOptions(block_size=blocksize)
    parse_options = pacsv.ParseOptions(delimiter=",")
    
    try:
        csv_reader = pa.csv.open_csv(str(tmp_filename),
                                     read_options=read_options,
                                     parse_options=parse_options,
                                     convert_options=convert_options)
    except Exception as e:
        logging.error(f"Error opening CSV with pyarrow for {event_filename}: {e}")
        tmp_filename.unlink()
        raise ValueError(f"Conversion failed for {event_filename}.")

    try:
        for chunk in csv_reader:
            # Add bucket column using your existing function.
            chunk = add_bucket_column(chunk, num_buckets=NUM_BUCKETS)
            bucket_array = chunk.column("bucket").to_pylist()
            unique_buckets = set(bucket_array)
            for bucket in unique_buckets:
                bucket_str = str(bucket)
                bucket_filter = pa.compute.equal(chunk.column("bucket"), bucket_str)
                bucket_table = chunk.filter(bucket_filter)
                # If the bucket table is a RecordBatch, convert to Table.
                if isinstance(bucket_table, pa.RecordBatch):
                    bucket_table = pa.Table.from_batches([bucket_table])
                if bucket_table.num_rows > 0:
                    out_file = out_dir / f"bucket={bucket_str}.parquet"
                    if out_file.exists():
                        existing = pq.read_table(str(out_file))
                        new_table = pa.concat_tables([existing, bucket_table])
                        pq.write_table(new_table, str(out_file), compression="snappy")
                    else:
                        pq.write_table(bucket_table, str(out_file), compression="snappy")
                    logging.info(f"Chunk processed: {bucket_table.num_rows} rows appended to bucket {bucket_str}.")
        logging.info(f"Conversion complete for {event_filename}.")
    except Exception as e:
        logging.error(f"Error processing chunks for {event_filename}: {e}")
        tmp_filename.unlink()
        raise ValueError(f"Conversion failed for {event_filename}.")
    
    tmp_filename.unlink()
    return out_dir

def load_events_for_key(key: Tuple[Any, Any, Any],
                        event_files: List[str],
                        parquet_paths: Dict[str, Path]) -> List[Dict[str, Any]]:
    subj, hadm, stay = map(str, key)
    bucket = compute_bucket_key_full(subj, hadm, stay, num_buckets=NUM_BUCKETS)
    events = []
    for filename in event_files:
        dir_path = parquet_paths.get(filename)
        if dir_path is None:
            raise ValueError(f"Conversion failed for {filename}.")
        bucket_file = dir_path / f"bucket={bucket}.parquet"
        if not bucket_file.exists():
            continue
        try:
            df = pd.read_parquet(bucket_file, engine="pyarrow")
            if not df.empty:
                df = df[(df["subject_id"] == subj) & (df["hadm_id"] == hadm) & (df["stay_id"] == stay)]
                if not df.empty:
                    df["source"] = filename
                    events.extend(df.to_dict(orient="records"))
        except Exception as e:
            logging.error(f"Error reading {bucket_file} for key {key}: {e}")
    return events

# =============================================================================
# Modified build_hourly_timeseries using items_dict and itemid_map for custom aggregation
# =============================================================================
def get_event_time(event: Dict[str, Any]) -> pd.Timestamp:
    """
    Return the event time for aggregation:
      - If 'starttime' exists and is not null, return it.
      - Otherwise, check 'charttime' and 'storetime' in that order.
      - If none are available, return pd.Timestamp.min.
    """
    if "starttime" in event and pd.notnull(event["starttime"]):
        return pd.Timestamp(event["starttime"])
    for t in ["charttime", "storetime"]:
        if t in event and pd.notnull(event[t]):
            return pd.Timestamp(event[t])
    return pd.Timestamp.min

def build_hourly_timeseries(events: List[Dict[str, Any]], icu_intime: Any, icu_outtime: Optional[Any],
                            items_dictionary: Dict[str, Dict[str, str]], itemid_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Build an hourly aggregated timeseries DataFrame.
    Aggregation is based on the event's param_type (from items_dictionary):
      - "Date and time": count events at the datetime value.
      - "Ingredient"/"Solution": sum "amount", if an endtime is present, distribute the event 'amount' across hourly bins 
      according to the fraction of the event's duration that falls into each bin.
      - "Numeric"/"Numeric with tag": average "valuenum".
      - "Checkbox"/"Text": if "valuenum" missing then count; else sum "valuenum".
      - "Processes": mark 1 for each hour between starttime and endtime.
    1. The starting time preserved (up to seconds). The time index is [icu_intime, icu_intime+1h, ...]
    2. Only events with event times within [icu_intime, end_time) are included.
    Ensures that the output contains one column per item in items_dictionary.
    The output column names will be the composite key "itemid_linksto".
    """
    start_time = pd.Timestamp(icu_intime)
    
    # Determine outtime: if icu_outtime is provided, use it; otherwise, use the latest event time.
    if icu_outtime is not None:
        out_time = pd.Timestamp(icu_outtime)
    else:
        event_times = [get_event_time(e) for e in events if get_event_time(e) != pd.Timestamp.min]
        out_time = max(event_times) if event_times else start_time
    
    # Compute total hours needed: smallest integer number such that start_time + n hours >= out_time.
    total_hours = int(np.ceil((out_time - start_time).total_seconds() / 3600.0))
    
    # Create a time index as [start_time, start_time + 1h, ..., start_time + total_hours * 1h]
    time_index = pd.DatetimeIndex([start_time + pd.Timedelta(hours=i) for i in range(total_hours + 1)])
    
    # Initialize DataFrame with one column per item from items_dictionary.
    # Column names are the composite keys.
    col_names = {key: key for key in items_dictionary.keys()}
    ts_df = pd.DataFrame(0.0, index=time_index, columns=col_names.values())
    
    # For numeric average, maintain sums and counts.
    numeric_sums: Dict[str, Dict[pd.Timestamp, float]] = {}
    numeric_counts: Dict[str, Dict[pd.Timestamp, int]] = {}
    
    # Only consider events within the [start_time, time_index[-1]) interval.
    valid_events = []
    for e in events:
        etime = get_event_time(e)
        if start_time <= etime < time_index[-1]:
            valid_events.append(e)
    
    # Process each valid event
    for e in valid_events:
        etime = get_event_time(e)
        # Find the hour bin index: floor relative to start_time.
        # Compute the time difference in hours (as float) then take int
        hour_offset = int(np.floor((etime - start_time).total_seconds() / 3600.0))
        # The actual timestamp in our time_index:
        hour_ts = start_time + pd.Timedelta(hours=hour_offset)
        
        event_itemid = str(e.get("itemid"))
        if event_itemid not in itemid_mapping:
            continue
        full_key = itemid_mapping[event_itemid]
        param_type = items_dictionary[full_key]["param_type"]
        
        if param_type in ["Ingredient", "Solution"]:
            # For these events, distribute the amount over the event duration.
            try:
                total_amount = float(e.get("amount", 0))
            except Exception:
                continue
            # Determine event start and end times
            event_start = pd.Timestamp(e.get("starttime", etime))
            event_end = pd.Timestamp(e.get("endtime", event_start))
            # Clip the event duration to [start_time, time_index[-1])
            event_start = max(event_start, start_time)
            event_end = min(event_end, time_index[-1])
            duration = (event_end - event_start).total_seconds()
            if duration <= 0:
                # If duration is zero, assign full amount to the bin that contains event_start.
                if event_start >= start_time and event_start < time_index[-1]:
                    bin_ts = start_time + pd.Timedelta(hours=int(np.floor((event_start - start_time).total_seconds() / 3600)))
                    ts_df.at[bin_ts, full_key] += total_amount
                continue
            # For each hourly bin, compute the overlap with the event interval.
            for i in range(total_hours):
                bin_start = start_time + pd.Timedelta(hours=i)
                bin_end = start_time + pd.Timedelta(hours=i+1)
                # Compute overlap interval in seconds.
                overlap_start = max(event_start, bin_start)
                overlap_end = min(event_end, bin_end)
                overlap = max((overlap_end - overlap_start).total_seconds(), 0)
                if overlap > 0:
                    fraction = overlap / duration
                    ts_df.at[bin_start, full_key] += total_amount * fraction
        elif param_type == "Date and time":
            etime = min(etime, pd.to_datetime(e.get("value"), errors="coerce"))
            if start_time <= etime < time_index[-1]:
                hour_offset = int(np.floor((etime - start_time).total_seconds() / 3600.0))
                # The actual timestamp in our time_index:
                hour_ts = start_time + pd.Timedelta(hours=hour_offset)
                ts_df.at[hour_ts, full_key] += 1
        
        elif param_type in ["Numeric", "Numeric with tag"]:
            # If the event links to the outputevent table, process its "value" similar to Ingredient/Solution.
            if full_key.lower().endswith("outputevents"):
                try:
                    total_amount = float(e.get("value", 0))
                except Exception:
                    continue
                # Determine event start and end times.
                ts_df.at[hour_ts, full_key] += total_amount
            else:
                # Default processing for Numeric events: compute average from "valuenum".
                try:
                    val = float(e.get("valuenum", np.nan))
                    if not np.isnan(val):
                        numeric_sums.setdefault(full_key, {}).setdefault(hour_ts, 0.0)
                        numeric_counts.setdefault(full_key, {}).setdefault(hour_ts, 0)
                        numeric_sums[full_key][hour_ts] += val
                        numeric_counts[full_key][hour_ts] += 1
                except Exception:
                    continue
        elif param_type in ["Checkbox", "Text"]:
            val = e.get("valuenum") if full_key.lower().endswith("chartevents") else e.get("value")
            if pd.isnull(val):
                ts_df.at[hour_ts, full_key] += 1
            else:
                try:
                    ts_df.at[hour_ts, full_key] += float(val)
                except Exception:
                    ts_df.at[hour_ts, full_key] += 1
        elif param_type == "Processes":
            st = pd.Timestamp(e.get("starttime", etime))
            et = pd.Timestamp(e.get("endtime", st))
            # Distribute a value of 1 to all bins that fall within [st, et)
            for i in range(total_hours):
                bin_start = start_time + pd.Timedelta(hours=i)
                bin_end = start_time + pd.Timedelta(hours=i+1)
                if st < bin_end and et > bin_start:
                    ts_df.at[bin_start, full_key] = 1
        else:
            ts_df.at[hour_ts, full_key] += 1

    # Compute averages for numeric events.
    for key in numeric_sums:
        for t, s in numeric_sums[key].items():
            count = numeric_counts[key][t]
            ts_df.at[t, key] = s / count if count > 0 else np.nan

    return ts_df

# =============================================================================
# ICU death label extraction 
# =============================================================================
def compute_icu_death(row: pd.Series) -> int:
    if pd.isnull(row.get("deathtime")):
        return 0
    elif pd.notnull(row.get("outtime")):
        death_time = pd.to_datetime(row["deathtime"], errors="coerce")
        outtime = pd.to_datetime(row["outtime"], errors="coerce")
        return 1 if death_time <= outtime else 0
    else:
        return 1

def extract_icu_death(icu_static: pd.DataFrame) -> None:
    df = icu_static.copy()
    output_file = OUTPUT_DIR / "icu_death_labels.csv"
    df[["subject_id", "hadm_id", "stay_id", "icu_death"]].to_csv(output_file, index=False)
    logging.info(f"ICU death written to {output_file}")

# =============================================================================
# Modified process_icu_stay to output diagnoses one-hot CSV (hospital info removed)
# =============================================================================
selected_demo = ["icu_type", "gender", "anchor_age", "admission_type", "insurance",
                 "language", "marital_status", "race", "patientweight"]

def process_icu_stay(row: pd.Series, event_files: List[str], output_dir: Path,
                     parquet_paths: Dict[str, Path], write_raw: bool = False) -> str:
    subj, hadm, stay = row["subject_id"], row["hadm_id"], row["stay_id"]
    key = (subj, hadm, stay)
    
    # Create a subdirectory for this ICU stay.
    stay_dir = output_dir / f"{subj}_{stay}"
    stay_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if demo.csv, diagnoses.csv, and dynamic.csv already exist.
    if ((stay_dir / "demo.csv").exists() and 
        (stay_dir / "diagnoses.csv").exists() and 
        (stay_dir / "dynamic.csv").exists()):
        logging.info(f"ICU stay {subj}_{stay} already processed. Skipping.")
        return f"Skipped patient {subj}, stay {stay} (already processed)"
    
    # --- Process Diagnoses ---
    # Filter diagnoses for this subject and admission.
    diag_rows = diagnoses_icd_df[(diagnoses_icd_df["subject_id"] == subj) & (diagnoses_icd_df["hadm_id"] == hadm)]
    # Build one-hot encoding based on the 'root' column.
    onehot = {code: 0 for code in unique_converted}
    rows_converted = diag_rows['converted'].dropna().unique()
    for code in rows_converted:
        onehot[code] = 1
    
    diag_output_path = stay_dir / "diagnoses.csv"
    pd.DataFrame([onehot]).to_csv(diag_output_path, index=False)
    logging.info(f"Diagnoses one-hot encoded info written to {diag_output_path}")
    
    # Load events.
    events = load_events_for_key(key, event_files, parquet_paths)
    events.sort(key=lambda e: get_event_time(e))
    
    # Write static info.
    static_df = pd.DataFrame([row.to_dict()])[selected_demo]
    static_path = stay_dir / "demo.csv"
    static_df.to_csv(static_path, index=False)
    
    # Optionally write raw events.
    if write_raw:
        raw_df = pd.DataFrame(events)
        if not raw_df.empty:
            sort_col = next((col for col in ["charttime", "starttime", "storetime"] if col in raw_df.columns), None)
            if sort_col:
                raw_df.sort_values(by=sort_col, inplace=True)
        raw_path = stay_dir / "events.csv"
        raw_df.to_csv(raw_path, index=False)
    
    # Determine effective outtime: if the patient died in ICU, use death time; otherwise use outtime.
    if pd.notnull(row.get("deathtime")) and pd.notnull(row.get("outtime")): 
        if row["intime"] <= row["deathtime"] <= row["outtime"]:
            effective_outtime = row["deathtime"]
        else:
            effective_outtime = row["outtime"]
    elif pd.isnull(row.get("deathtime")) and pd.isnull(row.get("outtime")):
        if pd.notnull(row["edouttime"]):
            effective_outtime = row["edouttime"]
        else:
            effective_outtime = row["dischtime"]
    elif pd.notnull(row.get("deathtime")) and pd.isnull(row.get("outtime")):
        effective_outtime = row["deathtime"]
    else:
        effective_outtime = row["outtime"]
        
    # Build hourly timeseries using custom aggregation.
    hourly_df = build_hourly_timeseries(events, row["intime"], effective_outtime, items_dict, itemid_map)
    hourly_df.fillna(0, inplace=True)
    hourly_df.reset_index(inplace=True)
    hourly_df.rename(columns={"index": "timestamp"}, inplace=True)
    hourly_path = stay_dir / "dynamic.csv"
    hourly_df.to_csv(hourly_path, index=False)
    
    return f"Processed patient {subj}, stay {stay}"

# =============================================================================
# Main Execution: Convert events, extract static info, process ICU stays in parallel
# =============================================================================
EVENT_FILES = [
    "procedureevents.csv.gz", 
    "inputevents.csv.gz",
    "outputevents.csv.gz", 
    "ingredientevents.csv.gz",
    "datetimeevents.csv.gz",
    "chartevents.csv.gz"
]

parquet_paths: Dict[str, Path] = {}
for filename in EVENT_FILES:
    parquet_paths[filename] = convert_csv_gz_to_parquet_stream(filename, ICU_DIR, PARQUET_DIR)


if __name__ == '__main__':
    
    # =============================================================================
    # Load Static Tables from Hosp and ICU modules
    # =============================================================================
    patients_df = load_csv_gz(HOSP_DIR / "patients.csv.gz", infer_length=10000)
    admissions_df = load_csv_gz(HOSP_DIR / "admissions.csv.gz", infer_length=10000)
    if patients_df is None or admissions_df is None:
        raise ValueError("Patients or Admissions table not loaded.")

    if "dod" in patients_df.columns:
        patients_df["dod"] = pd.to_datetime(patients_df["dod"], errors="coerce")
    patient_static_cols = ["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"]
    patients_static = patients_df[patient_static_cols].copy()

    for col in ["admittime", "dischtime", "edregtime", "edouttime", "deathtime"]:
        if col in admissions_df.columns:
            admissions_df[col] = pd.to_datetime(admissions_df[col], errors="coerce")
    admission_static_cols = ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
                              "admission_type", "admit_provider_id", "admission_location",
                              "discharge_location", "insurance", "language", "marital_status", "race",
                              "edregtime", "edouttime", "hospital_expire_flag"]
    admissions_static = admissions_df[admission_static_cols].copy()

    logging.info("Sample admissions static data:")
    logging.info(admissions_static.head())

    icustays_df = load_csv_gz(ICU_DIR / "icustays.csv.gz", infer_length=10000)
    if icustays_df is None:
        raise ValueError("ICU stays table not loaded.")
    for col in ["intime", "outtime"]:
        if col in icustays_df.columns:
            icustays_df[col] = pd.to_datetime(icustays_df[col], errors="coerce")
    logging.info("Sample icustays data:")
    logging.info(icustays_df.head())

    # =============================================================================
    # Extract Patient Weight (unchanged)
    # =============================================================================
    def extract_weight(filepath: Path) -> Optional[pd.DataFrame]:
        df = load_csv_gz(filepath, infer_length=10000)
        if df is not None and "patientweight" in df.columns:
            df["patientweight"] = pd.to_numeric(df["patientweight"], errors="coerce")
            return df.dropna(subset=["patientweight"]).groupby(["subject_id", "hadm_id", "stay_id"])["patientweight"].median().reset_index()
        return None

    weight_input = extract_weight(ICU_DIR / "inputevents.csv.gz")
    weight_proc = extract_weight(ICU_DIR / "procedureevents.csv.gz")
    if weight_input is not None and weight_proc is not None:
        weight_all = pd.concat([weight_input, weight_proc], ignore_index=True)
    elif weight_input is not None:
        weight_all = weight_input
    elif weight_proc is not None:
        weight_all = weight_proc
    else:
        weight_all = pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "patientweight"])
    if not weight_all.empty:
        weight_all = weight_all.groupby(["subject_id", "hadm_id", "stay_id"])["patientweight"].median().reset_index()
    logging.info("Sample extracted patient weight:")
    logging.info(weight_all.head())

    # =========================================================================
    # Build Static Features per ICU Stay
    # =========================================================================

    static_info = pd.merge(patients_static, admissions_static, on="subject_id", how="left")
    icu_static = pd.merge(icustays_df, static_info, on=["subject_id", "hadm_id"], how="left")
    if not weight_all.empty:
        icu_static = pd.merge(icu_static, weight_all, on=["subject_id", "hadm_id", "stay_id"], how="left")
    if "first_careunit" in icustays_df.columns:
        icu_static.rename(columns={"first_careunit": "icu_type"}, inplace=True)
    else:
        icu_static["icu_type"] = "NA"
        
    icu_static["intime"] = pd.to_datetime(icu_static["intime"], errors="coerce")
    icu_static["outtime"] = pd.to_datetime(icu_static["outtime"], errors="coerce")
    icu_static["icu_death"] = icu_static.apply(compute_icu_death, axis=1)
    icu_static.to_csv(OUTPUT_DIR / "icu_demo.csv", index=False)

    logging.info("Static features per ICU stay (sample):")
    logging.info(icu_static.head())
    extract_icu_death(icu_static)
    
    # =========================================================================
    ## Extraction of Patient csv files
    rows_to_process = [(row, EVENT_FILES, OUTPUT_DIR, parquet_paths) for _, row in icu_static.iterrows()]
    total = len(rows_to_process)
    processed = 0
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() // 2)) as executor:
        futures = [executor.submit(process_icu_stay, row, EVENT_FILES, OUTPUT_DIR, parquet_paths, False)
                   for row, _, _, _ in rows_to_process]
        for future in as_completed(futures):
            try:
                result = future.result()
                processed += 1
                logging.info(f"[{processed}/{total}] {result}")
            except Exception as e:
                logging.error(f"Error processing ICU stay: {e}")
    
    logging.info("ICU stay-level data extraction complete.")

    
    # split of train, val, test
    from sklearn.model_selection import train_test_split
    
    # Generate quantile bins (quartiles) for age and LOS.
    icu_static['age_bin'] = pd.qcut(icu_static['anchor_age'], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    icu_static['los_bin'] = pd.qcut(icu_static['los'], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    
    # Build a composite stratification column.
    icu_static['stratum'] = (
        icu_static['age_bin'].astype(str) + "_" +
        icu_static['gender'].astype(str) + "_" +
        icu_static['race'].astype(str) + "_" +
        icu_static['icu_type'].astype(str) + "_" +
        icu_static['los_bin'].astype(str) + "_" +
        icu_static['icu_death'].astype(str)
    )
    
    # Identify rare strata in the full data.
    stratum_counts = icu_static['stratum'].value_counts()
    rare_strata = stratum_counts[stratum_counts < 2].index.tolist()
    # Remove these rare rows from icu_static.
    rare_rows = icu_static[icu_static['stratum'].isin(rare_strata)]
    common_data = icu_static.drop(rare_rows.index)
    
    # Now perform an 80-20 split on common_data.
    train_val, test = train_test_split(common_data, test_size=0.2, stratify=common_data['stratum'], random_state=42)
    
    # Identify strata in train_val that have at least 2 members.
    rare_train_val = train_val.groupby('stratum').filter(lambda x: len(x) < 2)
    # Identify the rare rows that have only 1 record.
    common_train_val = train_val.drop(rare_train_val.index)
    
    # Now perform the stratified split on the common data.
    train, val = train_test_split(
        common_train_val,
        test_size=0.125,  # so that overall validation becomes 10% of total data
        stratify=common_train_val['stratum'],
        random_state=42)

    # Append the rare rows to the training set (or assign them as you see fit).
    train = pd.concat([train, rare_train_val, rare_rows])
    
    # Extract (subject_id, stay_id) pairs.
    train_ids = train[['subject_id', 'stay_id']]
    val_ids = val[['subject_id', 'stay_id']]
    test_ids = test[['subject_id', 'stay_id']]
    
    # Save the lists to CSV files.
    train_ids.to_csv("train_ids.csv", index=False)
    val_ids.to_csv("val_ids.csv", index=False)
    test_ids.to_csv("test_ids.csv", index=False)
    print("Train IDs saved to train_ids.csv")
    print("Validation IDs saved to val_ids.csv")
    print("Test IDs saved to test_ids.csv")
    
    # Simple statistical summary of each split.
    def generate_summary(df: pd.DataFrame, filename: str) -> None:
        with open(filename, 'w') as f:
            f.write("=== Summary Report ===\n\n")
            
            # Count number of ICU stays and unique patients.
            num_stays = len(df)
            num_patients = df['subject_id'].nunique()
            f.write(f"Number of ICU Stays: {num_stays}\n")
            f.write(f"Number of Unique Patients: {num_patients}\n\n")
            
            # ICU death rate
            if 'icu_death' in df.columns:
                icu_death_rate = df['icu_death'].mean()
                f.write(f"ICU Death Rate: {icu_death_rate:.4f}\n\n")
            
            # Patient weight statistics
            if 'patientweight' in df.columns:
                weight_stats = df['patientweight'].describe()
                f.write("Patient Weight Statistics:\n")
                f.write(f"Mean: {weight_stats['mean']:.2f}\n")
                f.write(f"Std: {weight_stats['std']:.2f}\n")
                f.write(f"Min: {weight_stats['min']:.2f}\n")
                f.write(f"Max: {weight_stats['max']:.2f}\n")
                f.write(f"Median: {df['patientweight'].median():.2f}\n\n")
            
            # Hospital expire flag rate
            if 'hospital_expire_flag' in df.columns:
                hosp_exp_rate = df['hospital_expire_flag'].mean()
                f.write(f"Hospital Expire Flag Rate: {hosp_exp_rate:.4f}\n\n")
            
            # Race proportions
            if 'race' in df.columns:
                f.write("Race Proportions:\n")
                f.write(df['race'].value_counts(normalize=True).to_string())
                f.write("\n\n")
            
            # Marital status proportions
            if 'marital_status' in df.columns:
                f.write("Marital Status Proportions:\n")
                f.write(df['marital_status'].value_counts(normalize=True).to_string())
                f.write("\n\n")
            
            # Language proportions
            if 'language' in df.columns:
                f.write("Language Proportions:\n")
                f.write(df['language'].value_counts(normalize=True).to_string())
                f.write("\n\n")
            
            # Insurance proportions
            if 'insurance' in df.columns:
                f.write("Insurance Proportions:\n")
                f.write(df['insurance'].value_counts(normalize=True).to_string())
                f.write("\n\n")
            
            # Anchor age statistics
            if 'anchor_age' in df.columns:
                age_stats = df['anchor_age'].describe()
                f.write("Anchor Age Statistics:\n")
                f.write(f"Mean: {age_stats['mean']:.2f}\n")
                f.write(f"Std: {age_stats['std']:.2f}\n")
                f.write(f"Min: {age_stats['min']:.2f}\n")
                f.write(f"Max: {age_stats['max']:.2f}\n")
                f.write(f"Median: {df['anchor_age'].median():.2f}\n\n")
            
            # Gender proportions
            if 'gender' in df.columns:
                f.write("Gender Proportions:\n")
                f.write(df['gender'].value_counts(normalize=True).to_string())
                f.write("\n\n")
            
            # LOS statistics
            if 'los' in df.columns:
                los_stats = df['los'].describe()
                f.write("LOS Statistics:\n")
                f.write(f"Mean: {los_stats['mean']:.2f}\n")
                f.write(f"Std: {los_stats['std']:.2f}\n")
                f.write(f"Min: {los_stats['min']:.2f}\n")
                f.write(f"Max: {los_stats['max']:.2f}\n")
                f.write(f"Median: {df['los'].median():.2f}\n\n")
            
            # ICU type proportions
            if 'icu_type' in df.columns:
                f.write("ICU Type Proportions:\n")
                f.write(df['icu_type'].value_counts(normalize=True).to_string())
                f.write("\n\n")
                
            f.write("=== End of Report ===\n")
    
    generate_summary(train, "train_summary.txt")
    generate_summary(val, "val_summary.txt")
    generate_summary(test, "test_summary.txt")
    print("Summary files generated: train_summary.txt, val_summary.txt, test_summary.txt")
    
