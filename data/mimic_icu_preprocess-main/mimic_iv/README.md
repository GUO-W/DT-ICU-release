
# mimic4\_preprocess

Preprocessing pipeline for **MIMIC-IV v3.1** ICU module

We leverage two modules from MIMIC-IV:

* **hosp** (hospitalization tables)
* **icu** (intensive care unit tables)

This pipeline focuses on extracting and structuring ICU data while using HOSP tables for supporting features. For more details on data fields and table descriptions, visit the [official MIMIC-IV documentation](https://physionet.org/content/mimiciv/3.1/) or the [module overview](https://mimic.mit.edu/docs/iv/modules/).

---
## 🚀 Key Improvements Over Previous [SOTA Pipeline](https://github.com/healthylaife/MIMIC-IV-Data-Pipeline)

1. **On‑Demand Parquet Conversion**
   CSV.gz files are converted to bucketed Parquet partitions only when needed, reducing I/O and speeding up repeated runs.

2. **Expanded Event Coverage**
   Adds support for:

   * [datetimeevents](https://mimic.mit.edu/docs/iv/modules/icu/datetimesevents/)
   * [ingredientevents](https://mimic.mit.edu/docs/iv/modules/icu/ingredientevents/)

3. **Parallel ICU‑Stay Processing**
   Utilizes Python’s `concurrent.futures` to process stays concurrently, significantly reducing end‑to‑end runtime.

4. **Skip Already‑Processed Stays**
   Before processing each stay, checks for the presence of `demo.csv`, `diagnoses.csv`, and `dynamic.csv` in `processed_icu/{subject_id}_{stay_id}`. If **all three** exist, the stay is skipped to save compute.

---
## 📦 Installation

1. Place your **decompressed** MIMIC-IV v3.1 data directory alongside this repo.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---
## 🔧 Usage

1. **Run the main pipeline**

   ```bash
   python mimic_iv_process.py
   ```

   This will:

   * Convert ICU CSV.gz files to Parquet in `icu_parquet/`
   * Generate static ICU-stay metadata (`processed_icu/icu_demo.csv`)
   * Create individual stay folders (`processed_icu/{subject_id}_{stay_id}`) containing:

     * `demo.csv` (demographics)
     * `diagnoses.csv` (one‑hot ICD roots)
     * `dynamic.csv` (hourly timeseries)
   * Perform stratified train/val/test split and save ID lists & summaries.

2. **Validate outputs**
   To ensure completeness, run:

   ```bash
   python check.py
   ```

   This will verify that each stay folder contains the required CSVs.

3. **Generate Demo Dictionaries**

   ```bash
   python icu_demo_dict.py
   ```

   Builds lookup tables for demographic features.

---
## 🛠 Pipeline Steps

1. **CSV → Parquet Conversion**

   * Reads each `*.csv.gz` in `icu/`.
   * Inspects headers to cast timestamps and IDs correctly.
   * Partitions data into bucketed Parquet files under `icu_parquet/`.

2. **Static Feature Extraction**

   * Merges `patients`, `admissions`, and `icustays` tables.
   * Computes ICU death labels.
   * Saves `processed_icu/icu_demo.csv` and `processed_icu/icu_death_labels.csv`.

3. **Per‑Stay Data Extraction**

   * Iterates over each ICU stay in `icu_demo.csv`.
   * Creates a folder `processed_icu/{subject_id}_{stay_id}`.
   * **Skip‑if‑Done:** Checks for `demo.csv`, `diagnoses.csv`, and `dynamic.csv` before processing.
   * Writes:

     * `demo.csv`: demographic & static ICU features
     * `diagnoses.csv`: ICD root one‑hot encodings
     * `dynamic.csv`: hourly aggregated event timeseries

4. **Train/Val/Test Split & Summaries**

   * Stratified split on patient demographics, LOS, and outcomes.
   * Saves `train_ids.csv`, `val_ids.csv`, `test_ids.csv`.
   * Generates `*_summary.txt` reports for each split.

---
## 📖 References

* MIMIC‑IV v3.1 Documentation: [https://physionet.org/content/mimiciv/3.1/](https://physionet.org/content/mimiciv/3.1/)
* MIMIC‑IV Module Overview: [https://mimic.mit.edu/docs/iv/modules/](https://mimic.mit.edu/docs/iv/modules/)

---
*Feel free to open an issue for bugs, feature requests, or improvements!*
