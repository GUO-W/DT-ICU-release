from pathlib import Path
import pandas as pd
import mimic_iv_process as mivp  # Your processing pipeline module

def check_icu_stay_folders(root_dir: Path, required_files: list) -> list:
    """
    Iterate over ICU stay folders in root_dir and return a list of tuples:
      (folder, missing_files_list)
    for folders that do not contain all required CSV files.
    """
    incomplete = []
    for folder in root_dir.iterdir():
        if folder.is_dir():
            existing_files = {f.name for f in folder.glob("*.csv")}
            missing = [fname for fname in required_files if fname not in existing_files]
            if missing:
                incomplete.append((folder, missing))
    return incomplete

def regenerate_missing_csvs(export_missing: bool = True):
    """
    Loads the ICU demo CSV from the processed_icu folder, checks for incomplete ICU stay folders
    (named as {subject_id}_{stay}), exports the list of incomplete folders with missing files,
    and for each incomplete folder, reprocesses that ICU stay using process_icu_stay.
    """
    root_dir = mivp.OUTPUT_DIR  # Processed ICU stays folder (e.g., "processed_icu")
    required_files = ["diagnoses.csv", "demo.csv", "dynamic.csv"]
    
    incomplete = check_icu_stay_folders(root_dir, required_files)
    if not incomplete:
        print("All ICU stay folders have complete CSV files.")
    else:
        print("Incomplete ICU stay folders:")
        missing_list = []
        for folder, missing in incomplete:
            print(f"{folder.name}: missing {missing}")
            missing_list.append({"folder": folder.name, "missing_files": ";".join(missing)})
        
        # Export the list of incomplete folders to a CSV file.
        if export_missing:
            missing_df = pd.DataFrame(missing_list)
            export_path = "incomplete_folders.csv"
            missing_df.to_csv(export_path, index=False)
            print(f"Exported list of incomplete folders to {export_path}")
    
    # Load the ICU demo CSV (assumed to be in processed_icu/icu_demo.csv)
    icu_demo_file = root_dir / "icu_demo.csv"
    if not icu_demo_file.exists():
        print("icu_demo.csv not found in processed_icu folder.")
        return
    icu_demo = pd.read_csv(icu_demo_file)
    
    # Process each incomplete folder.
    for folder, missing in incomplete:
        # Folder naming format: "{subject_id}_{stay}"
        parts = folder.name.split("_")
        if len(parts) != 2:
            print(f"Folder {folder.name} does not match expected naming (subjectID_stay).")
            continue
        subject_id, stay_id = int(parts[0]), int(parts[1])
        # Look up the corresponding ICU stay row in the ICU demo data.
        row_df = icu_demo[(icu_demo["subject_id"] == subject_id) & (icu_demo["stay_id"] == stay_id)]
        if row_df.empty:
            print(f"No matching ICU stay found for subject {subject_id} stay {stay_id}.")
            continue
        row = row_df.iloc[0]
        try:
            result = mivp.process_icu_stay(row, mivp.EVENT_FILES, mivp.OUTPUT_DIR, mivp.parquet_paths, write_raw=False)
            print(f"Reprocessed ICU stay for subject {subject_id}, stay {stay_id}: {result}")
        except Exception as e:
            print(f"Error reprocessing subject {subject_id}, stay {stay_id}: {e}")

if __name__ == "__main__":
    regenerate_missing_csvs(export_missing=True)
