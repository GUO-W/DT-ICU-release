## Directory

```text
ROOT
|-- data
    |-- mimiciv2.0
        |-- csv
        '-- dict
    |-- mimiciii1.4
        |-- 1.4             # original raw data
        |-- processed_icu
        |-- dict
        '-- train_test_val_split
            |-- test_ids.csv
            |-- train_ids.csv
            '-- val_ids.csv
    |-- mimiciv3.1
        |-- 3.1              # original raw data
        |-- processed_icu
        |-- dict
        '-- train_test_val_split
            |-- test_ids.csv
            |-- train_ids.csv
            '-- val_ids.csv
|-- exps
    |-- exps_mimic31_balancedsampling_bceloss

|-- lib
'-- out
    |-- checkpoints
    |-- logs
    '-- tensorboard
```

---

## Data Preparation

This project has been validated on three ICU datasets:

- **MIMIC-III v1.4**
- **MIMIC-IV v2.0**
- **MIMIC-IV v3.1**

You may use any data extraction pipeline, as long as the final outputs follow the same directory structure and data format shown above. For convenience, we suggest two open-source pipelines:

### MIMIC-IV v2.0

Follow the preprocessing pipeline from:  
https://github.com/healthylaife/MIMIC-IV-Data-Pipeline  

(72-hour mortality extraction)

Place the generated CSV files in:
```
./data/mimiciv2.0/csv
```

---

### MIMIC-IV v3.1

Follow the preprocessing pipeline from:  
https://github.com/14110951D0/mimic4_preprocess  

Place the generated outputs in:
```
./data/mimiciv3.1
```

---

### Using Other Pipelines

You may also use your own data processing pipeline, provided that the extracted data follows the same format and directory structure.


## Training and Testing

The experiment scripts and configurations are therefore mainly maintained for MIMIC-IV v3.1.

Go to the experiment folder:

```bash
cd ./exps/exps_mimic31_itera_pred_debug
```

Run:

```bash
bash run.sh
```

---

## Log Visualization

- Run `log_analyze.py` for standard visualization  
- Run `log_analyze_diff_thres.py` to visualize results under different classification thresholds
