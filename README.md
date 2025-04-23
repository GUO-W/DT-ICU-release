
## Directory
ROOT
|-- data
    |-- mimiciv2.0
        |-- csv
        '-- dict
    |-- mimiciv3.1
        |-- 3.1 # orig raw data
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

## Data Preparation
for 2.0: Please follow the data preprocessing pipeline from [MIMIC-IV-Data-Pipeline]{https://github.com/healthylaife/MIMIC-IV-Data-Pipeline.git} and put the generated csv files in ./data/mimiciv2.0. (72h mordality extraction)
for 3.1: please follow the data preprocessing pipeline from [mimic4_preprocess]https://github.com/14110951D0/mimic4_preprocess and put the generated csv files in ./data/mimiciv2.0.

## Train and test
Go to the experiment folder in ./exps, 
bash run.sh
note: checkpoints are not saved by default

# log vis
on leomed: use log_analyze.py, or log_analyze_diff_thres.py for diff classification thresholds.
on other servers: setting up tensorboard is suggested.

