use https://github.com/healthylaife/MIMIC-IV-Data-Pipeline/blob/main/mainPipeline.ipynb to process mimiciv 2.0, put csv files and dict in ./data
run train.py to get our models v1.0

result (mimic iv 2.0):
	exps	                                AUROC	    AUPRC	    Accuracy	Precision	Recall	    settings
literatures2022 reported	(0.12)		    0.87		0.49
literatures2022, LSTM	    (0.50)          0.88±0.02	0.87±0.02	0.8±0.02	0.83±0.04	0.75±0.05	
our models	v1.0	        (0.50)          0.97±0.01	0.93±0.05	0.96±0.01	0.92±0.02	0.99±0.01	AdamW, bz 128, lr 3e-5, wp 0.1
our model v1.0              (0.12)          


- python ~/projs/mimic_baseline_reproduce/main_baseline.py 2>&1 | tee log_baseline_lstm.txt : baseline (LSTM).  test also balanced
- python ~/projs/mimic_baseline/main.py 2>&1 | tee log_ours_transformerv1.0.txt       : our model v1.0. test also balanced
-  test no balancing: run baseline, to see if correspond with literature reported result


REPOS:
- ~/projs/mimic_baseline: orig used for table above, res for baseline and our models. but modified and not able to run baseline now
- ~/projs/mimic_baseline_reproduce: cp from ~/projs/mimic_baseline to reproduced baseline
- ~/projs/mimic_baseline3.1: modify for data version 3.1 could be ignored and replaced by ./dt_mimic
