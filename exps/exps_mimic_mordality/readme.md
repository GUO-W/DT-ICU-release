use https://github.com/healthylaife/MIMIC-IV-Data-Pipeline/blob/main/mainPipeline.ipynb to process mimiciv 2.0, put csv files and dict in ./data
run train.py
result:
	exps	AUROC	AUPRC	Accuracy	Precision	Recall	settings
literatures	MIMIC-IV-Data-Pipeline, LSTM	0.88±0.02	0.87±0.02	0.8±0.02	0.83±0.04	0.75±0.05	
our models	v1.0	0.97±0.01	0.93±0.05	0.96±0.01	0.92±0.02	0.99±0.01	AdamW, bz 128, lr 3e-5, wp 0.1
