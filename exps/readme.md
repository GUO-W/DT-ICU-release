## exps_mimic_mordality
mimic2.0
follow setting of baseline literature, use data 72h preprocessed by baseline literature
could compare with baseline

## exps_mimic31
mimic 3.1, preprocessed by yanke's code
data all loaded in the beginning
data normalized
oversampling strategy problematic
do not work (preprocessed data debuged: nan -> bad res)

## exps_mimic31_debug
data load from pkl preprocessed from the csv files, by gendata.py -- for selection
data normalized
no oversampling (oversampling code not work)
+ data_seqlen_ana, log_ana
debug used to debug exps_mimic31: test 72h w/wo pred (multi-task), >=72h (multi-input len): work, but low precision/recall
exps seen in log
gendata_balance.py : gen 1:1 data to debug:precision/recall yes! -- should work on oversampling


## exps_mimic31_balancedsampling
no pre-load data -- no need of high mem
(todo: no data norm)
reduce train time 10h-30min/epoch (todo: check nb_workers) 
balance sampler: 0-1 / long-short (todo: threshold check) -- check 4 losses
lr / warm-up
model: + causal transformer mask (todo: abla)
pb: low precision
+ diff classification thres (todo: analyse)
bce + f1

## exps_mimic31_balancedsampling_bceloss:
loss bce (todo: abla loss)
oversampling 1:1 -> 2:1 (todo: abla test)
+ log_analyze_diff_thres.py
test diff lr

## exps_mimic31_itera
config:  train_max_length = 480, lr1e-5, long-short thres = 72/48
save model
+ pred + evaluation
slides

## todo
keep len>500 data
try focal loss
pred head exp + evaluation

checkpoint save, inference code 
inference metric

code clean, (+ tensorboard)

try sparse nn
