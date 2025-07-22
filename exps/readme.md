## exps_mimic_mordality
mimic2.0
follow setting of baseline literature, use data 72h preprocessed by baseline literature
could compare with baseline
[todo] test baseline: with balancing the test data -- if correspond with paper reported result?

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
+ TODO : + self-att (+ weight analy)

## exps_mimic31_itera_sepdyn [todo]
2025.05.20
dataset_util.py 
dynamic -> regression + multi class classdication + binary classification
regression just for partial events
Further: re-ornagize the input events/study the meaning and build up countrafactual, group the events (+ contra anay)

## exps_mimic31_2stage_regression
2025.05.25
vs _itera）precision: from <1 to 90+
## exps_mimic31_2stage_regression_newloss
2025.05.25
+log/loss
## exps_mimic31_2stage_regression_attnviz 
2025.05.25
+ data_sparsity_diff_mode_analy.py （from exps_mimic31_itera_sepdyn/dataset_util.py） analyse sparsity under different mode
+ exps_mimic_mordality/data_analy_sparsity.py compare sparsity of new preprocessing vs previous preprocessing
[TODO] test mordality updates in iterative inference
[TODO] analyze weight metrix (+gcn)

## exps_mimic31_itera
## exps_mimic31_itera_mlp, exps_mimic31_itera_rnn, exps_mimic31_itera_lstm
vs  exps_mimic31_itera, with diff models (set in config).
[todo] + vs transformer decoder

## exps_iii
from exps_mimic31_itera

- mimiciii pipeiline debug:
    - miss icu_death_labels.csv -- add in mimic_iii_split.py
    - bug: test_ids.csv do not correspond with processed_icu/sub_folders -- debug mimic_iii_process.py
        save path {hadm_id}_{icu_id}"  should be subject_id  (prepare + process 'HADM_ID', split stay_id) lin 178, 121
    - bug: empty dynamic.csv in many stays -- debug mimic_iii_process.py
    - still have empty dynamic.csv for some stays after debugging -- reason: in mimic_iii_prepare.py, some stays with 0 dynamic are also kept
        modified mimic_iii_prepare.py, just keep stays with >0 hours of dymanic data in mimic_iii_icu.csv
        so all the output files of prepare, process, split just contain dyn>0h data.
    - change: admission merge move from process to prepare, avoid overwrite mimic_iii_icu.csv -- cause pbs when re-run process - and as a result, mimic_iii_split.py and make_vocab.py are also needed to be changed accordingly.
    - change: modified saving paths
    - [todo] push code iii to github
    samples 60373 (orig raw samples 61522, 1149 ICU stays with no valid VALUENUM events )

- + data_seqlen_gen.py  gen data lens in ./data_seqlen_analy/, for data balancing [todo]
- [todo]  modify path, run pipeline





## todo

keep len>500 data
try focal loss
pred head exp + evaluation

checkpoint save, inference code 
inference metric

code clean, (+ tensorboard)

try sparse nn
