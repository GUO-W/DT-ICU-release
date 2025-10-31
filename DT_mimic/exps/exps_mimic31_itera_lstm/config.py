from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
from easydict import EasyDict as edict


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


C = edict()
cfg = C

""" ROOT_dir"""
C.abs_dir = os.getcwd()  # /cluster/home/guowen/projs/DT_mimic
C.this_dir = C.abs_dir.split(osp.sep)[-1]  # exps_mimic_mordality
C.repo_name = "DT_mimic"  #
C.root_dir = C.abs_dir[
    : C.abs_dir.index(C.repo_name) + len(C.repo_name)
]  # /cluster/home/guowen/projs/DT_mimic
### create out dirs
C.out_dir = osp.abspath(
    osp.join(C.root_dir, "out", C.this_dir)
)  #'/cluster/home/guowen/projs/DT_mimic/out/exps_mimic_mordality'

create_folder(C.out_dir)
C.log_dir = osp.join(C.out_dir, "log")
C.snapshot_dir = osp.join(C.out_dir, "snapshot")
C.tensorboard_dir = osp.join(C.out_dir, "tensorboard")
create_folder(C.log_dir)
create_folder(C.snapshot_dir)
create_folder(C.out_dir)

add_path(osp.join(C.root_dir, "lib"))

""" Device Config"""
C.device = "cuda:0"

""" Data Config"""
C.data_icu = True

C.proc_flag = True
C.out_flag = True
C.chart_flag = True
C.med_flag = True
C.date_flag = True
C.ing_flag = True
C.oversampling = False
C.long_short_threshold = 48

""" Train and Test Config"""
C.seed = 42
C.k_fold = int(5)

C.batch_size = 16
C.test_batch_size = 16 # 128
C.num_epochs = 20
C.patience = 5
C.num_workers = 20

C.train_min_length = 4 # 72
C.train_max_length = 480

C.lrn_rate = 5e-7 #1e-5
C.warmup_epochs = 2
C.grad_accum_steps = 16

C.test_input_min_length = 4 # 12
C.test_pos_threshold = 0.9
C.test_max_pred_length_ratio = 2.0

## todo
C.test_size = 0.2
C.val_size = 0.1

""" Model Config"""
C.model = edict()

C.model.type = "lstm"
C.model.pred = True
C.model.pred_loss = 0.5
C.model.posemb = True # False

C.model.embedding_size = 64
C.model.latent_size = 256

## todo
C.model.rnnLayers = 2
C.model.rnn_size = 256

