#!/usr/bin/env python
# encoding: utf-8
# Performance of ablation study under different test lengths. 
# For this evaluation, we selected only test cases with ICU stays longer than 96 hours. 
# We then truncated the input sequences to the last 4h, 8h, 12h, …, up to 96h, and compared model performance at different input lengths. 
# the last hour is always used so there is no prediction result here

import os
import argparse
import torch
from config import cfg
from dataset import LengthAwareBalancedBatchSampler, data_config, pad_to_bucket_max, build_val_test_length_info_dict, load_trainval_data, MimicDataset
from utils import init_read
from torch.utils.data import Dataset, Sampler, DataLoader, BatchSampler
from tqdm import tqdm
import numpy as np
from evaluation import Loss

def load_model_for_test(model, cfg, model_path):
    # snapshot_path = os.path.join(cfg.snapshot_dir, f"snapshot_epoch{epoch}.pth")
    # if not os.path.exists(snapshot_path):
    #     raise FileNotFoundError(f"❌ Snapshot not found at {snapshot_path}. Please check --epoch parameter.")
    
    checkpoint = torch.load(model_path, map_location='cpu')  # Change to 'cuda' if needed
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {model_path} (saved epoch {checkpoint['epoch']})")
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Testing Script")
    parser.add_argument('--model_path', type=str, default="snapshot_epoch19.pth",
                        help='Which epoch snapshot to load (default: 10)')
    args = parser.parse_args()
    return args


def rescale_iou(x, c=6):
    # map 0.5 to 0.1
    # increase sharply after 0.5
    # increase smoothly after 0.6
    k = 4 / (0.5**c)
    return x / (1 + k * (1 - x)**c)

def cal_precision(pred, gt):
    precision = torch.abs(pred - gt) / (torch.abs(pred) + torch.abs(gt))
    return 1 - precision.mean()

@torch.no_grad()
def test(model, test_loader, evaluation, mode="meds"):
    print("======= TESTING ========")
    print("Zero out: ", mode)
    outputs = []
    gts = []
    pred_all, gt_pred_all = None, None
    input_min_length = 4

    for step, batch in enumerate(tqdm(test_loader)):
        gt_meds, gt_chart, gt_out, gt_proc, gt_date, gt_ing, stat, demo, Y = batch
        if gt_meds.shape[1] <= input_min_length:
            continue

        if mode == "meds":
            meds = gt_meds[:, :-1].cuda() * 0.0
        else:
            meds = gt_meds[:, :-1].cuda()
        
        if mode == "chart":
            chart = gt_chart[:, :-1].cuda() * 0.0
        else:
            chart = gt_chart[:, :-1].cuda()

        if mode == "out":
            out = gt_out[:, :-1].cuda() * 0.0
        else:
            out = gt_out[:, :-1].cuda()
        
        if mode == "proc":
            proc = gt_proc[:, :-1].cuda() * 0.0
        else:
            proc = gt_proc[:, :-1].cuda()
        
        if mode == "date":
            date = gt_date[:, :-1].cuda() * 0.0
        else:
            date = gt_date[:, :-1].cuda()
        
        if mode == "ing":
            ing = gt_ing[:, :-1].cuda() * 0.0
        else:
            ing = gt_ing[:, :-1].cuda()
        
        if mode == "static":
            stat = stat.cuda() * 0.0
            demo = demo.cuda() * 0.0
        else:
            stat = stat.cuda()
            demo = demo.cuda()

        gt_preds = [
            gt_meds[:, -1:],
            gt_chart[:, -1:],
            gt_out[:, -1:],
            gt_proc[:, -1:],
            gt_date[:, -1:],
            gt_ing[:, -1:],
        ]

        input_list = [meds, chart, out, proc, date, ing]
        meds, chart, out, proc, date, ing = input_list
        output, logits, preds = model(
            meds, chart, out, proc, date, ing, stat, demo, None
        )

        for i in range(len(preds)):
            if pred_all is None:
                pred_all = [[] for _ in range(len(preds))]
                gt_pred_all = [[] for _ in range(len(gt_preds))]
            pred_all[i].append(preds[i].detach().cpu())
            gt_pred_all[i].append(gt_preds[i].detach().cpu())

        outputs.append(output.detach().cpu())
        gts.append(Y.detach().cpu())
        torch.cuda.empty_cache()

    pred_all = [torch.cat(i, dim=0) for i in pred_all]
    gt_pred_all = [torch.cat(i, dim=0) for i in gt_pred_all]
    outputs = torch.cat(outputs, dim=0)
    gts = torch.cat(gts, dim=0)

    evaluation(outputs, gts, pred_all, gt_pred_all, train=False, threshold=0.5)

    return


if __name__ == "__main__":
    args = parse_args()
    model_path = args.model_path

    evaluation = Loss(cfg.device)

    # build model
    if cfg.model.posemb:
        from model_posemb import DTmodel
    else:
        from model import DTmodel

    # build test set
    modalities = (
        cfg.med_flag
        + cfg.chart_flag
        + cfg.out_flag
        + cfg.proc_flag
        + cfg.date_flag
        + cfg.ing_flag
    )
    print("total modalities:", modalities)
    datacfg = data_config(cfg.data_icu, modalities, cfg.train_min_length, cfg.train_max_length, train=True)
    train_ids, val_ids, test_ids, labels = load_trainval_data()

    race_vocab, gender_vocab, insurance_vocab, admission_vocab, icu_vocab = init_read(cfg.root_dir)
    _, test_dataset_length_info = build_val_test_length_info_dict(datacfg)

    test_ids_new = []
    test_seq_lengths = []
    for id in test_ids:
        newid = str(id[1])
        if newid in test_dataset_length_info:
            test_ids_new.append(id)
            test_seq_lengths.append(int(test_dataset_length_info[newid]))

    test_dataset = MimicDataset(
        "test",
        test_ids_new,
        labels,
        datacfg.data_icu,
        root_dir=cfg.root_dir,
        gender_vocab=gender_vocab,
        race_vocab=race_vocab,
        insurance_vocab=insurance_vocab,
        admission_vocab=admission_vocab,
        icu_vocab=icu_vocab,
    )

    meds, chart, out, proc, date, ing, stat, _, _ = test_dataset[0]
    datacfg.stat_vocab_size = stat.shape[-1]
    datacfg.proc_vocab_size = proc.shape[-1]
    datacfg.med_vocab_size = meds.shape[-1]
    datacfg.out_vocab_size = out.shape[-1]
    datacfg.chart_vocab_size = chart.shape[-1]
    datacfg.date_vocab_size = date.shape[-1]
    datacfg.ing_vocab_size = ing.shape[-1]

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=10, #cfg.num_workers,
        shuffle=False,
    )

    model = DTmodel(datacfg, embed_size=cfg.model.embedding_size, latent_size=cfg.model.latent_size, pred=cfg.model.pred)
    if cfg.device != "cpu":
        model = model.cuda()

    # load model 
    model = load_model_for_test(model, cfg, model_path)
    model.eval()

    # evaluation
    test(model, test_loader, evaluation=evaluation, mode="meds")
    test(model, test_loader, evaluation=evaluation, mode="chart")
    test(model, test_loader, evaluation=evaluation, mode="out")
    test(model, test_loader, evaluation=evaluation, mode="proc")
    test(model, test_loader, evaluation=evaluation, mode="date")
    test(model, test_loader, evaluation=evaluation, mode="ing")
    test(model, test_loader, evaluation=evaluation, mode="static")
