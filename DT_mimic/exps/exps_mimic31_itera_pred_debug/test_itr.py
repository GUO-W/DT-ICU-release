#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import torch
from config import cfg
from dataset import LengthAwareBalancedBatchSampler, data_config, pad_to_bucket_max, build_val_test_length_info_dict, load_trainval_data, MimicDataset
from utils import init_read
from torch.utils.data import Dataset, Sampler, DataLoader, BatchSampler
from tqdm import tqdm
import numpy as np

def load_model_for_test(model, cfg, epoch):
    snapshot_path = os.path.join(cfg.snapshot_dir, f"snapshot_epoch{epoch}.pth")
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(f"❌ Snapshot not found at {snapshot_path}. Please check --epoch parameter.")
    
    checkpoint = torch.load(snapshot_path, map_location='cpu')  # Change to 'cuda' if needed
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {snapshot_path} (saved epoch {checkpoint['epoch']})")
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Testing Script")
    parser.add_argument('--epoch', type=int, default=10,
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
    # print("pred", pred)
    # print("gt:", gt)
    precision = torch.abs(pred - gt) / (torch.abs(pred) + torch.abs(gt) + 1e-6)
    return 1 - precision.mean()

@torch.no_grad()
def test(model, test_loader):
    print("======= TESTING ========")
    precisions = []
    recalls = []

    input_min_length = cfg.test_input_min_length
    max_pred_length_ratio = cfg.test_max_pred_length_ratio

    for step, batch in enumerate(tqdm(test_loader)):
        gt_meds, gt_chart, gt_out, gt_proc, gt_date, gt_ing, stat, demo, Y = batch
        if gt_meds.shape[1] <= input_min_length:
            continue
        #print(gt_meds.shape)
        max_pred_length = max_pred_length_ratio * gt_meds.shape[1]
        meds = gt_meds[:, :input_min_length].cuda()
        chart = gt_chart[:, :input_min_length].cuda()
        out = gt_out[:, :input_min_length].cuda()
        proc = gt_proc[:, :input_min_length].cuda()
        date = gt_date[:, :input_min_length].cuda()
        ing = gt_ing[:, :input_min_length].cuda()
        stat = stat.cuda()
        demo = demo.cuda()
        input_list = [meds, chart, out, proc, date, ing]
        stop_pred = False
        while meds.shape[1] < max_pred_length and not stop_pred:

            meds, chart, out, proc, date, ing = input_list
            output, logits, preds = model(
                meds, chart, out, proc, date, ing, stat, demo, None
            )
            if output.squeeze().item() > cfg.test_pos_threshold: # or meds.shape[1] >= gt_meds.shape[1]:
                print("pred prob:", output.squeeze().item())
                stop_pred = True

            for i in range(len(preds)):
                input_list[i] = torch.cat([input_list[i], preds[i]], dim=1)
                # input_list[i] = torch.cat([input_list[i], batch[i][:, meds.shape[1]:meds.shape[1]+1].cuda()], dim=1)

        #print(gt_meds.shape[1], input_list[0].shape[1])
        iou = min(gt_meds.shape[1] - input_min_length, input_list[0].shape[1] - input_min_length) / max(gt_meds.shape[1] - input_min_length, input_list[0].shape[1] - input_min_length)
        print("iou:", iou, gt_meds.shape[1], input_list[0].shape[1])
        # recall = rescale_iou(iou)
        # print("Recall: ", recall)
        recall = iou
        recalls.append(recall)

        precision_min_length = min(gt_meds.shape[1] - input_min_length, input_list[0].shape[1] - input_min_length)
        precision = 0
        for i in range(len(input_list)):
            precision += cal_precision(input_list[i].detach().cpu()[:, input_min_length:input_min_length + precision_min_length], batch[i][:, input_min_length:input_min_length + precision_min_length])
        precision = precision.item() / len(input_list)
        print("Precision: ", precision, len(input_list))
        precisions.append(precision)

        torch.cuda.empty_cache()
        # break

    return precisions, recalls


if __name__ == "__main__":
    args = parse_args()
    epoch_to_load = args.epoch

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
        num_workers=cfg.num_workers,
        shuffle=False,
    )

    model = DTmodel(datacfg, embed_size=cfg.model.embedding_size, latent_size=cfg.model.latent_size, pred=cfg.model.pred)
    if cfg.device != "cpu":
        model = model.cuda()

    # load model 
    model = load_model_for_test(model, cfg, epoch_to_load)
    model.eval()

    # evaluation
    precisions, recalls = test(model, test_loader)
    print("overall precision: ", np.mean(precisions))
    print("overall recall: ", np.mean(recalls))
