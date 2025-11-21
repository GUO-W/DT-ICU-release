#!/usr/bin/env python
# encoding: utf-8

## test only + lomo
## run_test_lomo.sh

## baseline
#python test_only.py --ckpt snapshot_epoch19.pth

## ablate meds
#python test_only.py --ckpt snapshot_epoch19.pth --zero_inputs meds

## ablate meds + chart
#python test_only.py --ckpt snapshot_epoch19.pth --zero_inputs meds,chart

## ablate static inputs
#python test_only.py --ckpt snapshot_epoch19.pth --zero_inputs static


import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

from config import cfg
from dataset import data_config, load_trainval_data, build_dataloaders, build_datasets
from utils import fix_seed
from evaluation import Loss


def peel_last_step_as_target(x, attention_mask, last_nonpad_idx, pad_value=0.0):
    B, T, D = x.shape
    device = x.device
    valid = last_nonpad_idx >= 0
    idx = last_nonpad_idx.clamp(min=0)

    batch_idx = torch.arange(B, device=device)
    y = x[batch_idx, idx, :]
    y = y.masked_fill(~valid.unsqueeze(-1), 0.0)

    x_mod = x.clone()
    if valid.any():
        x_mod[batch_idx[valid], idx[valid], :] = pad_value

    mask_mod = attention_mask.clone()
    if valid.any():
        mask_mod[batch_idx[valid], idx[valid]] = True

    return x_mod, mask_mod, y


def peel_last_step_from_list(tensor_list, attention_mask, last_nonpad_idx, pad_value=0.0):
    out_list = []
    ys = []
    mask_mod = attention_mask
    for xi in tensor_list:
        xi_mod, mask_mod, y = peel_last_step_as_target(xi, mask_mod, last_nonpad_idx, pad_value)
        out_list.append(xi_mod)
        ys.append(y.unsqueeze(1))
    return out_list, mask_mod, ys


def zero_selected_inputs(input_list, stat, demo, zero_names):
    """
    input_list: [meds, chart, out, proc, date, ing]
    zero_names: set of strings, e.g. {"meds", "chart"} or {"static"}
    """
    meds, chart, out, proc, date, ing = input_list

    if "meds" in zero_names:
        meds = torch.zeros_like(meds)
    if "chart" in zero_names:
        chart = torch.zeros_like(chart)
    if "out" in zero_names:
        out = torch.zeros_like(out)
    if "proc" in zero_names:
        proc = torch.zeros_like(proc)
    if "date" in zero_names:
        date = torch.zeros_like(date)
    if "ing" in zero_names:
        ing = torch.zeros_like(ing)

    # static features
    if "static" in zero_names or "stat" in zero_names:
        stat = torch.zeros_like(stat)
    if "static" in zero_names or "demo" in zero_names:
        demo = torch.zeros_like(demo)

    return [meds, chart, out, proc, date, ing], stat, demo


@torch.no_grad()
def test(model, test_loader, evaluation, zero_names=None):
    if zero_names is None:
        zero_names = set()

    print("======= TESTING ========")
    if len(zero_names) == 0:
        print("Zero out: none")
    else:
        print("Zero out:", ",".join(sorted(list(zero_names))))

    outputs, gts = [], []
    pred_all, gt_pred_all = None, None
    model.eval()

    #for step, batch in enumerate(tqdm(test_loader)):
    for step, batch in enumerate(tqdm(test_loader, disable=not sys.stdout.isatty())):
        meds, chart, out, proc, date, ing, stat, demo, Y, key_padding_mask, buckets, last_nonpad_idx = batch
        meds = meds.cuda()
        chart = chart.cuda()
        out = out.cuda()
        proc = proc.cuda()
        date = date.cuda()
        ing = ing.cuda()
        stat = stat.cuda()
        demo = demo.cuda()
        Y = Y.cuda()
        key_padding_mask = key_padding_mask.cuda()
        last_nonpad_idx = last_nonpad_idx.cuda()

        # peel last timestep first (keep the order the same as train/val/test)
        input_list, key_padding_mask, gt_preds = peel_last_step_from_list(
            [meds, chart, out, proc, date, ing],
            key_padding_mask,
            last_nonpad_idx
        )

        # zero selected inputs for ablation/LOMO
        input_list, stat, demo = zero_selected_inputs(input_list, stat, demo, zero_names)

        if cfg.model.pred:
            output, logits, preds = model(
                input_list[0], input_list[1], input_list[2], input_list[3],
                input_list[4], input_list[5], stat, demo, key_padding_mask,
            )
        else:
            # if you don't use pred, you may still want to zero original tensors;
            # but here we keep the same structure
            output, logits, preds = model(
                input_list[0], input_list[1], input_list[2], input_list[3],
                input_list[4], input_list[5], stat, demo, key_padding_mask,
            )

        if cfg.model.pred:
            for i in range(len(preds)):
                if pred_all is None:
                    pred_all = [[] for _ in range(len(preds))]
                    gt_pred_all = [[] for _ in range(len(gt_preds))]
                pred_all[i].append(preds[i].detach().cpu())
                gt_pred_all[i].append(gt_preds[i].detach().cpu())

        outputs.append(output.detach().cpu())
        gts.append(Y.detach().cpu())
        torch.cuda.empty_cache()

    if cfg.model.pred:
        pred_all = [torch.cat(i, dim=0) for i in pred_all]
        gt_pred_all = [torch.cat(i, dim=0) for i in gt_pred_all]
    outputs = torch.cat(outputs, dim=0)
    gts = torch.cat(gts, dim=0)

    class_threshold = 0.5
    if cfg.model.pred:
    	evaluation(outputs, gts, pred_all, gt_pred_all, train=False, threshold=class_threshold)
    else:
    	evaluation(outputs, gts, train=False, threshold=class_threshold)


    #for class_threshold in np.linspace(0.5, 0.95, 10).tolist():
    #    print("class threshold:", class_threshold)
    #    if cfg.model.pred:
    #        evaluation(outputs, gts, pred_all, gt_pred_all, train=False, threshold=class_threshold)
    #    else:
    #        evaluation(outputs, gts, train=False, threshold=class_threshold)


def parse_args():
    parser = argparse.ArgumentParser(description="Test-only Script with optional input ablation")
    parser.add_argument(
        "--type",
        type=str,
        default="posemb",
        help="Model type: posemb | mlp | rnn | lstm | default",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="snapshot_epoch19.pth",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--zero_inputs",
        type=str,
        default="",
        help="Comma-separated input names to zero out. "
             "Possible names: meds,chart,out,proc,date,ing,stat,demo,static",
    )
    return parser.parse_args()


if __name__ == "__main__":
    fix_seed(cfg.seed)
    args = parse_args()

    if args.type == "posemb":
        from model_posemb import DTmodel
    elif args.type == "mlp":
        from model_mlp import DTmodel
    elif args.type == "rnn":
        from model_rnn import DTmodel
    elif args.type == "lstm":
        from model_lstm import DTmodel
    #else:
    #    from model import DTmodel

    print("model type:", args.type)
    print("model pred:", cfg.model.pred)

    modalities = (
        cfg.med_flag
        + cfg.chart_flag
        + cfg.out_flag
        + cfg.proc_flag
        + cfg.date_flag
        + cfg.ing_flag
    )
    datacfg = data_config(
        cfg.data_icu,
        modalities,
        cfg.train_min_length,
        cfg.train_max_length,
        train=True,
    )

    train_ids, val_ids, test_ids, labels = load_trainval_data()

    train_dataset, val_dataset, eval_dataset = build_datasets(
        train_ids,
        val_ids,
        test_ids,
        labels,
        datacfg,
        cfg.root_dir,
    )

    meds, chart, out, proc, date, ing, stat, demo, _ = train_dataset[0]
    datacfg.stat_vocab_size = stat.shape[-1]
    datacfg.proc_vocab_size = proc.shape[-1]
    datacfg.med_vocab_size = meds.shape[-1]
    datacfg.out_vocab_size = out.shape[-1]
    datacfg.chart_vocab_size = chart.shape[-1]
    datacfg.date_vocab_size = date.shape[-1]
    datacfg.ing_vocab_size = ing.shape[-1]

    _, _, test_loader = build_dataloaders(
        train_dataset,
        val_dataset,
        eval_dataset,
        datacfg,
        train_ids,
        val_ids,
        test_ids,
    )

    model = DTmodel(
        datacfg,
        embed_size=cfg.model.embedding_size,
        latent_size=cfg.model.latent_size,
        pred=cfg.model.pred,
    )
    evaluation = Loss(cfg.device)
    if cfg.device != "cpu":
        model = model.cuda()

    ckpt_path = args.ckpt
    print(f"Loading checkpoint from: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cuda" if cfg.device != "cpu" else "cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded model_state_dict from epoch {ckpt.get('epoch', 'unknown')}")
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded state_dict from epoch {ckpt.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(ckpt)
        print("Loaded raw state_dict")

    if args.zero_inputs.strip() == "":
        zero_names = set()
    else:
        zero_names = set([z.strip() for z in args.zero_inputs.split(",") if z.strip() != ""])

    test(model, test_loader, evaluation, zero_names=zero_names)

