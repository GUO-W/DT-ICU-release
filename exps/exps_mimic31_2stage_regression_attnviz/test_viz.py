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
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Union
from pathlib import Path

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
    parser.add_argument('--epoch', type=int, default=19,
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
    precision = torch.abs(pred - gt) / (torch.abs(pred) + torch.abs(gt) + 1e-8)
    return 1 - precision.mean()

def format_pred(value, logit, threshold=0.5):
    click_prob = torch.sigmoid(logit)   # [0, 1] probability
    # Apply threshold to decide whether to keep regression output
    # print(click_prob, threshold)
    pred_value = torch.where(
        click_prob > threshold,
        torch.nn.functional.relu(value),   # allow only positive predictions
        torch.zeros_like(value)
    )
    return pred_value

def plot_attention(
    attn: torch.Tensor,
    tokens:  List[str]              = None,
    layer:   Union[int, str]        = 0,
    save_to: Union[str, Path, None] = "layer{layer}_heads.png",
    show:    bool                   = False,
    figsize: Tuple[int, int]        = (10, 12),
):
    """
    Plot/save one layer’s attention weights in a **4×2 grid**.

    * attn shape  (B, H, L, L)  or  (H, L, L)
    * up to 8 heads displayed; extra heads are ignored.
    * If fewer than 8 heads, empty panels are hidden.
    """
    # Pick first item in batch if needed
    if attn.dim() == 4:
        attn = attn[0]

    n_heads = attn.size(0)
    n_show  = min(n_heads, 8)                       # at most 8 panels

    fig, axes = plt.subplots(4, 2, figsize=figsize)
    axes = axes.flatten()                           # 1-D iterator

    vmax = attn.max().item()
    for i in range(8):
        ax = axes[i]
        if i < n_show:
            sns.heatmap(
                attn[i].cpu().numpy(),
                vmin=0.0,
                vmax=vmax,
                cmap="plasma",
                square=True,
                xticklabels=tokens if tokens is not None else False,
                yticklabels=tokens if tokens is not None else False,
                cbar=False,
                ax=ax,
            )
            ax.set_title(f"Head {i}")
            ax.set_xlabel("Key"), ax.set_ylabel("Query")
        else:
            ax.axis("off")                          # hide unused panel

    plt.suptitle(f"Layer {layer} — attention weights", y=1.02)
    plt.tight_layout()

    saved_path = None
    if save_to is not None:
        save_path = Path(str(save_to).format(layer=layer)).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        saved_path = save_path
        print(f"[✓] Saved attention map → {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_path


@torch.no_grad()
def test(model, test_loader):
    print("======= TESTING ========")
    precisions = []
    recalls = []

    # input_min_length = cfg.test_input_min_length
    input_min_length = 4
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
        gt_list = [gt_meds, gt_chart, gt_out, gt_proc, gt_date, gt_ing]
        skip_indexes = set([0, 3, 4, 5])
        map_indexes = {1:0, 2:1}
        stop_pred = False
        while meds.shape[1] < max_pred_length and not stop_pred:

            meds, chart, out, proc, date, ing = input_list
            output, logits, preds_value, preds_logit, scoremap = model(
                meds, chart, out, proc, date, ing, stat, demo, None, viz=True
            )
            # if output.squeeze().item() > cfg.test_pos_threshold or meds.shape[1] >= gt_meds.shape[1]:
            if meds.shape[1] >= gt_meds.shape[1]:
                stop_pred = True

            for i in range(len(input_list)):
                if i in skip_indexes:
                    input_list[i] = torch.cat([input_list[i], gt_list[i][:, meds.shape[1]:meds.shape[1]+1].cuda()], dim=1)
                else:
                    new_index = map_indexes[i]
                    final_pred = format_pred(preds_value[new_index], preds_logit[new_index])
                    # print("preds_value:", preds_value[new_index], "preds_logit:", torch.sigmoid(preds_logit[new_index]))
                    input_list[i] = torch.cat([input_list[i], final_pred], dim=1)
            break

        scoremap = scoremap.detach().cpu()
        bs, heads, nq, nk = scoremap.shape
        # scoremap = scoremap.reshape(bs, heads, 6, nq // 6, 6, nk // 6).mean(dim=-1).mean(dim=-2)
        # scoremap = scoremap.mean(dim=1, keepdim=True)
        plot_attention(scoremap)

        torch.cuda.empty_cache()
        break

    return 


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
        shuffle=True,
    )

    model = DTmodel(datacfg, embed_size=cfg.model.embedding_size, latent_size=cfg.model.latent_size, pred=cfg.model.pred)
    if cfg.device != "cpu":
        model = model.cuda()

    # load model 
    model = load_model_for_test(model, cfg, epoch_to_load)
    model.eval()

    # evaluation
    test(model, test_loader)
    # print("overall precision: ", np.mean(precisions))
    # print("overall recall: ", np.mean(recalls))
