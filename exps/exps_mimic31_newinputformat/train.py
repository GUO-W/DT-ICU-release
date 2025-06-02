import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim

from config import cfg
import numpy as np

from dataset import data_config, load_trainval_data, build_dataloaders, build_datasets
from utils import fix_seed, get_data_mean_std_new, save_checkpoint
from evaluation import Loss
from IPython import embed

import torch
import torch.nn as nn

class SoftF1Loss(nn.Module):
    """
    F1‑style loss that stays differentiable by replacing the hard counts
    (TP, FP, FN) with *expected* counts computed from the predicted
    probabilities.

    Args
    ----
    eps : float   # numerical stability (denominator)
    reduction : 'mean' | 'sum' | 'none'
    """

    def __init__(self, eps: float = 1e-7, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")
        self.eps = eps
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        y_pred : probabilities in [0, 1]  (B, …)
        y_true : binary 0/1                (B, …)
        """
        y_pred = y_pred.float()
        y_pred = torch.sigmoid(y_pred)
        y_true = y_true.float()

        # --- expected TP, FP, FN ------------------------------------------
        tp = (y_pred * y_true).sum(dim=0)
        fp = (y_pred * (1 - y_true)).sum(dim=0)
        fn = ((1 - y_pred) * y_true).sum(dim=0)

        soft_f1 = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)
        loss = 1 - soft_f1                              # maximise F1  ⇒  minimise 1‑F1

        # --- reduction -----------------------------------------------------
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss          # 'none'

# Build model
def build_model(datacfg):
    model = DTmodel(datacfg, embed_size=cfg.model.embedding_size, latent_size=cfg.model.latent_size, pred=cfg.model.pred)
    evaluation = Loss(cfg.device)
    if cfg.device != "cpu":
        model = model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lrn_rate)
    bce_loss = nn.BCEWithLogitsLoss()
    # f1_loss = SoftF1Loss()
    l2_loss = nn.MSELoss()
    return model, evaluation, optimizer, bce_loss, l2_loss


def build_scheduler(optimizer, train_loader, grad_accum_steps):
    ## Define learning rate scheduler
    scheduler1 = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=cfg.warmup_epochs * len(train_loader) // grad_accum_steps,
    )
    scheduler2 = optim.lr_scheduler.ConstantLR(
        optimizer,
        factor=1.0,
        total_iters=(cfg.num_epochs - cfg.warmup_epochs) * len(train_loader) // grad_accum_steps,
    )
    scheduler = optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])
    return scheduler

# Focal Loss for handling class imbalance in classification head
def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    with torch.no_grad():
        prob = torch.sigmoid(logits.detach())
    pt = prob * targets + (1 - prob) * (1 - targets)
    weight = alpha * (1 - pt).pow(gamma)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, weight)

def compute_regression_twostage_loss(value, prob, y_true):
    # y_true: [B] ground truth click counts
    y_bin = (y_true > 0).float()

    # BCE loss for classification
    # loss_bce = torch.nn.functional.binary_cross_entropy_with_logits(prob, y_bin)
    loss_bce = focal_loss(prob, y_bin)

    # MSE loss for regression on positive samples only
    if y_bin.sum() > 0:
        y_log = torch.log1p(y_true[y_bin == 1])
        pred_value_log = torch.log1p(torch.nn.functional.relu(value[y_bin == 1]))
        loss_mse = torch.nn.functional.mse_loss(pred_value_log, y_log)
    else:
        loss_mse = 0.0
    # y_click_log = torch.log1p(y_true)
    # pred_click_log = torch.log1p(torch.nn.functional.relu(value))
    # loss_mse = torch.nn.functional.mse_loss(pred_click_log[y_bin == 1], y_click_log[y_bin == 1]) if y_bin.sum() > 0 else 0.0

    total_loss = loss_bce + 5.0 * loss_mse
    return total_loss

def format_pred(value, logit, threshold=0.5):
    click_prob = torch.sigmoid(logit)   # [0, 1] probability
    # Apply threshold to decide whether to keep regression output
    pred_value = torch.where(
        click_prob > threshold,
        torch.nn.functional.relu(value),   # allow only positive predictions
        torch.zeros_like(value)
    )

    return pred_value

def train(model, evaluation, optimizer, logit_loss, l2_loss, datacfg, train_loader, val_loader, test_loader, grad_accum_steps):

    scheduler = build_scheduler(optimizer, train_loader, grad_accum_steps)

    print(f"======== Training ======")
    min_loss = float("inf")
    counter = 0
    for epoch in range(cfg.num_epochs):
        print("lr:", scheduler.get_last_lr())

        if counter == cfg.patience:
            print(
                "STOPPING THE TRAINING BECAUSE VALIDATION ERROR DID NOT IMPROVE FOR {:.1f} EPOCHS".format(
                    cfg.patience
                )
            )
            break
        outputs, gts = [], []
        pred_all, gt_pred_all = None, None
        model.train()

        print("======= EPOCH {:.1f} ========".format(epoch))
        # i = 0
        for step, batch in enumerate(tqdm(train_loader)):
            # i += 1
            # optimizer.zero_grad()
            dynamic_train, stat_train, demo_train, Y_train, key_padding_mask = batch
            dynamic_train = dynamic_train.cuda()
            stat_train = stat_train.cuda()
            demo_train = demo_train.cuda()
            Y_train = Y_train.cuda()
            key_padding_mask = key_padding_mask.cuda()

            if cfg.model.pred:
                output, logits, _ = model(
                    dynamic_train, stat_train, demo_train, key_padding_mask
                )

            else:
                raise NotImplementedError
                # output, logits, preds = model(
                #     dynamic_train, stat_train, demo_train, key_padding_mask
                # )

            bs = dynamic_train.shape[0]
            out_loss = logit_loss(
                logits, Y_train.float()
            )
            pred_loss = 0.0
            # if cfg.model.pred:
            #     for pred_value, pred_logit, gt_pred in zip(preds_value, preds_logit, gt_preds):
            #         pred_loss += compute_regression_twostage_loss(pred_value, pred_logit, gt_pred)

            loss = out_loss + pred_loss * cfg.model.pred_loss
            loss = loss / grad_accum_steps
            loss.backward()
            # print("loss:", loss)
            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                # print("loss:", loss)

            # if cfg.model.pred:
            #     for i in range(len(preds_value)):
            #         if pred_all is None:
            #             pred_all = [[] for _ in range(len(preds_value))]
            #             gt_pred_all = [[] for _ in range(len(gt_preds))]
            #         pred_all[i].append(format_pred(preds_value[i].detach().cpu(), preds_logit[i].detach().cpu()))
            #         gt_pred_all[i].append(gt_preds[i].detach().cpu())

            outputs.append(output.detach().cpu())
            gts.append(Y_train.detach().cpu())
            torch.cuda.empty_cache()
            # if i > 50:
            #     break

        # if cfg.model.pred:
        #     pred_all = [torch.cat(i, dim=0) for i in pred_all]
        #     gt_pred_all = [torch.cat(i, dim=0) for i in gt_pred_all]
        outputs = torch.cat(outputs, dim=0)
        gts = torch.cat(gts, dim=0)

        for class_threshold in np.linspace(0.5, 0.95, 10).tolist():
            print("class threshold:", class_threshold)
            # if cfg.model.pred:
            #     evaluation(outputs, gts, pred_all, gt_pred_all, train=False, threshold=class_threshold)
            # else:
            evaluation(outputs, gts, train=False, threshold=class_threshold)

        ## Val
        val(model, val_loader, evaluation)

        ## Test
        test(model, test_loader, evaluation)

        ## save model
        save_checkpoint(model, optimizer, epoch, cfg.snapshot_dir)

@torch.no_grad()
def val(model, val_loader, evaluation):
    print("======= VALIDATION ========")
    outputs, gts = [], []
    pred_all, gt_pred_all = None, None
    model.eval()

    for batch in tqdm(val_loader):

        dynamic, stat, demo, Y, key_padding_mask = batch
        dynamic = dynamic.cuda()
        stat = stat.cuda()
        demo = demo.cuda()
        Y = Y.cuda()
        key_padding_mask = key_padding_mask.cuda()

        if cfg.model.pred:
            output, logits, _ = model(
                dynamic, stat, demo, key_padding_mask
            )
            # output, logits, preds_value, preds_logit = model(
            #     meds[:, :-1], chart[:, :-1], out[:, :-1], proc[:, :-1], date[:, :-1], ing[:, :-1], stat, demo, key_padding_mask[:, :-1]
            # )
            # gt_preds = [chart[:, -1:], out[:, -1:]]
        else:
            raise NotImplementedError
            # output, logits, preds = model(
            #     meds, chart, out, proc, date, ing, stat, demo, key_padding_mask
            # )

        # if cfg.model.pred:
        #     for i in range(len(preds_value)):
        #         if pred_all is None:
        #             pred_all = [[] for _ in range(len(preds_value))]
        #             gt_pred_all = [[] for _ in range(len(gt_preds))]
        #         pred_all[i].append(format_pred(preds_value[i].detach().cpu(), preds_logit[i].detach().cpu()))
        #         gt_pred_all[i].append(gt_preds[i].detach().cpu())

        outputs.append(output.detach().cpu())
        gts.append(Y.detach().cpu())
        torch.cuda.empty_cache()

    # if cfg.model.pred:
    #     pred_all = [torch.cat(i, dim=0) for i in pred_all]
    #     gt_pred_all = [torch.cat(i, dim=0) for i in gt_pred_all]
    outputs = torch.cat(outputs, dim=0)
    gts = torch.cat(gts, dim=0)

    for class_threshold in np.linspace(0.5, 0.95, 10).tolist():
        print("class threshold:", class_threshold)
        # if cfg.model.pred:
        #     val_loss = evaluation(outputs, gts, pred_all, gt_pred_all, train=False, threshold=class_threshold)
        # else:
        val_loss = evaluation(outputs, gts, train=False, threshold=class_threshold)
    return

@torch.no_grad()
def test(model, test_loader, evaluation):
    print("======= TESTING ========")
    outputs, gts  = [], []
    pred_all, gt_pred_all = None, None
    model.eval()

    for step, batch in enumerate(tqdm(test_loader)):
        dynamic, stat, demo, Y, key_padding_mask = batch
        dynamic = dynamic.cuda()
        stat = stat.cuda()
        demo = demo.cuda()
        Y = Y.cuda()
        key_padding_mask = key_padding_mask.cuda()

        if cfg.model.pred:
            output, logits, _ = model(
                dynamic, stat, demo, key_padding_mask
            )
            # output, logits, preds_value, preds_logit = model(
            #     meds[:, :-1], chart[:, :-1], out[:, :-1], proc[:, :-1], date[:, :-1], ing[:, :-1], stat, demo, key_padding_mask[:, :-1]
            # )
            # gt_preds = [chart[:, -1:], out[:, -1:]]
        else:
            raise NotImplementedError
            # output, logits, preds = model(
            #     meds, chart, out, proc, date, ing, stat, demo, key_padding_mask
            # )

        # if cfg.model.pred:
        #     for i in range(len(preds_value)):
        #         if pred_all is None:
        #             pred_all = [[] for _ in range(len(preds_value))]
        #             gt_pred_all = [[] for _ in range(len(gt_preds))]
        #         pred_all[i].append(format_pred(preds_value[i].detach().cpu(), preds_logit[i].detach().cpu()))
        #         gt_pred_all[i].append(gt_preds[i].detach().cpu())

        outputs.append(output.detach().cpu())
        gts.append(Y.detach().cpu())
        torch.cuda.empty_cache()

    # if cfg.model.pred:
    #     pred_all = [torch.cat(i, dim=0) for i in pred_all]
    #     gt_pred_all = [torch.cat(i, dim=0) for i in gt_pred_all]
    outputs = torch.cat(outputs, dim=0)
    gts = torch.cat(gts, dim=0)

    for class_threshold in np.linspace(0.5, 0.95, 10).tolist():
        print("class threshold:", class_threshold)
        # if cfg.model.pred:
        #     test_loss = evaluation(outputs, gts, pred_all, gt_pred_all, train=False, threshold=class_threshold)
        # else:
        test_loss = evaluation(outputs, gts, train=False, threshold=class_threshold)
    return


if __name__ == "__main__":
    fix_seed(cfg.seed)
    print("seed:", cfg.seed)

    print("Config:")
    for key, value in cfg.items():
        print(f"{key}: {value}")

    if cfg.model.posemb:
        from model_posemb import DTmodel
    else:
        from model import DTmodel
    print("model poseembed:", cfg.model.posemb)
    print("model pred:", cfg.model.pred)

    # Init dataset cfg
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

    print(train_ids[0])
    train_dataset, val_dataset, eval_dataset = build_datasets(train_ids, val_ids, test_ids, labels, datacfg, cfg.root_dir)
    train_loader, val_loader, test_loader = build_dataloaders(
        train_dataset, val_dataset, eval_dataset, datacfg, train_ids, val_ids, test_ids,
    )

    dynamic, stat, _, _ = train_dataset[0]
    datacfg.stat_vocab_size = stat.shape[-1]
    # datacfg.proc_vocab_size = proc.shape[-1]
    # datacfg.med_vocab_size = meds.shape[-1]
    # datacfg.out_vocab_size = out.shape[-1]
    # datacfg.chart_vocab_size = chart.shape[-1]
    # datacfg.date_vocab_size = date.shape[-1]
    # datacfg.ing_vocab_size = ing.shape[-1]

    model, evaluation, optimizer, logit_loss, l2_loss = build_model(datacfg=datacfg)

    # means, stds = get_data_mean_std_new(train_dataset, val_dataset)
    # model.set_mean_std(means, stds)

    train(model, evaluation, optimizer, logit_loss, l2_loss, datacfg, train_loader, val_loader, test_loader, cfg.grad_accum_steps)
