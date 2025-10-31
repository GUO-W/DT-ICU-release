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
    # l2_loss = nn.L1Loss() # nn.MSELoss()
    # l2_loss = nn.HuberLoss(delta=0.5)
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


def train(model, evaluation, optimizer, logit_loss, l2_loss, datacfg, train_loader, val_loader, test_loader, grad_accum_steps):

    scheduler = build_scheduler(optimizer, train_loader, grad_accum_steps)

    print(f"======== Training ======")
    min_loss = float("inf")
    counter = 0
    for epoch in range(cfg.num_epochs):
        print("lr:", scheduler.get_last_lr())

        #save_checkpoint(model, optimizer, epoch, cfg.snapshot_dir)
        #test(model, test_loader, evaluation)

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
        for step, batch in enumerate(tqdm(train_loader)):
            
            # optimizer.zero_grad()
            # meds, chart, out, proc, date, ing, stat_train, demo_train, Y_train, key_padding_mask, buckets = batch
            meds, chart, out, stat_train, demo_train, Y_train, key_padding_mask, buckets = batch
            meds = torch.log(meds.cuda() + 1)
            chart = torch.log(chart.cuda() + 1)
            out = torch.log(out.cuda() + 1)
            # proc = proc.cuda()
            # date = date.cuda()
            # ing = ing.cuda()
            stat_train = stat_train.cuda() 
            demo_train = demo_train.cuda()
            Y_train = Y_train.cuda()
            key_padding_mask = key_padding_mask.cuda()


            if cfg.model.pred:
                output, logits, preds = model(
                    #meds[:, :-1], chart[:, :-1], out[:, :-1], proc[:, :-1], date[:, :-1], ing[:, :-1], stat_train, demo_train, key_padding_mask[:, :-1]
                    meds[:, :-1], chart[:, :-1], out[:, :-1], stat_train, demo_train, key_padding_mask[:, :-1]
                )

                gt_preds = [
                meds[:, -1:],
                chart[:, -1:],
                out[:, -1:],
                # proc[:, -1:],
                # date[:, -1:],
                # ing[:, -1:],
                ]

            else:
                output, logits, preds = model(
                    #meds, chart, out, proc, date, ing, stat_train, demo_train, key_padding_mask
                    meds, chart, out, stat_train, demo_train, key_padding_mask
                )

            bs = meds.shape[0]

            out_loss = logit_loss(
                logits, Y_train.float()
            )
            # print(["short", "long"][buckets[0].item()], ["neg", "pos"][Y_train[0].item()], out_loss.detach().mean().item())
            pred_loss = 0.0
            if cfg.model.pred:
                for pred, gt_pred in zip(preds, gt_preds):
                    pred_loss += l2_loss(
                        pred, gt_pred
                    )
                    # print("pred and gt", pred.shape, gt_pred.shape, gt_pred.min().item(), gt_pred.max().item(), pred.min().item(), pred.max().item(), pred_loss.item(), flush=True)

            loss = out_loss + pred_loss * cfg.model.pred_loss
            loss = loss / grad_accum_steps
            loss.backward()
            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if cfg.model.pred:
                for i in range(len(preds)):
                    if pred_all is None:
                        pred_all = [[] for _ in range(len(preds))]
                        gt_pred_all = [[] for _ in range(len(gt_preds))]
                    pred_all[i].append(preds[i].detach().cpu())
                    gt_pred_all[i].append(gt_preds[i].detach().cpu())

            outputs.append(output.detach().cpu())
            gts.append(Y_train.detach().cpu())
            torch.cuda.empty_cache()

        if cfg.model.pred:
            pred_all = [torch.cat(i, dim=0) for i in pred_all]
            gt_pred_all = [torch.cat(i, dim=0) for i in gt_pred_all]
        outputs = torch.cat(outputs, dim=0)
        gts = torch.cat(gts, dim=0)

        for class_threshold in np.linspace(0.5, 0.95, 10).tolist():
            print("class threshold:", class_threshold)

            if cfg.model.pred:
                evaluation(outputs, gts, pred_all, gt_pred_all, train=False, threshold=class_threshold)
            else:
                evaluation(outputs, gts, train=False, threshold=class_threshold)


        ## Val
        val(model, val_loader, evaluation)
        #val_loss = val(model, val_loader, evaluation)
        # if val_loss <= min_loss + 0.001:
        #     print("Validation results improved")
        #     min_loss = val_loss
        #     print("Updating Model")
        #     #torch.save(model, cfg.save_path)
        #     counter = 0
        # else:
        #     print("No improvement in Validation results")
        #     counter = counter + 1

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

        #meds, chart, out, proc, date, ing, stat, demo, Y, key_padding_mask, buckets = batch
        meds, chart, out, stat, demo, Y, key_padding_mask, buckets = batch
        meds = torch.log(meds.cuda() + 1)
        chart = torch.log(chart.cuda() + 1)
        out = torch.log(out.cuda() + 1)
        # proc = proc.cuda()
        # date = date.cuda()
        # ing = ing.cuda()
        stat = stat.cuda()
        demo = demo.cuda()
        Y = Y.cuda()
        key_padding_mask = key_padding_mask.cuda()

        if cfg.model.pred:
            output, logits, preds = model(
                #meds[:, :-1], chart[:, :-1], out[:, :-1], proc[:, :-1], date[:, :-1], ing[:, :-1], stat, demo, key_padding_mask[:, :-1]
                meds[:, :-1], chart[:, :-1], out[:, :-1], stat, demo, key_padding_mask[:, :-1]
            )
            gt_preds = [
                meds[:, -1:],
                chart[:, -1:],
                out[:, -1:],
                # proc[:, -1:],
                # date[:, -1:],
                # ing[:, -1:],
            ]
        else:
            output, logits, preds = model(
                #meds, chart, out, proc, date, ing, stat, demo, key_padding_mask
                meds, chart, out, stat, demo, key_padding_mask
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

    for class_threshold in np.linspace(0.5, 0.95, 10).tolist():
        print("class threshold:", class_threshold)
        if cfg.model.pred:
            val_loss = evaluation(outputs, gts, pred_all, gt_pred_all, train=False, threshold=class_threshold)
        else:
            val_loss = evaluation(outputs, gts, train=False, threshold=class_threshold)
    return

@torch.no_grad()
def test(model, test_loader, evaluation):
    print("======= TESTING ========")
    outputs, gts  = [], []
    pred_all, gt_pred_all = None, None
    model.eval()

    for step, batch in enumerate(tqdm(test_loader)):
        #meds, chart, out, proc, date, ing, stat, demo, Y, key_padding_mask, buckets = batch
        meds, chart, out, stat, demo, Y, key_padding_mask, buckets = batch
        meds = torch.log(meds.cuda() + 1)
        chart = torch.log(chart.cuda() + 1)
        out = torch.log(out.cuda() + 1)
        # proc = proc.cuda()
        # date = date.cuda()
        # ing = ing.cuda()
        stat = stat.cuda()
        demo = demo.cuda()
        Y = Y.cuda()
        key_padding_mask = key_padding_mask.cuda()

        if cfg.model.pred:
            output, logits, preds = model(
                #meds[:, :-1], chart[:, :-1], out[:, :-1], proc[:, :-1], date[:, :-1], ing[:, :-1], stat, demo, key_padding_mask[:, :-1]
                meds[:, :-1], chart[:, :-1], out[:, :-1], stat, demo, key_padding_mask[:, :-1]
            )
            gt_preds = [
                meds[:, -1:],
                chart[:, -1:],
                out[:, -1:],
                # proc[:, -1:],
                # date[:, -1:],
                # ing[:, -1:],
            ]
        else:
            output, logits, preds = model(
                #meds, chart, out, proc, date, ing, stat, demo, key_padding_mask
                meds, chart, out,  stat, demo, key_padding_mask
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

    for class_threshold in np.linspace(0.5, 0.95, 10).tolist():
        print("class threshold:", class_threshold)
        if cfg.model.pred:
            test_loss = evaluation(outputs, gts, pred_all, gt_pred_all, train=False, threshold=class_threshold)
        else:
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
        # + cfg.proc_flag
        # + cfg.date_flag
        # + cfg.ing_flag
    )
    print("total modalities:", modalities)
    datacfg = data_config(cfg.data_icu, modalities, cfg.train_min_length, cfg.train_max_length, train=True)
    train_ids, val_ids, test_ids, labels = load_trainval_data()

    print(train_ids[0]) #iv:[12574949, 31423742] #train 66875 iii:[19243, 219174] #train 42355
    train_dataset, val_dataset, eval_dataset = build_datasets(train_ids, val_ids, test_ids, labels, datacfg, cfg.root_dir)
    train_loader, val_loader, test_loader = build_dataloaders(
        train_dataset, val_dataset, eval_dataset, datacfg, train_ids, val_ids, test_ids,
    )

    meds, chart, out, stat, _, _ = train_dataset[0]
    #meds, chart, out, proc, date, ing, stat, _, _ = train_dataset[0]
    datacfg.stat_vocab_size = stat.shape[-1]
    #datacfg.proc_vocab_size = proc.shape[-1]
    datacfg.med_vocab_size = meds.shape[-1]
    datacfg.out_vocab_size = out.shape[-1]
    datacfg.chart_vocab_size = chart.shape[-1]
    #datacfg.date_vocab_size = date.shape[-1]
    #datacfg.ing_vocab_size = ing.shape[-1]

    model, evaluation, optimizer, logit_loss, l2_loss = build_model(datacfg=datacfg)

    # means, stds = get_data_mean_std_new(train_dataset, val_dataset)
    # model.set_mean_std(means, stds)

    train(model, evaluation, optimizer, logit_loss, l2_loss, datacfg, train_loader, val_loader, test_loader, cfg.grad_accum_steps)
