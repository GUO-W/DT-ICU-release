import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim

from config import cfg
from model import DTmodel
from dataset import data_config, create_kfolds, split_train_val_test, build_dataloaders
from utils import fix_seed, get_data_mean_std
from evaluation import Loss


# Build model
def build_model(datacfg):
    model = DTmodel(datacfg, embed_size=cfg.model.embedding_size, latent_size=cfg.model.latent_size)
    evaluation = Loss(cfg.device)
    if cfg.device != "cpu":
        model = model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lrn_rate)
    bce_loss = nn.BCEWithLogitsLoss()
    return model, evaluation, optimizer, bce_loss


def build_scheduler(optimizer, train_loader):
    ## Define learning rate scheduler
    scheduler1 = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=cfg.warmup_epochs * len(train_loader),
    )
    scheduler2 = optim.lr_scheduler.ConstantLR(
        optimizer,
        factor=1.0,
        total_iters=(cfg.num_epochs - cfg.warmup_epochs) * len(train_loader),
    )
    scheduler = optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])
    return scheduler


def train(model, evaluation, optimizer, bce_loss, datacfg):
    k_hids, labels = create_kfolds()
    for i in range(cfg.k_fold):
        train_hids, val_hids, test_hids = split_train_val_test(k_hids, i)
        train_loader, val_loader, test_loader = build_dataloaders(
            train_hids, val_hids, test_hids, labels, datacfg
        )
        means, stds = get_data_mean_std(train_loader, val_loader)
        model.set_mean_std(means, stds)

        scheduler = build_scheduler(optimizer, train_loader)

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
            model.train()
            print("======= EPOCH {:.1f} ========".format(epoch))
            for (
                meds,
                chart,
                out,
                proc,
                lab,
                stat_train,
                demo_train,
                Y_train,
            ) in tqdm(train_loader):

                optimizer.zero_grad()
                output, logits = model(
                    meds, chart, out, proc, lab, stat_train, demo_train
                )
                out_loss = bce_loss(
                    logits, Y_train.float()
                )
                out_loss.backward()
                optimizer.step()

                outputs.append(output)
                gts.append(Y_train)
                scheduler.step()

            outputs = torch.cat(outputs, dim=0)
            gts = torch.cat(gts, dim=0)
            evaluation(outputs, gts, train=False)

            ## Val
            val_loss = val(model, val_loader, evaluation)
            if val_loss <= min_loss + 0.001:
                print("Validation results improved")
                min_loss = val_loss
                print("Updating Model")
                #torch.save(model, cfg.save_path)
                counter = 0
            else:
                print("No improvement in Validation results")
                counter = counter + 1

            ## Test
            test(model, test_loader, evaluation)


def val(model, val_loader, evaluation):
    print("======= VALIDATION ========")
    outputs = []
    gts = []
    model.eval()

    for meds, chart, out, proc, lab, stat, demo, Y_train in tqdm(val_loader):
        output, logits = model(meds, chart, out, proc, lab, stat, demo)
        outputs.append(output)
        gts.append(Y_train)

    outputs = torch.cat(outputs, dim=0)
    gts = torch.cat(gts, dim=0)
    val_loss = evaluation(outputs, gts, train=False)
    return val_loss.item()


def test(model, test_loader, evaluation):
    print("======= TESTING ========")
    outputs = []
    gts = []
    model.eval()

    with torch.no_grad():
        for meds, chart, out, proc, lab, stat, demo, Y_train in tqdm(test_loader):
            output, logits = model(meds, chart, out, proc, lab, stat, demo)
        outputs.append(output)
        gts.append(Y_train)

    outputs = torch.cat(outputs, dim=0)
    gts = torch.cat(gts, dim=0)
    evaluation(outputs, gts, train=False)


if __name__ == "__main__":
    fix_seed(cfg.seed)
    # Init dataset cfg
    modalities = (
        cfg.diag_flag
        + cfg.proc_flag
        + cfg.out_flag
        + cfg.chart_flag
        + cfg.med_flag
        + cfg.lab_flag
    )
    datacfg = data_config(cfg.data_icu, modalities, train=True)
    model, evaluation, optimizer, bce_loss = build_model(datacfg=datacfg)
    train(model, evaluation, optimizer, bce_loss, datacfg)