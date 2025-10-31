import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim

from config import cfg
from model import DTmodel
from dataset import data_config, load_trainval_data, build_dataloaders, build_datasets
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
    l2_loss = nn.MSELoss()
    return model, evaluation, optimizer, bce_loss, l2_loss


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


def train(model, evaluation, optimizer, bce_loss, l2_loss, datacfg, train_loader, val_loader, test_loader):

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
            date,
            ing,
            stat_train,
            demo_train,
            Y_train,
        ) in tqdm(train_loader):

            # inputs = [meds, chart, out, proc, date, ing, stat_train, demo_train]
            # for index, i in enumerate(inputs):
            #     print("data info:", index, i.shape, i.mean(), i.std())

            optimizer.zero_grad()
            output, logits, preds = model(
                meds[:, :-1], chart[:, :-1], out[:, :-1], proc[:, :-1], date[:, :-1], ing[:, :-1], stat_train, demo_train
            )
            bs = meds.shape[0]
            gt_preds = [
                meds[:, -1:],
                chart[:, -1:],
                out[:, -1:], 
                proc[:, -1:], 
                date[:, -1:],
                ing[:, -1:],
            ]

            out_loss = bce_loss(
                logits, Y_train.float()
            )
            pred_loss = 0.0
            for pred, gt_pred in zip(preds, gt_preds):
                pred_loss += l2_loss(
                    pred, gt_pred
                )

            loss = out_loss + pred_loss * 0
            loss.backward()
            optimizer.step()

            outputs.append(output)
            gts.append(Y_train)
            scheduler.step()

        outputs = torch.cat(outputs, dim=0)
        gts = torch.cat(gts, dim=0)
        evaluation(outputs, gts, train=False)

        # ## Val
        # val_loss = val(model, val_loader, evaluation)
        # if val_loss <= min_loss + 0.001:
        #     print("Validation results improved")
        #     min_loss = val_loss
        #     print("Updating Model")
        #     #torch.save(model, cfg.save_path)
        #     counter = 0
        # else:
        #     print("No improvement in Validation results")
        #     counter = counter + 1

        # ## Test
        # test(model, test_loader, evaluation)


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
    print("seed:", cfg.seed)
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

    train_dataset, val_dataset, eval_dataset = build_datasets(train_ids, val_ids, test_ids, labels, datacfg)
    train_loader, val_loader, test_loader = build_dataloaders(
        train_dataset, val_dataset, eval_dataset, datacfg
    )

    meds, chart, out, proc, date, ing, stat, _, _, _ = train_dataset[0]
    datacfg.stat_vocab_size = stat.shape[-1]
    datacfg.proc_vocab_size = proc.shape[-1]
    datacfg.med_vocab_size = meds.shape[-1]
    datacfg.out_vocab_size = out.shape[-1]
    datacfg.chart_vocab_size = chart.shape[-1]
    datacfg.date_vocab_size = date.shape[-1]
    datacfg.ing_vocab_size = ing.shape[-1]

    model, evaluation, optimizer, bce_loss, l2_loss = build_model(datacfg=datacfg)

    means, stds = get_data_mean_std(train_dataset, val_dataset)
    model.set_mean_std(means, stds)

    train(model, evaluation, optimizer, bce_loss, l2_loss, datacfg, train_loader, val_loader, test_loader)