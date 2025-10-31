import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import *
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import sys
from config import cfg

# Ensure output directory exists
if not os.path.exists("./data/output"):
    os.makedirs("./data/output")

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')

class Loss(nn.Module):
    def __init__(self, device):
        super(Loss, self).__init__()
        self.classify_loss = nn.BCELoss()  # Only BCELoss is kept
        self.l2_loss = nn.MSELoss()
        self.device = device
        self.acc = True
        self.ppv = True
        self.sensi = True
        self.tnr = True
        self.npv = True
        self.auroc = True
        self.aurocPlot = False
        self.auprc = True
        self.auprcPlot = True
        self.callb = True
        self.callbPlot = False

    def forward(self, prob, labels, preds=None, gt_preds=None, train=True, mode="train", threshold=0.5):
        # Initialize variables
        auc, apr, base, accur, prec, recall, spec, npv_val, ECE, MCE, pred_loss = 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'

        prob = prob.float().to(self.device)
        labels = labels.float().to(self.device)

        #################           BCE Loss            #######################
        classify_loss = self.classify_loss(prob, labels)  # Compute BCE loss for the entire batch

        if train:
            return classify_loss  # If in training phase, return the loss

        #################           AUROC            #######################
        labels_np = labels.cpu().numpy()
        prob_np = prob.detach().cpu().numpy()

        if self.auroc:
            fpr, tpr, _ = metrics.roc_curve(labels_np, prob_np)
            auc = metrics.auc(fpr, tpr)
        if self.aurocPlot:
            self.auroc_plot(labels_np, prob_np)

        #################           AUPRC            #######################
        if self.auprc:
            base = ((labels_np == 1).sum()) / labels_np.shape[0]
            precision, recall, _ = metrics.precision_recall_curve(labels_np, prob_np)
            apr = metrics.auc(recall, precision)

        #################           Metrics Calculation            #######################
        prob_bin = prob_np >= threshold
        pos_p = (prob_bin & (labels_np == 1)).sum()  # True Positives (TP)
        neg_p = (~prob_bin & (labels_np == 0)).sum()  # True Negatives (TN)
        fn = (~prob_bin & (labels_np == 1)).sum()  # False Negatives (FN)
        fp = (prob_bin & (labels_np == 0)).sum()  # False Positives (FP)

        #################           Accuracy            #######################
        if self.acc:
            accur = metrics.accuracy_score(labels_np, prob_bin)

        #################           Precision/PPV            #######################
        if self.ppv:
            prec = metrics.precision_score(labels_np, prob_bin)

        #################           Recall/Sensitivity            #######################
        if self.sensi:
            recall = pos_p / (pos_p + fn) if (pos_p + fn) > 0 else 0

        #################           Specificity/TNR            #######################
        if self.tnr:
            spec = neg_p / (neg_p + fp) if (neg_p + fp) > 0 else 0

        #################           NPV            #######################
        if self.npv:
            npv_val = neg_p / (neg_p + fn) if (neg_p + fn) > 0 else 0

        #################           Calibration Metrics            #######################
        if self.callb:
            ECE, MCE = self.calb_metrics(prob_np, labels_np, self.callbPlot)

        #################           pred loss           #######################
        if preds is not None and gt_preds is not None:
            pred_loss = 0.0
            for pred, gt_pred in zip(preds, gt_preds):
                pred_loss += self.l2_loss(pred, gt_pred)
            #pred_loss = pred_loss * cfg.model.pred_loss
            total_loss = classify_loss + pred_loss * cfg.model.pred_loss
        # Output the results
        print("Mode: ", mode)
        print("BCE Loss: {:.2f}".format(classify_loss))
        if preds is not None and gt_preds is not None:
            print("PRED Loss: {:.2f}".format(pred_loss))
            print("Total Loss: {:.2f}".format(total_loss))
        print("AU-ROC: {:.2f}".format(auc))
        print("AU-PRC: {:.2f}".format(apr))
        print("AU-PRC Baseline: {:.2f}".format(base))
        print("Accuracy: {:.2f}".format(accur))
        print("Precision: {:.2f}".format(prec))
        print("Recall: {:.2f}".format(recall))
        print("Specificity: {:.2f}".format(spec))
        print("NPV: {:.2f}".format(npv_val))
        print("ECE: {:.2f}".format(ECE))
        print("MCE: {:.2f}".format(MCE))
        return classify_loss

    def auroc_plot(self, label, pred):
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'r--')
        fpr, tpr, _ = metrics.roc_curve(label, pred)
        auc = metrics.roc_auc_score(label, pred)
        plt.plot(fpr, tpr, label=f'Deep Learning Model, auc = {str(round(auc, 3))}')
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.title("AUC-ROC")
        plt.legend()
        plt.savefig('./data/output/auroc_plot.png')

    def calb_curve(self, bins, bin_accs, ECE, MCE):
        import matplotlib.patches as mpatches
        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1)
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        ax.set_axisbelow(True)
        ax.grid(color='gray', linestyle='dashed')
        plt.bar(bins, bins, width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')
        plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b')
        plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=2)
        plt.gca().set_aspect('equal', adjustable='box')
        ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE * 100))
        MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE * 100))
        plt.legend(handles=[ECE_patch, MCE_patch])
        plt.savefig('./data/output/callibration_plot.png')

    def calb_bins(self, preds, labels):
        num_bins = 10
        bins = np.linspace(0.1, 1, num_bins)
        binned = np.digitize(preds, bins)
        bin_accs = np.zeros(num_bins)
        bin_confs = np.zeros(num_bins)
        bin_sizes = np.zeros(num_bins)

        for bin in range(num_bins):
            bin_sizes[bin] = len(preds[binned == bin])
            if bin_sizes[bin] > 0:
                bin_accs[bin] = (labels[binned == bin]).sum() / bin_sizes[bin]
                bin_confs[bin] = (preds[binned == bin]).sum() / bin_sizes[bin]

        return bins, binned, bin_accs, bin_confs, bin_sizes

    def calb_metrics(self, preds, labels, curve):
        ECE, MCE = 0, 0
        bins, _, bin_accs, bin_confs, bin_sizes = self.calb_bins(preds, labels)

        for i in range(len(bins)):
            abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
            ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
            MCE = max(MCE, abs_conf_dif)

        if curve:
            self.calb_curve(bins, bin_accs, ECE, MCE)

        return ECE, MCE

