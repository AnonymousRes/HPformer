import pickle
import logging
import pickle
import numpy as np
import sklearn.metrics
import torch
from torch.utils.data import Dataset
from sklearn import metrics

# Loss
def ihmp_predictor_loss(y_true, y_pred):
    ytr = y_true
    ypr = y_pred
    batchsize_ = y_true.size()[0]
    loss = ytr * torch.log(ypr + 1e-7) + (1 - ytr) * torch.log(1 - ypr + 1e-7)
    loss = torch.neg(torch.sum(loss)) / batchsize_
    return loss


def dp_predictor_loss(y_true, y_pred, mask_):
    ytr = y_true
    ypr = y_pred

    ymask = torch.unsqueeze(mask_, -1)
    masked_y = ypr * ymask

    batchsize_ = y_true.size()[0]
    loss = ytr * torch.log(masked_y + 1e-7) + (1 - ytr) * torch.log(1 - masked_y + 1e-7)
    loss = torch.sum(loss, dim=1) / (torch.sum(ymask, dim=1) + 1e-7)
    loss = torch.neg(torch.sum(loss)) / batchsize_
    return loss


def los_predictor_loss(y_true, y_pred, mask_):
    ytr = y_true
    ypr = y_pred

    ymask = torch.unsqueeze(mask_, -1)
    masked_y = ypr * ymask
    # batchsize_ = y_true.size()[0]
    # loss = ytr * torch.log(masked_y + 1e-7) + (1 - ytr) * torch.log(1 - masked_y + 1e-7)
    # loss = torch.sum(loss, dim=1) / (torch.sum(ymask, dim=1) + 1e-7)
    # loss = torch.neg(torch.sum(loss)) / batchsize_
    loss = torch.nn.CrossEntropyLoss(reduction='sum')(masked_y, ytr)
    loss = loss / (10 * (torch.sum(ymask) + 1e-7))
    return loss


# Metrics
# Modified from https://github.com/v1xerunt/StageNet/blob/master/utils/metrics.py

# for decompensation, in-hospital mortality
def print_metrics_binary(y_true, predictions, verbose=0):
    y_true = np.array(y_true)
    predictions = np.array(predictions)
    auroc = metrics.roc_auc_score(y_true, predictions)
    (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, predictions)
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

    if verbose:
        print("AUROC = {:.6f}".format(auroc))
        print("AUPRC = {:.6f}".format(auprc))
        print("Min(Se, P+) = {:.6f}".format(minpse))

    return {"AUROC": round(auroc, 6),
            "AUPRC": round(auprc, 6),
            "Min(Se, P+)": round(minpse, 6)}


# for length of stay
def print_metrics_multiclass(y_true, predictions, verbose=0):
    y_true = np.array(y_true)
    predictions = np.array(predictions)
    y_true_from_one_hot = np.array([np.argmax(one_hot) for one_hot in y_true])
    predictions_from_one_hot = np.array([np.argmax(one_hot) for one_hot in predictions])
    kappa = metrics.cohen_kappa_score(y_true_from_one_hot, predictions_from_one_hot, weights='linear')
    micro_auroc = metrics.roc_auc_score(y_true, predictions, average='micro', multi_class='ovr')
    macro_auroc = metrics.roc_auc_score(y_true, predictions, average='macro', multi_class='ovr')

    if verbose:
        print("Kappa = {}".format(kappa))
        print("Micro F1 = {}".format(micro_auroc))
        print("Macro F1 = {}".format(macro_auroc))

    return {"Kappa": kappa,
            "Micro AUROC": micro_auroc,
            "Macro AUROC": macro_auroc
            }


class EHR_Dataset(Dataset):
    def __init__(self, data_file):
        self.data_file = data_file
        data_dict_file = open(self.data_file, 'rb')
        data_dict = pickle.load(data_dict_file)
        data_dict_file.close()
        self.X = data_dict['X']
        self.MASK = data_dict['MASK']
        self.CUR_MASK = data_dict['CUR_MASK']
        self.INTERVAL = data_dict['INTERVAL']
        self.Y = data_dict['Y']


    def __len__(self):
        return len(self.X)


    def __getitem__(self, idx):
        x = self.X[idx]
        mask = self.MASK[idx]
        curmask = self.CUR_MASK[idx]
        interval = self.INTERVAL[idx]
        y = self.Y[idx]
        return x, mask, curmask, interval, y


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = 0.0
        self.early_stop = False
        self.delta = delta
        self.path = path
        # self.trace_func = trace_func
    def __call__(self, metric_, n_epoch, emodel):

        score = metric_

        if self.best_score is None:
            self.save_checkpoint(score, n_epoch, emodel)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}. ({self.best_score:.6f} --> {score:.6f})')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(score, n_epoch, emodel)
            self.counter = 0

    def save_checkpoint(self, score, n_epoch, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logging.info(f'Epoch:{n_epoch+1:d} Monitoring Metric ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.best_score = score
