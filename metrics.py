from sklearn import metrics

import pandas as pd
import numpy as np


def _auc_score(y_true, y_hat, pos_label=1):
    fpr, tpr, _ = metrics.roc_curve(y_true=y_true, y_score=y_hat, pos_label=pos_label)
    return metrics.auc(fpr, tpr)


def _precision_at_k_values(y_true, y_hat, pos_label=1, k_values=None):
    order = np.argsort(y_hat)
    if k_values is None:
        return metrics.precision_score(y_true=y_true, y_pred=y_hat, pos_label=pos_label, average='micro')
    precision_ak_ks = []
    for k in k_values:
        pak = metrics.precision_score(
            y_true=y_true[order][:k], y_pred=y_hat[order][:k], pos_label=pos_label, average='micro')
        precision_ak_ks += [{'k': k, 'value': pak}]
    return precision_ak_ks


def compile_metrics(results_lod, by, target, agg=False):
    df = pd.DataFrame(results_lod)
    if agg:
        return df.groupby(by)[target].agg([np.mean, np.std]).reset_index()
    else:
        return df


class Metrics:

    def __init__(self, names=None, pos_label=1, k_values=None):
        self._names = names
        self._pos_label = pos_label
        self._k_values = k_values
        self.scores = []

    def __compute(self, y_true, y_hat):
        for name in self._names:
            if name == 'acc':
                acc = metrics.accuracy_score(y_true=y_true, y_pred=y_hat)
                self.scores.append({'metrics': 'Accuracy', 'value': acc})
            elif name == 'pre':
                pre = metrics.precision_score(
                    y_true=y_true, y_pred=y_hat, pos_label=self._pos_label, average='micro')
                self.scores.append({'metrics': 'Precision', 'value': pre})
            elif name == 'rec':
                rec = metrics.recall_score(
                    y_true=y_true, y_pred=y_hat, pos_label=self._pos_label, average='micro')
                self.scores.append({'metrics': 'Recall', 'value': rec})
            elif name == 'pak':
                paks = _precision_at_k_values(
                    y_true=y_true, y_hat=y_hat, pos_label=self._pos_label, k_values=self._k_values)
                self.scores += [{**{'metrics': 'Precision@K'}, **pak} for pak in paks]
            elif name == 'mic':
                mic = metrics.f1_score(
                    y_true=y_true, y_pred=y_hat, pos_label=self._pos_label, average='micro')
                self.scores.append({'metrics': 'F1-Micro', 'value': mic})
            elif name == 'mac':
                mac = metrics.f1_score(
                    y_true=y_true, y_pred=y_hat, pos_label=self._pos_label, average='macro')
                self.scores.append({'metrics': 'F1-Macro', 'value': mac})
            elif name == 'auc':
                auc = _auc_score(y_true=y_true, y_hat=y_hat, pos_label=self._pos_label)
                self.scores.append({'metrics': 'AUC', 'value': auc})

    def __call__(self, y_true, y_hat):
        self.__compute(y_true=y_true, y_hat=y_hat)
