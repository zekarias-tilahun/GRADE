from sklearn import metrics

import pandas as pd
import numpy as np


def _auc_score(y_true, y_hat, pos_label=1):
    fpr, tpr, _ = metrics.roc_curve(y_true=y_true, y_score=y_hat, pos_label=pos_label)
    return metrics.auc(fpr, tpr)


def _precision_at_k_values(probabilities, labels, k_values=None):
    sorted_probabilities = probabilities.sort_values(by='scores', ascending=False)
    dim = labels.ndim
    if dim == 1 or dim == 2:
        if k_values is None:
            return labels[labels == 1] / labels.size
        precision_aks = []
        for k in k_values:
            df_k = sorted_probabilities.iloc[:k]
            indices = df_k.loc.values if dim == 1 else (df_k.values[0], df_k.values[1])
            ordered_labels_k = labels[indices]
            pak = ordered_labels_k[ordered_labels_k == 1].size / k
            precision_aks += [{'k': k, 'value': pak}]
        return precision_aks
    else:
        raise ValueError(f"Expecting a 1d or 2d labels argument, but found a {dim}d array")


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

    def __compute(self, y_true, y_hat, probabilities, labels):
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
                p_aks = _precision_at_k_values(
                    probabilities=probabilities, labels=labels, k_values=self._k_values)
                self.scores += [{**{'metrics': 'Precision@K'}, **pak} for pak in p_aks]
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

    def __call__(self, y_true=None, y_hat=None, probabilities=None, labels=None):
        self.__compute(y_true=y_true, y_hat=y_hat, probabilities=probabilities, labels=labels)
