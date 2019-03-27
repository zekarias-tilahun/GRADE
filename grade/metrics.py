"""
Author: Zekarias Tilahun Kefato

The module provides wrappers around the sklearn metrics objects
"""
from sklearn import metrics

import pandas as pd
import numpy as np


def _auc_score(y_true, y_hat, pos_label=1):
    """
    A wrapper for the `sklearn.metrics.auc method`

    Parameters
    ----------
    y_true: The ground truth
    y_hat: The predicted value
    pos_label: The positive label

    Returns
    -------
    Evaluation score
    """
    fpr, tpr, _ = metrics.roc_curve(y_true=y_true, y_score=y_hat, pos_label=pos_label)
    return metrics.auc(fpr, tpr)


def _precision_at_k_values(probabilities, labels, k_values=None):
    """
    Compute the precision at k for different values of k

    Parameters
    ----------
    probabilities: pandas.DataFrame object
            It should have three columns, each row (tuple) (left, right, score) corresponds
            to an edge prediction between left node and right node with a score probability.

    labels: An nd-array
            It should be either a 1d-numpy array, then the i-th entry will be the true label
            for the i-th index of the probabilities data frame, or 2d-numpy array, which
            corresponds to the adjacency matrix.
    k_values:

    :return:
    """
    dim = labels.ndim
    assert dim == 1 or dim == 2, f"Expecting a 1d or 2d array for labels argument, but found a {dim}d array"
    sorted_probabilities = probabilities.sort_values(by='score', ascending=False)
    if k_values is None:
        return labels[labels == 1].shape[0] / labels.shape[0]
    precision_aks = []
    for k in k_values:
        df_k = sorted_probabilities.iloc[:k]
        indices = df_k.index if dim == 1 else (df_k.values[:, 0].astype(int).tolist(), 
                                               df_k.values[:, 1].astype(int).tolist())
        ordered_labels_k = labels[indices]
        pak = ordered_labels_k[ordered_labels_k == 1].size / k
        precision_aks += [{'k': k, 'value': pak}]
    return precision_aks


def compile_metrics(results_lod, by=None, target=None):
    """
    Performs a final pass on a list of dictionary object - results_lod, to convert it into a
    data frame. And if the result is the output of a k-fold cross validation experiment, then
    the data frame will be summarized depending on the columns specified using the `by`
    and `target` argument.

    Parameters
    ----------
    results_lod: A list of dictionary
        A list containing dictionary objects where each dictionary should have the same
        skeleton, that is each dictionary should use the same set and number of keys,
        the keys will be used as a column name.

    by: The list of columns to aggregate by

    target: The target value to be aggregated

    Returns
    -------
    A pandas.DataFrame

    """
    df = pd.DataFrame(results_lod)
    if by is None or target is None:
        return df
    else:
        return df.groupby(by)[target].agg([np.mean, np.std]).reset_index()


class Metrics:

    def __init__(self, names=None, pos_label=1, k_values=None):
        """
        A wrapper class to execute different evaluation metrics depending
        on the specified names

        Parameters
        ----------
        names: set, list
            The names of the set of evaluation metrics, The possible options are
                name    --  Description
                `acc`   --  Accuracy
                `pre`   --  Precision
                `rec`   --  Recall
                `pak`   --  Precision@k
                `mic`   --  F1-Micro
                `mac`   --  F1-Macro
                `auc`   --  AUC
        pos_label: int
            The positive label
        k_values: set, list
            The list of values, relevant when `pak` is in names

        """
        self._names = names
        self._pos_label = pos_label
        self._k_values = k_values
        self.scores = []

    def __compute(self, y_true, y_hat, probabilities, labels):
        """
        Carries out the actual execution of the specified evaluation metrics

        Parameters
        ----------
        y_true: 1d numpy array
            The ground truth, mostly used in classification tasks
        y_hat: 1d numpy array
            The corresponding prediction of the ground truth
        probabilities: pandas.DataFrame
            A data frame with three columns, where each row - (left, right, score)
            corresponds to an edge prediction between left node and right node
            with score probability
        labels: nd numpy array
            It should be either a 1d-numpy array, then the i-th entry will be the true label
            for the i-th index of the probabilities data frame, or 2d-numpy array, which
            corresponds to the adjacency matrix.

        """
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
        """
        A Metrics object functional wrapper around __compute

        Parameters
        ----------
        y_true: 1d numpy array
            The ground truth, mostly used in classification tasks
        y_hat: 1d numpy array
            The corresponding prediction of the ground truth
        probabilities: pandas.DataFrame
            A data frame with three columns, where each row - (left, right, score)
            corresponds to an edge prediction between left node and right node
            with score probability
        labels: nd numpy array
            It should be either a 1d-numpy array, then the i-th entry will be the true label
            for the i-th index of the probabilities data frame, or 2d-numpy array, which
            corresponds to the adjacency matrix.
        """
        self.__compute(y_true=y_true, y_hat=y_hat, probabilities=probabilities, labels=labels)
