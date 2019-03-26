"""
Author: Zekarias Tilahun Kefato

Node Classification
====

This module provides utilities for evaluating the performance of
a graph embedding algorithm in node classification task.
It expects an embedding matrix, a label matrix, and an
estimator.Estimator object or a proper subclass that implements
its magic method `__call__`.
"""


from grade.metrics import Metrics, compile_metrics
from grade.estimators import Estimator
from grade.nrlio import Reader
from grade.helpers import *

import sys


def predict(embeddings, labels, estimator):
    """
    Predicts node labels using their embeddings

    Parameters
    ----

    embeddings: a numpy array
                Node embedding matrix
    labels: a numpy array
                Node labels
    estimator: estimator.Estimator or its subclass
                An estimator object
    :return: A dictionary
    """
    estimator(features=embeddings, labels=labels)
    return estimator.results


def main():
    if sys.argv[0] == 'tasks/node_classification.py' and len(sys.argv) > 1:
        parser = ArgParser(task=Const.NODE_CLASSIFICATION_TASK)
        options = parser.args
    else:
        options = ConfigParser(task=Const.NODE_CLASSIFICATION_TASK)
        
    reader = Reader(task=Const.NODE_CLASSIFICATION_TASK, options=options)
    metrics = Metrics(names=options.metric_names) \
        if options.metric_names is not None and len(options.metric_names) > 0 else None
    results = []
    for ts in options.train_size:
        estimator = Estimator(cv=options.cv, train_size=ts, metrics=metrics, random_state=options.seed)
        scores = predict(reader.embeddings, labels=reader.labels, estimator=estimator)
        results += [{**{'train_size': ts}, **score} for score in scores]
        
    if metrics is not None:
        by = ['train_size', 'metrics']
        return compile_metrics(results_lod=results, by=by, target='value', agg=options.cv > 1)
    else:
        return results


if __name__ == '__main__':
    print(main())
