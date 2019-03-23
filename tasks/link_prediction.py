"""
Author: Zekarias Tilahun Kefato

Link Prediction
===============

The following module is used for evaluating an embedding algorithm's
performance in link prediction. At least the inputs expected by the
`predict` method should be provided for a proper execution.

Following the most common practices in related literature, one can
specify, through `options.lp_method`, either of the following methods.

Link Prediction methods
  1. The first method computes a link probability between every pair of edges in the true and false edge collections. Then the the entire collection of edges will be sorted according to the predicted link probabilities. A good embedding algorithm should rank the true edges higher than the false edges.
  2. The second method builds features for edges using different feature construction techniques. One can use the existing edge feature builders in the `helpers.Builder` class or can use a custom class that extends this class and also implements its `custom_compile` method. Once features are built for the edges, a binary label will be assigned for each edge. That is, 1 if an edge is a true edge and 0 otherwise. Finally, a binary classifier is trained using the estimators.Estimator or any other custom object that is a subclass of the Estimator and implements its `__call__` magic method. Then depending on the cv and train_size options of this module a fraction of the edges and performance is reported on the quality of the classifiers prediction on the remaining fraction of edges.

Note on the embeddings: The number of embeddings per node can be one or two.
If the underlying graph is undirected, one embedding is sufficient. However,
for directed graphs two embeddings can be provided. For every node, there is
one embedding that encodes its outgoing neighborhood and another one that
encodes its incoming neighborhood. Therefore, for a given candidate directed
edge (u, v), the prediction is performed by using u's outgoing embedding and
v's incoming embedding. Consequently predictions are asymmetric, the prediction
for (u, v) is different from (v, u). Nonetheless, this is not mandatory and
one can simply provide a single embedding, and hence the link prediction does
not entail directionality. In other words a predict edge is assumed to be
symmetric.

Code snippets are indicated using `>>>`
"""

from estimators import Estimator, ScoreEstimator
from metrics import Metrics, compile_metrics
from helpers import *
from nrlio import *

import numpy as np

import sys


def _construct_features(left_nodes, right_nodes, embeddings, builders):
    """
    Builds a symmetric or an asymmetric features using the embeddings of the
    left_nodes and right_nodes. If embeddings is a numpy array symmetric
    features will be constructed, otherwise if a tuple of two numpy arrays
    is specified, then asymmetric features are constructed.
    That is, for an edge (u, v) the edge feature (u, v) and (v, u) are the
    same in the case of symmetric construction. However the two features
    different in the asymmetric construction.

    The feature building mechanism should be specified using the builders
    dictionary, where each key, value pair corresponds to the name and
    definition of the builder function, respectively. 


    Parameters
    ----------
    left_nodes : list, numpy array
                Indices of left nodes
    right_nodes : list, numpy array
                Indices of the right nodes
    embeddings : numpy array, a tuple of two numpy arrays
                An array is the embedding of nodes
    builders : dict
                A dictionary of builders, where the key is the name and value
                is the definition of the builder function.

    :return: A dictionary of edge features, keys are the name of the function used to
    construct edge features, and values are the constructed feature matrices.

    Example
    -------
        >>> builders = {'element_wise': lambda l: l[0] * l[1], 'avg': lambda l: (l[0] + l[1]) / 2}
        ... # The keys element_wise and avg will also be the keys used for the returned dictionary
    """
    left_embedding = (embeddings[0] if isinstance(embeddings, tuple) else embeddings)[left_nodes]
    right_embedding = (embeddings[1] if isinstance(embeddings, tuple) else embeddings)[right_nodes]
    features = {}
    for name, builder in builders.items():
        features[name] = builder([left_embedding, right_embedding])
    return features


def _predict_score_based(true_edges, false_edges, embeddings, estimator):
    """
    Predicts symmetric or asymmetric link probabilities between the pair of nodes
    in the true_edges and false_edges samples using their embeddings.
    If embeddings is a numpy array the prediction is symmetric, otherwise if it
    is a tuple of two numpy arrays then the prediction is asymmetric.

    Parameters
    ----------
    true_edges : list, set
                True edge samples
    false_edges : list, set
                False edge samples
    embeddings : numpy array, a tuple of two numpy arrays
                An array is the embedding of nodes
    estimator : estimator.ScoreEstimator
                A score based estimator

    :return: A dictionary

    """
    true_left_nodes, true_right_nodes = unzip_edges(true_edges)
    false_left_nodes, false_right_nodes = unzip_edges(false_edges)

    results_true = estimator(embeddings=embeddings, left_nodes=true_left_nodes, right_nodes=true_right_nodes, labels=1)
    results_false = estimator(embeddings=embeddings, left_nodes=false_left_nodes, right_nodes=false_right_nodes,
                              labels=0)
    results = results_true + results_false
    probs, y_true = list(zip(*sorted(results, key=lambda l: l[0], reverse=True)))
    return {'y_true': y_true, 'probs': probs}


def _predict_feature_based(true_edges, false_edges, embeddings, builder, estimator):
    """
    Predicts link samples in the true_edges and false_edges by using node
    embeddings. It leverages a helper.Builder or its subclass object and 
    the estimators.Estimator object.

    Parameters
    ----
    true_edges : list, set
                True edge samples
    false_edges : list, set
                False edge samples
    embeddings : numpy array, a tuple of two numpy arrays
                An array is the embedding of nodes
    builder: helper.Builder or its subclass
                If a subclass is used it should implement the custom_compile method,
                which should return a dictionary where the keys and values are the
                names and definitions of the edge feature builder functions
    estimator : estimator.ScoreEstimator
                A score based estimator

    :return: A dictionary.
    """
    true_left_nodes, true_right_nodes = unzip_edges(true_edges)
    false_left_nodes, false_right_nodes = unzip_edges(false_edges)
    edge_feature_builders = builder.compile()
    true_edge_features = _construct_features(
        left_nodes=true_left_nodes, right_nodes=true_right_nodes,
        embeddings=embeddings, builders=edge_feature_builders)
    false_edge_features = _construct_features(
        left_nodes=false_left_nodes, right_nodes=false_right_nodes,
        embeddings=embeddings, builders=edge_feature_builders)

    results = []
    for name in true_edge_features:
        current_true_features = true_edge_features[name]
        true_edge_labels = np.ones(current_true_features.shape[0])
        current_false_features = false_edge_features[name]
        false_edge_labels = np.zeros(current_false_features.shape[0])
        edge_features = np.concatenate([current_true_features, current_false_features], axis=0)
        edge_labels = np.concatenate([true_edge_labels, false_edge_labels])
        estimator(features=edge_features, labels=edge_labels)
        results += [{**{'feature_builder': name}, **result} for result in estimator.results]
    return results


def predict(true_edges, false_edges, embeddings, options, builder=None):
    """
    Executes link prediction according to the specified options

    Parameters
    ----
    true_edges : list, set
                True edge samples
    false_edges : list, set
                False edge samples
    embeddings : numpy array, a tuple of two numpy arrays
                An array is the embedding of nodes
    options : A helpers.ConfigParser or helpers.ArgParser, or any other object
                that provides the required options
    builder: helper.Builder or its subclass
                If a subclass is used it should implement the custom_compile method,
                which should return a dictionary where the keys and values are the
                names and definitions of the edge feature builder functions
    :return: A list of dictionaries or a dictionary
    """
    results = []
    metrics = Metrics(names=options.metric_names, k_values=options.k_values) \
        if options.metric_names is not None and len(options.metric_names) > 0 else None

    if options.lp_method == Const.FEATURE_BASED_LP_METHOD:
        print('Feature based link prediction')
        for ts in options.train_size:
            estimator = Estimator(
                algorithm=Const.LOGISTIC_REGRESSION, cv=options.cv, train_size=ts,
                random_state=options.seed, metrics=metrics)
            outputs = _predict_feature_based(
                true_edges=true_edges, false_edges=false_edges,
                embeddings=embeddings, builder=builder, estimator=estimator)
            results += [{**{'train_size': ts}, **output} for output in outputs]
    elif options.lp_method == Const.SCORE_BASED_LP_METHOD:
        print('Score based link prediction')
        estimator = ScoreEstimator(
            train_size=options.train_size, random_state=options.seed, metrics=metrics)
        results = _predict_score_based(
            true_edges=true_edges, false_edges=false_edges,
            embeddings=embeddings, estimator=estimator)
    return results


def main():
    if sys.argv[0] == 'tasks/node_classification.py' and len(sys.argv) > 1:
        parser = ArgParser(task=Const.LINK_PREDICTION_TASK)
        options = parser.args
    else:
        options = ConfigParser(task=Const.LINK_PREDICTION_TASK)

    reader = Reader(task=Const.LINK_PREDICTION_TASK, options=options)
    builder = Builder(names=options.feature_builders)
    results = predict(
        true_edges=reader.true_edges, false_edges=reader.false_edges,
        embeddings=reader.embeddings, builder=builder, options=options)
    if options.lp_method == Const.FEATURE_BASED_LP_METHOD and \
            options.metric_names is not None and len(options.metric_names) > 0:
        if 'pak' in options.metric_names:
            by = ['train_size', 'metrics', 'feature_builder', 'k']
        else:
            by = ['train_size', 'metrics', 'feature_builder']
        return compile_metrics(results_lod=results, by=by, target='value', agg=options.cv > 1)
    else:
        return results


if __name__ == '__main__':
    main()
