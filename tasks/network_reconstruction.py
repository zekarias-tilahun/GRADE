"""
Author: Zekarias Tilahun Kefato

Network Reconstruction
===============

This module provides a utility for evaluating the performance of a
graph embedding algorithm in effectively capturing the structural
property of the network by reconstructing the network from the
learned embeddings.

Depending on the the size of the network one can choose to do a full
network reconstruction or a batch network reconstruction.
For both cases the reconstruction could be symmetric or asymmetric
depending on the number of embedding matrices provided. That is,
symmetric reconstruction will be carried out if one embedding
matrix is specified, and an asymmetric one will be carried out if
two matrices are provided.

Note that at least an embedding matrix should be provided to execute
this module by calling the `reconstruct` method
"""

from estimators import ScoreEstimator
from nrlio import Reader
from helpers import *

import pandas as pd
import numpy as np

import sys


def _full_reconstruction(embeddings, estimator, ground_truth=None):
    """
    Network reconstruction computed based on the similarity between
    every pair of nodes (n*n - run-time), where the similarity
    between the nodes is computed as a function of the dot product
    of the embeddings of each pair of nodes using sigmoid:

    Parameters
    ----
    embeddings : A numpy array or a tuple of two numpy arrays
                An array should be (n x d)
                    n - the number of nodes and

                    d - the embedding dimension
                    
                    
    :return: list: A list of edges along with their weights estimated using the
                similarity between the embedding of the nodes

    Reconstruction Strategy
    ----
    >>> sim = sigmoid(np.dot(embeddings, embeddings.T))
    ... # For symmetric reconstruction
    >>> forward_sim = sigmoid(np.dot(embeddings[0], embeddings[1].T))
    >>> backward_sim = sigmoid(np.dot(embeddings[1], embeddings[0].T))
    ... # For asymmetric reconstruction

    """
    estimator(embeddings=embeddings, labels=ground_truth)
    return [estimator.results]

def _batch_reconstruction(embeddings, ground_truth, estimator, batch_size, threshold=0.7):
    """
    Batch network reconstruction; useful when the number of nodes is large.
    For every batch the similarity between the nodes in the batch with the
    rest of the nodes is computed (number_of_batches * batch_size * num_nodes
    - run-time). And only those pair of nodes above a given similarity threshold
    are kept and the rest is discarded.
    The similarity between nodes u and v is computed as a function of the
    dot product of the embeddings of u and v using sigmoid:
                sim = sigmoid(dot(embedding(u), embedding(v).T)).

    Parameters
    ----
    embeddings : A numpy array or a tuple of two numpy arrays
                An array should be (n x d)
                    n - the number of nodes and

                    d - the embedding dimension
    batch_size : int
                Batch size
    threshold : float
                A threshold for pruning

    :return: list,
                A list of edges along with their weight estimated using the
                similarity between the embedding of the nodes
    """

    def delegate(source_emb, target_emb):
        num_nodes = source_emb.shape[0]
        partial_reconstruction = []
        estimator.skip_eval(True)
        for i in range(0, source_emb.shape[0], batch_size):
            end = i + batch_size if num_nodes - batch_size > i else num_nodes
            batch_nodes = list(range(i, end))
            batch_embedding = source_emb[batch_nodes]
            estimator(embeddings=(batch_embedding, target_emb), labels=ground_truth)
            filtered_batch = estimator.results[estimator.results[:, 2] > threshold]
            if len(partial_reconstruction) == 0:
                partial_reconstruction = filtered_batch
            else:
                partial_reconstruction = np.concatenate([partial_reconstruction, filtered_batch])
            
        return partial_reconstruction

    if isinstance(embeddings, tuple):
        print(f'Running asymmetric network reconstruction in batches of {batch_size} '
              f'and a threshold {threshold}')
        source_target_reconstruction = delegate(embeddings[0], embeddings[1])
        target_source_reconstruction = delegate(embeddings[1], embeddings[0])
        reconstruction = np.concatenate([source_target_reconstruction, target_source_reconstruction])
    else:
        print(f'Running symmetric network reconstruction in batches of {batch_size} '
              f'and a threshold {threshold}')
        reconstruction = delegate(embeddings, embeddings)
    if estimator._metrics is not None:
        estimator.skip_eval(False)
        df = pd.DataFrame(reconstruction, columns=['left', 'right', 'score'])
        estimator._metrics(probabilities=df, labels=ground_truth)
        return estimator._metrics.scores
    return reconstruction


def reconstruct(embeddings, ground_truth, options):
    """
    Symmetric or asymmetric network reconstruction. If embeddings is
    a numpy array, the reconstruction is symmetric, i.e. the reconstruction
    is undirected. If embeddings is a tuple of numpy arrays, then reconstruction
    is asymmetric, and hence directed.

    Parameters
    ----------
    embeddings : An embedding array of (n x d), or a tuple of embedding arrays
                of (n x d) each.
    ground_truth : scipy csr sparse matrix or numpy dense matrix
                The ground truth adjacency matrix for evaluation, this becomes
                optional if the options argument has not metric names.
    options : config options
    
    Returns
    -------
    """
    metrics = Metrics(names=options.metric_names, k_values=options.k_values) \
        if options.metric_names is not None and len(options.metric_names) > 0 else None
    def filter_edges(recon):
        indices = np.where(recon > threshold)
        return list(zip(indices[0], indices[1], recon[indices]))
    estimator = ScoreEstimator(element_wise=False, metrics=metrics)
    if batch_size is None or batch_size == 0:
        link_probabilities = _full_reconstruction(
            embeddings=embeddings, estimator=estimator, ground_truth=ground_truth)
        if len(link_probabilities) == 2:
            edges = filter_edges(link_probabilities[0]) + filter_edges(link_probabilities[1])
        else:
            edges = filter_edges(link_probabilities)
    else:
        edges = _batch_reconstruction(
            embeddings=embeddings, estimator=estimator,
            batch_size=batch_size, threshold=threshold)
    
    return sorted(edges, key=lambda l: l[2], reverse=True)


def main():
    """
    Program entry point

    :return:
    """
    if len(sys.argv) > 1:
        parser = ArgParser(task=Const.NET_RECONSTRUCTION_TASK)
        options = parser.args
    else:
        parser = ConfigParser(task=Const.NET_RECONSTRUCTION_TASK)
        options = parser
    reader = Reader(task=Const.NET_RECONSTRUCTION_TASK, options=options)
    reconstruction = reconstruct(embeddings=reader.embeddings, ground_truth=reader.network, options=options)
    return reconstruction


if __name__ == '__main__':
    main()
