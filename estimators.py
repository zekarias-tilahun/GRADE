"""
Author: Zekarias Tilahun Kefato <zekarias.kefato@unitn.it>

This module defines a set of Estimators that can be used
in classification and regression tasks. Any class that
inherits from the classes defined here should implement
the `__call__` magic method.
"""
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing

from helpers import Const, sigmoid, row_dot

import pandas as pd
import numpy as np


def _deserialize_alg_name(name):
    """
    A utility to deserialize a string algorithm name to the corresponding
    sklearn model.

    :param name: The algorithm name
    :return: An sklearn model
    """
    if name == Const.LOGISTIC_REGRESSION:
        return LogisticRegression()
    elif name == Const.RANDOM_FOREST:
        pass
    elif name == Const.SVM:
        pass


class Estimator:

    def __init__(self, algorithm='log_reg', cv=None,
                 train_size=0.7, random_state=0, metrics=None):
        """
        A basic estimator for a classification task

        Parameters
        ----
        algorithm: string, or a valid classifier from sklearn, an sklearn
            like model for classification. If a custom model is provided, it should
            have a method - `fit` for training and inference - `predict`.
        cv: int
            The number of folds for a k-fold cross-validation experiment. If it is
            None or < 2, then a one-shot training and inference will be carried out

        train_size: float
            The fraction of the data points to be used as a training set

        random_state: int
            A seed for the pseudo-random number generator

        metrics: A metrics.Metrics object
            If evaluation of the estimators performance is desired, then
            this should not be None.
        """
        self._algorithm = _deserialize_alg_name(name=algorithm) if isinstance(algorithm, str) else algorithm
        self._cv = cv
        self._train_size = train_size
        self._random_state = random_state
        self._metrics = metrics
        self.results = {} if metrics is None else []

    def __estimate_cv(self, features, labels):
        """
        Trains an estimator model using a fraction of the features and labels, according
        the specified training size. The model is then used to predict the labels of the
        test set. This procedure is carried out using a k-fold cross-validation technique.

        If a metrics object is passed to the estimator, then a score will be computed
        using the test set labels as a ground truth, otherwise the ground-truth test
        labels and the corresponding predicted labels will be returned.

        Parameters
        ---------
        features : numpy array
                    The features
        labels : numpy array
                    The labels

        Returns
        -------
        A dictionary or a list of dictionaries

        See Also
        --------
        __estimate : Similar function without cross-validation

        """
        ss_iter = ShuffleSplit(
            self._cv, test_size=1 - self._train_size)
        for train_index, test_index in ss_iter.split(features):
            x_train, x_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            pipe = make_pipeline(preprocessing.Normalizer(), self._algorithm)
            y_hat = pipe.fit(X=x_train, y=y_train).predict(x_test)
            if self._metrics is None:
                self.results['y_true'] = y_test
                self.results['y_hat'] = y_hat
            else:
                self._metrics(y_true=y_test, y_hat=y_hat)
                self.results = self._metrics.scores

    def __estimate(self, features, labels):
        """
        Trains an estimator model using a fraction of the features and labels, according
        the specified training size. The model is then used to predict the labels of the
        test set. If a metrics object is passed to the estimator, then a score will be
        computed using the test set labels as a ground truth, otherwise the test labels
        and the predict labels will be returned.

        Parameters
        ----------
        features : The features
        labels : The labels

        Returns
        -------
        A dictionary or a list of dictionaries

        See Also
        --------
        __estimate_cv : A similar function with k-fold cross validation

        """
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, train_size=self._train_size, test_size=1 - self._train_size,
            random_state=self._random_state)
        self._algorithm.fit(X=x_train, y=y_train)
        y_hat=self._algorithm.predict(X=x_test)
        if self._metrics is None:
            self.results = {'y_true': y_test, 'y_hat': y_hat}
        else:
            self._metrics(y_true=y_test, y_hat=y_hat)
            self.results = self._metrics.scores

    def __call__(self, **kwargs):
        if self._cv is None or self._cv < 2:
            self.__estimate(features=kwargs['features'], labels=kwargs['labels'])
        else:
            self.__estimate_cv(features=kwargs['features'], labels=kwargs['labels'])


class ScoreEstimator(Estimator):

    def __init__(self, random_state=0, metrics=None, element_wise=False):
        """
        A basic estimator for regression task

        Parameters
        ----------
        random_state :
        metrics : A metrics.Metrics object
        element_wise : bool, if true a row-wise otherwise a matrix multiplication
            will be used to estimate scores
        """
        self._element_wise = element_wise
        super().__init__(algorithm='', train_size=1., random_state=random_state, metrics=metrics)

    def __call__(self, **kwargs):
        """
        Parameters
        ----------
        embeddings : An embedding matrix, this should be specified
        left_nodes : A list of indices to the embedding matrix, optional.
        right_nodes : A list of indices to the embedding matrix, optional.
        labels: If the left_nodes and right_nodes are provided, then labels
            should specify the label associated to each pair of left_node,
            right_node. Otherwise, Should be a sparse or dense adjacency
            matrix like numpy array.

        :return:
        """
        if 'embeddings' in kwargs:
            if self._metrics is not None and 'labels' in kwargs:
                emb = kwargs['embeddings']
                labels = kwargs['labels']
                left_emb = emb[0] if isinstance(emb, tuple) else emb
                right_emb = emb[1] if isinstance(emb, tuple) else emb
                if 'left_nodes' in kwargs and 'right_nodes' in kwargs:
                    left_indices = kwargs['left_nodes']
                    right_indices = kwargs['right_nodes']
                else:
                    left_indices = list(range(left_emb.shape[0]))
                    right_indices = list(range(right_emb.shape[0]))

                left_emb = left_emb[left_indices]
                right_emb = right_emb[right_indices]
                similarity_fun = row_dot if self._element_wise else np.dot
                probabilities = list(zip(
                    left_indices, right_indices, sigmoid(similarity_fun(left_emb, right_emb))))
                if self._metrics is None:
                    self.results = probabilities
                else:
                    probabilities = pd.DataFrame(probabilities, columns=['left', 'right', 'score'])
                    self._metrics(probabilities=probabilities, labels=labels)
            else:
                raise ValueError("The argument labels can not be empty when a metrics object is specified")
        else:
            raise ValueError('The __call__ magic method expects the embeddings and labels argument')
