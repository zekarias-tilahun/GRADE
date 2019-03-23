from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing

from helpers import Const, sigmoid, row_dot

import numpy as np


def _deserialize_alg_name(name):
    if name == Const.LOGISTIC_REGRESSION:
        return LogisticRegression()
    elif name == Const.RANDOM_FOREST:
        pass
    elif name == Const.SVM:
        pass


class Estimator:

    def __init__(self, algorithm='log_reg', cv=None,
                 train_size=0.7, random_state=0, metrics=None):
        self._algorithm = _deserialize_alg_name(name=algorithm) if isinstance(algorithm, str) else algorithm
        self._cv = cv
        self._train_size = train_size
        self._random_state = random_state
        self._metrics = metrics
        self.results = {} if metrics is None else []

    def __classify_cv(self, features, labels):
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

    def __classify(self, features, labels):
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
            self.__classify(features=kwargs['features'], labels=kwargs['labels'])
        else:
            self.__classify_cv(features=kwargs['features'], labels=kwargs['labels'])


class ScoreEstimator(Estimator):

    def __init__(self, random_state=0, metrics=None, element_wise=False, use_labels=True):
        self._element_wise = element_wise
        self._use_labels = use_labels
        super().__init__(algorithm='', train_size=1., random_state=random_state, metrics=metrics)

    def __call__(self, **kwargs):
        if 'embeddings' in kwargs:
            emb = kwargs['embeddings']
            left_emb = emb[0] if isinstance(emb, tuple) else emb
            right_emb = emb[1] if isinstance(emb, tuple) else emb
            if 'left_nodes' in kwargs and 'right_nodes' in kwargs:
                left_indices = kwargs['left_nodes']
                right_indices = kwargs['right_nodes']
            else:
                left_indices = list(range(left_emb.shape[0]))
                right_indices = list(range(right_emb.shape[0]))

            size = len(left_indices)
            left_emb = left_emb[left_indices]
            right_emb = right_emb[right_indices]
            similarity_fun = row_dot if self._element_wise else np.dot
            probabilities = sigmoid(similarity_fun(left_emb, right_emb))
            if self._use_labels:
                lbl = kwargs['labels']
                labels = [lbl] * size
                return list(zip(probabilities, labels))
            return probabilities
        else:
            raise ValueError('The __call__ magic method expects the embeddings argument')
