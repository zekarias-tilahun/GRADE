"""
Author: Zekarias Tilahun Kefato

This module provides the necessary helper utilities. These includes, Parser
classes to read options from a command line or a config.ini file.
In addition it contains a Builder class for specifying edge feature construction
methods and a few other helper functions for numerical and analytical operations.
"""

import pandas as pd
import numpy as np

import configparser
import argparse


class Const:

    NET_SAMPLING_TASK = 0
    NET_RECONSTRUCTION_TASK = 1
    LINK_PREDICTION_TASK = 2
    NODE_CLASSIFICATION_TASK = 3

    FEATURE_BASED_LP_METHOD = 'f'
    SCORE_BASED_LP_METHOD = 's'

    EMB_FILE_NAME_KEY = 'unified'
    LEFT_EMB_FILE_NAME_KEY = 'outgoing'
    RIGHT_EMB_FILE_NAME_KEY = 'incoming'

    LOGISTIC_REGRESSION = 'log_reg'
    RANDOM_FOREST = 'forest'
    SVM = 'svm'
    

class Builder:

    def __init__(self, names=None, custom=False):
        """
        A helper class for compiling a set of predefined edge feature building
        methods

        Parameters
        ----------
        names : set, list
                A set of names of the feature builder functions
        custom: bool
                It should be set to True whenever it's extended by a custom
                feature builder to specify a desired set of feature builders.
                Note that any subclass of Builder also have to implement
                the `_custom_compile` method in addition to setting this
                argument to True, and the method should return a dictionary
                where the keys are the names and values are the definitions
                of the feature builder functions.
        """
        self._names = names
        self._custom = custom

    def __compile(self):
        """
        A method for compiling the set of predefined feature builder functions

        """
        builders = {}
        for name in self._names:
            if name == 'hd' or name == 'all':
                builders['Hadamard'] = lambda l: l[0] * l[1]
            if name == 'avg' or name == 'all':
                builders['Average'] = lambda l: (l[0] + l[1]) / 2
            if name == 'wl1' or name == 'all':
                builders['Weighted-L1'] = lambda l: np.abs(l[0] - l[1])
            if name == 'wl2' or name == 'all':
                builders['Weighted-L2'] = lambda l: np.square(l[0] - l[1])
            if name not in {'hd', 'avg', 'wl1', 'wl2', 'all'}:
                raise ValueError(f"Found unexpected feature builders option 'builders: {name}' while expecting either "
                                 f"hd, avg, wl1, wl2, all.")
        return builders

    def _custom_compile(self):
        """
        The implementation of this method should return a dictionary, where
        the keys are the names and values are the definitions of the feature
        builder functions.

        Note that the subclass which implements this method should set
        custom=True when invoking the constructor (__init__) of the superclass
        as:

            >>> super().__init__(names={'name_1', 'name_2'}, custom=True)
            ... # names is optional

        :return: dict
        """
        pass
        
    def compile(self):
        """
        A wrapper around the `__compile` and `_custom_compile` methods

        """
        if self._custom:
            return self._custom_compile()
        else:
            return self.__compile()


class ArgParser:

    def __init__(self, task):
        """
        A parser for options provided via command line (terminal)

        Parameters
        ----------
        task : string
                The identifier of the desired task, check out
                the Const class to see the existing tasks
        """
        if task == Const.NET_SAMPLING_TASK:
            self._args = _net_sampling_parse_args()
        elif task == Const.NET_RECONSTRUCTION_TASK:
            self._args = _net_reconstruction_parse_args()
        elif task == Const.LINK_PREDICTION_TASK:
            self._args = _link_prediction_parse_args()
        elif task == Const.NODE_CLASSIFICATION_TASK:
            self._args = _node_classification_parse_args()

    @property
    def args(self):
        return self._args


class ConfigParser:

    def __init__(self, task, path='config.ini'):
        """
        A parser for options provided via config file

        Parameters
        ----------
        task : string
                The identifier of the desired task, check out
                the Const class to see the existing tasks
        path : string
                A path to the config.ini file
        """
        self._task = task
        self._config = configparser.ConfigParser()
        self._config.read(path)
        self.__parse()
        
    def has_metrics(self):
        """
        Checks if a set of metric names have been provided

        Returns
        -------
        True metric_names is defined, otherwise False
        """
        return self.metric_names is not None and len(self.metric_names) > 0

    def __graph_args(self):
        """
        Options for reading and writing a graph

        """
        self.net_file = self._config.get(section='graph-args', option='net_file')
        self.net_format = self._config.get(section='graph-args', option='net_format')
        self.directed = self._config.getboolean(section='graph-args', option='directed')
        self.weighted = self._config.getboolean(section='graph-args', option='weighted')

    def __embedding_args(self):
        """
        Options for reading and writing embeddings

        """
        file_patterns = self._config.get(section='embedding-args', option='emb_file').strip().split()
        formats = self._config.get(section='embedding-args', option='emb_format').strip().split()
        if len(file_patterns) == 1 and len(formats) == 1:
            self.emb_file = file_patterns[0]
            self.emb_format = formats[0]
        elif len(file_patterns) == 1 and len(formats) > 1:
            if formats[0] == 'npz':
                self.emb_file = file_patterns[0]
                self.emb_format = {'format': formats[0], 'keys': [key for key in formats[1:]]}
            else:
                raise ValueError(f"Unfortunately, the emb_file pattern {file_patterns} and "
                                 f"the format {formats} combination is not supported yet. "
                                 f"It looks like you want to read multiple files compressed together, "
                                 f"if so, use the format npz followed by the list of keys for each "
                                 f"file. Othwerise use multiple file path patterns and multiple formats "
                                 "associated to each pattern.")
        else:
            self.emb_file = file_patterns
            self.emb_format = formats

    def __edge_sample_args(self):
        """
        Options for reading and writing sampled edges

        """
        self.pos_file = self._config.get(section='edge-sample-args', option='pos_file')
        self.neg_file = self._config.get(section='edge-sample-args', option='neg_file')

    def __net_sampling_args(self):
        """
        Options for carrying out network sampling

        """
        self.res_file = self._config.get(section='net-sampling-args', option='res_file')
        self.rate = self._config.getfloat(section='net-sampling-args', option='rate')
        self.hard_neg = self._config.getboolean(section='net-sampling-args', option='hard_neg')
        self.__graph_args()
        self.__edge_sample_args()

    def __eval_metric_args(self):
        """
        Options for evaluation

        """
        self.metric_names = self._config.get(section='eval-metric-args', option='metrics').strip().split()
        self.k_values = [int(op) for op in
                         self._config.get(section='eval-metric-args', option='k_values').strip().split()]

    def __hyper_param_args(self):
        """
        Options for specifying hyper-parameters

        """
        self.train_size = [float(ts) for ts in
                           self._config.get(section='hyper-param-args', option='train_size').strip().split()]
        self.seed = self._config.getint(section='hyper-param-args', option='seed')
        self.cv = self._config.getint(section='hyper-param-args', option='cv')

    def __net_reconstruction_args(self):
        """
        Options for carrying out network reconstruction experiment

        """
        self.threshold = self._config.getfloat(section='net-reconstruction-args', option='threshold')
        self.batch_size = self._config.getint(section='net-reconstruction-args', option='batch_size')
        self.__graph_args()
        self.__embedding_args()
        self.__eval_metric_args()

    def __link_prediction_args(self):
        """
        Options for carrying out link prediction experiment

        """
        self.lp_method = self._config.get(section='link-prediction-args', option='lp_method')
        self.feature_builders = self._config.get(section='link-prediction-args', option='builders').strip().split()
        self.__embedding_args()
        self.__edge_sample_args()
        self.__eval_metric_args()
        self.__hyper_param_args()

    def __node_classification_args(self):
        """
        Options for carrying out node classification experiment

        """
        self.label_file = self._config.get(section='node-classification-args', option='label_file')
        self.__embedding_args()
        self.__eval_metric_args()
        self.__hyper_param_args()

    def __parse(self):
        """
        A wrapper for reading options

        """
        if self._task == Const.NET_SAMPLING_TASK:
            self.__net_sampling_args()
        elif self._task == Const.NET_RECONSTRUCTION_TASK:
            self.__net_reconstruction_args()
        elif self._task == Const.LINK_PREDICTION_TASK:
            self.__link_prediction_args()
        elif self._task == Const.NODE_CLASSIFICATION_TASK:
            self.__node_classification_args()


def sort_array(array, by_column=0, ascending=True):
    """
    Sorts an array by a given column in ascending or descending order

    Parameters
    ----------
    array : numpy array
            The array
    by_column : int
            A column index to sort the array by
    ascending : bool
            Whether the sort should be in ascending order or not

    Returns
    -------
    An array sorted by a given column
    """
    arr = array[array[:, by_column].argsort()]
    return arr if ascending else arr[::-1]


def unzip_edges(edges):
    """
    Deconstructs a list of edges (pairs) into a tuple of a list of left nodes
    and list of right nodes.

    :param edges:
    :return: tuple: a tuple of two lists
    """
    left_nodes, right_nodes = list(zip(*edges))
    return list(left_nodes), list(right_nodes)


def sigmoid(z):
    """
    Applies a sigmoid function to the input z

    Parameters
    ----------
    z : a scalar value or an nd-array

    Returns
    -------
    A scalar value or an nd-array depending on the input
    """
    return 1 / (1 + np.exp(-z))


def dot(x, y):
    """
    Computes the dot product between x, y transpose, nd-arrays.
    x and y should have the same kind of shape

    Parameters
    ----------
    x : nd-array
        A numpy array or scipy sparse matrix
    y : nd-array
        A numpy array or scipy sparse matrix

    Returns
    -------
    A scalar value or a nd-array

    See Also
    --------
    row_dot: A row wise dot product
    """
    return np.dot(x, y.T)


def row_dot(x, y):
    """
    Computes a row wise dot product between x and y, nd-arrays. Which is
    similar to:
        
            >>> for i in range(x.shape[0]): np.dot(x[i], y[i])

    Parameters
    ----------
    x : nd-array
        A numpy array or scipy sparse matrix
    y : nd-array
        A numpy array or scipy sparse matrix
    
    Returns
    -------
    A 1d numpy array
    
    See Also
    --------
    dot: A matrix multiplication between x and y
    """
    msg = "The number of rows for matrix x and y should be the same"
    assert x.shape[0] == y.shape[0], msg
    return np.einsum("ij,ij->i", x, y)


def _add_network_args(parser):
    """
    Add network file related options to a command line argument parser.

    :param parser: The parse object
    :return:
    """
    parser.add_argument('--net-file', type=str, default='',
                        help='A path to read a network file')
    parser.add_argument('--net-format', type=str, default='edgelist',
                        help='The network file format; Possible values are '
                             '(edgelist | adjlist | mattxt | matnpy). '
                             'Default is edgelist')
    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is directed, Default is True')
    parser.add_argument('--undirected', dest='directed', action='store_false',
                        help='Graph is undirected')
    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Graph is directed, Default is False')
    parser.add_argument('--unweighted', dest='weighted', action='store_false',
                        help='Graph is undirected')
    parser.set_defaults(directed=True)
    parser.set_defaults(weighted=False)


def _add_embedding_args(parser):
    """
    Adds embedding related options to a command line argument parser

    :param parser: The parser
    :return:
    """
    parser.add_argument('--emb-file', type=str, default='',
                        help='A path to node embeddings file')
    parser.add_argument('--emb-format', type=str, default='npz',
                        help='The format of the embedding file. '
                             'Possible values are (npy - numpy binary file |'
                             'w2v - word2vec format, npz). Default is npz')


def _add_metrics_args(parser):
    """
    Adds evaluation metric related options to a command line argument parser

    :param parser: The parser
    :return:
    """
    parser.add_argument('--metrics', type=str, nargs='*',
                        help='List of evaluation metrics, supported values are '
                             '(pk - precision at k | pre - precision, |'
                             'rec - recall | auc - area under the curve | '
                             'acc - accuracy | mic - micro-f1 | mac - macro-f1). '
                             'Default is [pk, auc, mic, mac]')
    parser.add_argument('--k-values', type=int, nargs='*',
                        help='List of k values if precision at k is specified '
                             'among the evaluation metrics (--metrics arg). '
                             'Default is [100, 1000, 10000, 50000, 100000]')


def _add_train_hyper_param_args(parser):
    """
    Adds generic training hyper parameters related options to a command
    line argument parser

    :param parser: The parser
    :return:
    """
    parser.add_argument('--trs', type=float, nargs='*',
                        help='A list of training ratios. '
                             'Default is [.1, .3, .5]')
    parser.add_argument('--vr', type=float, default=0.,
                        help='A fraction of the training set to be used as'
                             'a validation set')


def _net_sampling_parse_args():
    """
    Prepares command line arguments relevant for network sampling

    :return: parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    _add_network_args(parser)
    parser.add_argument('--pos-file', type=str, default='',
                        help='A path to save the positive samples')
    parser.add_argument('--neg-file', type=str, default='',
                        help='A path to save the negative samples')
    parser.add_argument('--res-file', type=str, default='',
                        help='A path to save the residual network')
    parser.add_argument('--rate', type=str, default='',
                        help='The sampling rate ')
    return parser.parse_args()


def _net_reconstruction_parse_args():
    """
    Prepares command line arguments relevant for network reconstruction

    :return: parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    _add_network_args(parser)
    _add_embedding_args(parser)
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='If the similarity between a pair of nodes is'
                             'greater than this threshold, the nodes will'
                             'be considered as True edges. Default is 0.5')
    parser.add_argument('--batch-size', type=int, default=0,
                        help='Useful when the number of nodes is large, '
                             'with no batch size the current reconstruction '
                             'algorithm requires an O(n*n) space. '
                             'Default is 0 = no batch size')
    return parser.parse_args()


def _link_prediction_parse_args():
    """
    Prepares command line arguments relevant for network sampling

    :return: parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    _add_embedding_args(parser)
    _add_metrics_args(parser)
    parser.add_argument('--pos-file', type=str, default='',
                        help="A path to true edge samples")
    parser.add_argument('--neg-file', type=str, default='',
                        help="A path to false edge samples")
    parser.add_argument('--method', type=str, default='f',
                        help='Method of link prediction, possible values are '
                             '(f | s). '
                             'f - using edge features constructed '
                             'from the embeddings of incident nodes.'
                             's - using edge scores computed based on '
                             'the similarity of incident node embeddings. '
                             'Default is f.')

    parser.set_defaults(sep_space=False)
    return parser.parse_args()


def _node_classification_parse_args():
    """
    Prepares command line arguments relevant for node classification

    :return: parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('label-file', type=str, default='',
                        help='A path to node labels file')
    _add_embedding_args(parser)
    _add_metrics_args(parser)
    _add_train_hyper_param_args(parser)
    return parser.parse_args()
