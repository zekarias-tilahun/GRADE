"""
Author: Zekarias Tilahun Kefato

The module provides a set of input and output utilities that are necessary
for GRADE's execution.
"""
from sklearn.preprocessing import normalize
from scipy import sparse as sp
from grade.helpers import Const

import networkx as nx
import pandas as pd
import numpy as np

import os


class Reader(object):

    def __init__(self, task, options):
        """
        Reads the necessary inputs required to execute a specified task

        Parameters
        ----------
        task : string
            The specified task
        options : helpers.ArgParser or helpers.ConfigParser options
            Options compiled from command line or config input
        """
        self.task = task
        self._options = options
        self.__read()

    def __read_embedding(self):
        """
        A wrapper around read_embeddings

        """
        self.embeddings = read_embeddings(
            files=self._options.emb_file, formats=self._options.emb_format)

    def __read_network(self):
        """
        A wrapper around read_network

        """
        opts = self._options
        self.network = read_network(
            path=opts.net_file, directed=opts.directed, weighted=opts.weighted,
            input_format=opts.net_format)

    def __read_sampled_edges(self):
        """
        A wrapper around read_network, invoked twice

        """
        opts = self._options
        true_graph = read_network(path=opts.pos_file)
        self.true_edges = list(true_graph.edges)
        false_graph = read_network(path=opts.neg_file)
        self.false_edges = list(false_graph.edges)

    def __read_labels(self):
        """
        A wrapper around read_labels

        """
        self.labels = read_labels(path=self._options.label_file)

    def __read(self):
        """
        A wrapper for reading the necessary inputs for the specified task

        """
        if self.task == Const.NET_SAMPLING_TASK:
            self.__read_network()
        else:
            self.__read_embedding()
            if self.task == Const.NET_RECONSTRUCTION_TASK:
                if self._options.has_metrics():
                    self.__read_network()
                else:
                    self.network = None
            elif self.task == Const.LINK_PREDICTION_TASK:
                self.__read_sampled_edges()
            elif self.task == Const.NODE_CLASSIFICATION_TASK:
                self.__read_labels()

    @property
    def has_sym_embedding(self):
        """
        Checks if Reader.embeddings is symmetric embedding or not.

        """
        return not isinstance(self.embeddings, tuple)


class Writer:

    def __init__(self, task, options, data):
        self.task = task
        self._options = options
        self._data = data

    def __write(self):
        opts = self._options
        if self.task == Const.NET_SAMPLING_TASK:
            write_list_of_tuples(
                self._data[Const.NET_SAMPLING_TASK]['true'],
                path=opts.pos_file)
            write_list_of_tuples(
                self._data[Const.NET_SAMPLING_TASK]['false'],
                path=opts.neg_file)
            write_nx_graph(
                self._data[Const.NET_SAMPLING_TASK]['res_net'],
                path=opts.res_file)


def read_network(path, directed=True, weighted=False, input_format='edgelist',
                 in_norm=None, out_norm=None):
    """
    Reads a graph from a path
    :param path: The path
    :param directed: A flag to indicate if the graph is directed or not.
                    Default is True
    :param weighted: A flag to indicate whether the graph is weighted or not.
                    Default is False
    :param input_format: The format of the file, possible values are
                    (edgelist - Default | adjlist | mattxt | matnpy)
    :param in_norm: A flag to indicate to normalize incoming weights.
                    Possible values are l1 and l2. Default is None, un-normalized
    :param out_norm: A flag that indicate the outgoing weights are normalized.
                    If the graph is undirected, this will be used to normalize
                    the weights of all the neighbors of a node.
                    Possible values are l1 and l2. Default is None, un-normalized
    :return:
    """
    print('INFO: Reading network file from {} stored as {} format'.format(
        path, input_format))
    create_using = nx.DiGraph() if directed else nx.Graph()
    if input_format == 'edgelist':
        reader = nx.read_weighted_edgelist if weighted else nx.read_edgelist
        network = reader(
            path, nodetype=int, create_using=create_using)
        adj_mat = nx.to_scipy_sparse_matrix(network, sorted(network.nodes()))
    elif input_format == 'adjlist':
        if weighted:
            raise ValueError("The combination of input format 'adjlist' "
                             "and weighted=True is not supported")
        network = nx.read_adjlist(
            path, nodetype=int, create_using=create_using)
        adj_mat = nx.to_scipy_sparse_matrix(network, sorted(network.nodes()))
    elif input_format == 'mattxt':
        adj_mat = sp.csr_matrix(np.loadtxt(path))
    elif input_format == 'npy':
        adj_mat = sp.csr_matrix(np.load(path))

    if out_norm is not None:
        norm_adj_mat = normalize(adj_mat, norm=out_norm)
        network = nx.from_scipy_sparse_matrix(
            A=norm_adj_mat, create_using=create_using)
    elif in_norm is not None:
        norm_adj_mat = normalize(adj_mat, norm=in_norm, axis=0)
        network = nx.from_scipy_sparse_matrix(
            A=norm_adj_mat, create_using=create_using)
    else:
        network = nx.from_scipy_sparse_matrix(A=adj_mat, create_using=create_using)

    print('\n\tNumber of nodes: {}\n\tNumber of edges: {}'.format(
        network.number_of_nodes(), network.number_of_edges()))
    return network


def read_embedding(path, input_format=None, keys=None):
    """
    Reading embeddings of nodes from a path.

    :param path: The path
    :param input_format: The format of the embedding file (npy or w2v)
    :param keys A tuple of keys for left and right embedding matrices if
    path is points to an npz files containing these embeddings.
    :return:
    """
    if input_format is not None:
        print(f'INFO: Reading embedding file from {path} stored in {input_format} format')
        if input_format == 'npy':
            return np.load(path)
        elif input_format == 'npz':
            files = np.load(path)
            return files[list(files.keys())[0]]
        elif input_format == 'mattxt':
            return np.loadtxt(path)
        elif input_format == 'w2v':
            emb_frame = pd.read_csv(path, header=None, skiprows=1)
            emb_frame.sort_values(0, inplace=True)
            emb_frame.dropna(axis=1, how='all', inplace=True)
            return emb_frame.values[:, 1:]
    elif keys is not None and len(keys) == 2:
        print(f'INFO: Reading left and right embedding files from a .npz file located in {path}')
        files = np.load(path)
        return files[keys[0]], files[keys[1]]
    elif keys is not None and len(keys) == 1:
        print(f'INFO: Reading node embeddings from a .npz file located in {path}')
        files = np.load(path)
        return files[keys[0]]
    

def read_embeddings(files, formats):
    """
    Reads 1 or more embeddings embeddings

    files: str, list
        It could be a path to a file containing an embedding or multiple embeddings,
        for example *.npz file. It could also be a list of paths for multiple embedding
        files.
    formats: str, list, dict
        The format of each file in files

    """
    if isinstance(files, str) and isinstance(formats, str):
        return read_embedding(path=files, input_format=formats)
    elif isinstance(formats, dict):
        return read_embedding(path=files, keys=formats['keys'])
    elif isinstance(files, list) and isinstance(formats, list):
        embeddings = {}
        for path, f in zip(files, formats):
            file_name = os.path.basename(path)
            name = file_name[:file_name.rfind('.')]
            embeddings[name] = read_embedding(path=path, input_format=f)
        return embeddings


def read_labels(path, sep=r"\s+"):
    """
    Reads labels associated with nodes.
    :param path: A path to node's label file
    :param sep: Separator, the default is white space separator
    :return: A numpy array of node labels
    """
    label_frame = pd.read_csv(path, header=None, sep=sep, names=['node_id', 'label'])
    return label_frame.sort_values(by='node_id').label.values


def write_list_of_tuples(lot, path):
    """
    Writes a list of tuple (a tuple per line) to a file
    :param lot: list of tuples
    :param path: The file to write to
    :return:
    """
    print(f'INFO: Writing to a file {path}')
    with open(path, 'w') as f:
        for u, v in lot:
            f.write('{} {}\n'.format(u, v))


def write_nx_graph(graph, path, writer='edgelist'):
    print(f'INFO: Writing to a file {path}')
    if writer == 'edgelist':
        nx.write_edgelist(graph, path=path, data=False)
    elif writer == 'adjlist':
        nx.write_adjlist(graph, path=path)
