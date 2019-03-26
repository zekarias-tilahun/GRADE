from grade.helpers import Const, ConfigParser, ArgParser
from grade.nrlio import Reader, write_nx_graph, write_list_of_tuples

import datetime as dt
import networkx as nx
import numpy as np

import random
import sys


def sample_true_edges_asp(network, rate=0.1):
    print(f'Sampling true edges with {rate} rate')
    start = dt.datetime.now()
    g = nx.Graph(network.edges)
    num_samples = int(np.ceil(g.number_of_edges() * rate))
    edges = list(g.edges())
    sampled_edges = []
    tested_edges = set()
    while len(sampled_edges) < num_samples and g.number_of_edges() > 0:
        s, t = random.sample(edges, 1)[0]
        if (s, t) not in tested_edges:
            tested_edges.add((s, t))
            g.remove_edge(s, t)
            if nx.has_path(g, s, t):
                sampled_edges.append((s, t))
            else:
                g.add_edge(s, t)
                
            sys.stdout.write(f'\r{len(sampled_edges)}/{num_samples}')
            sys.stdout.flush()
    print()
    delta = dt.datetime.now() - start
    print(f'Sampling finished in {delta.seconds} seconds')
    return sampled_edges


def sample_false_edges(network, rate=.1):
    """
    Samples a fraction of node pairs (u, v) that are not connected
    by an edge in a given network
    :param network: The network
    :param rate: The fraction of pairs to sample
    :return: list: A list of pair of nodes
    """
    print('Sampling false edges with rate: {}%'.format(rate))
    non_edges = nx.non_edges(network)
    num_samples = int(network.number_of_edges() * rate)
    sampled_edges = []
    for ne in non_edges:
        sampled_edges.append(ne)
        if len(sampled_edges) == num_samples:
            break
    return sampled_edges


def main():
    if len(sys.argv) > 1:
        parser = ArgParser(task=Const.NET_SAMPLING_TASK)
        options = parser.args
    else:
        options = ConfigParser(task=Const.NET_SAMPLING_TASK)
        
    reader = Reader(task=Const.NET_SAMPLING_TASK, options=options)
    
    true_edges = sample_true_edges_asp(network=reader.network, rate=options.rate)
    false_edges = sample_false_edges(network=reader.network, rate=options.rate)
    if options.res_file.strip() != '':
        reader.network.remove_edges_from(true_edges)
        write_nx_graph(reader.network, options.res_file)
    if options.pos_file.strip() != '':
        write_list_of_tuples(true_edges, options.pos_file)
    if options.neg_file.strip() != '':
        write_list_of_tuples(false_edges, options.neg_file)
        
    return reader.network, true_edges, false_edges
    

if __name__ == '__main__':
    main()
