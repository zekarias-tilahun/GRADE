# GRADE (**Gra**ph Embe**d**ding ***E***valuation) - 0.0.1 (beta)

GRADE is a library that provides a set of utilities for evaluating the performance of graph embedding (aka: Network Representation Learning - NRL) algorithms. It is intended to ease the work load during the experimental evaluation phase of NRL research. For example, one can use GRADE to compare several algorithms across a range of network analysis tasks. In this version, we have added three of the most common tasks.

## Applications/Tasks

- Network Reconstruction
- Link Prediction
- Node Classification

The following tasks will be included soon

- Vertex Recommendation
- Visualization

In addition, in the future version we will provide utilities for hyper-parameter tunning.

## Requirements

- Numpy
- Pandas
- Scikit-learn
- networkx

## Usage

To run GRADE a few inputs should be provided, currently there are two ways to do that

- Using the config.ini (Recommended): The required input options vary depending on the task under consideration. Inline documentation is provided, in addition refer to the **Input Arguments** Section. 
- Using the command line arguments (Not-complet):

### Example usage
```sh
$ cd tasks
$ python link_prediction.py
```

## Input Arguments

The following are the list of options that can be passed as a command line arguments or using the config.ini file. If no command line argument is provided as in **Example Usage** the config.ini will be used for reading the options.

### Graph file options
`--net-file:` 
A path to a graph file 

`--net-format:`
The format of the graph file, the possible choices are
>`edgelist`, `adjlist` - The standard edge list and adjacency list formats, respectively. 

>`mattxt` - A matrix file stored in txt format. The number of lines should be equal to the number of nodes and each line should have a number of values separated by white space and equal to the number of nodes .

>`npy` - A numpy matrix file

`--directed:`
A flag to indicate the graph is directed. The default is false

`--weighted:`
A flag to indicate the input graph is weighted. The default is false

### Embedding file options
`--emb-file:`
A path to an a graph embedding file

`--emb-format:`
The format of the graph embedding file, the possible choices are

>`w2v` - A word2vec format

>`mattxt` - A matrix file stored in txt format. The number of lines should be equal to the number of nodes and each line should have a number of values separated by white space and equal to the embedding dimension

>`npy` - A numpy matrix file

>`npz` - A numpy matrix file

### Evaluation metric options
The following options will be used by all the tasks, however if no evaluation metric is specified then the outputs are simply the predictions, for example a python dictionary containing the true and predicted labels of nodes

`--metrics:`
The name(s) of the evaluation metrics separated by white space. The following are the supported choices

>`acc` - Accuracy

>`pre` - Precision

>`rec` - Recall

>`mic` - F1-Micro

>`mac` - F1-Macro

>`auc` - Area Under the Curve

>`pak` - Precision@K, if this metric is selected the `--k-values` should also be specified

`--k-values:`
A list of integers, the different k values for the Precision@K metric

### Edge sample options:
The following options are relevant for link-prediction experiment and will be used during **network sampling** as well. In the sampling phase, depending on the rate a fraction of true and false edges will be sampled and saved to files specified by the following options.

`--pos-file:`
A path to the true edge samples

`--neg-file:`
A path to the false edge samples, each pair of node in this list is sampled the incident nodes are not connected by an edge.

### Network sampling options
Normally this should be the first step of a link-prediction experiment. According to the specified `--rate` option, a fraction of true and false edges will be sampled from the graph. The sampled edges will be exported to files specified by `--pos-file` and `--neg-file`. Once edges are sampled the true edges will be removed from the graph and the sampling is done by ensuring that the residual graph remains connected.

`--res-file:`
A path to the residual graph. Usually, for the purpose of link prediction experiment, the graph located in this path is used to train a particular graph embedding algorithm

`--rate:`
A value in (0, 1): the sampling rate, the fraction of edges to be sampled.

### Link prediction options
The following options will be required for link prediction experiment

`--lp-method:` 
A link prediction method, Currently there are two alternatives

>`f` - Feature based, link prediction is done by first constructing edge features using the embeddings of the left and right incident nodes.

>`s` - Score based, link prediction is done by computing link probabilities using the embeddings of the left and right indicent nodes

`--builder:`
The name of the edge feature builder methods, if only the --lp-method is *f*.

Given the embeddings of the left and right incident nodes of a list of edges, their features can be construted using one or more of the following techniques

>`hd` - Hadamard (Element-wise multiplication)

>`avg` - Average

>`wl1` - Weighted-L1

>`wl2` - Weighted-L2

>`all` - All of the above

### Network reconstruction options
The following options will be used for network reconstruction experiment, but they are not mandatory. For large graphs, the following options will enable efficient space and time usage.

`--threshold:`
A value in [0, 1], it will be used to prune reconstructed edges having a link probability less than the specified threshold

`--batch-size:`
A integer value, when it is specified reconstruction will be carried out in batches as opposed to all pairs alternative.

### Node classification options
The following option will be required for node classification experiment

`--label-file:`
A path to a file containing node labels. Should be a white space separated file, where the first value is the node id and the second one is the corresponding label.

### Hyper parameter options
`--train-size:`
A list of values between (0, 1), some experiments, such as node classification and link prediction might require this value

`--seed`
A integer value to be used as a seed for the pseudo random number generators

`--cv:`
A integer value, if a number greater than 1 is specified a k-fold cross-validation experiment, using `cv`-folds will be carried out. It can be used during node classification and link prediction with `f` method.