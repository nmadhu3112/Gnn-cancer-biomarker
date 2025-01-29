import networkx as nx
import numpy as np
import torch

import pickle
import random

from graph_sampler import GraphSampler
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from itertools import combinations

def prepare_val_data(graphs, graphs_labels,args, val_idx, max_nodes=0):

    genes =[]
    with open('data/HPA/enhanced_gene.txt', 'r') as f1:
        for line in f1.readlines():
            genes.append(line.strip("\n"))

    # multi-stratified data
    mskf = MultilabelStratifiedKFold(n_splits=11, random_state=1234)
    folds = []
    X = np.arange(len(graphs_labels))
    y = np.array(graphs_labels)
    for train_index, test_index in mskf.split(X, y):
        folds.append(test_index)
    # check the split data
    for a, b in combinations(folds, 2):
        assert len(set(a) & set(b)) == 0

    fold_idx = val_idx
    records = []
    train_graphs = []
    val_graphs = []
    test_graphs = []

    train_genes = []
    val_genes = []
    test_genes = []

    for i, indices in enumerate(folds):
        # if i == (fold_idx * 2) or i == (fold_idx * 2) + 1: # 5-fold cross-validation
        if i == fold_idx : # 10-fold cross-validation
            for j in indices:
                val_graphs.append(graphs[j])
                val_genes.append(genes[j])
        elif i == 10:
            for j in indices:
                test_graphs.append(graphs[j])
                test_genes.append(genes[j])
        else:
            for j in indices:
                train_graphs.append(graphs[j])
                train_genes.append(genes[j])

    # random.shuffle(train_graphs) #shuffle training data while training
    print('fold:',val_idx)
    print('Num training graphs: ', len(train_graphs), 
          '; Num validation graphs: ', len(val_graphs),
          '; Num test graphs: ', len(test_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ', 
            max([G.number_of_nodes() for G in graphs]), ', '
            "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
            "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    # shuffle= True wihle training 
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers)

    dataset_sampler = GraphSampler(test_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    test_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers)

    # return train\val\test genes
    return train_genes, val_genes, test_genes, train_dataset_loader, val_dataset_loader,test_dataset_loader, \
            dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.attention_feat_dim
