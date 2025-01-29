#!/usr/bin/env python
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import scipy as sc
import os
import re
import matplotlib.pyplot as plt

import util
import random
import math
#from data.HPA import constant as c
from sklearn.neighbors import DistanceMetric

NUM_CLASSES = 6

label_map = {
    "Nuclear membrane": 0,
    "Cytoplasm": 1,
    "Vesicles": 2,
    "Mitochondria": 3,
    "Golgi Apparatus": 4,
    "Nucleoli": 0,
    "Plasma Membrane": 1,
    "Nucleoplasm": 0,
    "Endoplasmic Reticulum": 5
}


four_tissue_list = ['liver', 'breast', 'prostate', 'bladder']


def get_gene_pics(gene, tissue_list=four_tissue_list):
    pics = []
    for t in tissue_list:
        tp = os.path.join(c.TISSUE_DIR, t, "%s.txt" % gene)
        if os.path.exists(tp):
            with open(tp, 'r') as f:
                pics.extend([l.strip("\n") for l in f.readlines()])
    return pics

def get_enhanced_gene(gene_file):
    genes=[]
    # gene_file = 'enhanced_gene.txt'
    with open(gene_file, 'r') as f:
        for line in f.readlines():
            gene = line.strip("\n")
            genes.append(gene)
    return genes

def get_enhanced_gene_list():
    '''some gene marked as enhanced but do not have enhanced label'''
    return [x for x in os.listdir(c.DATA_DIR)
            if len(os.listdir(os.path.join(c.DATA_DIR, x)))]


def get_supported_gene_list():
    return [x for x in os.listdir(c.SUPP_DATA_DIR)
            if len(os.listdir(os.path.join(c.SUPP_DATA_DIR, x)))]


def get_approved_gene_list():
    return [x for x in os.listdir(c.APPROVE_DATA_DIR)
            if len(os.listdir(os.path.join(c.APPROVE_DATA_DIR, x)))]

def get_gene_list(size=1):
    '''not consider train/val/test'''
    gene_list = []
    if size >= 0:
        gene_list += get_enhanced_gene_list()
    if size >= 1:
        gene_list += get_supported_gene_list()
    if size >= 2:
        gene_list += get_approved_gene_list()

    return gene_list

def load_enhanced_label():
    return _load_label_from_file("enhanced_label.txt")

def _load_label_from_file(label_file):
    d = {}
    # pardir = os.path.join(os.path.dirname(__file__), os.pardir)
    # label_file = os.path.join(pardir, "label", fname)
    # label_file = os.path.join("label", fname)
    with open(label_file, 'r') as f:
        for line in f.readlines():
            gene, label = line.strip("\n").split("\t")
            labels = [label_map[x] for x in label.split(",") if x]
            if labels:
                d[gene] = labels
    return d

def read_graphfile(datadir, dataname, adj_method, max_nodes=None):
    ''' Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    '''
    # prefix = os.path.join(datadir,dataname, "enhanced_gene.txt")
    prefix = os.path.join(datadir,'HPA', "enhanced_gene.txt")
    Gene = get_enhanced_gene(prefix)
    print(len(Gene))

    # save_path = 'data/HPA/'+'res18_128' 
    # adj_method = 'dist' #dist, dist3,dist01(disttance after one threshold), simmilarity, dist_umap, minkowski
    save_path = 'data/HPA/Adjacent/'+'res18_128_'+ adj_method
    filename_adj_weight = save_path + '_A.npy'
    filename_graph_indic = save_path +'_graph_indicator.npy'
    filename_graph_labels = save_path + '_graph_labels.npy'
    filename_node_attrs = save_path + '_node_attributes.npy'
    filename_per_proein_img_number = save_path + '_per_protein_img_number.npy'

    adj_weight_list = np.load(filename_adj_weight)
    graph_indicator = np.load(filename_graph_indic)
    graph_labels = np.load(filename_graph_labels)
    fv_img = np.load(filename_node_attrs)
    per_proein_img_number = np.load(filename_per_proein_img_number)    

    adj_weight_list=adj_weight_list.tolist()
    graph_indicator=graph_indicator.tolist()
    graph_labels=graph_labels.tolist()
    fv_img=fv_img.tolist()
    per_proein_img_number=per_proein_img_number.tolist()

    print(len(adj_weight_list)) # graph number
    print(len(per_proein_img_number)) # graph number
    print(len(graph_indicator)) #node number
    print(len(graph_labels)) # graph number
    print(len(fv_img)) # node number

    # index of graphs that a given node belongs to
    graph_indic={}
    i=1
    for ind in graph_indicator:
        graph_indic[i]=int(ind)
        i+=1
    # print(graph_indic)

    # node feature
    node_attrs = fv_img

    # graph label
    graph_labels = np.array(graph_labels) # start form 0

    # Ajacent matrix
    start=0
    # node and graph start from 1
    adj_list={i:[] for i in range(1,len(graph_labels)+1)} 
    for j in range(len(adj_weight_list)):
        adj_weight_i = adj_weight_list[j]
        # print(adj_weight_i)
        adj_i=[]
        for colum in range(np.shape(adj_weight_i)[1]):
            for row in range(np.shape(adj_weight_i)[0]):
                e0 = start+row+1
                e1 = start+colum+1
                # print(e0,e1,adj_weight_i[row,colum])
                if e0 != e1:
                    if adj_weight_i[row,colum]!=0:
                        adj_list[j+1].append((e0,e1,adj_weight_i[row,colum]))
        start=start + np.shape(adj_weight_i)[0]

    # print(adj_list.keys()) # adj_list is a dict form, contain 1186 graphs(1,...,1186)
    print(adj_list[1]) # all edges of one graph
    # print(adj_list[2])
    # print(len(adj_list)) #1186

    graphs=[]
    graphs_labels = []
    for i in range(1,1+len(adj_list)):
        # indexed from 1 here
        # G=nx.from_edgelist(adj_list[i])
        G = nx.Graph()
        G.add_weighted_edges_from(adj_list[i])
        # draw the graph G
        # nx.draw(G)
        # plt.show()
        # print(G.get_edge_data(1,2))

        if max_nodes is not None and G.number_of_nodes() > max_nodes:
            continue
      
        # add features and labels
        ################################## one-hot label ###########
        # G.graph['label'] = graph_labels[i-1]
        # print(graph_labels[i-1])
        label = [0 for _ in range(6)] # class number is 6
        for l in graph_labels[i-1]:
            label[l] = 1
        label = np.array(label)
        # print(label)
        # print(stop)
        G.graph['label'] = label
        graphs_labels.append(label)
        for u in util.node_iter(G):
            # print(u)
            # if len(node_labels) > 0:
            #     node_label_one_hot = [0] * num_unique_node_labels
            #     node_label = node_labels[u-1]
            #     node_label_one_hot[node_label] = 1
            #     util.node_dict(G)[u]['label'] = node_label_one_hot
            if len(node_attrs) > 0:
                util.node_dict(G)[u]['feat'] = np.array(node_attrs[u-1])
                # print(np.array(node_attrs[u-1])) # evry last row from attribute matrix 1 [11.,15.887014,37.78,-0.51,1.701,93.9,4.,5.,2.,4.,4.,3.,3.,4.,4.,3.,6.,2.]
                # print(a)

        if len(node_attrs) > 0:
            G.graph['feat_dim'] = np.array(node_attrs[0]).shape[0]
        # print(G.graph['feat_dim']) # 512

        # relabeling
        mapping={}
        it=0
        for n in util.node_iter(G):
            mapping[n]=it
            it+=1
            
        # indexed from 0
        graphs.append(nx.relabel_nodes(G, mapping))
        # print(a)
        
    return graphs,graphs_labels

