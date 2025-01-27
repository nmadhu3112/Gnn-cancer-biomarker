#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import math
from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

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
	with open(label_file, 'r') as f:
		for line in f.readlines():
			gene, label = line.strip("\n").split("\t")
			labels = [label_map[x] for x in label.split(",") if x]
			if labels:
				d[gene] = labels
	return d

if __name__ == "__main__":
	prefix = 'data/HPA/'
	gene_file = prefix + 'enhanced_gene.txt'
	Gene = get_enhanced_gene(gene_file)
	print('protein number: ',len(Gene))

	fv_loader = prefix + 'res18_128'# image features extracted from CNN model
	method = 'dist' #disttance function: 
	
	Gene_Label = _load_label_from_file("data/HPA/label/enhanced_label.txt")

	# ########## count the number of multi-label proteins
	# len_list = []
	# for gene_i in Gene:
	# 	len_list.append(len(Gene_Label[gene_i]))
	# print(Counter(len_list)) #Counter({1: 948, 2: 208, 3: 29, 4: 1})
	
	graph_indicator = []
	fv_img = []
	graph_labels = []
	per_proein_img_number = []
	adj_weight_list = []
	for i in range(len(Gene)):
		gene_i=Gene[i]
		# graph label
		gene_i_label = Gene_Label[gene_i]
		graph_labels.append(gene_i_label)

		#node feature
		fv_path = fv_loader+'/'+ gene_i +'.npy'
		fv_protein = np.load(fv_path) #(img_number, 512)
		per_proein_img_number.append(np.shape(fv_protein)[0])
		per_protein_img = []
		for j in range(np.shape(fv_protein)[0]):
			fv_img.append(fv_protein[j])
			per_protein_img.append(fv_protein[j])
			graph_indicator.append(i+1) # start from 1 
		if method == 'dist':
			dist = DistanceMetric.get_metric('euclidean')
			Euclidean = dist.pairwise(np.array(per_protein_img))
			adj = np.reciprocal(np.add(Euclidean,np.eye(np.shape(Euclidean)[0])))-np.eye(np.shape(Euclidean)[0])
			adj_weight_list.append(adj)		
		if method == 'dist3':
			dist = DistanceMetric.get_metric('euclidean')
			Euclidean = dist.pairwise(np.array(per_protein_img))
			adj = np.reciprocal(np.add(Euclidean,np.eye(np.shape(Euclidean)[0])))-np.eye(np.shape(Euclidean)[0])
			adj = adj/3.0
			adj_weight_list.append(adj)
		if method == 'dist01':
			dist = DistanceMetric.get_metric('euclidean')
			Euclidean = dist.pairwise(np.array(per_protein_img))
			Euclidean[Euclidean <= 0.5] = -1
			Euclidean[Euclidean > 0.5] = 0
			Euclidean[Euclidean < 0] = 1
			Euclidean = Euclidean - np.eye(np.shape(Euclidean)[0])
			adj = Euclidean
			adj_weight_list.append(adj)
		if method == 'simmilarity':
			adj = cosine_similarity(np.array(per_protein_img))
			adj[np.diag_indices_from(adj)] = 0
			adj_weight_list.append(adj)
		if method == 'chebyshev':
			dist = DistanceMetric.get_metric('chebyshev')
			Euclidean = dist.pairwise(np.array(per_protein_img))
			adj = np.reciprocal(np.add(Euclidean,np.eye(np.shape(Euclidean)[0])))-np.eye(np.shape(Euclidean)[0])
			adj_weight_list.append(adj)	
		if method == 'minkowski':
			dist = DistanceMetric.get_metric('minkowski')
			Euclidean = dist.pairwise(np.array(per_protein_img))
			adj = np.reciprocal(np.add(Euclidean,np.eye(np.shape(Euclidean)[0])))-np.eye(np.shape(Euclidean)[0])
			adj_weight_list.append(adj)	
		if method == 'wminkowski':
			dist = DistanceMetric.get_metric('wminkowski')
			Euclidean = dist.pairwise(np.array(per_protein_img))
			adj = np.reciprocal(np.add(Euclidean,np.eye(np.shape(Euclidean)[0])))-np.eye(np.shape(Euclidean)[0])
			adj_weight_list.append(adj)		

	save_prefix = prefix + 'Adjacent'
	if not os.path.exists(save_prefix):
		os.mkdir(save_prefix)

	save_path = save_prefix + '/'+ fv_loader.split('/')[-1] + '_' + method
	filename_adj_weight = save_path + '_A.npy'
	filename_graph_indic = save_path +'_graph_indicator.npy'
	filename_graph_labels = save_path + '_graph_labels.npy'
	filename_node_attrs = save_path + '_node_attributes.npy'
	filename_per_proein_img_number = save_path + '_per_protein_img_number.npy'

	np.save(filename_adj_weight,np.array(adj_weight_list))
	np.save(filename_graph_indic,np.array(graph_indicator))
	np.save(filename_graph_labels,np.array(graph_labels))
	np.save(filename_node_attrs,np.array(fv_img))
	np.save(filename_per_proein_img_number,np.array(per_proein_img_number))

	# print(len(adj_weight_list)) # graph number
	# print(len(per_proein_img_number)) # graph number
	# print(per_proein_img_number)
	# print(len(graph_indicator)) #node number
	# print(len(graph_labels)) # graph number
	# print(len(fv_img)) # node number