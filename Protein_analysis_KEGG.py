import pandas as pd 
import numpy as np
from scipy import spatial
import random
import math

def get_enhanced_gene(gene_file):
    genes=[]
    with open(gene_file, 'r') as f:
        for line in f.readlines():
            gene = line.strip("\n")
            genes.append(gene)
    return genes

def get_nodes(df):
	df['x'] = df['x'].map(lambda x: x[7:])
	nodes_list = df['x'].values.tolist()
	return nodes_list

def generate_dict(list1,list2):
	dict_test = {}
	for i in range(len(list1)):
		dict_test[list1[i]]=list2[i]
	return dict_test

def transfer_id2Name(df,dict_id2name):
	df = df.reset_index()
	for i in range(len(df)):
		df['protein1'][i] = dict_id2name[df.iloc[i]['protein1']]
		df['protein2'][i] = dict_id2name[df.iloc[i]['protein2']]
	return df

def euclideanDistance(instance1,instance2,dimension):
	distance = 0
	for i in range(dimension):
		distance += (instance1[i]-instance2[i])**2
	return math.sqrt(distance)

# load ImPloc data
df_reference_geneList = pd.read_csv('data/HPA/subcellular_location_ImpLoc.tsv',sep='\t')
gene_list = get_enhanced_gene('data/HPA/enhanced_gene.txt')
df_reference_geneList = df_reference_geneList[df_reference_geneList['Gene'].isin(gene_list)]
ImPLoc_genelist = df_reference_geneList['Gene name'].values.tolist()
ImPLoc_ENSGlist = df_reference_geneList['Gene'].values.tolist()
dict_gene_ENSG = {}
for i in range(len(ImPLoc_ENSGlist)):
    dict_gene_ENSG[ImPLoc_ENSGlist[i]] = ImPLoc_genelist[i] 

####### load STRING data
df_STRING = pd.read_csv('protein_network/STRING/9606.protein.physical.links.v11.5.csv')
df_STRING_info = pd.read_csv('protein_network/STRING/9606.protein.info.v11.5.csv')
string_protein_id = df_STRING_info['string_protein_id'].values.tolist()
preferred_name = df_STRING_info['preferred_name'].values.tolist()
dict_id2name = generate_dict(string_protein_id,preferred_name)
dict_name2id = generate_dict(preferred_name,string_protein_id)
inter_geneName_list = [i for i in ImPLoc_genelist if i in preferred_name] #338
id_list_Imploc_STRING = []
for geneName in inter_geneName_list:
	id_list_Imploc_STRING.append(dict_name2id[geneName])

#load KEGG data, compute the combined score for each KEGG network
KEGG_prefix = 'protein_network/KEGG/human/'

network_list = []
average_score_list = []
frequency_list = []
p_value_list = []
new_num_list = []
for network_ind in range(1,313):
	print(network_ind)
	nodes_file = KEGG_prefix + str(network_ind)+'_nodes.csv'
	edges_file = KEGG_prefix + str(network_ind)+'_edges.csv'
	df_nodes = pd.read_csv(nodes_file)
	nodes_list = get_nodes(df_nodes)

	inter_KEGG_Imploc = [i for i in nodes_list if i in ImPLoc_genelist]
	# find gene network that have at least two proteins in imploc dataset
	if len(inter_KEGG_Imploc) > 2:
		network_list.append(network_ind)
		# compute the simmilarity score of KEGG and STRING
		inter_KEGG_STRING = [i for i in nodes_list if i in preferred_name]

		id_list_KEGG_STRING = []
		for geneName in inter_KEGG_STRING:
			id_list_KEGG_STRING.append(dict_name2id[geneName])

		df_temp = df_STRING[df_STRING['protein1'].isin(id_list_KEGG_STRING)]
		df_KEGG_STRING = df_temp[df_temp['protein2'].isin(id_list_KEGG_STRING)]
		score = np.sum(df_KEGG_STRING['combined_score'].values)

		m = len(df_KEGG_STRING)
		average_score = score / float(m)
		average_score_list.append(average_score)
		frequency =  len(inter_KEGG_STRING)/ float(len(nodes_list))
		frequency_list.append(frequency)

network_list, average_score_list, frequency_list = (list(t) for t in zip(*sorted(zip(network_list,average_score_list,frequency_list),key=lambda x: x[1],reverse = True)))
df = pd.DataFrame({'network':network_list,'average_score':average_score_list,'frequency':frequency_list})
print(df)

# load hidden features of each protein through inference_HPA.py
geneList = []
gene_i = get_enhanced_gene('data/HPA/GCN_feat/featWithGene/fold0/test_gene.txt')
geneList.extend(gene_i)
geneFeat = np.load('data/HPA/GCN_feat/featWithGene/fold0/Test_feats.npy')	
for fold_index in range(10):
	filePath_prefix = 'data/HPA/GCN_feat/featWithGene/'+'fold'+str(fold_index)+'/'
	geneList_path = filePath_prefix + 'val_gene.txt'
	geneFeat_path = filePath_prefix + 'Val_feats.npy'
	gene_i = get_enhanced_gene(geneList_path)
	geneFeat_i = np.load(geneFeat_path)
	geneList.extend(gene_i)
	geneFeat = np.vstack((geneFeat,geneFeat_i))

dict_gene_feat = {}
for i in range(len(geneList)):
    dict_gene_feat[dict_gene_ENSG[geneList[i]]] = geneFeat[i,:] 

# predic candidate protein for top 10 network
for ind in range(10):
	dict_protein_correlation = {}
	network_ind = int(df.iloc[ind]['network'])
	print('network_ind: ',network_ind)
	nodes_file = KEGG_prefix + str(network_ind)+'_nodes.csv'
	df_nodes = pd.read_csv(nodes_file)
	nodes_list = get_nodes(df_nodes)
	inter_KEGG_Imploc = [i for i in nodes_list if i in ImPLoc_genelist]
	print(inter_KEGG_Imploc)

	dict_protein_correlation = {}
	dict_protein_distacne = {}
	for protein1 in dict_gene_feat.keys():
		if protein1 not in inter_KEGG_Imploc:
			protein1_feat = dict_gene_feat[protein1]
			inter_correlation = []
			inter_distance = []
			for protein2 in inter_KEGG_Imploc:
				protein2_feat = dict_gene_feat[protein2]
				correlation = spatial.distance.correlation(protein1_feat, protein2_feat)
				distance = euclideanDistance(protein1_feat,protein2_feat,np.shape(protein2_feat)[0])
				inter_correlation.append(correlation)
				inter_distance.append(distance)
			average_correlation = np.mean(np.array(inter_correlation))
			average_distance = np.mean(np.array(inter_distance))
			dict_protein_correlation[protein1] = average_correlation
			dict_protein_distacne[protein1] = average_distance

	sorted_protein_correlation = sorted(dict_protein_correlation.items(),key=lambda kv:(kv[1],kv[0]),reverse=True)
	sorted_protein_distance = sorted(dict_protein_correlation.items(),key=lambda kv:(kv[1],kv[0]))
	# print(sorted_protein_correlation[0:5])
	print(sorted_protein_distance[0:5])
