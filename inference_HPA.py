import pandas as pd
import numpy as np
import torch
import os
import networkx as nx
import util
from graph_sampler import GraphSampler
from torch.autograd import Variable
import npmetrics
from torch import nn
import argparse
import cross_val_inference as cross_val
import random

seed = 0 
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed) #cpu
torch.cuda.manual_seed_all(seed)  #gpu
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = True  

def load_gene_list():
    geneList =[]
    with open('data/HPA/enhanced_gene.txt', 'r') as f1:
        for line in f1.readlines():
            geneList.append(line.strip("\n"))

    return geneList

def save_txt(split,genes):
    file_path = 'data/HPA/GCN_feat/featWithGene/'+split+'_gene.txt'
    with open(file_path, 'w') as f1:
        for line in genes:
            f1.write(line)
            f1.write("\n")

def load_Adjacent(prefix,max_nodes=None):
    # file_path = prefix+'res18_128_minkowski'
    file_path = prefix
    filename_adj_weight = file_path + '_A.npy'
    filename_graph_indic = file_path +'_graph_indicator.npy'
    filename_graph_labels = file_path + '_graph_labels.npy'
    filename_node_attrs = file_path + '_node_attributes.npy'
    filename_per_proein_img_number = file_path + '_per_protein_img_number.npy'

    adj_weight_list = np.load(filename_adj_weight,allow_pickle=True)
    graph_indicator = np.load(filename_graph_indic)
    graph_labels = np.load(filename_graph_labels,allow_pickle=True)
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

    # node feature
    node_attrs = fv_img

    # graph label
    graph_labels = np.array(graph_labels) # start form 0

    # Ajacent matrix
    start=0
    adj_list={i:[] for i in range(1,len(graph_labels)+1)} 
    for j in range(len(adj_weight_list)):
        adj_weight_i = adj_weight_list[j]
        adj_i=[]
        for colum in range(np.shape(adj_weight_i)[1]):
            for row in range(np.shape(adj_weight_i)[0]):
                e0 = start+row+1
                e1 = start+colum+1
                if e0 != e1:
                    if adj_weight_i[row,colum]!=0:
                        adj_list[j+1].append((e0,e1,adj_weight_i[row,colum]))
        start=start + np.shape(adj_weight_i)[0]

    print(adj_list[1]) # all edges of one graph


    graphs=[]
    graphs_labels = []
    for i in range(1,1+len(adj_list)):
        G = nx.Graph()
        G.add_weighted_edges_from(adj_list[i])

        if max_nodes is not None and G.number_of_nodes() > max_nodes:
            continue
      
        # add features and labels
        ################################## one-hot label ###########
        label = [0 for _ in range(6)] # class number is 6
        for l in graph_labels[i-1]:
            label[l] = 1
        label = np.array(label)
        G.graph['label'] = label
        graphs_labels.append(label)
        for u in util.node_iter(G):
            if len(node_attrs) > 0:
                util.node_dict(G)[u]['feat'] = np.array(node_attrs[u-1])

        if len(node_attrs) > 0:
            G.graph['feat_dim'] = np.array(node_attrs[0]).shape[0]

        # relabeling
        mapping={}
        it=0
        for n in util.node_iter(G):
            mapping[n]=it
            it+=1
            
        # indexed from 0
        graphs.append(nx.relabel_nodes(G, mapping))
        
    return graphs,graphs_labels    

def sigmoid(x):
    # TODO: Implement sigmoid function
    return 1/(1 + np.exp(-x))

def compute_sigmoid(probs):
    for i in range(np.shape(probs)[0]):
        for j in range(np.shape(probs)[1]):
            probs[i,j] = sigmoid(np.array(probs[i,j]))
    return probs

def threshold(probs):
    preds = []
    for i in range(np.shape(probs)[0]):
        ypred = probs[i,:]
        ypred_new = (ypred > 0.5).astype(int)
        if np.all(ypred_new == 0):
            max_index = np.where(ypred==np.max(ypred))
            ypred_new[max_index] =1
        preds.append(ypred_new)
    return np.array(preds) 

def get_best_sitaAndcons(np_label,np_prob,best_cons, best_sita,savePath):
    if best_cons == None:
        cons = np.arange(0,1,0.01)
        sita = np.arange(0,1,0.01) 
    else:
        cons = (0,best_cons)
        sita = (0,best_sita) 

    best_subset_acc=0.0
    best_F1_score=0.0
    best_cons=0.0
    best_sita=0.0
    y_pred_label=np.zeros(np_prob.shape,dtype='uint8')
    for choose_cons in cons:      
        for choose_sita in sita:
            for i in range(np_prob.shape[0]):
                output=np_prob[i,:]
                maxi=max(output)
                maxi_index=np.where(output==maxi)[0][0]
                for j in range(np_prob.shape[1]): 
                    if output[j]>=choose_cons:
                        y_pred_label[i][j]=1
                    if output[j]>=maxi*choose_sita:
                        y_pred_label[i][j]=1
            metric_result=evaluate_original('dynamic',y_pred_label,np_label,savePath)
            if metric_result['sub_acc']>best_subset_acc:
                best_subset_acc=metric_result['sub_acc']
                best_result=metric_result
                best_cons=choose_cons
                best_sita=choose_sita
                best_y_pred_label = y_pred_label
            y_pred_label=np.zeros(np_prob.shape,dtype='uint8')
    print ("best_cons", best_cons)
    print("best_sita",best_sita)
    print ("Best_result", best_result)
    return best_cons,best_sita,best_result,best_y_pred_label,np_label

def evaluate(dataset, model, savePath, name='Validation', max_num_examples=None):
    # model.eval()

    labels = []
    preds = []
    probs = []
    feats = []
    right_predict = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        h0 = Variable(data['feats'].float()).cuda()
        labels.append(data['label'].float().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()
        attention_input = Variable(data['attention_feats'].float(), requires_grad=False).cuda()

        ypred = model(h0, adj, batch_num_nodes, attention_x=attention_input)
        ypred = ypred.cpu().data.numpy()
        probs.append(ypred)
        ypred = (ypred > 0.5).astype(int)
        preds.append(ypred)

        if max_num_examples is not None:
            if (batch_idx+1)*20 > max_num_examples:
                break

    labels = np.vstack(labels)
    probs = np.vstack(probs)
    preds = np.vstack(preds)
    for i in range(np.shape(probs)[0]):
        for j in range(np.shape(probs)[1]):
            probs[i,j] = sigmoid(np.array(probs[i,j]))

    ### save hiden feat for location biomarker
    # if name == 'Train':
    #     outfile = 'data/HPA/GCN_feat/featWithGene/Train_'
    # if name == 'Validation':
    #     outfile = 'data/HPA/GCN_feat/featWithGene/Val_'
    # if name == 'Test':
    #     outfile = 'data/HPA/GCN_feat/featWithGene/Test_'
    # labels_file = outfile + 'labels.npy'
    # probs_file = outfile + 'probs.npy'
    # preds_file = outfile + 'preds.npy'
    # feats_file = outfile + 'feats.npy'

    # np.save(labels_file, labels)
    # np.save(probs_file, probs)
    # np.save(preds_file, preds)
    # np.save(feats_file, feats)

    npmetrics.write_metrics(labels,preds,savePath)

    result = {'ex_acc': npmetrics.example_accuracy(labels, preds),
              'ex_precision': npmetrics.example_precision(labels, preds),
              'ex_recall': npmetrics.example_recall(labels, preds),
              'ex_f1': npmetrics.compute_f1(npmetrics.example_precision(labels, preds), npmetrics.example_recall(labels, preds)),
              
              'lab_acc_macro': npmetrics.label_accuracy_macro(labels, preds),
              'lab_precision_macro': npmetrics.label_precision_macro(labels, preds),
              'lab_recall_macro': npmetrics.label_recall_macro(labels, preds),
              'lab_f1_macro': npmetrics.compute_f1(npmetrics.label_precision_macro(labels, preds), npmetrics.label_recall_macro(labels, preds)),

              'lab_acc_micro': npmetrics.label_accuracy_micro(labels, preds),
              'lab_precision_micro': npmetrics.label_precision_micro(labels, preds),
              'lab_recall_micro': npmetrics.label_recall_micro(labels, preds),
              'lab_f1_micro': npmetrics.compute_f1(npmetrics.label_precision_micro(labels, preds), npmetrics.label_recall_micro(labels, preds)),

              'sub_acc': npmetrics.example_subset_accuracy(labels, preds)}
    print(result)
    return result

def evaluate_original(name,labels,preds,savePath):
    if savePath is not None:
        npmetrics.write_metrics(labels,preds,savePath)

    result = {'ex_acc': npmetrics.example_accuracy(labels, preds),
              'ex_precision': npmetrics.example_precision(labels, preds),
              'ex_recall': npmetrics.example_recall(labels, preds),
              'ex_f1': npmetrics.compute_f1(npmetrics.example_precision(labels, preds), npmetrics.example_recall(labels, preds)),
              
              'lab_acc_macro': npmetrics.label_accuracy_macro(labels, preds),
              'lab_precision_macro': npmetrics.label_precision_macro(labels, preds),
              'lab_recall_macro': npmetrics.label_recall_macro(labels, preds),
              'lab_f1_macro': npmetrics.compute_f1(npmetrics.label_precision_macro(labels, preds), npmetrics.label_recall_macro(labels, preds)),

              'lab_acc_micro': npmetrics.label_accuracy_micro(labels, preds),
              'lab_precision_micro': npmetrics.label_precision_micro(labels, preds),
              'lab_recall_micro': npmetrics.label_recall_micro(labels, preds),
              'lab_f1_micro': npmetrics.compute_f1(npmetrics.label_precision_micro(labels, preds), npmetrics.label_recall_micro(labels, preds)),

              'sub_acc': npmetrics.example_subset_accuracy(labels, preds)}
    return result

def inference(dataset, model,saveprefix, name='Validation', max_num_examples=None,best_cons=None, best_sita=None):
    # model.eval()

    labels = []
    preds = []
    probs = []
    feats = []
    right_predict = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        h0 = Variable(data['feats'].float()).cuda()
        labels.append(data['label'].float().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()
        attention_input = Variable(data['attention_feats'].float(), requires_grad=False).cuda()

        ypred = model(h0, adj, batch_num_nodes, attention_x=attention_input)
        ypred = ypred.cpu().data.numpy()
        probs.append(ypred)
        ypred = (ypred > 0.5).astype(int)
        preds.append(ypred)

        if max_num_examples is not None:
            # if (batch_idx+1)*args.batch_size > max_num_examples:
            if (batch_idx+1)*20 > max_num_examples:
                break
    labels = np.vstack(labels)
    probs = np.vstack(probs)
    preds = np.vstack(preds)

    probs = compute_sigmoid(probs)
    preds = threshold(probs)

    ############## dynamic threshold
    if saveprefix is not None:
        savePath = saveprefix+ '_dynamic_threshold'+'.txt'
    else:
        savePath = None
    best_cons,best_sita,best_result,best_y_pred_label,np_label = get_best_sitaAndcons(labels,probs,best_cons,best_sita,savePath)
    np.save(saveprefix+'_y_pred_label',best_y_pred_label)
    np.save(saveprefix+'_np_label',np_label)
    return best_cons,best_sita,best_result


def bag_predict(gene_list,args):
    '''filter mislocation by bag predict'''
    torch.nn.Module.dump_patches = True
    compare_method = 'res18_128' #'res18_128_distance', 'res18_128_pool'
    method = 'dist' # 'dist','chebyshev', 'minkowski','simmilarity'
    process_list = ['dynamic_val','dynamic_test'] 

    for fold_num in range(10):
        for process in process_list:
            fold = 'fold'+str(fold_num)
            print('fold: ',fold)
            model_pth = os.path.join('models', compare_method, method, fold, "l3_model.pth")

            model = torch.load(model_pth).cuda()
            model.eval()

            prefix = 'data/HPA/Adjacent/' + 'res18_128_' + method
            graphs,graphs_labels = load_Adjacent(prefix)

            ## split dataset into train, val, test
            train_genes, val_genes, test_genes,train_dataset, val_dataset,test_dataset,max_num_nodes, input_dim, attention_input_dim = \
                    cross_val.prepare_val_data(graphs,graphs_labels, args, fold_num, max_nodes=0)

            paraPath = 'inference/'+ compare_method + '/' + method+ '_' + fold + '_dynamic_val'+'.txt'
            
            ## compute theta and tao from valid dataset
            if process == 'dynamic_val':
                saveprefix = 'inference/'+ compare_method + '/' +method+'_val_theta/'+ method+ '_' + fold
                np.save(saveprefix+'_genes',val_genes)
                best_cons_val,best_sita_val,best_result_val = inference(dataset = val_dataset, saveprefix = saveprefix,model = model,name='Validation')
                with open(paraPath, 'w') as f:
                    f.write("best_cons_val:   %.4f\n" % best_cons_val)
                    f.write("best_sita_val:   %.4f\n" % best_sita_val)
                f.close()

            ## compute dynamic threshold with theta and tao from valid dataset
            if process == 'dynamic_test':
                para_dict = {}
                fo =  open(paraPath,'r')
                for line in fo.readlines():
                    line = line.strip()
                    parameters = line.split(' ')
                    para_dict[parameters[0]] = float(parameters[-1]) 
                fo.close()
                print(para_dict)
                saveprefix = 'inference/'+ compare_method + '/' +method+'_test_theta/'+ method+ '_' + fold
                np.save(saveprefix+'_genes',test_genes)
                inference(dataset = test_dataset, model = model, saveprefix = saveprefix, name='Test',best_cons = para_dict['best_cons_val:'],best_sita = para_dict['best_sita_val:'])

    
def arg_parse():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset', 
            help='Input dataset.')
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument('--bmname', dest='bmname',
            help='Name of the benchmark dataset')
    io_parser.add_argument('--pkl', dest='pkl_fname',
            help='Name of the pkl data file')

    softpool_parser = parser.add_argument_group()
    softpool_parser.add_argument('--attention-ratio', dest='attention_ratio', type=float,
            help='ratio of number of nodes in consecutive layers')
    softpool_parser.add_argument('--num-pool', dest='num_pool', type=int,
            help='number of pooling layers')
    parser.add_argument('--linkpred', dest='linkpred', action='store_const',
            const=True, default=False,
            help='Whether link prediction side objective is used')

    parser.add_argument('--adj_method', dest='adj_method',
            help='Method to produce the adj matrix, such as dist, dist01, simmilarity, minkowski ...')

    parser.add_argument('--datadir', dest='datadir',
            help='Directory where benchmark is located')
    parser.add_argument('--logdir', dest='logdir',
            help='Tensorboard log directory')
    parser.add_argument('--cuda', dest='cuda',
            help='CUDA.')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
            help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
            help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
            help='Number of epochs to train.')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
            help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
            help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
            help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
            help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
            help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,
            help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
            const=False, default=True,
            help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
            help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
            const=False, default=True,
            help='Whether to add bias. Default to True.')
    parser.add_argument('--no-log-graph', dest='log_graph', action='store_const',
            const=False, default=True,
            help='Whether disable log graph')

    parser.add_argument('--method', dest='method',
            help='Method. Possible values: base, base-set2set, GNN')
    parser.add_argument('--name-suffix', dest='name_suffix',
            help='suffix added to the output filename')

    parser.set_defaults(adj_method='dist',
                        datadir='data',
                        logdir='log',
                        dataset='syn1v2',
                        max_nodes=100,
                        cuda='1',
                        feature_type='default',
                        lr=0.001,
                        clip=2.0,
                        batch_size=20, #20
                        num_epochs=2500,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=1,
                        input_dim=10,
                        hidden_dim=20,
                        output_dim=20,
                        num_classes=2,
                        num_gc_layers=3,
                        dropout=0.0,
                        method='base',
                        name_suffix='',
                        attention_ratio=0.1,
                        num_pool=1
                       )
    return parser.parse_args()

if __name__ == "__main__":

    prog_args = arg_parse()
    gene_list = load_gene_list()
    bag_predict(gene_list,prog_args)
