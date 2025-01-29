import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import networkx as nx
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.autograd import Variable
import tensorboardX
from tensorboardX import SummaryWriter

import argparse
import os
import pickle
import random
import shutil
import time

import cross_val
import encoders
import gen.feat as featgen
import gen.data as datagen
from graph_sampler import GraphSampler
import load_data_HPA
import util
import npmetrics

from torchsummary import summary

def evaluate(dataset, model, args, name='Validation', max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        h0 = Variable(data['feats'].float()).cuda()
        labels.append(data['label'].float().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()
        attention_input = Variable(data['attention_feats'].float(), requires_grad=False).cuda()

        ypred = model(h0, adj, batch_num_nodes, attention_x=attention_input)
        ypred = ypred.cpu().data.numpy()
        ypred = (ypred > 0.5).astype(int)
        preds.append(ypred)

        if max_num_examples is not None:
            if (batch_idx+1)*args.batch_size > max_num_examples:
                break

    labels = np.vstack(labels)
    preds = np.vstack(preds)

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
    print(name, " sub_acc:", result['sub_acc'])
    return result

def evaluate_test(dataset, model, args, result_path,max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        h0 = Variable(data['feats'].float()).cuda()
        labels.append(data['label'].float().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()
        attention_input = Variable(data['attention_feats'].float(), requires_grad=False).cuda()

        ypred = model(h0, adj, batch_num_nodes, attention_x=attention_input)
        ypred = ypred.cpu().data.numpy()
        ypred = (ypred > 0.5).astype(int)
        preds.append(ypred)

        if max_num_examples is not None:
            if (batch_idx+1)*args.batch_size > max_num_examples:
                break

    labels = np.vstack(labels)
    preds = np.vstack(preds)

    npmetrics.write_metrics(labels, preds, result_path)


def gen_prefix(args):
    if args.bmname is not None:
        name = args.bmname
    else:
        name = args.dataset
    name += '_' + args.method

    name += '_l' + str(args.num_gc_layers)
    name += '_h' + str(args.hidden_dim) + '_o' + str(args.output_dim)
    if not args.bias:
        name += '_nobias'
    if len(args.name_suffix) > 0:
        name += '_' + args.name_suffix 
    # name += '_nosplit'
    return name

def gen_train_plt_name(args,val_idx):    
    #### different pool method
    if args.bmname == 'res18_128': 
        return 'results/'+ args.bmname +'/'+ args.adj_method + '/'+ gen_prefix(args) + '_fold'+str(val_idx)+'.png'
    if args.bmname == 'res18_128_distance': 
        return 'results/'+ args.bmname +'/'+ args.adj_method + '/'+ gen_prefix(args) + '_fold'+str(val_idx)+'.png'
    if args.bmname == 'res18_128_pool': 
        return 'results/'+ args.bmname +'/'+ args.method + '/'+ gen_prefix(args) + '_fold'+str(val_idx)+'.png'


def train(dataset, model, args,val_idx, same_feat=True, val_dataset=None, test_dataset=None, writer=None,
        mask_nodes = True):
    writer_batch_idx = [0, 3, 6, 9]
    
    # try different weight decay
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=0.001)
    iter = 0
    best_val_result = {
            'epoch': 0,
            'loss': 0,
            'sub_acc': 0}
    test_result = {
            'epoch': 0,
            'loss': 0,
            'sub_acc': 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []

    # load trained model and continue training
    if args.bmname == 'res18_128': 
        model_dir = 'models/'+args.bmname +'/'+args.adj_method +'/' + 'fold' + str(val_idx) + '/'

    isExists = os.path.exists(model_dir)
    if not isExists:
        os.makedirs(model_dir)

    model_pth = model_dir + 'l' + str(args.num_gc_layers)+'_'+"model.pth"
    print(model_pth)

    for epoch in range(args.num_epochs):
        total_time = 0
        avg_loss = 0.0
        model.train()
        print('Epoch: ', epoch)
        for batch_idx, data in enumerate(dataset):
            begin_time = time.time()
            model.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
            label = Variable(data['label'].float()).cuda()
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            attention_input = Variable(data['attention_feats'].float(), requires_grad=False).cuda()
            ypred = model(h0, adj, batch_num_nodes, attention_x=attention_input)
            if not args.method == 'GNN' or not args.linkpred:
                loss = model.loss(ypred, label)
            else:
                loss = model.loss(ypred, label, adj, batch_num_nodes)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss
            #if iter % 20 == 0:
            #    print('Iter: ', iter, ', loss: ', loss.data[0])
            elapsed = time.time() - begin_time
            total_time += elapsed


        avg_loss /= batch_idx + 1
        if writer is not None:
            writer.add_scalar('loss/avg_loss', avg_loss, epoch)
            if args.linkpred:
                writer.add_scalar('loss/linkpred_loss', model.link_loss, epoch)
        print('Avg loss: ', avg_loss, '; epoch time: ', total_time)
        result = evaluate(dataset, model, args, name='Train', max_num_examples=100)
        train_accs.append(result['sub_acc'])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate(val_dataset, model, args, name='Validation')
            val_accs.append(val_result['sub_acc'])
        if val_result['sub_acc'] > best_val_result['sub_acc'] - 1e-7:
            best_val_result['sub_acc'] = val_result['sub_acc']
            best_val_result['epoch'] = epoch
            best_val_result['loss'] = avg_loss
            # save model:
            torch.save(model, model_pth)
            # only plot the test result of the best chekpoint
            if test_dataset is not None:
                test_epochs.append(test_result['epoch'])
                test_accs.append(test_result['sub_acc'])

        if test_dataset is not None:
            test_result = evaluate(test_dataset, model, args, name='Test')
            test_result['epoch'] = epoch
            # save result of test dataset
            model_pth = model_dir + 'l' + str(args.num_gc_layers)+'_'+"model.pth"
            result_name = 'result_l'+str(args.num_gc_layers)+"_epoch%d.txt" % epoch
            result_path = os.path.join(model_dir, result_name)
            evaluate_test(test_dataset, model, args, result_path)
        if writer is not None:
            writer.add_scalar('acc/train_acc', result['sub_acc'], epoch)
            writer.add_scalar('acc/val_acc', val_result['sub_acc'], epoch)
            writer.add_scalar('loss/best_val_loss', best_val_result['loss'], epoch)
            if test_dataset is not None:
                writer.add_scalar('acc/test_acc', test_result['sub_acc'], epoch)

        print('Best val result: ', best_val_result)
        best_val_epochs.append(best_val_result['epoch'])
        best_val_accs.append(best_val_result['sub_acc'])
        # if test_dataset is not None:
        #     print('Test result: ', test_result)
        #     test_epochs.append(test_result['epoch'])
        #     test_accs.append(test_result['sub_acc'])

    # plot the epoch-acc figure
    matplotlib.style.use('seaborn')
    plt.switch_backend('agg')
    plt.figure()
    plt.plot(train_epochs, util.exp_moving_avg(train_accs, 0.85), '-', lw=1)
    if test_dataset is not None:
        plt.plot(best_val_epochs, best_val_accs, 'bo', test_epochs, test_accs, 'go')
        plt.legend(['train', 'val', 'test'])
    else:
        plt.plot(best_val_epochs, best_val_accs, 'bo')
        plt.legend(['train', 'val'])
    plt.savefig(gen_train_plt_name(args,val_idx), dpi=600)
    plt.close()
    matplotlib.style.use('default')

    return model, val_accs

def prepare_data(graphs, args, test_graphs=None, max_nodes=0):

    random.shuffle(graphs)
    if test_graphs is None:
        train_idx = int(len(graphs) * args.train_ratio)
        test_idx = int(len(graphs) * (1-args.test_ratio))
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx: test_idx]
        test_graphs = graphs[test_idx:]
    else:
        train_idx = int(len(graphs) * args.train_ratio)
        train_graphs = graphs[:train_idx]
        val_graphs = graph[train_idx:]
    print('Num training graphs: ', len(train_graphs), 
          '; Num validation graphs: ', len(val_graphs),
          '; Num testing graphs: ', len(test_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ', 
            max([G.number_of_nodes() for G in graphs]), ', '
            "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
            "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    # minibatch
    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=True,
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

    return train_dataset_loader, val_dataset_loader, test_dataset_loader, \
            dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.attention_feat_dim



def pkl_task(args, feat=None):
    with open(os.path.join(args.datadir, args.pkl_fname), 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    graphs = data[0]
    labels = data[1]
    test_graphs = data[2]
    test_labels = data[3]

    for i in range(len(graphs)):
        graphs[i].graph['label'] = labels[i]
    for i in range(len(test_graphs)):
        test_graphs[i].graph['label'] = test_labels[i]

    if feat is None:
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)
        for G in test_graphs:
            featgen_const.gen_node_features(G)

    train_dataset, test_dataset, max_num_nodes = prepare_data(graphs, args, test_graphs=test_graphs)
    model = encoders.GcnEncoderGraph(
            args.input_dim, args.hidden_dim, args.output_dim, args.num_classes, 
            args.num_gc_layers, bn=args.bn).cuda()
    train(train_dataset, model, args, test_dataset=test_dataset)
    evaluate(test_dataset, model, args, 'Validation')


def benchmark_task_val(args, writer=None, feat='node-label'):
    all_vals = []
    print('adj method:',args.adj_method)
    graphs,graphs_labels = load_data_HPA.read_graphfile(args.datadir, args.bmname, args.adj_method, max_nodes=args.max_nodes)  
    
    if 'feat_dim' in graphs[0].graph:
        print('Using node features')
        input_dim = graphs[0].graph['feat_dim']
    elif feat == 'node-label' and 'label' in example_node:
        print('Using node labels')
        for G in graphs:
            for u in G.nodes():
                util.node_dict(G)[u]['feat'] = np.array(util.node_dict(G)[u]['label'])
    else:
        print('Using constant labels')
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    for i in range(10):
        train_dataset, val_dataset,test_dataset,max_num_nodes, input_dim, attention_input_dim = \
                cross_val.prepare_val_data(graphs,graphs_labels, args, i, max_nodes=0)
        print(max_num_nodes)
        if args.method == 'GNN':
            print('Method: GNN')
            model = encoders.GNNPoolingGcnEncoder(
                    max_num_nodes, 
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                    args.hidden_dim, attention_ratio=args.attention_ratio, num_pooling=args.num_pool,
                    bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                    attention_input_dim=attention_input_dim).cuda()
        elif args.method == 'set2set':
            print('Method: set2set')
            model = encoders.GcnSet2SetEncoder(
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes,
                    args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).cuda()

        elif args.method == 'SortK':
            print('Method: SortK')
            model = encoders.GcnSortKPoolEncoder(
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes,
                    args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).cuda()

        _, val_accs = train(train_dataset, model, args,i, val_dataset=val_dataset, test_dataset=test_dataset,
            writer=writer)
        all_vals.append(np.array(val_accs))
    all_vals = np.vstack(all_vals)
    all_vals = np.mean(all_vals, axis=0)
    print(all_vals)
    print(np.max(all_vals))
    print(np.argmax(all_vals))
    
    
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
                        batch_size=20,
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

def main():
    prog_args = arg_parse()

    # export scalar data to JSON for external processing
    path = os.path.join(prog_args.logdir, gen_prefix(prog_args))
    if os.path.isdir(path):
        print('Remove existing log dir: ', path)
        shutil.rmtree(path)
    writer = SummaryWriter(path)
    #writer = None

    os.environ['CUDA_VISIBLE_DEVICES'] = prog_args.cuda
    print('CUDA', prog_args.cuda)

    if prog_args.bmname is not None:
        benchmark_task_val(prog_args, writer=writer)
    elif prog_args.pkl_fname is not None:
        pkl_task(prog_args)
    elif prog_args.dataset is not None:
        if prog_args.dataset == 'syn1v2':
            syn_community1v2(prog_args, writer=writer)
        if prog_args.dataset == 'syn2hier':
            syn_community2hier(prog_args, writer=writer)

    writer.close()

if __name__ == "__main__":
    main()

