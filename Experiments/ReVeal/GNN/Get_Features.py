import argparse
import os
import pickle
import sys
import json
import torch
from dgl.nn import GatedGraphConv
from torch import nn
import torch.nn.functional as f

import numpy as np
import torch
from torch.nn import BCELoss
from torch.optim import Adam

from data_loader.dataset import DataSet
from tqdm import trange
from trainer import train
from utils import tally_param, debug


class GGNNSum(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):
        super(GGNNSum, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim, n_steps=num_steps,
                                   n_etypes=max_edge_types)
        self.classifier = nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, cuda=False, device=None):
        # print("from model", cuda, device)
        graph, features, edge_types = batch.get_network_inputs(
            cuda=cuda, device=device)
        # print(next(self.ggnn.parameters()).is_cuda, graph.device)
        outputs = self.ggnn(graph, features, edge_types)
        h_i, _ = batch.de_batchify_graphs(outputs)
        ggnn_sum = self.classifier(h_i.sum(dim=1))
        return ggnn_sum, h_i.sum(dim=1)


if __name__ == '__main__':
    torch.manual_seed(1000)
    np.random.seed(1000)
    path = ""  # "/project/def-m2nagapp/partha9/Devign/"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='Type of the model (devign/ggnn)',
                        choices=['devign', 'ggnn'], default='ggnn')
    parser.add_argument('--dataset', type=str, required=False,
                        help='Name of the dataset for experiment.', default=path)
    parser.add_argument('--input_dir', type=str, required=False,
                        help='Input Directory of the parser', default=path + "input_dir/")
    parser.add_argument('--node_tag', type=str,
                        help='Name of the node feature.', default='node_features')
    parser.add_argument('--graph_tag', type=str,
                        help='Name of the graph feature.', default='graph')
    parser.add_argument('--label_tag', type=str,
                        help='Name of the label feature.', default='targets')

    parser.add_argument('--feature_size', type=int,
                        help='Size of feature vector for each node', default=169)
    parser.add_argument('--graph_embed_size', type=int,
                        help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int,
                        help='Number of steps in GGNN', default=6)
    parser.add_argument('--batch_size', type=int,
                        help='Batch Size for training', default=128)
    parser.add_argument('--max_iter', type=int,
                        help='Maximum Iteration', default=500000)
    args = parser.parse_args()
    CUDA = False
    if args.feature_size > args.graph_embed_size:
        print('Warning!!! Graph Embed dimension should be at least equal to the feature dimension.\n'
              'Setting graph embedding size to feature size', file=sys.stderr)
        args.graph_embed_size = args.feature_size

    model_dir = os.path.join('models', args.dataset)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    input_dir = args.input_dir
    processed_data_path = os.path.join(input_dir, 'processed.bin')
    if os.path.exists(processed_data_path):
        debug('Reading already processed data from %s!' % processed_data_path)
        dataset = pickle.load(open(processed_data_path, 'rb'))
        debug(len(dataset.train_examples), len(
            dataset.valid_examples), len(dataset.test_examples))
    else:
        dataset = DataSet(train_src=os.path.join(input_dir, 'train_GGNNinput.json'),
                          valid_src=os.path.join(
                              input_dir, 'valid_GGNNinput.json'),
                          test_src=os.path.join(
                              input_dir, 'test_GGNNinput.json'),
                          batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
                          l_ident=args.label_tag, return_feature=True)
        file = open(processed_data_path, 'wb')
        pickle.dump(dataset, file)
        file.close()
    debug('Dataset feature size %s!' % dataset.feature_size)
    assert args.feature_size == dataset.feature_size, \
        'Dataset contains different feature vector than argument feature size. ' \
        'Either change the feature vector size in argument, or provide different dataset.'
    if args.model_type == 'ggnn':

        model = GGNNSum(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                        num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)
        if os.path.exists(model_dir + '/GGNNSumModel' + '-model.bin'):
            model.load_state_dict(torch.load(
                model_dir + '/GGNNSumModel' + '-model.bin', map_location='cuda:0' if CUDA else "cpu"))
            debug('Loaded from previous')
    else:
        model = DevignModel(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                            num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)
        if os.path.exists(model_dir + '/DevignModel' + '-model.bin'):

            model.load_state_dict(torch.load(
                model_dir + '/DevignModel' + '-model.bin', map_location='cuda:0' if CUDA else "cpu"))
            debug('Loaded from previous')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # print(device)

    model.eval()
    num_batches = dataset.initialize_train_batch()
    data_iter = dataset.get_next_train_batch
    data = []
    for _ in trange(num_batches):
        graph, targets, ids = data_iter()
        targets = targets.cuda() if CUDA else targets
        _, sum_value = model(graph, cuda=CUDA, device=device)
        graph_features = sum_value.cpu().detach().numpy().tolist()
        for target, graph_feature, id in zip(targets, graph_features, ids):
            data.append({
                "target": target.tolist(),
                "graph_feature": graph_feature,
                "id": int(id.tolist())
            })
    with open("input_dir/train_GGNNinput_graph.json", "w") as f:
        json.dump(data, f)


    num_batches = dataset.initialize_valid_batch()
    data_iter = dataset.get_next_valid_batch
    data = []
    for _ in trange(num_batches):
        graph, targets, ids = data_iter()
        targets = targets.cuda() if CUDA else targets
        _, sum_value = model(graph, cuda=CUDA, device=device)
        graph_features = sum_value.cpu().detach().numpy().tolist()
        for target, graph_feature, id in zip(targets, graph_features, ids):
            data.append({
                "target": target.tolist(),
                "graph_feature": graph_feature,
                "id": int(id.tolist())
            })
    with open("input_dir/valid_GGNNinput_graph.json", "w") as f:
        json.dump(data, f)



    num_batches = dataset.initialize_test_batch()
    data_iter = dataset.get_next_test_batch
    data = []
    for _ in trange(num_batches):
        graph, targets, ids = data_iter()
        targets = targets.cuda() if CUDA else targets
        _, sum_value = model(graph, cuda=CUDA, device=device)
        graph_features = sum_value.cpu().detach().numpy().tolist()
        for target, graph_feature, id in zip(targets, graph_features, ids):
            data.append({
                "target": target.tolist(),
                "graph_feature": graph_feature,
                "id": int(id.tolist())
            })
    with open("input_dir/test_GGNNinput_graph.json", "w") as f:
        json.dump(data, f)
