import argparse
import os
import pickle
import sys

import numpy as np
import torch
from torch.nn import BCELoss
from torch.optim import Adam

from data_loader.dataset import DataSet
from modules.model import DevignModel, GGNNSum
from trainer import train
from utils import tally_param, debug


if __name__ == '__main__':
    torch.manual_seed(1000)
    np.random.seed(1000)
    path = "/project/def-m2nagapp/partha9/Devign/"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='Type of the model (devign/ggnn)',
                        choices=['devign', 'ggnn'], default='ggnn')
    parser.add_argument('--dataset', type=str, required=False,
                        help='Name of the dataset for experiment.', default=path)
    parser.add_argument('--input_dir', type=str, required=False,
                        help='Input Directory of the parser', default=path)
    parser.add_argument('--node_tag', type=str,
                        help='Name of the node feature.', default='node_features')
    parser.add_argument('--graph_tag', type=str,
                        help='Name of the graph feature.', default='graph')
    parser.add_argument('--label_tag', type=str,
                        help='Name of the label feature.', default='targets')

    parser.add_argument('--feature_size', type=int,
                        help='Size of feature vector for each node', default=100)
    parser.add_argument('--graph_embed_size', type=int,
                        help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int,
                        help='Number of steps in GGNN', default=6)
    parser.add_argument('--batch_size', type=int,
                        help='Batch Size for training', default=128)
    parser.add_argument('--max_iter', type=int,
                        help='Maximum Iteration', default=500000)
    args = parser.parse_args()
    CUDA = True
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
                          l_ident=args.label_tag)
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

    debug('Total Parameters : %d' % tally_param(model))
    debug('#' * 100)
    loss_function = BCELoss(reduction='sum')
    optim = Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    model_name = '/GGNNSumModel' if args.model_type == 'ggnn' else "DevignModel"
    train(model=model, dataset=dataset, max_steps=args.max_iter, dev_every=128,
          loss_function=loss_function, optimizer=optim,
          save_path=model_dir + model_name, max_patience=100, log_every=None, cuda=CUDA)
