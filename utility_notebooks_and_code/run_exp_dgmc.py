import os.path as osp

import argparse
import torch
import numpy as np
from torch_geometric.datasets import DBP15K
from matplotlib import pyplot as plt
import json
import numpy as np

from dgmc.models import DGMC, RelCNN


parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, default='fr_en')
parser.add_argument('--train_size', type=int, default=4121)
parser.add_argument('--dim', type=int, default=100)
parser.add_argument('--rnd_dim', type=int, default=50)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--num_steps', type=int, default=10)
parser.add_argument('--k', type=int, default=10)
args = parser.parse_args()


class SumEmbedding(object):
    def __call__(self, data):
        data.x1, data.x2 = data.x1.sum(dim=1), data.x2.sum(dim=1)
        return data
    
def train():
    model.train()
    optimizer.zero_grad()
    _, S_L = model(data.x1, data.edge_index1, None, None, data.x2,
                   data.edge_index2, None, None, train_y)

    loss = model.loss(S_L, train_y)
    hits1 = model.acc(S_L, train_y)
    hits10 = model.hits_at_k(10, S_L, train_y)

    loss.backward()
    optimizer.step()


    return loss, hits1, hits10


@torch.no_grad()
def test():
    model.eval()

    _, S_L = model(data.x1, data.edge_index1, None, None, data.x2,
                   data.edge_index2, None, None)

    loss = model.loss(S_L, test_y)
    hits1 = model.acc(S_L, test_y)
    hits10 = model.hits_at_k(10, S_L, test_y)

    return loss, hits1, hits10


device = 'cuda' if torch.cuda.is_available() else 'cpu'

path = osp.join('..', 'data', 'DBP15K')

results_dict = {'hits1b': {}, 'hits10b': {}, 'hits1a': {}, 'hits10a': {}}

args.dim=100
args.rnd_dim=25
args.num_layers=1
args.num_steps=10
args.k=5

for category in ['fr_en', 'en_fr', 'zh_en', 'en_zh', 'en_ja', 'ja_en']:
    print(f'======================={category}=======================')
    data = DBP15K(path, category, transform=SumEmbedding())[0].to(device)

    data.x1 = data.x1 
    data.x2 = data.x2

    psi_1 = RelCNN(data.x1.shape[-1], args.dim, args.num_layers, 
                   cat=True, lin=True, dropout=0.5)

    psi_2 = RelCNN(args.rnd_dim, args.rnd_dim, 
                             args.num_layers, cat=True, lin=False, dropout=0.0)

    model = DGMC(psi_1, psi_2, num_steps=None, k=args.k).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)


    train_y = data.train_y
    test_y = data.test_y


    print('Optimize initial feature matching...')
    best_hits1 = 0
    best_hits10 = 0
    model.num_steps = 0
    for epoch in range(100):
        if epoch == 50:
            print('Refine correspondence matrix...')
            model.num_steps = args.num_steps
            model.detach = True
            results_dict['hits1b'][category] = np.round(best_hits1, 3)
            results_dict['hits10b'][category] = np.round(best_hits10, 3)
            best_hits1 = 0
            best_hits10 = 0

        loss, hits1, hits10 = train()
    #     print((f'{epoch:03d}: Loss: {loss:.4f}, Hits@1: {hits1:.4f}, '
    #                f'Hits@10: {hits10:.4f}'))


        if epoch % 1 == 0 or epoch == 50 or epoch == 100:
            loss, hits1, hits10 = test()
            if hits1 >= best_hits1:
                best_hits1 = hits1
                best_hits10 = hits10

            print((f'{epoch:03d}: Loss: {loss:.4f}, Hits@1: {hits1:.4f}, '
                   f'Hits@10: {hits10:.4f}'))

    results_dict['hits1a'][category] = np.round(best_hits1, 3)
    results_dict['hits10a'][category] = np.round(best_hits10, 3)


with open(f'results_dict_dgmc_new.json', 'w') as fp:
    json.dump(results_dict, fp)