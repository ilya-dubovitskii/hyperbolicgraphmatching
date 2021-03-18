import os.path as osp

import argparse
import torch
import numpy as np
from torch_geometric.datasets import DBP15K
from matplotlib import pyplot as plt
import numpy as np
import json

from manifolds.hyperboloid import Hyperboloid
from hdgmc.hdgmc import RelCNN, HyperbolicRelCNN, HDGMC
from models.encoders import MyHGCN

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

results_dict = {'hits1': {}, 'hits10': {}}

for concat in ['hyp1', 'hyp2']:
    for category in ['fr_en', 'en_fr', 'zh_en', 'en_zh', 'en_ja', 'ja_en']:
        print(f'======================={category}=======================')
        data = DBP15K(path, category, transform=SumEmbedding())[0].to(device)

        data.x1 = data.x1[..., :100] / 30 
        data.x2 = data.x2[..., :100] / 30

        psi_1 = HyperbolicRelCNN(Hyperboloid(), data.x1.shape[-1], args.dim, 1, args.num_layers, 
                       cat=concat, lin=True, dropout=0.5, use_bias=True, use_att=False)

        psi_2 = HyperbolicRelCNN(Hyperboloid(), args.rnd_dim, args.rnd_dim, 1, 
                                 args.num_layers, cat=concat, lin=False, dropout=0.0, use_bias=True, use_att=False)
        
        model = HDGMC(psi_1, psi_2, num_steps=None, k=args.k).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

        hyp = Hyperboloid()


        h_s = data.x1
        h_t = data.x2

        h_s = hyp.proj_tan0(h_s, c=1)
        h_s = hyp.expmap0(h_s, c=1)
        h_s = hyp.proj(h_s, c=1)

        h_t = hyp.proj_tan0(h_t, c=1)
        h_t = hyp.expmap0(h_t, c=1)
        h_t = hyp.proj(h_t, c=1)

        data.x1 = h_s
        data.x2 = h_t

        train_y = data.train_y
        test_y = data.test_y


        print('Optimize initial feature matching...')
        best_hits1 = 0
        best_hits10 = 0
        model.num_steps = 0
        for epoch in range(40):
            if epoch == 20:
                print('Refine correspondence matrix...')
                model.num_steps = args.num_steps
                model.detach = True

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

        results_dict['hits1'][category] = np.round(best_hits1, 3)
        results_dict['hits10'][category] = np.round(best_hits10, 3)

    
    with open(f'results_dict_{concat}.json', 'w') as fp:
        json.dump(results_dict, fp)