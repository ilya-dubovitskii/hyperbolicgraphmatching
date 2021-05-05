import sys
sys.path.append('../..')
import os.path as osp
import os
import argparse
import torch
import numpy as np
from torch_geometric.datasets import DBP15K
import json
from dgmc.models import RelCNN
from matching.models import EuclideanGCN, HyperbolicGCN, EuclideanGraphMatching, HyperbolicGraphMatching
from manifolds.hyperboloid import Hyperboloid



class SumEmbedding(object):
    def __call__(self, data):
        data.x1, data.x2 = data.x1.sum(dim=1), data.x2.sum(dim=1)
        return data

parser = argparse.ArgumentParser()

parser.add_argument('--train_size', type=int, default=4000)
parser.add_argument('--space', type=str, default='euclidean')
parser.add_argument('--experiment_num', type=int, default=0)
args = parser.parse_args()


device = 'cuda:0'
path = os.path.join('..', '..', '..', 'data', 'DBP15K')
data = DBP15K(path, 'en_fr', transform=SumEmbedding())[0].to(device)


gt = torch.cat([data.train_y, data.test_y], dim=-1)

c = 1


if args.space == 'euclidean':
    
    psi = EuclideanGCN(data.x1.size(-1), 30, 1,
                         cat=True, lin=True, dropout=0).to(device)
    model = EuclideanGraphMatching(psi, k=10).to(device)
    h_s, h_t = data.x1, data.x2
    
elif args.space == 'hyperbolic':
    manifold = Hyperboloid()
    psi = HyperbolicGCN(manifold, data.x1.size(-1), 30, c,
                          1, cat=True, lin=True, dropout=0).to(device)
    model = HyperbolicGraphMatching(psi, k=10, sim='dot').to(device)
    
    norm = 5.5
    h_s, h_t = data.x1 / norm, data.x2 / norm

    h_s = manifold.proj_tan0(h_s, c=c)
    h_s = manifold.expmap0(h_s, c=c)
    h_s = manifold.proj(h_s, c=c)

    h_t = manifold.proj_tan0(h_t, c=c)
    h_t = manifold.expmap0(h_t, c=c)
    h_t = manifold.proj(h_t, c=c)

    norm_s = manifold.minkowski_dot(h_s, h_s).squeeze()
    norm_t = manifold.minkowski_dot(h_t, h_t).squeeze()

    mask_s = (norm_s < -1.1/c) | (norm_s > -0.9/c)
    mask_t = (norm_t < -1.1/c) | (norm_t > -0.9/c)

    h_s[mask_s] = manifold.expmap0(data.x1[mask_s]/(2*norm**2), c=c)
    h_t[mask_t] = manifold.expmap0(data.x2[mask_t]/(2*norm**2), c=c)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


total_y = gt
train_size = args.train_size
total_size = total_y.shape[-1]
total_indices = np.arange(total_size)
train_indices = np.random.choice(total_indices, size=train_size, replace=False)
test_indices = np.setdiff1d(total_indices, train_indices, assume_unique=True)
train_y = total_y[:, train_indices]

test_y = total_y[:, test_indices]



def train():
    model.train()
    optimizer.zero_grad()

    S_L = model(h_s, data.edge_index1, None, None, h_t,
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

    S_L = model(h_s, data.edge_index1, None, None, h_t,
                   data.edge_index2, None, None)
    
    hits1 = model.acc(S_L, test_y)
    hits10 = model.hits_at_k(10, S_L, test_y)

    return S_L, hits1, hits10

best_ts_hits1 = 0

for epoch in range(50):
    loss, tr_hits1, tr_hits10 = train()
    S_L, ts_hits1, ts_hits10 = test()

    if ts_hits1 > best_ts_hits1:
        best_ts_hits1 = ts_hits1
        best_ts_hits10 = ts_hits10
        best_tr_hits1 = tr_hits1
        best_tr_hits10 = tr_hits10
        
res = {'ts_hits1': ts_hits1,
        'ts_hits10': ts_hits10,
        'tr_hits1': tr_hits1,
        'tr_hits10': tr_hits10}
        
        
with open(f'results/{train_size}/{args.experiment_num}_{args.space}_res.json', 'w') as fp:
    json.dump(res, fp)
    

    
