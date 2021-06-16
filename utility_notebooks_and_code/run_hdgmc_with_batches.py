import os.path as osp

import argparse
import torch
import numpy as np
from torch_geometric.datasets import DBP15K
from matplotlib import pyplot as plt

from manifolds.hyperboloid import Hyperboloid
from hdgmc.hdgmc import RelCNN, HyperbolicRelCNN, HDGMC, HDGMC_ver1
from models.encoders import MyHGCN

from torch_geometric.data import Data, DataLoader
from torch_sparse import SparseTensor

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, default='fr_en')
parser.add_argument('--train_size', type=int, default=4121)
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--rnd_dim', type=int, default=50)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--num_steps', type=int, default=10)
parser.add_argument('--k', type=int, default=10)
args = parser.parse_args()


class SumEmbedding(object):
    def __call__(self, data):
        data.x1, data.x2 = data.x1.sum(dim=1), data.x2.sum(dim=1)
        return data
    
class DBP15kSampler(torch.utils.data.DataLoader):
    def __init__(self, x_s, edge_index_s, x_t, edge_index_t, y,
                 sizes, node_idx=None,
                 transform=None, **kwargs):

        edge_index_s = edge_index_s.to('cpu')
        edge_index_t = edge_index_t.to('cpu')

        # Save for Pytorch Lightning...
        self.edge_index_s = edge_index_s
        self.edge_index_t = edge_index_t
        self.x_s = x_s
        self.x_t = x_t
        self.y = y
        self.node_idx = node_idx
        self.num_nodes_s = x_s.size(0)
        self.num_nodes_t = x_t.size(0)

        self.sizes = sizes
        self.is_sparse_tensor = isinstance(edge_index_s, SparseTensor)
        self.__val__ = None

        # Obtain a *transposed* `SparseTensor` instance.
        if not self.is_sparse_tensor:
            num_nodes = max(x_s.size(0), x_t.size(0))

            self.adj_s = SparseTensor(row=edge_index_s[0], col=edge_index_s[1],
                                      value=None,
                                      sparse_sizes=(self.num_nodes_s, self.num_nodes_s)).t()
            self.adj_t = SparseTensor(row=edge_index_t[0], col=edge_index_t[1],
                                      value=None,
                                      sparse_sizes=(self.num_nodes_t, self.num_nodes_t)).t()
        self.adj_s.storage.rowptr()
        self.adj_t.storage.rowptr()

        super(DBP15kSampler, self).__init__(
            self.y[0], collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        batch_size = len(batch)

        adjs_s = []
        adjs_t = []
        sizes_s = [batch_size]
        sizes_t = [batch_size]
        
        n_id_s, n_id_t = batch, batch
        
        for size in self.sizes:
            adj_s, n_id_s = self.adj_s.sample_adj(n_id_s, size, replace=False)
            size_s = adj_s.sparse_sizes()[::-1]
            adj_t, n_id_t = self.adj_t.sample_adj(n_id_t, size, replace=False)
            size_t = adj_t.sparse_sizes()[::-1]
            
            row, col, _ = adj_s.coo()
            edge_index_s = torch.stack([col, row], dim=0)
            adjs_s.append(edge_index_s.to(device))
            
            row, col, _ = adj_t.coo()
            edge_index_t = torch.stack([col, row], dim=0)
            adjs_t.append(edge_index_t.to(device))
            sizes_s.append(size_s[0])
            sizes_t.append(size_t[0])

        adjs_s = adjs_s[0] if len(adjs_s) == 1 else adjs_s[::-1]
        adjs_t = adjs_t[0] if len(adjs_t) == 1 else adjs_t[::-1]
        
        sizes_s = sizes_s[0] if len(sizes_s) == 1 else sizes_s[::-1]
        sizes_t = sizes_t[0] if len(sizes_t) == 1 else sizes_t[::-1]
        
        x_s = self.x_s[n_id_s]
        x_t = self.x_t[n_id_t]
        
        y = torch.arange(batch_size).unsqueeze(0).repeat_interleave(2, dim=0)
        
        out = (batch_size, x_s.to(device), sizes_s, adjs_s, x_t.to(device), sizes_t, adjs_t, y.to(device))
        
        return out

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)







device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

path = osp.join('..', 'data', 'DBP15K')
data = DBP15K(path, args.category, transform=SumEmbedding())[0].to(device)


data.x1 = data.x1 / 10
data.x2 = data.x2 / 10

# args.dim=50

psi_1 = HyperbolicRelCNN(Hyperboloid(), data.x1.shape[-1], args.dim, 1, args.num_layers, 
               cat='hyp2', lin=True, dropout=0.0, use_bias=True, use_att=False)

psi_2 = HyperbolicRelCNN(Hyperboloid(), args.rnd_dim, args.rnd_dim, 1, 
                         args.num_layers, cat='hyp1', lin=False, dropout=0.0, use_bias=False, use_att=False)

# psi_2 = RelCNN(args.rnd_dim, args.rnd_dim, 
#                          args.num_layers, cat='eucl', lin=False, dropout=0.0)
model = HDGMC(psi_1, psi_2, num_steps=None, k=args.k).to(device)
# model = HDGMC_ver1(psi_1, psi_2, num_steps=None, k=args.k).to(device)

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

sampler = DBP15kSampler(data.x1, data.edge_index1, data.x2, data.edge_index2, data.train_y, [-1]*args.num_layers, batch_size=1000)

def train():
    model.train()
    loss_total = 0
    hits1_total = 0
    hits10_total = 0
    for batch_size, x_s, sizes_s, adjs_s, x_t, sizes_t, adjs_t, y in sampler:
        optimizer.zero_grad()
        _, S_L = model(x_s, adjs_s, None, x_t,
                       adjs_t, None, sizes_s=sizes_s, sizes_t=sizes_t, y=y)

        loss = model.loss(S_L, y)
        hits1 = model.acc(S_L, y)
        hits10 = model.hits_at_k(10, S_L, y)

        loss.backward()
        optimizer.step()
        loss_total += loss.item()
        hits1_total += hits1
        hits10_total += hits10
    
    loss_total /= len(sampler)
    hits1_total /= len(sampler)
    hits10_total /= len(sampler)
    

    return loss_total, hits1_total, hits10_total


@torch.no_grad()
def test():
    print('TEST')
    model.eval()

    _, S_L = model(data.x1, data.edge_index1, None, data.x2,
                   data.edge_index2, None)
    
    loss = model.loss(S_L, test_y)
    hits1 = model.acc(S_L, test_y)
    hits10 = model.hits_at_k(10, S_L, test_y)

    return loss, hits1, hits10

loss_history_train, hits1_history_train, hits10_history_train = [], [], []
loss_history_test, hits1_history_test, hits10_history_test = [], [], []

print('Optimize initial feature matching...')
model.num_steps = 0
for epoch in range(100):
    if epoch == 2:
        print('Refine correspondence matrix...')
        model.num_steps = args.num_steps
        model.detach = True

    loss, hits1, hits10 = train()
#     print((f'{epoch:03d}: Loss: {loss:.4f}, Hits@1: {hits1:.4f}, '
#                f'Hits@10: {hits10:.4f}'))
    
    loss_history_train.append(loss)
    hits1_history_train.append(hits1)
    hits10_history_train.append(hits10)

    if epoch % 1 == 0 or epoch == 50 or epoch == 100:
        loss, hits1, hits10 = test()
        if loss.isnan().item():
            model.set_verbose(True)
        loss_history_test.append(loss)
        hits1_history_test.append(hits1)
        hits10_history_test.append(hits10)
        
        print((f'{epoch:03d}: Loss: {loss:.4f}, Hits@1: {hits1:.4f}, '
               f'Hits@10: {hits10:.4f}'))
    

# plt.figure(figsize=(20,20))
# plt.subplot(2, 2, 1)
# plt.plot(loss_history_train, label='train')
# plt.plot(loss_history_test, label='test')
# plt.title('loss')
# plt.xlabel('epoch')
# plt.legend()



# plt.subplot(2, 2, 2)
# plt.plot(hits1_history_train, label='train')
# plt.plot(hits1_history_test, label='test')
# plt.title('hits1')
# plt.xlabel('epoch')
# plt.legend()



# plt.subplot(2, 2, 3)
# plt.plot(hits10_history_train, label='train')
# plt.plot(hits10_history_test, label='test')
# plt.title('hits10')
# plt.xlabel('epoch')
# plt.legend()
# plt.savefig('progress_report/results_initial_ver3.png')