import os.path as osp

import argparse
import torch
from torch_geometric.datasets import DBP15K
from matching.models import EuclideanGraphMatching, HyperbolicGraphMatching, HyperbolicRelCNN
from manifolds.hyperboloid import Hyperboloid

from dgmc.models import DGMC, RelCNN

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, default='fr_en')
parser.add_argument('--dim', type=int, default=100)
parser.add_argument('--num_layers', type=int, default=2)
args = parser.parse_args()

c = 0.5
out_channels = 100
num_layers = 2
sim = 'dot'
cat = False
lin = True
lr = 0.0003
dropout = 0.

k = 10


class SumEmbedding(object):
    def __call__(self, data):
        data.x1, data.x2 = data.x1.sum(dim=1), data.x2.sum(dim=1)
        return data


device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
path = osp.join('..', 'data', 'DBP15K')
data = DBP15K(path, args.category, transform=SumEmbedding())[0].to(device)

hyp = Hyperboloid()
data.x1 = data.x1 / 10
data.x2 = data.x2 / 10
h_s = data.x1
h_t = data.x2

h_s = hyp.proj_tan0(h_s, c=c)
h_s = hyp.expmap0(h_s, c=c)
h_s = hyp.proj(h_s, c=c)

h_t = hyp.proj_tan0(h_t, c=c)
h_t = hyp.expmap0(h_t, c=c)
h_t = hyp.proj(h_t, c=c)
data.x1 = h_s
data.x2 = h_t

psi_1 = HyperbolicRelCNN(Hyperboloid(), data.x1.size(-1), out_channels, c, num_layers, 
               cat=cat, lin=lin, dropout=0.5).to(device)

model = HyperbolicGraphMatching(psi_1, k=k, sim=sim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def train():
    model.train()
    optimizer.zero_grad()

    S_L = model(data.x1, data.edge_index1, None, None, data.x2,
                   data.edge_index2, None, None, data.train_y)

    loss = model.loss(S_L, data.train_y)
    hits1 = model.acc(S_L, data.train_y)
    loss.backward()
    optimizer.step()
    return loss, hits1


@torch.no_grad()
def test():
    model.eval()

    S_L = model(data.x1, data.edge_index1, None, None, data.x2,
                   data.edge_index2, None, None)

    hits1 = model.acc(S_L, data.test_y)
    hits10 = model.hits_at_k(10, S_L, data.test_y)

    return hits1, hits10

for i in range(200):
    

    loss, hits1 = train()
    print(f'epoch {i} loss {loss.item()} hits1 {hits1}')

    hits1, hits10 = test()
    print(f' test loss {loss.item()} hits1 {hits1}')
    



# for i in range(200):
#     # train()
#     model.train()
#     optimizer.zero_grad()

#     S_L = model(data.x1, data.edge_index1, None, None, data.x2,
#                    data.edge_index2, None, None, data.train_y)
    
#     loss = model.loss(S_L, data.train_y)
#     hits1 = model.acc(S_L, data.train_y)
#     print(f'epoch {i} loss {loss.item()} hits1 {hits1}')
#     loss.backward()
#     optimizer.step()
    
#     # test()
#     model.eval()

#     S_L = model(data.x1, data.edge_index1, None, None, data.x2,
#                    data.edge_index2, None, None)

#     hits1 = model.acc(S_L, data.test_y)
#     hits10 = model.hits_at_k(10, S_L, data.test_y)
    
# for i in range(200):
#         model.train()
#         optimizer.zero_grad()
        
#         S = model(data.x1, data.edge_index1, None, None, data.x2,
#                        data.edge_index2, None, None, data.train_y)

#         # TODO: check for logging
#         tr_loss = model.loss(S, data.train_y)
#         tr_hits1 = model.acc(S, data.train_y)
#         tr_hits10 = model.hits_at_k(10, S, data.train_y)

#         tr_loss.backward()
#         optimizer.step()
#         print(f'epoch {i} tr loss {tr_loss} h1: {tr_hits1}')
#         with torch.no_grad():
#             model.eval()
#             S = model(data.x1, data.edge_index1, None, None, data.x2,
#                            data.edge_index2, None, None, None)

#             # TODO: check for logging
#             vl_loss = model.loss(S, data.test_y)
#             vl_hits1 = model.acc(S, data.test_y)
#             vl_hits10 = model.hits_at_k(10, S, data.test_y)



