import os.path as osp

import argparse
import torch
from torch_geometric.datasets import DBP15K
from matching.models import EuclideanGraphMatching, HyperbolicGraphMatching, HyperbolicGCN, EuclideanGCN
from manifolds.hyperboloid import Hyperboloid


parser = argparse.ArgumentParser()
parser.add_argument('--space', type=str, default='hyperbolic')
args = parser.parse_args()

class SumEmbedding(object):
    def __call__(self, data):
        data.x1, data.x2 = data.x1.sum(dim=1), data.x2.sum(dim=1)
        return data


c = 0.5
out_channels = 100
num_layers = 2
sim = 'sqdist'
cat = False
lin = True
lr = 0.0003
dropout = 0.

k = 10



device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
path = osp.join('..', 'data', 'DBP15K')
data = DBP15K(path, 'fr_en', transform=SumEmbedding())[0].to(device)
x_s = data.x1
x_t = data.x2
edge_index_s = data.edge_index1
edge_index_t = data.edge_index2
gt = torch.cat([data.train_y, data.test_y], dim=-1)
train_y = gt[:,  :5000]
test_y = gt[:, 5000:]



if args.space == 'euclidean':
    psi = EuclideanGCN(300, out_channels, num_layers,
                 cat=cat, lin=lin, dropout=dropout).to(device)
    model = EuclideanGraphMatching(psi, k=k).to(device)

elif args.space == 'hyperbolic':
    manifold = Hyperboloid()
    psi = HyperbolicGCN(manifold, 300, out_channels, c,
                          num_layers, cat=cat, lin=lin, dropout=dropout).to(device)
    model = HyperbolicGraphMatching(psi, k=k, sim=sim).to(device)
    
if args.space == 'hyperbolic':
    hyp = Hyperboloid()
    x_s = x_s / 5
    x_t = x_t / 5
    x_s = hyp.proj_tan0(x_s, c=c)
    x_s = hyp.expmap0(x_s, c=c)
    x_s = hyp.proj(x_s, c=c)

    x_t = hyp.proj_tan0(x_t, c=c)
    x_t = hyp.expmap0(x_t, c=c)
    x_t = hyp.proj(x_t, c=c)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def train():
    model.train()
    optimizer.zero_grad()

    S_L = model(x_s, edge_index_s, None, None, x_t,
                   edge_index_t, None, None, train_y)

    loss = model.loss(S_L, data.train_y)
    hits1 = model.acc(S_L, data.train_y)
    loss.backward()
#     print(f'nans in grad: {torch.isnan(model.psi.convs[0].weight.grad).sum().item()}')
#     print(f'grad: {model.psi.convs[0].weight.grad}')
    optimizer.step()
    return loss, hits1


@torch.no_grad()
def test():
    model.eval()

    S_L = model(x_s, edge_index_s, None, None, x_t,
                   edge_index_t, None, None)

    hits1 = model.acc(S_L, data.test_y)
    hits10 = model.hits_at_k(10, S_L, data.test_y)

    return hits1, hits10

for i in range(200):
    

    loss, hits1 = train()
    print(f'epoch {i}\n\tloss {loss.item()} hits1 {hits1}')

    hits1, hits10 = test()
    print(f'\ttest loss {loss.item()} hits1 {hits1}')
    



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



