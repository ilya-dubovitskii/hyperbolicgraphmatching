import os.path as osp

import argparse
import torch
import numpy as np
from torch_geometric.datasets import DBP15K

from dgmc.models import DGMC, RelCNN
from hdgmc import HyperbolicRelCNN, HDGMC, RiemannianAdam
from hdgmc import MyHGCN

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, default='fr_en')
parser.add_argument('--train_size', type=int, default=4121)
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--rnd_dim', type=int, default=50)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--num_steps', type=int, default=10)
parser.add_argument('--k', type=int, default=10)
args = parser.parse_args()


class SumEmbedding(object):
    def __call__(self, data):
        data.x1, data.x2 = data.x1.sum(dim=1), data.x2.sum(dim=1)
        return data


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
path = osp.join('..', 'data', 'DBP15K')
data = DBP15K(path, args.category, transform=SumEmbedding())[0].to(device)



# psi_1 = HyperbolicRelCNN(50, 50, args.num_layers, batch_norm=False,
#                cat=True, lin=False, dropout=0.5)

psi_1 = MyHGCN()
psi_2 = RelCNN(args.rnd_dim, args.rnd_dim, args.num_layers, batch_norm=False,
               cat=True, lin=False, dropout=0.0)
model = HDGMC(psi_1, psi_2, num_steps=None, k=args.k).to(device)
optimizer = torch.optim.Adam(list(model.psi_2.parameters()) + list(model.mlp.parameters()), lr=0.001)
hyp_optimizer = RiemannianAdam(list(model.psi_1.layers[0].parameters()), lr=0.001)

data.x1 = data.x1[:, :50]
data.x2 = data.x2[:, :50]


total_y = torch.cat([data.train_y, data.test_y], dim=-1)
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
    hyp_optimizer.zero_grad()

    _, S_L = model(data.x1, data.edge_index1, None, None, data.x2,
                   data.edge_index2, None, None, train_y)

    loss = model.loss(S_L, train_y)
    loss.backward()
    optimizer.step()
    hyp_optimizer.step()
    return loss


@torch.no_grad()
def test():
    model.eval()

    _, S_L = model(data.x1, data.edge_index1, None, None, data.x2,
                   data.edge_index2, None, None)

    hits1 = model.acc(S_L, test_y)
    hits10 = model.hits_at_k(10, S_L, test_y)

    return hits1, hits10


#print('Optimize initial feature matching...')
model.num_steps = 0
for epoch in range(0, 100):
    print(f'epoch: {epoch}')
    if epoch == 50:
        #print('Refine correspondence matrix...')
        model.num_steps = args.num_steps
        model.detach = True

    loss = train()

    if epoch % 1 == 0 or epoch == 50 or epoch == 100:
        hits1, hits10 = test()
        print((f'{epoch:03d}: Loss: {loss:.4f}, Hits@1: {hits1:.4f}, '
               f'Hits@10: {hits10:.4f}'))
