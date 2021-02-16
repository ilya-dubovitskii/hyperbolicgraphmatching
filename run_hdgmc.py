import os.path as osp

import argparse
import torch
import numpy as np
from torch_geometric.datasets import DBP15K
from matplotlib import pyplot as plt


from hdgmc.hdgmc import RelCNN, HyperbolicRelCNN, HDGMC
from models.encoders import MyHGCN

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
path = osp.join('..', 'data', 'DBP15K')
data = DBP15K(path, args.category, transform=SumEmbedding())[0].to(device)



psi_1 = HyperbolicRelCNN(100, 100, args.num_layers, batch_norm=False,
               cat=True, lin=False, dropout=0.5)

# psi_1 = MyHGCN()
# psi_1 = HyperbolicRelCNN(50, 50, 2)
psi_2 = RelCNN(args.rnd_dim, args.rnd_dim, args.num_layers, batch_norm=False,
               cat=True, lin=False, dropout=0.0)
model = HDGMC(psi_1, psi_2, num_steps=None, k=args.k).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

data.x1 = data.x1[:, :100]
data.x2 = data.x2[:, :100]

train_y = data.train_y
test_y = data.test_y



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

loss_history_train, hits1_history_train, hits10_history_train = [], [], []
loss_history_test, hits1_history_test, hits10_history_test = [], [], []

print('Optimize initial feature matching...')
model.num_steps = 0
for epoch in range(100):
    if epoch == 50:
        print('Refine correspondence matrix...')
        model.num_steps = args.num_steps
        model.detach = True

    loss, hits1, hits10 = train()
    
    loss_history_train.append(loss)
    hits1_history_train.append(hits1)
    hits10_history_train.append(hits10)

    if epoch % 1 == 0 or epoch == 50 or epoch == 100:
        loss, hits1, hits10 = test()
        
        loss_history_test.append(loss)
        hits1_history_test.append(hits1)
        hits10_history_test.append(hits10)
        
        print((f'{epoch:03d}: Loss: {loss:.4f}, Hits@1: {hits1:.4f}, '
               f'Hits@10: {hits10:.4f}'))
    

plt.figure(figsize=(10,10))
plt.plot(loss_history_train, label='train')
plt.plot(loss_history_test, label='test')
plt.title('loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig('../loss.png')


plt.figure(figsize=(10,10))
plt.plot(hits1_history_train, label='train')
plt.plot(hits1_history_test, label='test')
plt.title('hits1')
plt.xlabel('epoch')
plt.legend()
plt.savefig('../hits1.png')


plt.figure(figsize=(10,10))
plt.plot(hits10_history_train, label='train')
plt.plot(hits10_history_test, label='test')
plt.title('hits10')
plt.xlabel('epoch')
plt.legend()
plt.savefig('../hits10.png')