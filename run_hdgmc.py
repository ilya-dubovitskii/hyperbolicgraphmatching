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
    


device = 'cuda:0'
device_1 = 'cuda:0'
device_2 = 'cuda:1'
device_3 = 'cuda:2'
path = osp.join('..', 'data', 'DBP15K')
data = DBP15K(path, args.category, transform=SumEmbedding())[0].to(device)


data.x1 = data.x1 / 10
data.x2 = data.x2 / 10

args.dim=100
args.rnd_dim=25
args.num_layers=1
args.num_steps=5
args.k=5

psi_1 = HyperbolicRelCNN(Hyperboloid(), data.x1.shape[-1], args.dim, 1, args.num_layers, 
               cat='hyp1', lin=True, dropout=0.0, use_bias=True, use_att=False)

psi_2 = HyperbolicRelCNN(Hyperboloid(), args.rnd_dim, args.rnd_dim, 1, 
                         args.num_layers, cat='hyp1', lin=False, dropout=0.0, use_bias=False, use_att=False)

model = HDGMC(psi_1, psi_2, num_steps=None, k=args.k).to(device)
model.multi_gpu(device_1, device_2, device_3)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


hyp = Hyperboloid()


h_s = data.x1
h_t = data.x2

h_s = hyp.proj_tan0(h_s, c=1)
h_s = hyp.expmap0(h_s, c=1)
h_s = hyp.proj(h_s, c=1)

h_t = hyp.proj_tan0(h_t, c=1)
h_t = hyp.expmap0(h_t, c=1)
h_t = hyp.proj(h_t, c=1)


norm_s = hyp.minkowski_dot(h_s, h_s)
valid_s = ((norm_s > -1.1) & (norm_s < -0.9)).sum()
valid_s = valid_s.float() / h_s.shape[-2] 

norm_t = hyp.minkowski_dot(h_t, h_t)
valid_t = ((norm_t > -1.1) & (norm_t < -0.9)).sum()
valid_t = valid_t.float() / h_t.shape[-2] 

print('AT THE START')
print(f'on hyperboloid: {valid_s:.02f}, {valid_t:.02f}')
print(f'norms: {norm_s.mean().cpu().detach().numpy().round(2)}, {norm_t.mean().cpu().detach().numpy().round(2)}')

data.x1 = h_s
data.x2 = h_t

train_y = data.train_y
test_y = data.test_y



def train():
    model.train()
    optimizer.zero_grad()
    _, S_L = model(data.x1, data.edge_index1, None, data.x2,
                   data.edge_index2, None, y=train_y)
    
    loss = model.loss(S_L, train_y)
    hits1 = model.acc(S_L, train_y)
    hits10 = model.hits_at_k(10, S_L, train_y)
    
    loss.backward()
    optimizer.step()
    

    return loss, hits1, hits10


@torch.no_grad()
def test():
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
    if epoch == 50:
        print('Refine correspondence matrix...')
        model.num_steps = args.num_steps
#         model.detach = True

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