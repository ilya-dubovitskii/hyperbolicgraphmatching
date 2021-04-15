import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from dgmc.models import RelCNN
from matching.models import HyperbolicRelCNN, EuclideanGraphMatching, HyperbolicGraphMatching
from manifolds.hyperboloid import Hyperboloid
from patience import Patience

_supported_manifold_list = ['Euclidean', 'Hyperbolic']



class ModelWrapper:
    def __init__(self, input_dim, config, device='cuda'):
        self.space = config.space
        self.input_dim = input_dim
        self.out_channels = config.out_channels
        self.c = config.c
        self.num_layers = config.num_layers
        self.cat = config.cat
        self.lin = config.lin
        self.dropout = config.dropout
        self.sim = config.sim
        self.k = config.k
        self.lr = config.lr
        self.gamma = config.gamma
        self.max_epochs = config.max_epochs
        self.device = device
        
    
    def run(self, dataset, tr_idx, val_idx):
        device = self.device
        if self.space == 'Euclidean':
            psi = RelCNN(self.input_dim, self.out_channels, self.num_layers,
                         batch_norm=False, cat=self.cat, lin=self.lin, dropout=self.dropout).to(device)
            model = EuclideanGraphMatching(psi, k=self.k).to(device)
            h_s, h_t = dataset.x_s, dataset.x_t
        elif self.space == 'Hyperbolic':
            manifold = Hyperboloid()
            psi = HyperbolicRelCNN(manifold, self.input_dim, self.out_channels, self.c,
                                  self.num_layers, cat=self.cat, lin=self.lin, dropout=self.dropout).to(device)
            model = HyperbolicGraphMatching(psi, k=self.k, sim=self.sim).to(device)
            
            manifold = Hyperboloid()

            h_s, h_t = dataset.x_s / 15, dataset.x_t / 15

            h_s = manifold.proj_tan0(h_s, c=self.c)
            h_s = manifold.expmap0(h_s, c=self.c)
            h_s = manifold.proj(h_s, c=self.c)

            h_t = manifold.proj_tan0(h_t, c=self.c)
            h_t = manifold.expmap0(h_t, c=self.c)
            h_t = manifold.proj(h_t, c=self.c)
            
        else:
            raise ValueError(f'Wrong manifold! Expected one of: {_supported_manifold_list}')
            
        dataset.to(device)
        print(model)
        
        optimizer = Adam(model.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=20, gamma=self.gamma)
        early_stopper = Patience()
        
        y_train = dataset.y[:, tr_idx]
        y_test = dataset.y[:, val_idx]
        
#         hyp = manifold
#         norm_s = hyp.minkowski_dot(h_s, h_s)
#         valid_s = ((norm_s > -1.1/self.c) & (norm_s < -0.9/self.c)).sum()
#         valid_s = valid_s.float() / h_s.shape[-2] 

#         norm_t = hyp.minkowski_dot(h_t, h_t)
#         valid_t = ((norm_t > -1.1/self.c) & (norm_t < -0.9/self.c)).sum()
#         valid_t = valid_t.float() / h_t.shape[-2] 

#         print('AT THE START')
#         print(f'on hyperboloid: {valid_s:.02f}, {valid_t:.02f} c = {self.c}')
#         print(f'norms: {norm_s.mean().cpu().detach().numpy().round(2)}, {norm_t.mean().cpu().detach().numpy().round(2)}')
#         print(f'dropout: {self.dropout}')
        
        for i in range(self.max_epochs):
            model.train()
            optimizer.zero_grad()
            S = model(h_s, dataset.edge_index_s, dataset.edge_attr_s, None, h_t,
                           dataset.edge_index_t, dataset.edge_attr_t, None, y=y_train)
            
            # TODO: check for logging
            tr_loss = model.loss(S, y_train)
            tr_hits1 = model.acc(S, y_train)
            tr_hits10 = model.hits_at_k(10, S, y_train)

            tr_loss.backward()
            optimizer.step()
#             print(f'epoch {i} tr loss {tr_loss} h1: {tr_hits1}')
            with torch.no_grad():
                model.eval()
                S = model(h_s, dataset.edge_index_s, dataset.edge_attr_s, None, h_t,
                           dataset.edge_index_t, dataset.edge_attr_t, None, y=None)

                # TODO: check for logging
                vl_loss = model.loss(S, y_test)
                vl_hits1 = model.acc(S, y_test)
                vl_hits10 = model.hits_at_k(10, S, y_test)

#             print(f'\t val loss {vl_loss} h1: {vl_hits1}') 
            if torch.isnan(tr_loss) or torch.isnan(vl_loss):
                print('nan loss encountered')
                tr_loss = torch.tensor([10000])
                vl_loss = torch.tensor([10000])
                break
            
            if early_stopper.stop(i, vl_loss, vl_hits1, vl_hits10, tr_loss, tr_hits1, tr_hits10):
                break
        print('val loss: ', vl_loss.item(), end=' ')
        tr_loss, tr_hits1, tr_hits10, vl_loss, vl_hits1, vl_hits10, best_epoch = early_stopper.get_best_vl_metrics()
        print('best epoch: ', best_epoch)
        
        return tr_loss, tr_hits1, tr_hits10, vl_loss, vl_hits1, vl_hits10