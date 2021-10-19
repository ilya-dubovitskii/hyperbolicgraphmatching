import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset

from torch_geometric.data import DataLoader
from dgmc.models import RelCNN
from matching.models import EuclideanGCN, HyperbolicGCN, EuclideanGraphMatching, HyperbolicGraphMatching, MobiusGCN, \
    MobiusGraphMatching
from manifolds.hyperboloid import Hyperboloid
from multigraph.patience import Patience
import geoopt

_supported_manifold_list = ['Euclidean', 'Hyperbolic', 'Mobius']
_c_to_norm = {0.005: 0.5, 0.01: 0.6, 0.015: 0.8, 0.02: 0.9, 0.025: 1, 0.04: 1.2, 0.08: 1.7, 0.16: 2.4, 0.32: 3.4,
              0.64: 4.8, 0.5: 3.5, 1: 5.5, 2: 7, 4: 9}


class ModelWrapper:
    def __init__(self, input_dim, config, device='cuda', dataset_type='pascal'):
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
        self.dataset_type = dataset_type

    def generate_y(self, data):
        if self.dataset_type == 'pascal':
            y_row = torch.arange(data.y.size(0), device=self.device)
            y = torch.stack([y_row, data.y], dim=0)
        elif self.dataset_type == 'pascal_pf':
            y = torch.stack([data.y_index_s, data.y_t], dim=0)
        elif self.dataset_type == 'willow':
            raise NotImplementedError()
        else:
            raise ValueError('wrong dataset')
        return y

    def preprocess_input(self, dataset, manifold):
        if self.space == 'Euclidean':
            h_s, h_t = dataset.x_s, dataset.x_t
        elif self.space == 'Hyperbolic':
            norm = _c_to_norm[self.c]
            h_s, h_t = dataset.x_s / norm, dataset.x_t / norm

            h_s = manifold.proj_tan0(h_s, c=self.c)
            h_s = manifold.expmap0(h_s, c=self.c)
            h_s = manifold.proj(h_s, c=self.c)

            h_t = manifold.proj_tan0(h_t, c=self.c)
            h_t = manifold.expmap0(h_t, c=self.c)
            h_t = manifold.proj(h_t, c=self.c)

            norm_s = manifold.minkowski_dot(h_s, h_s).squeeze()
            norm_t = manifold.minkowski_dot(h_t, h_t).squeeze()

            mask_s = (norm_s < -1.1 / self.c) | (norm_s > -0.9 / self.c)
            mask_t = (norm_t < -1.1 / self.c) | (norm_t > -0.9 / self.c)

            h_s[mask_s] = manifold.expmap0(dataset.x_s[mask_s] / (2 * norm ** 2), c=self.c)
            h_t[mask_t] = manifold.expmap0(dataset.x_t[mask_t] / (2 * norm ** 2), c=self.c)
        elif self.space == 'Mobius':
            h_s = manifold.expmap0(dataset.x_s)
            h_s = manifold.projx(h_s)

            h_t = manifold.expmap0(dataset.x_t)
            h_t = manifold.projx(h_t)

        return h_s, h_t

    def run(self, dataset, tr_idx, val_idx, val_dataset=None):
        device = self.device
        if self.space == 'Euclidean':
            manifold = None
            psi = EuclideanGCN(self.input_dim, self.out_channels, self.num_layers,
                               cat=self.cat, lin=self.lin, dropout=self.dropout).to(device)
            model = EuclideanGraphMatching(psi, k=self.k).to(device)

        elif self.space == 'Hyperbolic':
            manifold = Hyperboloid()
            psi = HyperbolicGCN(manifold, self.input_dim, self.out_channels, self.c,
                                self.num_layers, cat=self.cat, lin=self.lin, dropout=self.dropout).to(device)
            model = HyperbolicGraphMatching(psi, k=self.k, sim=self.sim).to(device)


        elif self.space == 'Mobius':
            print('Mobius NNs are initializing...')
            manifold = geoopt.PoincareBall()
            psi = MobiusGCN(manifold, self.input_dim, self.out_channels,
                            self.num_layers, cat=self.cat, lin=self.lin, dropout=self.dropout).to(device)
            model = MobiusGraphMatching(psi, k=self.k, sim=self.sim).to(device)

        else:
            raise ValueError(f'Wrong manifold! Expected one of: {_supported_manifold_list}')

        # dataset.to(device)

        if self.space == 'Mobius':
            optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=self.lr)
        else:
            optimizer = Adam(model.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=20, gamma=self.gamma)
        early_stopper = Patience()

        train_dataset = Subset(dataset, tr_idx)
        if not val_dataset:
            val_dataset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_dataset, 64, shuffle=True,
                                  follow_batch=['x_s', 'x_t'])
        val_loader = DataLoader(val_dataset, 64, shuffle=False,
                                follow_batch=['x_s', 'x_t'])
        for i in range(self.max_epochs):
            model.train()
            total_loss = 0
            correct_at_1 = correct_at_10 = num_examples = 0

            for data in train_loader:
                optimizer.zero_grad()
                data = data.to(device)
                h_s, h_t = self.preprocess_input(data, manifold)
                S = model(h_s, data.edge_index_s, data.edge_attr_s,
                          data.x_s_batch, h_t, data.edge_index_t,
                          data.edge_attr_t, data.x_t_batch)

                y = self.generate_y(data)
                tr_loss = model.loss(S, y)
                total_loss += tr_loss.item() * (data.x_s_batch.max().item() + 1)
                correct_at_1 += model.acc(S, y, reduction='sum')
                correct_at_10 += model.hits_at_k(10, S, y, reduction='sum')
                num_examples += y.size(1)
                tr_loss.backward()
                optimizer.step()

            tr_loss = total_loss / len(train_loader.dataset)
            tr_hits1 = correct_at_1 / num_examples
            tr_hits10 = correct_at_10 / num_examples

            with torch.no_grad():
                model.eval()
                correct_at_1 = correct_at_10 = num_examples = 0
                if self.dataset_type == 'pascal_pf' and isinstance(val_dataset, list):
                    vl_loss = vl_hits1 = vl_hits10 = 0
                    for dataset in val_dataset:

                        for pair in dataset.pairs:
                            data_s, data_t = dataset[pair[0]], dataset[pair[1]]
                            data_s, data_t = data_s.to(device), data_t.to(device)
                            S = model(data_s.x, data_s.edge_index, data_s.edge_attr, None,
                                      data_t.x, data_t.edge_index, data_t.edge_attr, None)
                            y = torch.arange(data_s.num_nodes, device=device)
                            y = torch.stack([y, y], dim=0)
                            vl_loss = model.loss(S, y)
                            total_loss += vl_loss.item() * (data.x_s_batch.max().item() + 1)
                            correct_at_1 += model.acc(S, y, reduction='sum')
                            correct_at_10 += model.hits_at_k(10, S, y, reduction='sum')
                            num_examples += y.size(1)
                        vl_loss += total_loss / len(val_loader.dataset)
                        vl_hits1 += correct_at_1 / num_examples
                        vl_hits10 += correct_at_10 / num_examples
                    vl_loss = vl_loss / len(dataset.pairs)
                    vl_hits1 = vl_hits1 / len(dataset.pairs)
                    vl_hits10 = vl_hits10 / len(dataset.pairs)

                else:
                    for data in val_loader:
                        data = data.to(device)
                        h_s, h_t = self.preprocess_input(data, manifold)
                        S = model(h_s, data.edge_index_s, data.edge_attr_s,
                                  data.x_s_batch, h_t, data.edge_index_t,
                                  data.edge_attr_t, data.x_t_batch)

                        y = self.generate_y(data)
                        vl_loss = model.loss(S, y)
                        total_loss += vl_loss.item() * (data.x_s_batch.max().item() + 1)
                        correct_at_1 += model.acc(S, y, reduction='sum')
                        correct_at_10 += model.hits_at_k(10, S, y, reduction='sum')
                        num_examples += y.size(1)

                    vl_loss = total_loss / len(val_loader.dataset)
                    vl_hits1 = correct_at_1 / num_examples
                    vl_hits10 = correct_at_10 / num_examples

            # if float.isnan(tr_loss) or float.isnan(vl_loss):
            #     print('NAN LOSS')
            #     tr_loss = torch.tensor([10000])
            #     vl_loss = torch.tensor([10000])
            #     break

            if early_stopper.stop(i, vl_loss, vl_hits1, vl_hits10, tr_loss, tr_hits1, tr_hits10):
                break
        tr_loss, tr_hits1, tr_hits10, vl_loss, vl_hits1, vl_hits10, best_epoch = early_stopper.get_best_vl_metrics()
        #         print(f'TRAINING FINISHED, BEST EPOCH: {best_epoch}\n\tTR: L {tr_loss:.03f}, H1 {tr_hits1:.03f};\n\tVL: L: {vl_loss:.03f} H1:{vl_hits1:.03f}')

        print(f'BEST EPOCH: {best_epoch}\n\tTR: L {tr_loss}, H1 {tr_hits1};\n\tVL: L: {vl_loss} H1:{vl_hits1}')

        return tr_loss, tr_hits1, tr_hits10, vl_loss, vl_hits1, vl_hits10
