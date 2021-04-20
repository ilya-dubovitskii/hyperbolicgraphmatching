import torch
from torch import nn
import torch.nn.functional as F

from torch.nn import Parameter, Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch.nn import init

from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.inits import reset

from scipy.special import beta

from manifolds.hyperboloid import Hyperboloid
from layers.hyp_layers import MyHyperbolicGraphConvolution, HypLinear, HypReLU
from layers.rel import RelConv

import math
from utils.math_utils import arcosh, cosh, sinh 



try:
    from pykeops.torch import LazyTensor
except ImportError:
    LazyTensor = None

    
class HyperbolicRelCNN(torch.nn.Module):
    def __init__(self, manifold, in_channels, out_channels, c, num_layers,
                 cat=True, lin=True, dropout=0.0, use_att=False, use_bias=True):
        super(HyperbolicRelCNN, self).__init__()
        
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.cat = cat
        self.lin = lin
        self.dropout = dropout
        self.manifold = manifold
        self.c = c
        
        self.convs = torch.nn.ModuleList()

        for _ in range(self.num_layers):
            self.convs.append(MyHyperbolicGraphConvolution(manifold, in_channels, out_channels, c, dropout=dropout, use_att=use_att, use_bias=use_bias))
            in_channels = out_channels

        if self.cat:
            in_channels = self.in_channels + num_layers * out_channels
        else:
            in_channels = out_channels

        if self.lin:
            self.out_channels = out_channels
            self.final = HypLinear(manifold, in_channels, out_channels, c, dropout, use_bias)
        else:
            self.out_channels = in_channels
        
        total_channels = self.in_channels + self.num_layers * self.out_channels
        
        self.beta_in = beta(self.in_channels/2., 1./self.in_channels)
        self.beta_out = beta(self.out_channels/2., 1./self.out_channels)
        self.beta_total = beta(total_channels/2., 1./total_channels)

        self.reset_parameters()
    
    @property
    def device(self):
        return self.convs[0].weight.device
    
    def set_verbose(self, verbose=True):
        for conv in self.convs:
            conv.set_verbose(verbose)
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.lin:
            self.final.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        
        xs = [x]
        for conv in self.convs:
            x = conv(xs[-1], edge_index)
            xs.append(x)
                        
        if self.cat:
            xs[0] = self.manifold.logmap0(xs[0], c=self.c) * self.beta_total / self.beta_in
            
            for x in xs[1:]:
                x = self.manifold.logmap0(x, c=self.c) * self.beta_total / self.beta_out    
            x = torch.cat(xs, dim=-1)
            x = self.manifold.expmap0(x, c=self.c)
        else:
            x = xs[-1]
            
        x = self.final(x) if self.lin else x
        
        return x

    def __repr__(self):
        return ('{}({}, {}, c={} num_layers={}, cat={}, lin={}, '
                'dropout={})').format(self.__class__.__name__,
                                      self.in_channels, self.out_channels, self.c,
                                      self.num_layers,
                                      self.cat, self.lin, self.dropout)



EPS = 1e-8


def masked_softmax(src, mask, dim=-1):
    out = src.masked_fill(~mask, float('-inf'))
    out = torch.softmax(out, dim=dim)
    out = out.masked_fill(~mask, 0)
    return out


def to_sparse(x, mask):
    return x[mask]


def to_dense(x, mask):
    out = x.new_zeros(tuple(mask.size()) + (x.size(-1), ))
    out[mask] = x
    return out


class HyperbolicGraphMatching(torch.nn.Module):
    def __init__(self, psi, k=-1, sim='dot'):
        super().__init__()
        assert sim in ['dot', 'sqdist']
        self.psi = psi
        self.k = k
        self.backend = 'auto'
        self.manifold = psi.manifold
        self.c = psi.c
        self.sim = sim
        
    def reset_parameters(self):
        self.psi_1.reset_parameters()
        reset(self.mlp)

    def __top_k__(self, x_s, x_t):  # pragma: no cover
        S_ij = (x_s @ x_t.transpose(-1, -2)) - 2 * torch.einsum('bi,bj->bij', (x_s[..., :, 0], x_t[..., :, 0]))
        
        if self.sim == 'sqdist':
            K = 1. / self.c
            theta = torch.clamp(-S_ij / K, min=1.0 + 1e-15)
            S_ij = -K * arcosh(theta) ** 2
        
        return S_ij.topk(self.k, dim=2)[1]

    def __include_gt__(self, S_idx, s_mask, y):
        r"""Includes the ground-truth values in :obj:`y` to the index tensor
        :obj:`S_idx`."""
        (B, N_s), (row, col), k = s_mask.size(), y, S_idx.size(-1)

        gt_mask = (S_idx[s_mask][row] != col.view(-1, 1)).all(dim=-1)

        sparse_mask = gt_mask.new_zeros((s_mask.sum(), ))
        sparse_mask[row] = gt_mask

        dense_mask = sparse_mask.new_zeros((B, N_s))
        dense_mask[s_mask] = sparse_mask
        last_entry = torch.zeros(k, dtype=torch.bool, device=gt_mask.device)
        last_entry[-1] = 1
        dense_mask = dense_mask.view(B, N_s, 1) * last_entry.view(1, 1, k)

        return S_idx.masked_scatter(dense_mask, col[gt_mask])

    def forward(self, x_s, edge_index_s, edge_attr_s, batch_s, x_t,
                edge_index_t, edge_attr_t, batch_t, y=None):
        
        h_s = self.psi(x_s, edge_index_s, edge_attr_s)
        h_t = self.psi(x_t, edge_index_t, edge_attr_t)
        

        h_s, s_mask = to_dense_batch(h_s, batch_s, fill_value=0)
        h_t, t_mask = to_dense_batch(h_t, batch_t, fill_value=0)
        
        
        assert h_s.size(0) == h_t.size(0), 'Encountered unequal batch-sizes'
        (B, N_s, C_out), N_t = h_s.size(), h_t.size(1)
        
        if self.k < 1:
            # ------ Dense variant ------ #
            S_hat = (h_s @ h_t.transpose(-1, -2)) - 2 * torch.einsum('bi,bj->bij', (h_s[..., :, 0], h_t[..., :, 0]))
            if self.sim == 'sqdist':
                K = 1. / self.c
                theta = torch.clamp(-S_hat / K, min=1.0 + 1e-15)
                S_hat = -K * arcosh(theta) ** 2
                
            S_mask = s_mask.view(B, N_s, 1) & t_mask.view(B, 1, N_t)
            S_0 = masked_softmax(S_hat, S_mask, dim=-1)[s_mask]

            return S_0
        
        else:
            # ------ Sparse variant ------ #
            S_idx = self.__top_k__(h_s, h_t)  # [B, N_s, k]

            # In addition to the top-k, randomly sample negative examples and
            # ensure that the ground-truth is included as a sparse entry.
            if self.training and y is not None:
                rnd_size = (B, N_s, min(self.k, N_t - self.k))
                S_rnd_idx = torch.randint(N_t, rnd_size, dtype=torch.long,
                                          device=S_idx.device)
                S_idx = torch.cat([S_idx, S_rnd_idx], dim=-1)
                S_idx = self.__include_gt__(S_idx, s_mask, y)

            k = S_idx.size(-1)
            tmp_s = h_s.view(B, N_s, 1, C_out)
            idx = S_idx.view(B, N_s * k, 1).expand(-1, -1, C_out)
            tmp_t = torch.gather(h_t.view(B, N_t, C_out), -2, idx)
            S_hat = (tmp_s * tmp_t.view(B, N_s, k, C_out)).sum(dim=-1) - 2 * (tmp_s[..., 0] * tmp_t.view(B, N_s, k, C_out)[..., 0])

            if self.sim == 'sqdist':
                K = 1. / self.c
                theta = torch.clamp(-S_hat / K, min=1.0 + 1e-15)
                S_hat = -K * arcosh(theta) ** 2

            S_0 = S_hat.softmax(dim=-1)[s_mask]

            S_idx = S_idx[s_mask]

            # Convert sparse layout to `torch.sparse_coo_tensor`.
            row = torch.arange(x_s.size(0), device=S_idx.device)
            row = row.view(-1, 1).repeat(1, k)
            idx = torch.stack([row.view(-1), S_idx.view(-1)], dim=0)
            size = torch.Size([x_s.size(0), N_t])

            S_sparse_0 = torch.sparse_coo_tensor(idx, S_0.view(-1).cuda(0),
                                                 size, requires_grad=S_0.requires_grad).to(idx.device)
            S_sparse_0.__idx__ = S_idx
            S_sparse_0.__val__ = S_0


            return S_sparse_0

    def loss(self, S, y, reduction='mean'):
        r"""Computes the negative log-likelihood loss on the correspondence
        matrix.

        Args:
            S (Tensor): Sparse or dense correspondence matrix of shape
                :obj:`[batch_size * num_nodes, num_nodes]`.
            y (LongTensor): Ground-truth matchings of shape
                :obj:`[2, num_ground_truths]`.
            reduction (string, optional): Specifies the reduction to apply to
                the output: :obj:`'none'|'mean'|'sum'`.
                (default: :obj:`'mean'`)
        """
        assert reduction in ['none', 'mean', 'sum']
        if not S.is_sparse:
            val = S[y[0], y[1]]
        else:
            assert S.__idx__ is not None and S.__val__ is not None
            mask = S.__idx__[y[0]] == y[1].view(-1, 1)
            val = S.__val__[[y[0]]][mask]
            
        nll = -torch.log(val + EPS)

        return nll if reduction == 'none' else getattr(torch, reduction)(nll)

    def acc(self, S, y, reduction='mean'):
        r"""Computes the accuracy of correspondence predictions.

        Args:
            S (Tensor): Sparse or dense correspondence matrix of shape
                :obj:`[batch_size * num_nodes, num_nodes]`.
            y (LongTensor): Ground-truth matchings of shape
                :obj:`[2, num_ground_truths]`.
            reduction (string, optional): Specifies the reduction to apply to
                the output: :obj:`'mean'|'sum'`. (default: :obj:`'mean'`)
        """
        assert reduction in ['mean', 'sum']
        if not S.is_sparse:
            pred = S[y[0]].argmax(dim=-1)
        else:
            assert S.__idx__ is not None and S.__val__ is not None
            pred = S.__idx__[y[0], S.__val__[y[0]].argmax(dim=-1)]

        correct = (pred == y[1]).sum().item()
        return correct / y.size(1) if reduction == 'mean' else correct

    def hits_at_k(self, k, S, y, reduction='mean'):
        r"""Computes the hits@k of correspondence predictions.

        Args:
            k (int): The :math:`\mathrm{top}_k` predictions to consider.
            S (Tensor): Sparse or dense correspondence matrix of shape
                :obj:`[batch_size * num_nodes, num_nodes]`.
            y (LongTensor): Ground-truth matchings of shape
                :obj:`[2, num_ground_truths]`.
            reduction (string, optional): Specifies the reduction to apply to
                the output: :obj:`'mean'|'sum'`. (default: :obj:`'mean'`)
        """
        assert reduction in ['mean', 'sum']
        if not S.is_sparse:
            pred = S[y[0]].argsort(dim=-1, descending=True)[:, :k]
        else:
            assert S.__idx__ is not None and S.__val__ is not None
            perm = S.__val__[y[0]].argsort(dim=-1, descending=True)[:, :k]
            pred = torch.gather(S.__idx__[y[0]], -1, perm)

        correct = (pred == y[1].view(-1, 1)).sum().item()
        return correct / y.size(1) if reduction == 'mean' else correct

    def __repr__(self):
        return ('{}(\n'
                '    psi={},\n'
                '    k={}\n)').format(self.__class__.__name__,
                                                    self.psi.__repr__(),
                                                    self.k)

    
class EuclideanGraphMatching(torch.nn.Module):
    def __init__(self, psi, k=-1, sim='dot'):
        super().__init__()

        self.psi = psi
        self.k = k
        self.sim = sim
        self.backend = 'auto'

    def reset_parameters(self):
        self.psi.reset_parameters()
        reset(self.mlp)

    def __top_k__(self, x_s, x_t):  # pragma: no cover
        r"""Memory-efficient top-k correspondence computation."""
        
        if self.sim == 'dot':
            x_s = x_s  # [..., n_s, d]
            x_t = x_t.transpose(-1, -2)  # [..., d, n_t]
            S_ij = x_s @ x_t
        elif self.sim == 'sqdist':
            S_ij = -(torch.cdist(x_s, x_t) ** 2)
        else:
            raise ValueError('wrong sim')
        return S_ij.topk(self.k, dim=2)[1]

    def __include_gt__(self, S_idx, s_mask, y):
        r"""Includes the ground-truth values in :obj:`y` to the index tensor
        :obj:`S_idx`."""
        (B, N_s), (row, col), k = s_mask.size(), y, S_idx.size(-1)

        gt_mask = (S_idx[s_mask][row] != col.view(-1, 1)).all(dim=-1)

        sparse_mask = gt_mask.new_zeros((s_mask.sum(), ))
        sparse_mask[row] = gt_mask

        dense_mask = sparse_mask.new_zeros((B, N_s))
        dense_mask[s_mask] = sparse_mask
        last_entry = torch.zeros(k, dtype=torch.bool, device=gt_mask.device)
        last_entry[-1] = 1
        dense_mask = dense_mask.view(B, N_s, 1) * last_entry.view(1, 1, k)

        return S_idx.masked_scatter(dense_mask, col[gt_mask])

    def forward(self, x_s, edge_index_s, edge_attr_s, batch_s, x_t,
                edge_index_t, edge_attr_t, batch_t, y=None):
        r"""
        Args:
            x_s (Tensor): Source graph node features of shape
                :obj:`[batch_size * num_nodes, C_in]`.
            edge_index_s (LongTensor): Source graph edge connectivity of shape
                :obj:`[2, num_edges]`.
            edge_attr_s (Tensor): Source graph edge features of shape
                :obj:`[num_edges, D]`. Set to :obj:`None` if the GNNs are not
                taking edge features into account.
            batch_s (LongTensor): Source graph batch vector of shape
                :obj:`[batch_size * num_nodes]` indicating node to graph
                assignment. Set to :obj:`None` if operating on single graphs.
            x_t (Tensor): Target graph node features of shape
                :obj:`[batch_size * num_nodes, C_in]`.
            edge_index_t (LongTensor): Target graph edge connectivity of shape
                :obj:`[2, num_edges]`.
            edge_attr_t (Tensor): Target graph edge features of shape
                :obj:`[num_edges, D]`. Set to :obj:`None` if the GNNs are not
                taking edge features into account.
            batch_s (LongTensor): Target graph batch vector of shape
                :obj:`[batch_size * num_nodes]` indicating node to graph
                assignment. Set to :obj:`None` if operating on single graphs.
            y (LongTensor, optional): Ground-truth matchings of shape
                :obj:`[2, num_ground_truths]` to include ground-truth values
                when training against sparse correspondences. Ground-truths
                are only used in case the model is in training mode.
                (default: :obj:`None`)

        Returns:
            Initial and refined correspondence matrices :obj:`(S_0, S_L)`
            of shapes :obj:`[batch_size * num_nodes, num_nodes]`. The
            correspondence matrix are either given as dense or sparse matrices.
        """
        h_s = self.psi(x_s, edge_index_s, edge_attr_s)
        h_t = self.psi(x_t, edge_index_t, edge_attr_t)

        h_s, s_mask = to_dense_batch(h_s, batch_s, fill_value=0)
        h_t, t_mask = to_dense_batch(h_t, batch_t, fill_value=0)

        assert h_s.size(0) == h_t.size(0), 'Encountered unequal batch-sizes'
        (B, N_s, C_out), N_t = h_s.size(), h_t.size(1)

        if self.k < 1:
            # ------ Dense variant ------ #
            if self.sim == 'dot':
                S_hat = h_s @ h_t.transpose(-1, -2)
            elif self.sim == 'sqdist':
                S_hat = -(torch.cdist(h_s, h_t) ** 2)
            else:
                raise ValueError('wrong sim')
           
            S_mask = s_mask.view(B, N_s, 1) & t_mask.view(B, 1, N_t)
            S_0 = masked_softmax(S_hat, S_mask, dim=-1)[s_mask]

            return S_0
        else:
            # ------ Sparse variant ------ #
            S_idx = self.__top_k__(h_s, h_t)  # [B, N_s, k]

            # In addition to the top-k, randomly sample negative examples and
            # ensure that the ground-truth is included as a sparse entry.
            if self.training and y is not None:
                rnd_size = (B, N_s, min(self.k, N_t - self.k))
                S_rnd_idx = torch.randint(N_t, rnd_size, dtype=torch.long,
                                          device=S_idx.device)
                S_idx = torch.cat([S_idx, S_rnd_idx], dim=-1)
                S_idx = self.__include_gt__(S_idx, s_mask, y)

            k = S_idx.size(-1)
            tmp_s = h_s.view(B, N_s, 1, C_out)
            idx = S_idx.view(B, N_s * k, 1).expand(-1, -1, C_out)
            tmp_t = torch.gather(h_t.view(B, N_t, C_out), -2, idx)
            
            if self.sim == 'dot':
                S_hat = (tmp_s * tmp_t.view(B, N_s, k, C_out)).sum(dim=-1)
            elif self.sim == 'sqdist':
                S_hat = -((tmp_s - tmp_t.view(B, N_s, k, C_out)) ** 2).sum(dim=-1)
                
            S_0 = S_hat.softmax(dim=-1)[s_mask]

            S_idx = S_idx[s_mask]

            # Convert sparse layout to `torch.sparse_coo_tensor`.
            row = torch.arange(x_s.size(0), device=S_idx.device)
            row = row.view(-1, 1).repeat(1, k)
            idx = torch.stack([row.view(-1), S_idx.view(-1)], dim=0)
            size = torch.Size([x_s.size(0), N_t])

            S_sparse_0 = torch.sparse_coo_tensor(
                idx, S_0.view(-1).cuda(0), size, requires_grad=S_0.requires_grad).to(idx.device)
            S_sparse_0.__idx__ = S_idx
            S_sparse_0.__val__ = S_0

            return S_sparse_0

    def loss(self, S, y, reduction='mean'):
        r"""Computes the negative log-likelihood loss on the correspondence
        matrix.

        Args:
            S (Tensor): Sparse or dense correspondence matrix of shape
                :obj:`[batch_size * num_nodes, num_nodes]`.
            y (LongTensor): Ground-truth matchings of shape
                :obj:`[2, num_ground_truths]`.
            reduction (string, optional): Specifies the reduction to apply to
                the output: :obj:`'none'|'mean'|'sum'`.
                (default: :obj:`'mean'`)
        """
        assert reduction in ['none', 'mean', 'sum']
        if not S.is_sparse:
            val = S[y[0], y[1]]
        else:
            assert S.__idx__ is not None and S.__val__ is not None
            mask = S.__idx__[y[0]] == y[1].view(-1, 1)
            val = S.__val__[[y[0]]][mask]
        nll = -torch.log(val + EPS)
        return nll if reduction == 'none' else getattr(torch, reduction)(nll)

    def acc(self, S, y, reduction='mean'):
        r"""Computes the accuracy of correspondence predictions.

        Args:
            S (Tensor): Sparse or dense correspondence matrix of shape
                :obj:`[batch_size * num_nodes, num_nodes]`.
            y (LongTensor): Ground-truth matchings of shape
                :obj:`[2, num_ground_truths]`.
            reduction (string, optional): Specifies the reduction to apply to
                the output: :obj:`'mean'|'sum'`. (default: :obj:`'mean'`)
        """
        assert reduction in ['mean', 'sum']
        if not S.is_sparse:
            pred = S[y[0]].argmax(dim=-1)
        else:
            assert S.__idx__ is not None and S.__val__ is not None
            pred = S.__idx__[y[0], S.__val__[y[0]].argmax(dim=-1)]

        correct = (pred == y[1]).sum().item()
        return correct / y.size(1) if reduction == 'mean' else correct

    def hits_at_k(self, k, S, y, reduction='mean'):
        r"""Computes the hits@k of correspondence predictions.

        Args:
            k (int): The :math:`\mathrm{top}_k` predictions to consider.
            S (Tensor): Sparse or dense correspondence matrix of shape
                :obj:`[batch_size * num_nodes, num_nodes]`.
            y (LongTensor): Ground-truth matchings of shape
                :obj:`[2, num_ground_truths]`.
            reduction (string, optional): Specifies the reduction to apply to
                the output: :obj:`'mean'|'sum'`. (default: :obj:`'mean'`)
        """
        assert reduction in ['mean', 'sum']
        if not S.is_sparse:
            pred = S[y[0]].argsort(dim=-1, descending=True)[:, :k]
        else:
            assert S.__idx__ is not None and S.__val__ is not None
            perm = S.__val__[y[0]].argsort(dim=-1, descending=True)[:, :k]
            pred = torch.gather(S.__idx__[y[0]], -1, perm)

        correct = (pred == y[1].view(-1, 1)).sum().item()
        return correct / y.size(1) if reduction == 'mean' else correct

    def __repr__(self):
        return ('{}(\n'
                '    psi={},\n'
                '    k={}\n)').format(self.__class__.__name__,
                                                    self.psi,
                                                    self.k)
