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



try:
    from pykeops.torch import LazyTensor
except ImportError:
    LazyTensor = None

def show_memory(device, msg=None):
    r = torch.cuda.memory_reserved(device) 
    a = torch.cuda.memory_allocated(device)
    f = r-a  # free inside reserved
    print(f'{msg} FREE MEMORY ON DEVICE{device} IS {f/1024.**3:.03f}')
    
def hook_fn(grad, msg=None):
#     print(f'\n------------{msg}------------')
#     print(f'the norm is: {grad.norm()}')
#     print(f'++++++++++++{msg}++++++++++++\n')
    
    return grad

class RelCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, batch_norm=False,
                 cat=True, lin=True, dropout=0.0):
        super(RelCNN, self).__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.cat = cat
        self.lin = lin
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(RelConv(in_channels, out_channels))
            self.batch_norms.append(BN(out_channels))
            in_channels = out_channels

        if self.cat:
            in_channels = self.in_channels + num_layers * out_channels
        else:
            in_channels = out_channels

        if self.lin:
            self.out_channels = out_channels
            self.final = Lin(in_channels, out_channels)
        else:
            self.out_channels = in_channels

        self.reset_parameters()

    def reset_parameters(self):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            conv.reset_parameters()
            batch_norm.reset_parameters()
        if self.lin:
            self.final.reset_parameters()

    def forward(self, x, edge_index, *args):
        """"""
        xs = [x]

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(xs[-1], edge_index)
            x = batch_norm(F.relu(x)) if self.batch_norm else F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        x = self.final(x) if self.lin else x
        return x

    def __repr__(self):
        return ('{}({}, {}, num_layers={}, batch_norm={}, cat={}, lin={}, '
                'dropout={})').format(self.__class__.__name__,
                                      self.in_channels, self.out_channels,
                                      self.num_layers, self.batch_norm,
                                      self.cat, self.lin, self.dropout)
    
    
    
class HyperbolicRelCNN(torch.nn.Module):
    def __init__(self, manifold, in_channels, out_channels, c, num_layers,
                 cat=True, lin=True, dropout=0.0, use_att=False, use_bias=True):
        super(HyperbolicRelCNN, self).__init__()
        
        assert cat in ['none', 'eucl', 'hyp1', 'hyp2']
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

        if self.cat == 'hyp2':
            in_channels = self.in_channels + num_layers * (out_channels - 1)
        elif self.cat in ['eucl', 'hyp1']:
            in_channels = self.in_channels + num_layers * out_channels
        else:
            in_channels = out_channels

        if self.lin:
            self.out_channels = out_channels
            print(f'in channels in lin: ', in_channels)
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

    def forward(self, x, edge_index, sizes=None):
        """"""
#         print(f'x shape at the start is: {x.shape}')
        if sizes is not None:
            batch_size = sizes[-1]
#             print('batch size', batch_size)
#             print(batch_size)
            xs = [x[:batch_size]]
            x_temp = x
            for i, conv in enumerate(self.convs):
                x_temp = (conv(x_temp, edge_index[i]))[:sizes[i+1]]
                xs.append(x_temp[:batch_size])
        else:
            xs = [x]
            for conv in self.convs:
                x = conv(xs[-1], edge_index)
                xs.append(x)
            
        if self.cat == 'eucl':
            x = torch.cat(xs, dim=-1)
            
        elif self.cat == 'hyp1':
            xs[0] = self.manifold.logmap0(xs[0], c=self.c) * self.beta_total / self.beta_in
            
            for x in xs[1:]:
                x = self.manifold.logmap0(x, c=self.c) * self.beta_total / self.beta_out    
            x = torch.cat(xs, dim=-1)
            x = self.manifold.expmap0(x, c=self.c)
            
        elif self.cat == 'hyp2':
            xs_temp = []
            xs_0 = torch.zeros(*xs[0].shape[:-1], 1).to(xs[0].device)
            for x in xs:
#                 print(f'shapes: {x.shape}, narrowed: {x.narrow(-1, 1, x.shape[-1]-1).shape}, 0: {((x[..., 0] * x[..., 0]).unsqueeze(-1)).shape}')
                xs_0 = xs_0 + ((x[..., 0] * x[..., 0]).unsqueeze(-1))
                xs_temp.append(x.narrow(-1, 1, x.shape[-1]-1))
            xs_0 = torch.sqrt(xs_0)
            x = torch.cat([xs_0, *xs_temp], dim=-1)
#             print(f'x shape -1 = {x.shape[-1]}')
        else:
            x = xs[-1]
            
#         print(f'x shape before lin is: {x.shape}')
        x = self.final(x) if self.lin else x
        
        return x

    def __repr__(self):
        return ('{}({}, {}, num_layers={}, cat={}, lin={}, '
                'dropout={})').format(self.__class__.__name__,
                                      self.in_channels, self.out_channels,
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


class HDGMC(torch.nn.Module):
    r"""The *Deep Graph Matching Consensus* module which first matches nodes
    locally via a graph neural network :math:`\Psi_{\theta_1}`, and then
    updates correspondence scores iteratively by reaching for neighborhood
    consensus via a second graph neural network :math:`\Psi_{\theta_2}`.

    .. note::
        See the `PyTorch Geometric introductory tutorial
        <https://pytorch-geometric.readthedocs.io/en/latest/notes/
        introduction.html>`_ for a detailed overview of the used GNN modules
        and the respective data format.

    Args:
        psi_1 (torch.nn.Module): The first GNN :math:`\Psi_{\theta_1}` which
            takes in node features :obj:`x`, edge connectivity
            :obj:`edge_index`, and optional edge features :obj:`edge_attr` and
            computes node embeddings.
        psi_2 (torch.nn.Module): The second GNN :math:`\Psi_{\theta_2}` which
            takes in node features :obj:`x`, edge connectivity
            :obj:`edge_index`, and optional edge features :obj:`edge_attr` and
            validates for neighborhood consensus.
            :obj:`psi_2` needs to hold the attributes :obj:`in_channels` and
            :obj:`out_channels` which indicates the dimensionality of randomly
            drawn node indicator functions and the output dimensionality of
            :obj:`psi_2`, respectively.
        num_steps (int): Number of consensus iterations.
        k (int, optional): Sparsity parameter. If set to :obj:`-1`, will
            not sparsify initial correspondence rankings. (default: :obj:`-1`)
        detach (bool, optional): If set to :obj:`True`, will detach the
            computation of :math:`\Psi_{\theta_1}` from the current computation
            graph. (default: :obj:`False`)
    """
    def __init__(self, psi_1, psi_2, num_steps, k=-1, detach=False):
        super(HDGMC, self).__init__()
        self.psi_1 = psi_1
        self.psi_2 = psi_2
        self.num_steps = num_steps
        self.k = k
        self.detach = detach
        self.backend = 'auto'
        self.manifold = psi_1.manifold
        self.c = psi_1.c
#         self.mlp = Seq(
#             HypLinear(self.manifold, psi_2.out_channels, 10, self.c, 0.0, True),
#             HypReLU(self.manifold, self.c),
#             HypLinear(self.manifold, 10, 1, self.c, 0.0, True),
#         )
        
        self.mlp = Seq(
            Lin(psi_2.out_channels, 10),
            ReLU(),
            Lin(10, 1),
        )
        
    def set_verbose(self, verbose=True):
        self.psi1.set_verbose(verbose)
        self.psi2.set_verbose(verbose)

    def reset_parameters(self):
        self.psi_1.reset_parameters()
        self.psi_2.reset_parameters()
        reset(self.mlp)
        
    def multi_gpu(self, device1, device2, device3):
        self.psi_1 = self.psi_1.to(device1)
        self.psi_2 = self.psi_2.to(device2)
        self.mlp = self.mlp.to(device3)

    def __top_k__(self, x_s, x_t):  # pragma: no cover
        S_ij = (x_s @ x_t.transpose(-1, -2)) - 2 * torch.einsum('bi,bj->bij', (x_s[..., : , 0], x_t[..., : , 0]))
        
        return S_ij.topk(self.k, dim=2)[1]

    def __include_gt__(self, S_idx, s_mask, y):
        r"""Includes the ground-truth values in :obj:`y` to the index tensor
        :obj:`S_idx`."""
        (B, N_s), (row, col), k = s_mask.size(), y, S_idx.size(-1)
#         print(f'shapes: mask, {s_mask.shape}, row col {row.shape}, {col.shape}, S_idx {S_idx.shape}')
#         print(f'mask:\n{s_mask}\nrow{row}\ncol{col}\nS_idx\n{S_idx}\n')
        gt_mask = (S_idx[s_mask][row] != col.view(-1, 1)).all(dim=-1)

        sparse_mask = gt_mask.new_zeros((s_mask.sum(), ))
        sparse_mask[row] = gt_mask

        dense_mask = sparse_mask.new_zeros((B, N_s))
        dense_mask[s_mask] = sparse_mask
        last_entry = torch.zeros(k, dtype=torch.bool, device=gt_mask.device)
        last_entry[-1] = 1
        dense_mask = dense_mask.view(B, N_s, 1) * last_entry.view(1, 1, k)

        return S_idx.masked_scatter(dense_mask, col[gt_mask])

    def forward(self, x_s, edge_index_s, batch_s, x_t,
                edge_index_t, batch_t, sizes_s=None, sizes_t=None, y=None):
#         print('EE 1 here is: ', edge_index_s_total.shape  if edge_index_s_total is not None else 'what')
        device_3 = 'cuda:2'
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
        h_s = self.psi_1(x_s, edge_index_s, sizes=sizes_s)
        h_t = self.psi_1(x_t, edge_index_t, sizes=sizes_t)
        h_s_len = h_s.shape[0]
#         print(f'shapes: hs: {h_s.shape} y: {y.shape if y is not None else "wops"}')

        h_s, h_t = (h_s.detach(), h_t.detach()) if self.detach else (h_s, h_t)
        
#         print('-----------HYPERBOLOID CHECK IN HDGMC--------------')
       
        
#         norm_s = Hyperboloid().minkowski_dot(h_s, h_s)
#         valid_s = ((norm_s > -1.1) & (norm_s < -0.9)).sum()
#         valid_s = valid_s.float() / h_s.shape[-2] 
        
#         norm_t = Hyperboloid().minkowski_dot(h_t, h_t)
#         valid_t = ((norm_t > -1.1) & (norm_t < -0.9)).sum()
#         valid_t = valid_t.float() / h_t.shape[-2] 
        
#         print(f'on hyperboloid: {valid_s:.02f}, {valid_t:.02f}')
#         print(f'norms: {norm_s.mean().cpu().detach().numpy().round(2)}, {norm_t.mean().cpu().detach().numpy().round(2)}')
        
#         print('++++++++++++HYPERBOLOID CHECK IN HDGMC++++++++++++++')
        
        

        h_s, s_mask = to_dense_batch(h_s, batch_s, fill_value=0)
        h_t, t_mask = to_dense_batch(h_t, batch_t, fill_value=0)
        
        #print(f'================= nans: {torch.isnan(h_s).sum().item()}, {torch.isnan(h_t).sum().item()} =============')
        
        assert h_s.size(0) == h_t.size(0), 'Encountered unequal batch-sizes'
        (B, N_s, C_out), N_t = h_s.size(), h_t.size(1)
        
        R_in, R_out = self.psi_2.in_channels, self.psi_2.out_channels

    
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
        
#             print('....................')
#             print((tmp_s * tmp_t.view(B, N_s, k, C_out)).sum(dim=-1).shape, (tmp_s[..., 0] * tmp_t.view(B, N_s, k, C_out)[..., 0]).shape)
#             print('.........................')
        S_0 = S_hat.softmax(dim=-1)[s_mask]

        for _ in range(self.num_steps):
            
            if sizes_s is not None:
                N_s, N_t = sizes_s[0], sizes_t[0]
            
            S = S_hat.softmax(dim=-1).to(self.psi_2.device)
            r_s = torch.randn((B, N_s, R_in), dtype=h_s.dtype,
                              device=self.psi_2.device) / 10

            tmp_t = r_s.view(B, N_s, 1, R_in) * S.view(B, N_s, k, 1)
            tmp_t = tmp_t.view(B, N_s * k, R_in)
            idx = S_idx.view(B, N_s * k, 1).to(self.psi_2.device)
            r_t = scatter_add(tmp_t, idx, dim=1, dim_size=N_t)

#                 r_s = self.manifold.proj_tan0(r_s, c=self.c)
            r_s = self.manifold.expmap0(r_s, c=self.c)
            r_s = self.manifold.proj(r_s, c=self.c)

#                 r_t = self.manifold.proj_tan0(r_t, c=self.c)
            r_t = self.manifold.expmap0(r_t, c=self.c)
            r_t = self.manifold.proj(r_t, c=self.c)

            r_s, r_t = to_sparse(r_s, s_mask), to_sparse(r_t, t_mask)
            
#             print(f'devices: r_t: {r_t.device}, idx: {idx.device}')

            if sizes_s is not None:
#                 print('ALRITE MATE')
#                 print(f'shapes: r: {r_s.shape}, {edge_index_s[-1].max()}')
                o_s = self.psi_2(r_s, edge_index_s, sizes=sizes_s)
                o_t = self.psi_2(r_t, edge_index_t, sizes=sizes_t)
            else:
                o_s = self.psi_2(r_s, edge_index_s.to(self.psi_2.device))
                o_t = self.psi_2(r_t, edge_index_t.to(self.psi_2.device))
            o_s, o_t = to_dense(o_s, s_mask), to_dense(o_t, t_mask)

            o_s = o_s.view(B, N_s, 1, R_out).expand(-1, -1, k, -1)
            idx = S_idx.view(B, N_s * k, 1).expand(-1, -1, R_out).to(self.psi_1.device)
#             print(f'devices: o_t: {o_t.device}, idx: {idx.device}')
            o_t = o_t.to(self.psi_1.device)
            tmp_t = torch.gather(o_t.view(B, N_t, R_out), -2, idx)
#             show_memory(device_3, msg='BEFORE D')
#             print(f'SHAPES: o_s: {o_s.shape}, tmp_t: {tmp_t.view(B, N_s, k, R_out).shape}')
            
            
#             D = self.manifold.mobius_add(-o_s.to(device_3), tmp_t.view(B, N_s, k, R_out).to(device_3), c=self.c)
            
            a1 = self.manifold.logmap0(o_s.to(device_3), self.c)
            a2 = self.manifold.logmap0(tmp_t.view(B, N_s, k, R_out).to(device_3), c=self.c)
            
            D = a1 - a2
#             show_memory(device_3, msg='AFTER D')
            S_hat = S_hat.to(device_3) + self.mlp(D).squeeze(-1).to(device_3)

        S_L = S_hat.softmax(dim=-1)[s_mask]
        S_idx = S_idx[s_mask]

        # Convert sparse layout to `torch.sparse_coo_tensor`.
        row = torch.arange(h_s_len, device=S_idx.device)
        row = row.view(-1, 1).repeat(1, k)
        idx = torch.stack([row.view(-1), S_idx.view(-1)], dim=0)
        size = torch.Size([h_s_len, N_t])

        S_sparse_0 = torch.sparse_coo_tensor(
            idx, S_0.view(-1), size, requires_grad=S_0.requires_grad)
        S_sparse_0.__idx__ = S_idx
        S_sparse_0.__val__ = S_0

        S_sparse_L = torch.sparse_coo_tensor(
            idx, S_L.view(-1).to('cuda:0'), size, requires_grad=S_L.requires_grad)
        S_sparse_L.__idx__ = S_idx
        S_sparse_L.__val__ = S_L

#             print(f'S 0: {torch.isnan(S_sparse_0).any().item()} nans')
#             print(f'S L: {torch.isnan(S_sparse_L).any().item()} nans')
#             S_L.register_hook(lambda grad: hook_fn(grad, msg='HDGMC FINAL'))

        return S_sparse_0, S_sparse_L

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
            
#         print(f'mask: {mask}')
#         print(f'mask: {mask.sum()}')
#         print(f'value: {val}')
        nll = -torch.log(val + EPS)
#         print(f'nll1: {torch.isnan(nll).any().item()} nans')
#         print(f'nll2: value is {nll}')
#         print(f'nll2: {torch.isnan(getattr(torch, reduction)(nll)).any().item()} nans')
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
        S, y = S.to('cuda:0'), y.to('cuda:0')
        assert reduction in ['mean', 'sum']
        if not S.is_sparse:
            pred = S[y[0]].argsort(dim=-1, descending=True)[:, :k]
        else:
            assert S.__idx__ is not None and S.__val__ is not None
            perm = S.__val__[y[0]].argsort(dim=-1, descending=True)[:, :k]
            pred = torch.gather(S.__idx__[y[0]].to('cuda:0'), -1, perm.to('cuda:0'))

        correct = (pred == y[1].view(-1, 1)).sum().item()
        return correct / y.size(1) if reduction == 'mean' else correct

    def __repr__(self):
        return ('{}(\n'
                '    psi_1={},\n'
                '    psi_2={},\n'
                '    num_steps={}, k={}\n)').format(self.__class__.__name__,
                                                    self.psi_1, self.psi_2,
                                                    self.num_steps, self.k)

    
    
    
    
    
######################################################################################
######################################################################################








class HDGMC_ver1(HDGMC):
    def __init__(self, psi_1, psi_2, num_steps, k=-1, detach=False):
        super().__init__(psi_1, psi_2, num_steps, k=k, detach=detach)
        self.manifold = psi_1.manifold
        self.c = psi_1.c
        
    def __top__k(self, x_s, x_t):
        x_s = self.manifold.proj_tan0(self.manifold.logmap0(x_s, c=self.c), c=self.c)
        x_t = self.manifold.proj_tan0(self.manifold.logmap0(x_t, c=self.c), c=self.c)
        
        S_ij = (x_s @ x_t.transpose(-1, -2))
        
        return S_ij.topk(self.k, dim=2)[1]
    
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
        h_s = self.psi_1(x_s, edge_index_s, edge_attr_s)
        h_t = self.psi_1(x_t, edge_index_t, edge_attr_t)
        
        h_s = self.manifold.proj_tan0(self.manifold.logmap0(h_s, c=self.c), c=self.c)
        h_t = self.manifold.proj_tan0(self.manifold.logmap0(h_t, c=self.c), c=self.c)
        
        h_s, h_t = (h_s.detach(), h_t.detach()) if self.detach else (h_s, h_t)
        
#         print('-----------HYPERBOLOID CHECK IN HDGMC--------------')
       
        
#         norm_s = Hyperboloid().minkowski_dot(h_s, h_s)
#         valid_s = ((norm_s > -1.1) & (norm_s < -0.9)).sum()
#         valid_s = valid_s.float() / h_s.shape[-2] 
        
#         norm_t = Hyperboloid().minkowski_dot(h_t, h_t)
#         valid_t = ((norm_t > -1.1) & (norm_t < -0.9)).sum()
#         valid_t = valid_t.float() / h_t.shape[-2] 
        
#         print(f'on hyperboloid: {valid_s:.02f}, {valid_t:.02f}')
#         print(f'norms: {norm_s.mean().cpu().detach().numpy().round(2)}, {norm_t.mean().cpu().detach().numpy().round(2)}')
        
#         print(f'nans: {torch.isnan(h_s).sum().item()} nans')
#         print('++++++++++++HYPERBOLOID CHECK IN HDGMC++++++++++++++')
        
        

        h_s, s_mask = to_dense_batch(h_s, batch_s, fill_value=0)
        h_t, t_mask = to_dense_batch(h_t, batch_t, fill_value=0)
        
        #print(f'================= nans: {torch.isnan(h_s).sum().item()}, {torch.isnan(h_t).sum().item()} =============')
        
        assert h_s.size(0) == h_t.size(0), 'Encountered unequal batch-sizes'
        (B, N_s, C_out), N_t = h_s.size(), h_t.size(1)
        R_in, R_out = self.psi_2.in_channels, self.psi_2.out_channels

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
        S_hat = (tmp_s * tmp_t.view(B, N_s, k, C_out)).sum(dim=-1)
        S_0 = S_hat.softmax(dim=-1)[s_mask]

        for _ in range(self.num_steps):
            S = S_hat.softmax(dim=-1)
            r_s = torch.randn((B, N_s, R_in), dtype=h_s.dtype,
                              device=h_s.device)

            tmp_t = r_s.view(B, N_s, 1, R_in) * S.view(B, N_s, k, 1)
            tmp_t = tmp_t.view(B, N_s * k, R_in)
            idx = S_idx.view(B, N_s * k, 1)
            r_t = scatter_add(tmp_t, idx, dim=1, dim_size=N_t)
            
            #                 r_s = self.manifold.proj_tan0(r_s, c=self.c)
            r_s = self.manifold.expmap0(r_s, c=self.c)
            r_s = self.manifold.proj(r_s, c=self.c)

#                 r_t = self.manifold.proj_tan0(r_t, c=self.c)
            r_t = self.manifold.expmap0(r_t, c=self.c)
            r_t = self.manifold.proj(r_t, c=self.c)
            
            r_s, r_t = to_sparse(r_s, s_mask), to_sparse(r_t, t_mask)
            o_s = self.psi_2(r_s, edge_index_s, edge_attr_s)
            o_t = self.psi_2(r_t, edge_index_t, edge_attr_t)
            o_s, o_t = to_dense(o_s, s_mask), to_dense(o_t, t_mask)

            o_s = o_s.view(B, N_s, 1, R_out).expand(-1, -1, k, -1)
            idx = S_idx.view(B, N_s * k, 1).expand(-1, -1, R_out)
            tmp_t = torch.gather(o_t.view(B, N_t, R_out), -2, idx)
            D = o_s - tmp_t.view(B, N_s, k, R_out)
            S_hat = S_hat + self.mlp(D).squeeze(-1)

        S_L = S_hat.softmax(dim=-1)[s_mask]
        S_idx = S_idx[s_mask]

        # Convert sparse layout to `torch.sparse_coo_tensor`.
        row = torch.arange(h_s.size(0), device=S_idx.device)
        row = row.view(-1, 1).repeat(1, k)
        idx = torch.stack([row.view(-1), S_idx.view(-1)], dim=0)
        size = torch.Size([h_s.size(0), N_t])

        S_sparse_0 = torch.sparse_coo_tensor(
            idx, S_0.view(-1), size, requires_grad=S_0.requires_grad)
        S_sparse_0.__idx__ = S_idx
        S_sparse_0.__val__ = S_0

        S_sparse_L = torch.sparse_coo_tensor(
            idx, S_L.view(-1), size, requires_grad=S_L.requires_grad)
        S_sparse_L.__idx__ = S_idx
        S_sparse_L.__val__ = S_L

#             print(f'S 0: {torch.isnan(S_sparse_0).any().item()} nans')
#             print(f'S L: {torch.isnan(S_sparse_L).any().item()} nans')
#             S_L.register_hook(lambda grad: hook_fn(grad, msg='HDGMC FINAL'))

        return S_sparse_0, S_sparse_L
