"""Hyperbolic layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

from torch_scatter import gather_csr, scatter, segment_csr
from torch_geometric.nn import MessagePassing


import numpy as np


def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers - 1
    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, local_agg):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout, local_agg)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, x, adj):
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
        
        return h


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(1))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, local_agg):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
       

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )


class MyHyperbolicGraphConvolution(MessagePassing):
    def __init__(self, in_channels, out_channels, manifold, c, verbose=False):
        super().__init__(aggr='add')
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
#         self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.manifold = manifold
        self.c = c
        self.verbose = verbose
        self.reset_parameters()
        
    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=1)
        print(f'shape: {self.weight.shape}, norm: {np.linalg.norm(self.weight.detach().cpu().numpy(), ord=2)}')
#         init.constant_(self.bias, 0)

    def aggregate(self, x_i, x_j, index, ptr=None, dim_size=None):
        if ptr is not None:
            ptr = expand_left(ptr, dim=self.node_dim, dims=inputs.dim())
            return segment_csr(inputs, ptr, reduce=self.aggr)
        else:
            if self.verbose:
                print('--------------------AGGREGATE--------------------------')
                print(f'x_i shape: {x_i.shape}, x_j shape: {x_j.shape}')
            x_j = self.manifold.mobius_matvec(self.weight, x_j, self.c, self.verbose)
            x_j = self.manifold.logmap(x_j, x_i, self.c, self.verbose)
            if self.verbose:
                print('++++++++++++++++++++AGGREGATE++++++++++++++++++++++++++')
            return scatter(x_j, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)
    
    def message(self, x_j):
        if self.verbose:
            print('--------------------MESSAGE--------------------------')
        out = self.manifold.mobius_matvec(self.weight, x_j, self.c, self.verbose)
        if self.verbose:
            print('++++++++++++++++++++MESSAGE++++++++++++++++++++++++++')
        if self.weight.grad is not None:
            print(f'THE GRADIENT OF THE WEIGHT IS {self.weight.grad.abs().sum()}')
        return out
        
    def update(self, x_j, x):
        if self.verbose:
            print('----------------------UPDATE-------------------------')
            print(f'x shape: {x.shape}, x_j shape: {x_j.shape}')
        x = self.manifold.mobius_matvec(self.weight, x, self.c, self.verbose)
        x = self.manifold.expmap(x_j, x, self.c, self.verbose)
        x = self.manifold.to_poincare(x, self.c, self.verbose)
        x = F.relu(x)
        x = self.manifold.to_hyperboloid(x, self.c, self.verbose)
        if self.verbose:
            print('++++++++++++++++++++++UPDATE++++++++++++++++++++++++++')
        return x
        
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)