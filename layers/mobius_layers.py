import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

from torch_scatter import gather_csr, scatter, segment_csr
from torch_geometric.nn import MessagePassing
import geoopt


class MobiusGC(MessagePassing):
    def __init__(self, manifold, in_channels, out_channels, dropout=0, use_att=False, use_bias=False, verbose=False):
        super().__init__(aggr='add')
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
        if use_bias:
            self.bias = geoopt.ManifoldParameter(nn.Parameter(torch.empty(out_channels)), manifold=manifold)
        else:
            self.bias = None    
        self.att = nn.Linear(2 * out_channels, 1) if use_att else None
        self.dropout = nn.Dropout(p=dropout)
        self.manifold = manifold
        self.verbose = verbose
        self.reset_parameters()

    def set_verbose(self, verbose):
        self.verbose = verbose

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=1)
        if self.bias is not None:
            init.constant_(self.bias, 0)

    def message(self, x_j, x_i):
        out = self.manifold.logmap(x_j, x_i)

        return out

    def aggregate(self, inputs, x_i, x_j, index, ptr=None, dim_size=None):
        if ptr is not None:
            ptr = expand_left(ptr, dim=self.node_dim, dims=inputs.dim())
            return segment_csr(inputs, ptr, reduce=self.aggr)
        else:
            if self.verbose:
                print('--------------------AGGREGATE--------------------------')
                print(f'inputs: {inputs.shape} x_i shape: {x_i.shape}, x_j shape: {x_j.shape}')
            if self.att is not None:
                x_i0 = self.manifold.logmap0(x_i)
                x_j0 = self.manifold.logmap0(x_j)
                att_scores = self.att(torch.cat([x_i0, x_j0], dim=-1))
                att_scores = torch.sigmoid(att_scores).reshape(-1, 1)
                out = inputs * att_scores
            else:
                out = inputs
            if self.verbose:
                print('++++++++++++++++++++AGGREGATE++++++++++++++++++++++++++')

            return scatter(out, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)

    def update(self, inputs, x):
        if self.verbose:
            print('----------------------UPDATE-------------------------')
            print(f'inputs shape: {inputs.shape}, x shape: {x.shape}')

        out = self.manifold.expmap(inputs, x)
        out = self.manifold.projx(out)
        out = self.manifold.logmap0(out)
        out = F.relu(out)
        out = self.manifold.expmap0(out)
        out = self.manifold.projx(out)
        if self.verbose:
            print('++++++++++++++++++++++UPDATE++++++++++++++++++++++++++')

        return out

    def forward(self, x, edge_index):
        drop_weight = self.dropout(self.weight)
        x = self.manifold.mobius_matvec(drop_weight, x, project=False)
        x = self.manifold.projx(x)
        if self.bias is not None:
            x = self.manifold.mobius_add(x, self.bias)

        return self.propagate(edge_index, x=x)

