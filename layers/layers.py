import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch_geometric.nn import MessagePassing
from torch_scatter import gather_csr, scatter, segment_csr
import torch.nn.init as init

class Linear(Module):
    """
    Simple Linear layer with dropout.
    """

    def __init__(self, in_features, out_features, dropout, use_bias):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        hidden = self.linear.forward(x)
        return hidden
    
    def reset_parameters(self):
        self.linear.reset_parameters()
    
    
class EuclideanGC(MessagePassing):
    def __init__(self, in_channels, out_channels, dropout=0, use_att=False, use_bias=False, verbose=False):
        super().__init__(aggr='add')
        self.linear = nn.Linear(in_channels, out_channels, bias=use_bias)
        self.att = nn.Linear(2 * out_channels, 1) if use_att else None
        self.dropout = nn.Dropout(p=dropout)
        self.verbose = verbose
        self.reset_parameters()
    
    def set_verbose(self, verbose):
        self.verbose = verbose
        
    def reset_parameters(self):
        self.linear.reset_parameters()
    
    def message(self, x_j):
        return x_j
    
    def aggregate(self, inputs, x_i, x_j, index, ptr=None, dim_size=None):
        if ptr is not None:
            ptr = expand_left(ptr, dim=self.node_dim, dims=inputs.dim())
            return segment_csr(inputs, ptr, reduce=self.aggr)
        else:
            if self.verbose:
                print('--------------------AGGREGATE--------------------------')
                print(f'inputs: {inputs.shape} x_i shape: {x_i.shape}, x_j shape: {x_j.shape}')
            if self.att is not None:
                att_scores = self.att(torch.cat([x_i, x_j], dim=-1))
                att_scores = torch.sigmoid(att_scores).reshape(-1, 1)
                out = inputs * att_scores
            else:
                out = inputs
            if self.verbose:
                print('++++++++++++++++++++AGGREGATE++++++++++++++++++++++++++')
                
            return scatter(out, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)
        
    def update(self, inputs):
        if self.verbose:
            print('----------------------UPDATE-------------------------')
            print(f'inputs shape: {inputs.shape}')
            print('++++++++++++++++++++++UPDATE++++++++++++++++++++++++++')
                        
        return F.relu(inputs)
        
    def forward(self, x, edge_index):
        x = self.linear(x)
            
        return self.propagate(edge_index, x=x)