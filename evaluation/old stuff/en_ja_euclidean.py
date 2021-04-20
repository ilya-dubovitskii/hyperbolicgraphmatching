import torch
import sys, os
sys.path.append("..")
from torch_geometric.datasets import DBP15K
from matching.models import HyperbolicRelCNN
from manifolds.hyperboloid import Hyperboloid
from KFoldAssessment import KFoldAssessment
from HoldOutSelection import HoldOutSelector
from dataset import GraphMatchingDataset


class SumEmbedding(object):
    def __call__(self, data):
        data.x1, data.x2 = data.x1.sum(dim=1), data.x2.sum(dim=1)
        return data

parameter_ranges = {'space': ['Euclidean'],
                    'c': [None],
                    'in_channels': [50],
                    'out_channels': [30, 50, 100, 200],
                    'num_layers': [1, 2, 3],
                    'cat': [True, False],
                    'lin': [True, False],
                    'dropout': [0.3, 0.5, 0.7],
                    'sim': ['dot'],
                    'k': [10],
                    'lr': [3e-4, 1e-3],
                    'gamma': [0.7],
                    'max_epochs': [200]
                   }

ass = KFoldAssessment(5, 'en_ja_euclidean', parameter_ranges, num_configs=100)
path = os.path.join('..', '..', 'data', 'DBP15K')
data = DBP15K(path, 'en_ja', transform=SumEmbedding())[0]



gt = torch.cat([data.train_y, data.test_y], dim=-1)
data = GraphMatchingDataset('en_ja', data.x1, data.edge_index1, None, data.x2, data.edge_index2, None, gt)
ass.risk_assessment(data, device='cuda')
