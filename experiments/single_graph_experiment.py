import torch
import sys, os, argparse
sys.path.append("../evaluation")
sys.path.append("..")
from torch_geometric.datasets import DBP15K
from matching.models import HyperbolicRelCNN
from manifolds.hyperboloid import Hyperboloid
from KFoldAssessment import KFoldAssessment
from HoldOutSelection import HoldOutSelector
from dataset import GraphMatchingDataset

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, required=True)
parser.add_argument('--space', type=str, required=True)
parser.add_argument('--sim', type=str, default='dot')
parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()

class SumEmbedding(object):
    def __call__(self, data):
        data.x1, data.x2 = data.x1.sum(dim=1), data.x2.sum(dim=1)
        return data

if args.space == 'hyperbolic':
    parameter_ranges = {'space': ['Hyperbolic'],
                        'c': [0.5, 1, 2, 4],
                        'in_channels': [50],
                        'out_channels': [30, 50, 100, 200],
                        'num_layers': [1, 2, 3],
                        'cat': [True],
                        'lin': [True],
                        'dropout': [0],
                        'sim': [args.sim],
                        'k': [10],
                        'lr': [1e-3, 3e-3],
                        'gamma': [0.7],
                        'max_epochs': [500]
                       }
else:
    parameter_ranges = {'space': ['Euclidean'],
                        'c': [None],
                        'in_channels': [50],
                        'out_channels': [100, 200],
                        'num_layers': [1, 2, 3],
                        'cat': [True],
                        'lin': [True],
                        'dropout': [0, 0.3, 0.5],
                        'sim': [args.sim],
                        'k': [10],
                        'lr': [1e-3, 3e-3],
                        'gamma': [0.7],
                        'max_epochs': [500]
                       }

    
model_selector = HoldOutSelector(parameter_ranges, full_search=True)

path = os.path.join('..', '..', 'data', 'DBP15K')
data = DBP15K(path, 'fr_en', transform=SumEmbedding())[0]

if args.space == 'hyperbolic' and args.category in ['zh_en', 'en_zh', 'ja_en', 'en_ja']:
    data.x1, data.x2 = data.x1 * 0.8, data.x2 * 0.8

gt = torch.cat([data.train_y, data.test_y], dim=-1)
data = GraphMatchingDataset(args.category, data.x1, data.edge_index1, None, data.x2, data.edge_index2, None, gt)

ass = KFoldAssessment(5, f'results/{args.category}_{args.space}_{args.sim}', model_selector, invert_folds=True)
ass.risk_assessment(data, device=args.device)