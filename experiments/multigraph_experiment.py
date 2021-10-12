import torch
import sys, os, argparse, json
import os.path as osp

import argparse
import torch
from torch_geometric.datasets import PascalVOCKeypoints as PascalVOC
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

from dgmc.utils import ValidPairDataset
from dgmc.models import DGMC, SplineCNN

sys.path.append("../evaluation")
sys.path.append("..")
from torch_geometric.datasets import DBP15K
from manifolds.hyperboloid import Hyperboloid
from multigraph.KFoldAssessment import KFoldAssessment
from multigraph.HoldOutSelection import HoldOutSelector
from multigraph.dataset import GraphMatchingDataset

parser = argparse.ArgumentParser()
parser.add_argument('--space', type=str, required=True)
parser.add_argument('--sim', type=str, default='dot')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='pascal')
parser.add_argument('--category', type=str, default='fr_en')
parser.add_argument('--num_folds', type=int, default='5')
parser.add_argument('--emb_dim', type=int, default=50)
parser.add_argument('--isotropic', action='store_true')

args = parser.parse_args()


class SumEmbedding(object):
    def __call__(self, data):
        data.x1, data.x2 = data.x1.sum(dim=1), data.x2.sum(dim=1)
        return data


EXP_NAME = f'exp1'
if args.space == 'hyperbolic':
    parameter_ranges = {'space': ['Hyperbolic'],
                        'c': [0.005, 0.01, 0.08, 0.16, 1],
                        'in_channels': [50],
                        'out_channels': [args.emb_dim],
                        'num_layers': [1],
                        'cat': [True],
                        'lin': [True],
                        'dropout': [0],
                        'sim': [args.sim],
                        'k': [10],
                        'lr': [0.9 * 1e-3, 1e-3, 1.1 * 1e-3, 1.2 * 1e-3],
                        'gamma': [0.8],
                        'max_epochs': [500]
                        }
elif args.space == 'euclidean':
    parameter_ranges = {'space': ['Euclidean'],
                        'c': [None],
                        'in_channels': [50],
                        'out_channels': [args.emb_dim],
                        'num_layers': [1],
                        'cat': [True],
                        'lin': [True],
                        'dropout': [0],
                        'sim': [args.sim],
                        'k': [10],
                        'lr': [0.9 * 1e-3, 1e-3, 1.1 * 1e-3, 1.2 * 1e-3],
                        'gamma': [0.8],
                        'max_epochs': [500]
                        }
elif args.space == 'mobius':
    parameter_ranges = {'space': ['Mobius'],
                        'c': [0.5, 1, 2, 3, 4],
                        'in_channels': [50],
                        'out_channels': [args.emb_dim],
                        'num_layers': [1],
                        'cat': [True],
                        'lin': [True],
                        'dropout': [0],
                        'sim': [args.sim],
                        'k': [10],
                        'lr': [0.9 * 1e-3, 1e-3, 1.1 * 1e-3, 1.2 * 1e-3],
                        'gamma': [0.8],
                        'max_epochs': [500]
                        }


else:
    raise ValueError(f'Wrong space!')

model_selector = HoldOutSelector(parameter_ranges, full_search=True)

if args.dataset == 'pascal':
    pre_filter = lambda data: data.pos.size(0) > 0  # noqa
    transform = T.Compose([
        T.Delaunay(),
        T.FaceToEdge(),
        T.Distance() if args.isotropic else T.Cartesian(),
    ])

    train_datasets = []
    test_datasets = []
    path = osp.join('..', 'data', 'PascalVOC')
    for category in PascalVOC.categories:
        dataset = PascalVOC(path, category, train=True, transform=transform,
                            pre_filter=pre_filter)
        train_datasets += [ValidPairDataset(dataset, dataset, sample=True)]
        dataset = PascalVOC(path, category, train=False, transform=transform,
                            pre_filter=pre_filter)
        test_datasets += [ValidPairDataset(dataset, dataset, sample=True)]
    data = torch.utils.data.ConcatDataset(train_datasets)
    # data = DataLoader(train_dataset, args.batch_size, shuffle=True,
    #                           follow_batch=['x_s', 'x_t'])
    # gt = torch.cat([data.train_y, data.test_y], dim=-1)
    # data = GraphMatchingDataset(args.category, data.x1, data.edge_index1,
    #                             None, data.x2, data.edge_index2, None, gt)
else:
    raise ValueError('wrong dataset')

results_path = f'results/{args.dataset}/{args.space}_{args.sim}_{args.emb_dim}_{EXP_NAME}'

if args.num_folds > 5:
    half_folds = True
else:
    half_folds = False

ass = KFoldAssessment(args.num_folds, results_path, model_selector, invert_folds=False, half_folds=half_folds)
ass.risk_assessment(data, device=args.device)

with open(f'{results_path}/parameter_ranges.json', 'w') as fp:
    json.dump(parameter_ranges, fp)
