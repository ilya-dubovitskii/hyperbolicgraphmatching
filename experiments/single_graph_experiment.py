import torch
import sys, os, argparse, json
sys.path.append("../evaluation")
sys.path.append("..")
from torch_geometric.datasets import DBP15K
from manifolds.hyperboloid import Hyperboloid
from single_graph.KFoldAssessment import KFoldAssessment
from single_graph.HoldOutSelection import HoldOutSelector
from single_graph.dataset import GraphMatchingDataset

parser = argparse.ArgumentParser()
parser.add_argument('--space', type=str, required=True)
parser.add_argument('--sim', type=str, default='dot')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='dbp15k')
parser.add_argument('--category', type=str, default='fr_en')
parser.add_argument('--num_folds', type=int, default='5')
parser.add_argument('--emb_dim', type=int, default=50)

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
                        'lr': [0.9*1e-3, 1e-3, 1.1*1e-3, 1.2*1e-3],
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
                        'lr': [0.9*1e-3, 1e-3, 1.1*1e-3, 1.2*1e-3],
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
                        'lr': [0.9*1e-3, 1e-3, 1.1*1e-3, 1.2*1e-3],
                        'gamma': [0.8],
                        'max_epochs': [500]
                       }


else:
    raise ValueError(f'Wrong space!')
         
model_selector = HoldOutSelector(parameter_ranges, full_search=True)

if args.dataset == 'dbp15k':
    path = os.path.join('..', '..', 'data', 'DBP15K')
    data = DBP15K(path, args.category, transform=SumEmbedding())[0]

    if args.space == 'hyperbolic' and args.category in ['zh_en', 'en_zh', 'ja_en', 'en_ja']:
        data.x1, data.x2 = data.x1 * 0.8, data.x2 * 0.8

    gt = torch.cat([data.train_y, data.test_y], dim=-1)
    data = GraphMatchingDataset(args.category, data.x1, data.edge_index1,
                                None, data.x2, data.edge_index2, None, gt)
elif args.dataset == 'anatomy':
    datapath = '../../data/anatomy'
    x_s = torch.load(f'{datapath}/mouse_embs.pt').float()
    x_t = torch.load(f'{datapath}/human_embs.pt').float()
    edge_index_s = torch.load(f'{datapath}/mouse_edge_index.pt').long()
    edge_index_t = torch.load(f'{datapath}/human_edge_index.pt').long()
    gt = torch.load(f'{datapath}/gt.pt').long()
    data = GraphMatchingDataset('anatomy', x_s, edge_index_s,
                                None, x_t, edge_index_t, None, gt)
elif args.dataset == 'largebio':
    datapath = '../../data/largebio'
    x_s = torch.load(f'{datapath}/fma_embs.pt').float()
    x_t = torch.load(f'{datapath}/nci_embs.pt').float()
    edge_index_s = torch.load(f'{datapath}/fma_edge_index.pt').long()
    edge_index_t = torch.load(f'{datapath}/nci_edge_index.pt').long()
    gt = torch.load(f'{datapath}/gt.pt').long()
    data = GraphMatchingDataset('largebio', x_s, edge_index_s,
                                None, x_t, edge_index_t, None, gt)
elif args.dataset == 'pubmed':
    datapath = '../../data/opentargets_pubmed'
    x_s = torch.load(f'{datapath}/pubmed_node_embeddings.pt').float()
    x_t = torch.load(f'{datapath}/ot_node_embeddings.pt').float()
    edge_index_s = torch.load(f'{datapath}/pubmed_edge_index.pt').long()
    edge_index_t = torch.load(f'{datapath}/ot_edge_index.pt').long()
    gt = torch.load(f'{datapath}/pubmed_to_ot_gt.pt').long()
    data = GraphMatchingDataset('opentargets_pubmed', x_s, edge_index_s,
                                None, x_t, edge_index_t, None, gt)
else:
    raise ValueError('wrong dataset')

if args.dataset == 'dbp15k':
    results_path = f'results/{args.dataset}/{args.category}/{args.space}_{args.sim}_{args.emb_dim}_{EXP_NAME}'
else:
    results_path = f'results/{args.dataset}/{args.space}_{args.sim}_{args.emb_dim}_{EXP_NAME}'

    
if args.num_folds > 5:
    half_folds = True
else:
    half_folds = False
    
ass = KFoldAssessment(args.num_folds, results_path, model_selector, invert_folds=False, half_folds=half_folds)
ass.risk_assessment(data, device=args.device)

with open(f'{results_path}/parameter_ranges.json', 'w') as fp:
    json.dump(parameter_ranges, fp)
