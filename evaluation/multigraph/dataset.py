import torch
from torch_geometric.data import Data
import random


class GraphMatchingDataset:
    def __init__(self, name, x_s, edge_index_s, edge_attr_s,
                 x_t, edge_index_t, edge_attr_t, y):
        self.x_s = x_s
        self.edge_index_s = edge_index_s
        self.edge_attr_s = edge_attr_s
        self.x_t = x_t
        self.edge_index_t = edge_index_t
        self.edge_attr_t = edge_attr_t
        self.y = y
        self.name = name

    def to(self, device):
        self.y = self.y.to(device)
        self.x_s, self.x_t = self.x_s.to(device), self.x_t.to(device)
        self.edge_index_s, self.edge_index_t = self.edge_index_s.to(device), self.edge_index_t.to(device)
        if self.edge_attr_s is not None:
            self.edge_attr_s, self.edge_attr_t = self.edge_attr_s.to(device), self.edge_attr_t.to(device)


class RandomGraphDataset(torch.utils.data.Dataset):
    def __init__(self, min_inliers, max_inliers, min_outliers, max_outliers,
                 min_scale=0.9, max_scale=1.2, noise=0.05, transform=None):

        self.min_inliers = min_inliers
        self.max_inliers = max_inliers
        self.min_outliers = min_outliers
        self.max_outliers = max_outliers
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.noise = noise
        self.transform = transform

    def __len__(self):
        return 1024

    def __getitem__(self, idx):
        num_inliers = random.randint(self.min_inliers, self.max_inliers)
        num_outliers = random.randint(self.min_outliers, self.max_outliers)

        pos_s = 2 * torch.rand((num_inliers, 2)) - 1
        pos_t = pos_s + self.noise * torch.randn_like(pos_s)

        y_s = torch.arange(pos_s.size(0))
        y_t = torch.arange(pos_t.size(0))

        pos_s = torch.cat([pos_s, 3 - torch.rand((num_outliers, 2))], dim=0)
        pos_t = torch.cat([pos_t, 3 - torch.rand((num_outliers, 2))], dim=0)

        data_s = Data(pos=pos_s, y_index=y_s)
        data_t = Data(pos=pos_t, y=y_t)

        if self.transform is not None:
            data_s = self.transform(data_s)
            data_t = self.transform(data_t)

        data = Data(num_nodes=pos_s.size(0))
        for key in data_s.keys:
            data['{}_s'.format(key)] = data_s[key]
        for key in data_t.keys:
            data['{}_t'.format(key)] = data_t[key]

        return data
