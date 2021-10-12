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