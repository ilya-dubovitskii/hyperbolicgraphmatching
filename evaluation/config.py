class Config:
    def __init__(self, space, c, out_channels,
                num_layers, cat, lin, dropout, sim, k, lr, gamma, max_epochs):
        self.space = space
        self.c = c
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.cat = cat
        self.lin = lin
        self.dropout = dropout
        self.sim = sim
        self.k = k
        self.lr = lr
        self.gamma = gamma
        self.max_epochs = max_epochs
        
    def __repr__(self):
        return f'{self.space} {self.c};\n \
        out_channels={self.out_channels}, num_layers={self.num_layers}, sim={self.sim}; cat={self.cat},\n \
        lin={self.lin}; k={self.k}, lr={self.lr}, gamma={self.gamma}, max_epochs={self.max_epochs}'
        



    
    
        
        
