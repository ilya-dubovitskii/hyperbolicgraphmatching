import json

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
        
    def save(self, fname):
        config_dict = {'space': self.space, 'c': self.c,
        'out_channels': self.out_channels, 'num_layers': self.num_layers, 'sim': self.sim, 'cat': self.cat,
        'lin': self.lin, 'dropout': self.dropout, 'k': self.k, 'lr': self.lr, 'gamma': self.gamma, 'max_epochs': self.max_epochs}
        with open(fname, 'w') as fp:
            json.dump(config_dict, fp)
    
    @classmethod
    def load(self, config_dict):
        return Config(space=config_dict['space'],
                c=config_dict['c'],
                out_channels=config_dict['out_channels'],
                num_layers=config_dict['num_layers'],
                cat=config_dict['cat'],
                lin=config_dict['lin'],
                dropout=config_dict['dropout'],
                sim=config_dict['sim'],
                k=config_dict['k'],
                lr=config_dict['lr'],
                gamma=config_dict['gamma'],
                max_epochs=config_dict['max_epochs'])
    
    
    def __repr__(self):
        return f'{self.space} {self.c};\n \
        out_channels={self.out_channels}, num_layers={self.num_layers}, sim={self.sim}; cat={self.cat}, dropout={self.dropout}\n \
        lin={self.lin}; k={self.k}, lr={self.lr}, gamma={self.gamma}, max_epochs={self.max_epochs}'
        



    
    
        
        
