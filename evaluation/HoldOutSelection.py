import os
import numpy as np, pickle as pkl

from sklearn.model_selection import train_test_split

from config import Config
from experiment import Experiment


class GridSampler:
    def __init__(self):
        self.prev_parameters = set()
        
    def sample(self, parameter_ranges):
        while True:
            space = np.random.choice(parameter_ranges['space'])
            c = np.random.choice(parameter_ranges['c'])
            out_channels = np.random.choice(parameter_ranges['out_channels'])
            num_layers = np.random.choice(parameter_ranges['num_layers'])
            cat = np.random.choice(parameter_ranges['cat'])
            lin = np.random.choice(parameter_ranges['lin'])
            dropout = np.random.choice(parameter_ranges['dropout'])
            sim = np.random.choice(parameter_ranges['sim'])
            k = int(np.random.choice(parameter_ranges['k']))
            lr = np.random.choice(parameter_ranges['lr'])
            gamma = np.random.choice(parameter_ranges['gamma'])
            max_epochs = np.random.choice(parameter_ranges['max_epochs'])
            param_tuple = (space, c, out_channels, num_layers,
            cat, dropout, sim, k, lr, gamma, max_epochs)
            
            if param_tuple not in self.prev_parameters:
                self.prev_parameters.add(param_tuple)
                config = Config(space=space, c=c, out_channels=out_channels,
                                num_layers=num_layers, cat=cat, lin=lin, dropout=dropout,
                                sim=sim, k=k, lr=lr, gamma=gamma, max_epochs=max_epochs)
                break
        
        return config

class HoldOutSelector:
    def __init__(self, num_configs):
        self.num_configs = num_configs
        self._CONFIG_FILENAME = 'config.pkl'
        self._RESULTS_FILENAME = 'results.pkl'
        self.WINNER_CONFIG_FILENAME = 'winner_config.pkl'
        self._FOLD_BASE = None    # to be specified in :meth: model_selection
        
    def process_results(self):
        best_vl_hits1 = 0.
        print('PROCESSING RESULTS')
        for i in range(self.num_configs):
            try:
                results_filename = os.path.join(self._FOLD_BASE, str(i+1), self._RESULTS_FILENAME)
                config_filename = os.path.join(self._FOLD_BASE, str(i+1), self._CONFIG_FILENAME)
                with open(results_filename, 'rb') as fp:
                    results_dict = pkl.load(fp)
                vl_hits1 = results_dict['VL_hits1']
                if vl_hits1 is None:
                    continue
                elif vl_hits1 > best_vl_hits1:
                    best_i = i+1
                    best_vl_hits1 = vl_hits1
                    with open(config_filename, 'rb') as fp:
                        winner_config = pkl.load(fp)
            except Exception as e:
                print(e)

        print(f'Model selection winner for experiment {self._FOLD_BASE}: {best_i} with hits1 {best_vl_hits1:.03f}')
        
        return winner_config
            
    
    def model_selection(self, dataset, idx, parameter_ranges, fold_dir, device, num_configs=100):
        self._FOLD_BASE = fold_dir
        grid_sampler = GridSampler()
        tr_idx, vl_idx = train_test_split(idx, test_size=0.2)
        for i in range(self.num_configs):
            config_folder = os.path.join(fold_dir, str(i+1))
            if not os.path.exists(config_folder):
                os.makedirs(config_folder)
            config = grid_sampler.sample(parameter_ranges)
            self._model_selection_helper(dataset, tr_idx, vl_idx, config, config_folder, device)
            print(f'{i}\n{config}')
            
        winner_config = self.process_results()
        
        return winner_config
            
    def _model_selection_helper(self, dataset, tr_idx, vl_idx, config, config_folder, device):
        exp = Experiment()
        results_dict = {}
        
        tr_h1, vl_h1, tr_h10, vl_h10 = exp.run_valid(dataset, tr_idx, vl_idx, config, device)
        
        results_dict['TR_hits1'] = tr_h1
        results_dict['VL_hits1'] = vl_h1
        results_dict['TR_hits10'] = tr_h10
        results_dict['VL_hits10'] = vl_h10
        print(f'@1: {vl_h1}', end=' ')
        with open(os.path.join(config_folder, self._RESULTS_FILENAME), 'wb') as fp:
            pkl.dump(results_dict, fp)
        with open(os.path.join(config_folder, self._CONFIG_FILENAME), 'wb') as fp:
            pkl.dump(config, fp)
        
        