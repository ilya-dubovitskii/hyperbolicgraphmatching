import os, json
import numpy as np

from sklearn.model_selection import train_test_split

from multigraph.config import Config
from multigraph.experiment import Experiment


class FullGrid:
    def __init__(self, parameter_ranges):
        self.parameter_ranges = parameter_ranges

    def generate_grid(self):
        space = self.parameter_ranges['space'][0]
        max_epochs = self.parameter_ranges['max_epochs'][0]
        sim = self.parameter_ranges['sim'][0]
        cat = self.parameter_ranges['cat'][0]
        lin = self.parameter_ranges['lin'][0]
        k = self.parameter_ranges['k'][0]
        for c in self.parameter_ranges['c']:
            for out_channels in self.parameter_ranges['out_channels']:
                for num_layers in self.parameter_ranges['num_layers']:
                    for dropout in self.parameter_ranges['dropout']:
                        for lr in self.parameter_ranges['lr']:
                            for gamma in self.parameter_ranges['gamma']:
                                config = Config(space=space, c=c, out_channels=out_channels,
                                                num_layers=num_layers, cat=cat, lin=lin, dropout=dropout,
                                                sim=sim, k=k, lr=lr, gamma=gamma, max_epochs=max_epochs)

                                yield config


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
    def __init__(self, parameter_ranges, full_search=True, num_configs=100, dataset_type='pascal'):
        self.parameter_ranges = parameter_ranges
        self.num_configs = num_configs
        self.full_search = full_search
        self._CONFIG_FILENAME = 'config.json'
        self._RESULTS_FILENAME = 'results.json'
        self.WINNER_CONFIG_FILENAME = 'winner_config.json'
        self._FOLD_BASE = None  # to be specified in :meth: model_selection
        self.dataset_type = dataset_type

    def process_results(self):
        best_vl_hits1 = 0.
        print(f'\n-------PROCESSING MODEL SELECTION RESULTS OF FOLD {self._FOLD_BASE}-------\n')
        for i in range(self.num_configs):
            try:
                results_filename = os.path.join(self._FOLD_BASE, str(i + 1), self._RESULTS_FILENAME)
                config_filename = os.path.join(self._FOLD_BASE, str(i + 1), self._CONFIG_FILENAME)
                with open(results_filename, 'r') as fp:
                    results_dict = json.load(fp)
                vl_hits1 = results_dict['VL_hits1']
                if vl_hits1 is None:
                    continue
                elif vl_hits1 > best_vl_hits1:
                    best_i = i + 1
                    best_vl_hits1 = vl_hits1
                    with open(config_filename, 'rb') as fp:
                        winner_config_dict = json.load(fp)
                        winner_config = Config.load(winner_config_dict)
                        winner_results = results_dict

            except Exception as e:
                print(e)

        print(f'Winner config: {i}')
        return winner_config

    def model_selection(self, dataset, idx, fold_dir, device='cuda', num_configs=100):
        self._FOLD_BASE = fold_dir
        tr_idx, vl_idx = train_test_split(idx, test_size=0.2)
        if self.full_search:
            grid_generator = FullGrid(self.parameter_ranges)
            i = 0
            for config in grid_generator.generate_grid():
                config_folder = os.path.join(fold_dir, str(i + 1))
                if not os.path.exists(config_folder):
                    os.makedirs(config_folder)

                #                 print(config)
                if not os.path.exists(os.path.join(config_folder, self._CONFIG_FILENAME)):
                    print(f'Config #{i + 1}')
                    self._model_selection_helper(dataset, tr_idx, vl_idx, config, config_folder, device)

                else:
                    print(f'Config {i + 1} was already processed!')
                i += 1
            self.num_configs = i
        else:
            grid_sampler = GridSampler()
            for i in range(self.num_configs):
                config_folder = os.path.join(fold_dir, str(i + 1))
                if not os.path.exists(config_folder):
                    os.makedirs(config_folder)
                config = grid_sampler.sample(self.parameter_ranges)
                print(f'Config #{i + 1}')
                self._model_selection_helper(dataset, tr_idx, vl_idx, config, config_folder, device)

        winner_config = self.process_results()

        return winner_config

    def _model_selection_helper(self, dataset, tr_idx, vl_idx, config, config_folder, device):
        exp = Experiment()
        results_dict = {}

        tr_hits1, tr_hits10, vl_hits1, vl_hits10 = exp.run_valid(dataset, tr_idx, vl_idx, config, device, self.dataset_type)

        results_dict['TR_hits1'] = tr_hits1
        results_dict['VL_hits1'] = vl_hits1
        results_dict['TR_hits10'] = tr_hits10
        results_dict['VL_hits10'] = vl_hits10

        with open(os.path.join(config_folder, self._RESULTS_FILENAME), 'w') as fp:
            try:
                json.dump(results_dict, fp)
            except Exception as e:
                print(results_dict, '\n', e)

        config.save(os.path.join(config_folder, self._CONFIG_FILENAME))


