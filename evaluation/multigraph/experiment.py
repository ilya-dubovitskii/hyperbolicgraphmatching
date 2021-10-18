from multigraph.wrapper import ModelWrapper
from torch_geometric.data import DataLoader

class Experiment:
    def run_valid(self, dataset, tr_idx, val_idx, model_config, device, dataset_type, val_dataset=None):

        model = ModelWrapper(input_dim=next(iter(DataLoader(dataset, 32,
                                                            shuffle=True,
                                                            follow_batch=['x_s', 'x_t']))).x_s.size(-1),
                             config=model_config, device=device, dataset_type=dataset_type)
        tr_loss, tr_hits1, tr_hits10, vl_loss, vl_hits1, vl_hits10 = model.run(dataset=dataset,
                                                                               tr_idx=tr_idx, val_idx=val_idx,
                                                                               val_dataset=val_dataset)

        return tr_hits1, tr_hits10, vl_hits1, vl_hits10

