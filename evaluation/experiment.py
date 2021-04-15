from wrapper import ModelWrapper

class Experiment:        
    def run_valid(self, dataset, tr_idx, val_idx, model_config, device):
        model = ModelWrapper(input_dim=dataset.x_s.size(-1), config=model_config, device=device)
        tr_loss, tr_hits1, tr_hits10, vl_loss, vl_hits1, vl_hits10 = model.run(dataset=dataset, 
                                                                                  tr_idx=tr_idx, val_idx=val_idx)
        
        return tr_hits1, tr_hits10, vl_hits1, vl_hits10
        
        