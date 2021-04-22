from wrapper import ModelWrapper

class Experiment:        
    def run_valid(self, train_loader, val_loader, model_config, device):
        model = ModelWrapper(input_dim=dataset.x_s.size(-1), config=model_config, device=device) # TODO
        tr_loss, tr_hits1, tr_hits10, vl_loss, vl_hits1, vl_hits10 = model.run(train_loader=train_loader, val_loader=val_loader, 
                                                                                  tr_idx=tr_idx, val_idx=val_idx)
        
        return tr_hits1, tr_hits10, vl_hits1, vl_hits10
        
        