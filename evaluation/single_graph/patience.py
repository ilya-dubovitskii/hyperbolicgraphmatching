class Patience:

    '''
    Implement common "patience" technique
    '''

    def __init__(self, patience=30):
        self.local_vl_optimum = -1
        self.patience = patience
        self.best_epoch = -1
        self.counter = -1

        self.tr_loss, self.tr_hits1, self.tr_hits10 = None, None, None
        self.vl_loss, self.vl_hits1, self.vl_hits10 = None, None, None

    def stop(self, epoch, vl_loss, vl_hits1=None, vl_hits10=None,
             tr_loss=None, tr_hits1=None, tr_hits10=None):
        if vl_hits1 >= self.local_vl_optimum:
            self.counter = 0
            self.local_vl_optimum = vl_hits1
            self.best_epoch = epoch
            self.tr_loss, self.tr_hits1, self.tr_hits10 = tr_loss, tr_hits1, tr_hits10
            self.vl_loss, self.vl_hits1, self.vl_hits10 = vl_loss, vl_hits1, vl_hits10
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
        
    def get_best_vl_metrics(self):
        return self.tr_loss, self.tr_hits1, self.tr_hits10, self.vl_loss, self.vl_hits1, self.vl_hits10, self.best_epoch