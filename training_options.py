class TrainOptions():

    def __init__(self, n_epochs=10, epoch_count=0, n_epochs_decays=0):
        self.n_epochs = n_epochs
        self.epoch_count = epoch_count
        self.n_epochs_decays = n_epochs_decays
        self.gpu_ids = 0
