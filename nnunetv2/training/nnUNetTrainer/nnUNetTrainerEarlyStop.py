from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainerEarlyStop(nnUNetTrainer):
    def __init__(self, *args, patience=20, min_delta=0.001, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def on_epoch_end(self):
        current_val_loss = self.logger.get_last_validation_loss()
        if current_val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = current_val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        if self.epochs_without_improvement >= self.patience:
            print(f"Early stopping triggered at epoch {self.epoch}.")
            self.done_training = True
