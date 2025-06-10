from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch


class nnUNetTrainerEarlyStop(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, output_folder,
                 batch_dice, stage, wandb_project_name=None, wandb_run_name=None):
        super().__init__(plans, configuration, fold, dataset_json, output_folder,
                         batch_dice, stage, wandb_project_name, wandb_run_name)

        # 设置 EarlyStopping 参数
        self.patience = 10  # 连续多少次 val 没提升就停止
        self.best_val_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def on_epoch_end(self):
        """
        每轮结束后，检查是否需要早停。
        """
        current_val_loss = self.logger.all_val_losses[-1] if len(self.logger.all_val_losses) > 0 else None

        if current_val_loss is not None:
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                self.counter = 0
            else:
                self.counter += 1
                print(f"Validation loss did not improve for {self.counter} epochs.")
                if self.counter >= self.patience:
                    self.early_stop = True
                    print("Early stopping triggered!")

    def run_training(self):
        """
        Override 原始训练主循环，加上 early stop 判断
        """
        self.on_train_start()

        while not self.early_stop and self.current_epoch < self.num_epochs:
            self.on_epoch_start()
            self.train_one_epoch()
            self.on_epoch_end()
            self.current_epoch += 1
            self.save_checkpoint()

        self.on_train_end()
