from typing import Any

from ml_collections import ConfigDict

from datasets.utils import DatasetModule


class Callback:
    def __init__(self, config: ConfigDict, trainer: Any, data_module: DatasetModule):
        self.config = config
        self.trainer = trainer
        self.data_module = data_module
        self.every_n_epochs = config.get("every_n_epochs", 1)

    def on_training_start(self):
        pass

    def on_training_end(self):
        pass

    def on_training_epoch_start(self, epoch_idx):
        if epoch_idx % self.every_n_epochs != 0:
            return
        self._on_training_epoch_start(epoch_idx)

    def _on_training_epoch_start(self, epoch_idx):
        pass

    def on_training_epoch_end(self, train_metrics, epoch_idx):
        if epoch_idx % self.every_n_epochs != 0:
            return
        self._on_training_epoch_end(train_metrics, epoch_idx)

    def _on_training_epoch_end(self, train_metrics, epoch_idx):
        pass

    def on_validation_epoch_end(self, eval_metrics, epoch_idx):
        if epoch_idx % self.every_n_epochs != 0:
            return
        self._on_validation_epoch_end(eval_metrics, epoch_idx)

    def _on_validation_epoch_end(self, eval_metrics, epoch_idx):
        pass

    def on_test_epoch_end(self, test_metrics, epoch_idx):
        # We apply it to every epoch, since we don't know when the test set is evaluated.
        self._on_test_epoch_end(test_metrics, epoch_idx)

    def _on_test_epoch_end(self, test_metrics, epoch_idx):
        pass
