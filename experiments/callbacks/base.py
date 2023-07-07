from ml_collections import ConfigDict
from datasets.utils import DatasetModule
from typing import Any


class Callback:

    def __init__(self, config: ConfigDict, trainer: Any, data_module: DatasetModule):
        self.config = config
        self.trainer = trainer
        self.data_module = data_module

    def on_training_start(self):
        pass

    def on_training_end(self):
        pass

    def on_training_epoch_start(self, epoch_idx):
        pass

    def on_training_epoch_end(self, train_metrics, epoch_idx):
        pass

    def on_validation_epoch_end(self, eval_metrics, epoch_idx):
        pass

    def on_test_epoch_end(self, test_metrics, epoch_idx):
        pass