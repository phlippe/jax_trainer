from typing import Any, Optional

from ml_collections import ConfigDict

from jax_trainer.datasets import DatasetModule


class Callback:
    """Base class for callbacks."""

    def __init__(
        self, config: ConfigDict, trainer: Any, data_module: Optional[DatasetModule] = None
    ):
        """Base class for callbacks.

        Args:
            config: Configuration dictionary.
            trainer: Trainer object.
            data_module (optional): Data module object.
        """
        self.config = config
        self.trainer = trainer
        self.data_module = data_module
        self.every_n_epochs = config.get("every_n_epochs", 1)

    def on_training_start(self):
        """Called at the beginning of training."""
        pass

    def on_training_end(self):
        """Called at the end of training."""
        pass

    def on_training_epoch_start(self, epoch_idx):
        """Called at the beginning of each training epoch.

        Args:
            epoch_idx: Index of the current epoch.
        """
        if epoch_idx % self.every_n_epochs != 0:
            return
        self._on_training_epoch_start(epoch_idx)

    def _on_training_epoch_start(self, epoch_idx):
        """Called at the beginning of each `every_n_epochs` training epoch. To be implemented by
        subclasses.

        Args:
            epoch_idx: Index of the current epoch.
        """
        pass

    def on_training_epoch_end(self, train_metrics, epoch_idx):
        """Called at the end of each training epoch.

        Args:
            train_metrics: Dictionary of training metrics of the current epoch.
            epoch_idx: Index of the current epoch.
        """
        if epoch_idx % self.every_n_epochs != 0:
            return
        self._on_training_epoch_end(train_metrics, epoch_idx)

    def _on_training_epoch_end(self, train_metrics, epoch_idx):
        """Called at the end of each `every_n_epochs` training epoch. To be implemented by
        subclasses.

        Args:
            train_metrics: Dictionary of training metrics of the current epoch.
        """
        pass

    def on_validation_epoch_end(self, eval_metrics, epoch_idx):
        """Called at the end of each validation epoch.

        Args:
            eval_metrics: Dictionary of evaluation metrics of the current epoch.
            epoch_idx: Index of the current epoch.
        """
        if epoch_idx % self.every_n_epochs != 0:
            return
        self._on_validation_epoch_end(eval_metrics, epoch_idx)

    def _on_validation_epoch_end(self, eval_metrics, epoch_idx):
        """Called at the end of each `every_n_epochs` validation epoch. To be implemented by
        subclasses.

        Args:
            eval_metrics: Dictionary of evaluation metrics of the current epoch.
            epoch_idx: Index of the current epoch.
        """
        pass

    def on_test_epoch_end(self, test_metrics, epoch_idx):
        """Called at the end of each test epoch.

        Args:
            test_metrics: Dictionary of test metrics of the current epoch.
            epoch_idx: Index of the current epoch.
        """
        # We apply it to every epoch, since we don't know when the test set is evaluated.
        self._on_test_epoch_end(test_metrics, epoch_idx)

    def _on_test_epoch_end(self, test_metrics, epoch_idx):
        """Called at the end of each test epoch. To be implemented by subclasses.

        Args:
            test_metrics: Dictionary of test metrics of the current epoch.
            epoch_idx: Index of the current epoch.
        """
        pass
