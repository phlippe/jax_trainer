import os
from typing import Any, Dict

import orbax
from flax.training import orbax_utils

from jax_trainer.callbacks.callback import Callback


class ModelCheckpoint(Callback):
    """Callback to save model parameters and mutable variables to the logging directory."""

    def __init__(self, config, trainer, data_module):
        super().__init__(config, trainer, data_module)
        self.log_dir = self.trainer.log_dir
        assert self.config.get("monitor", None) is not None, "Please specify a metric to monitor."

        options = orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=self.config.get("save_top_k", 1),
            best_fn=lambda m: m[self.config.monitor],
            best_mode=self.config.get("mode", "min"),
            step_prefix="checkpoint",
            cleanup_tmp_directories=True,
            create=True,
        )
        checkpointers = {
            "params": orbax.checkpoint.PyTreeCheckpointer(),
            "metadata": orbax.checkpoint.Checkpointer(orbax.checkpoint.JsonCheckpointHandler()),
        }
        if self.trainer.state.mutable_variables is not None:
            checkpointers["mutable_variables"] = orbax.checkpoint.PyTreeCheckpointer()
        if self.config.get("save_optimizer_state", False):
            checkpointers["optimizer"] = orbax.checkpoint.PyTreeCheckpointer()
        self.metadata = {
            "trainer": self.trainer.trainer_config.to_dict(),
            "model": self.trainer.model_config.to_dict(),
            "optimizer": self.trainer.optimizer_config.to_dict(),
        }
        self.manager = orbax.checkpoint.CheckpointManager(
            directory=os.path.join(self.log_dir, "checkpoints/"),
            checkpointers=checkpointers,
            options=options,
        )

    def on_filtered_validation_epoch_end(self, eval_metrics, epoch_idx):
        self.save_model(eval_metrics, epoch_idx)

    def save_model(self, eval_metrics, epoch_idx):
        """Saves model parameters and batch statistics to the logging directory.

        Args:
            eval_metrics: Dictionary of evaluation metrics.
            epoch_idx: Index of the current epoch.
        """
        save_items = {"params": self.trainer.state.params, "metadata": self.metadata}
        if self.trainer.state.mutable_variables is not None:
            save_items["mutable_variables"] = self.trainer.state.mutable_variables
        if self.config.get("save_optimizer_state", False):
            save_items["optimizer"] = self.trainer.state.optimizer
        eval_metrics = {
            k: eval_metrics[k]
            for k in eval_metrics
            if isinstance(eval_metrics[k], (int, float, str, bool))
        }
        self.manager.save(epoch_idx, save_items, metrics=eval_metrics)

    def load_model(self, epoch_idx=-1):
        """Loads model parameters and variables from the logging directory.

        Args:
            epoch_idx: Index of the epoch to load. If -1, loads the best epoch.

        Returns:
            Dictionary of loaded model parameters and additional variables.
        """
        if epoch_idx == -1:
            epoch_idx = self.manager.best_step()
        state_dict = self.manager.restore(epoch_idx)
        return state_dict
