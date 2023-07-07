from experiments.callbacks.base import Callback
from typing import Dict, Any
import orbax
from flax.training import orbax_utils
import os


class ModelCheckpoint(Callback):

    def __init__(self, config, trainer, data_module):
        super().__init__(config, trainer, data_module)
        self.log_dir = self.trainer.log_dir
        assert self.config.get('monitor', None) is not None, 'Please specify a metric to monitor.'

        options = orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=self.config.get('save_top_k', 1),
            best_fn=lambda m: m[self.config.monitor],
            best_mode=self.config.get('mode', 'min'),
            step_prefix='checkpoint',
            cleanup_tmp_directories=True,
            create=True
        )
        checkpointers = {
            'params': orbax.checkpoint.PyTreeCheckpointer()
        }
        if self.trainer.state.batch_stats is not None:
            checkpointers['batch_stats'] = orbax.checkpoint.PyTreeCheckpointer()
        if self.config.get('save_optimizer_state', False):
            checkpointers['optimizer'] = orbax.checkpoint.PyTreeCheckpointer()
        self.manager = orbax.checkpoint.CheckpointManager(
            directory=os.path.join(self.log_dir, 'checkpoints/'),
            checkpointers=checkpointers,
            options=options
        )

    def on_validation_epoch_end(self, eval_metrics, epoch_idx):
        self.save_model(eval_metrics, epoch_idx)

    def save_model(self, eval_metrics, epoch_idx):
        """
        Saves model parameters and batch statistics to the logging directory.
        """
        save_items = {'params': self.trainer.state.params}
        if self.trainer.state.batch_stats is not None:
            save_items['batch_stats'] = self.trainer.state.batch_stats
        if self.config.get('save_optimizer_state', False):
            save_items['optimizer'] = self.trainer.state.optimizer
        self.manager.save(epoch_idx, save_items, metrics=eval_metrics)

    def load_model(self, epoch_idx=-1):
        """
        Loads model parameters and batch statistics from the logging directory.
        """
        if epoch_idx == -1:
            epoch_idx = self.manager.best_step()
        state_dict = self.manager.restore(epoch_idx)
        return state_dict