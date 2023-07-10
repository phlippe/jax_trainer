from experiments.callbacks.base import Callback
from absl import logging


class LearningRateMonitor(Callback):

    def __init__(self, config, trainer, data_module):
        super().__init__(config, trainer, data_module)
        self.log_dir = self.trainer.log_dir

    def _on_training_epoch_start(self, epoch_idx):
        schedule = self.trainer.lr_schedule
        if schedule is None:
            logging.warning('No learning rate schedule found.')
        step = self.trainer.state.opt_state[-1][-1].count
        lr = schedule(step)
        self.trainer.logger.log_scalar('optimizer/lr', lr, epoch_idx)