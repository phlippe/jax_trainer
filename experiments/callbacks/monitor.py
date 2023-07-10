import optax
from absl import logging

from experiments.callbacks.base import Callback


class LearningRateMonitor(Callback):
    def __init__(self, config, trainer, data_module):
        super().__init__(config, trainer, data_module)
        self.log_dir = self.trainer.log_dir

    def _on_training_epoch_start(self, epoch_idx):
        schedule = self.trainer.lr_schedule
        if schedule is None:
            logging.warning("No learning rate schedule found.")
            return
        opt_state = [
            s
            for s in self.trainer.state.opt_state[-1]
            if isinstance(s, optax.ScaleByScheduleState)
        ]
        if len(opt_state) == 0:
            logging.warning("No state of a learning rate schedule found.")
            return
        if len(opt_state) > 1:
            logging.warning(
                "Found multiple states of a learning rate schedule. Using the last one."
            )
        step = opt_state[-1].count
        lr = schedule(step)
        self.trainer.logger.log_scalar("optimizer/lr", lr, epoch_idx)
