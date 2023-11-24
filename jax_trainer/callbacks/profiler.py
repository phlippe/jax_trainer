import time

import jax
import optax
from absl import logging

from jax_trainer.callbacks.callback import Callback


class JAXProfiler(Callback):
    """Callback to profile model training steps."""

    def __init__(self, config, trainer, data_module):
        super().__init__(config, trainer, data_module)
        self.log_dir = self.trainer.log_dir
        self.every_n_minutes = config.get("every_n_minutes", 60)
        self.first_step = config.get("first_step", 10)
        self.profiler_n_steps = config.get("profile_n_steps", 5)
        self.profiler_active = False
        self.profiler_last_time = None

    def on_training_start(self):
        self.profiler_active = False
        self.profiler_last_time = time.time()

    def on_training_step(self, step_metrics, epoch_idx, step_idx):
        if self.profiler_active:
            if step_idx >= self.profile_start_step + self.profiler_n_steps:
                self.stop_trace()
        else:
            if (step_idx == self.first_step) or (
                time.time() - self.profiler_last_time > self.every_n_minutes * 60
            ):
                self.start_trace(step_idx)

    def on_training_epoch_end(self, train_metrics, epoch_idx):
        self.stop_trace()

    def start_trace(self, step_idx):
        if not self.profiler_active:
            logging.info(f"Starting trace at step {step_idx}.")
            jax.profiler.start_trace(self.log_dir)
            self.profiler_active = True
            self.profile_start_step = step_idx
        else:
            logging.warning("Trace already active.")

    def stop_trace(self):
        if self.profiler_active:
            logging.info("Stopping trace")
            jax.tree_map(lambda x: x.block_until_ready(), self.trainer.state.params)
            jax.profiler.stop_trace()
            self.profiler_last_time = time.time()
            self.profiler_active = False
