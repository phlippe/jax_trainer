import os

import jax
import numpy as np
import optax
from absl import logging

from jax_trainer.callbacks.callback import Callback


class LearningRateMonitor(Callback):
    """Callback to monitor the learning rate."""

    def __init__(self, config, trainer, data_module):
        super().__init__(config, trainer, data_module)
        self.log_dir = self.trainer.log_dir

    def on_filtered_training_epoch_start(self, epoch_idx):
        """Logs the learning rate at the beginning of each training epoch.

        Args:
            epoch_idx: Index of the current epoch.
        """
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


class GradientSpikeMonitor(Callback):
    """Callback to monitor gradient spikes."""

    def __init__(self, config, trainer, data_module):
        super().__init__(config, trainer, data_module)
        assert (
            self.trainer.trainer_config.log_grad_norm
        ), "log_grad_norm must be True to use GradientSpikeMonitor."
        self.log_dir = self.trainer.log_dir
        self.threshold = config.get("threshold", 2.0)
        self.log_to_disk = config.get("log_to_disk", False)
        self.ema_decay = config.get("ema_decay", 0.995)
        self.max_elements = int(np.log(1e-3) / np.log(self.ema_decay))

    def on_training_start(self):
        self.grad_norms_buffer = []
        self.losses_buffer = []
        self.grad_norms = np.array([], dtype=np.float64)
        self.losses = np.array([], dtype=np.float64)

    def on_training_step(self, step_metrics, epoch_idx, step_idx):
        assert "optimizer/grad_global_norm" in step_metrics
        assert "loss" in step_metrics
        self.grad_norms_buffer.append(step_metrics["optimizer/grad_global_norm"])
        self.losses_buffer.append(step_metrics["loss"])

    def on_filtered_training_epoch_end(self, train_metrics, epoch_idx):
        del train_metrics
        epoch_grad_norms = np.asarray(jax.device_get(self.grad_norms_buffer))
        epoch_losses = np.asarray(jax.device_get(self.losses_buffer))
        self.grad_norms = np.concatenate([self.grad_norms, epoch_grad_norms])
        self.losses = np.concatenate([self.losses, epoch_losses])
        self.grad_norms_buffer = []
        self.losses_buffer = []
        if self.log_to_disk:
            np.savez(
                os.path.join(os.path.join(self.log_dir, "gradient_spikes.npz")),
                grad_norms=self.grad_norms,
                losses=self.losses,
            )
        self.log_gradients_spikes(num_elements=epoch_losses.shape[0], epoch_idx=epoch_idx)

    def log_gradients_spikes(self, num_elements: int, epoch_idx: int):
        grad_norms = self.grad_norms
        losses = self.losses
        if num_elements + self.max_elements < self.grad_norms.shape[0]:
            grad_norms = grad_norms[-(num_elements + self.max_elements) :]
            losses = losses[-(num_elements + self.max_elements) :]
        # Calculate EMA by giving each element the weight for the final EMA element.
        weights = np.zeros_like(grad_norms) + self.ema_decay
        weights = np.power(weights, np.flip(np.arange(grad_norms.shape[0])))
        # Normalize all means with respect to all previous weights.
        weight_cumsum = np.cumsum(weights)
        grad_norms_cumsum = np.cumsum(grad_norms * weights)
        losses_cumsum = np.cumsum(losses * weights)
        grad_norms_ema = grad_norms_cumsum / weight_cumsum
        losses_ema = losses_cumsum / weight_cumsum
        # Check for elements that are spikes compared to the previous EMA.
        grad_norms_spike = (
            grad_norms[-min(num_elements, grad_norms.shape[0] - 1) :]
            > self.threshold * grad_norms_ema[-num_elements - 1 : -1]
        )
        losses_spike = (
            losses[-min(num_elements, losses.shape[0] - 1) :]
            > self.threshold * losses_ema[-num_elements - 1 : -1]
        )
        # Remove spikes that directly come after spikes.
        grad_norms_spike = grad_norms_spike & np.logical_not(
            np.concatenate([[False], grad_norms_spike[:-1]])
        )
        losses_spike = losses_spike & np.logical_not(np.concatenate([[False], losses_spike[:-1]]))
        # Log the spikes.
        grad_norms_num_spikes = np.sum(grad_norms_spike)
        losses_num_spikes = np.sum(losses_spike)
        self.trainer.logger.log_scalar(
            "optimizer/spikes_grad_norms", grad_norms_num_spikes, epoch_idx
        )
        self.trainer.logger.log_scalar("optimizer/spikes_losses", losses_num_spikes, epoch_idx)
        synchronous_spikes = np.sum(grad_norms_spike & losses_spike)
        self.trainer.logger.log_scalar(
            "optimizer/spikes_synchronous", synchronous_spikes, epoch_idx
        )
        grad_spike_before_loss = np.sum(
            grad_norms_spike & np.concatenate([[False], losses_spike[:-1]])
        )
        self.trainer.logger.log_scalar(
            "optimizer/spikes_grad_before_loss", grad_spike_before_loss, epoch_idx
        )
        loss_spike_before_grad = np.sum(
            losses_spike & np.concatenate([[False], grad_norms_spike[:-1]])
        )
        self.trainer.logger.log_scalar(
            "optimizer/spikes_loss_before_grad", loss_spike_before_grad, epoch_idx
        )
