from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from jax import random

from jax_trainer.datasets import SupervisedBatch
from jax_trainer.logger import LogFreq, LogMetricMode, LogMode
from jax_trainer.trainer import TrainerModule, TrainState


class ImgClassifierTrainer(TrainerModule):
    def batch_to_input(self, batch: SupervisedBatch) -> Any:
        return batch.input

    def loss_function(
        self,
        params: Any,
        state: TrainState,
        batch: SupervisedBatch,
        rng: random.PRNGKey,
        train: bool = True,
    ) -> Tuple[Any, Tuple[Any, Dict]]:
        """Loss function for image classification.

        Args:
            params: Parameters of the model.
            state: State of the trainer.
            batch: Batch of data. Assumes structure of SupervisedBatch or subclasses.
            rng: Key for random number generation.
            train: Whether the model is in training mode.

        Returns:
            Tuple of loss and tuple of mutable variables and metrics.
        """
        imgs = batch.input
        labels = batch.target
        logits, mutable_variables = self.model_apply(
            params=params, state=state, input=imgs, rng=rng, train=train
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        preds = logits.argmax(axis=-1)
        acc = (preds == labels).mean()
        conf_matrix = jnp.zeros((logits.shape[-1], logits.shape[-1]))
        conf_matrix = conf_matrix.at[preds, labels].add(1)
        metrics = {
            "acc": acc,
            "acc_std": {"value": acc, "mode": LogMetricMode.STD, "log_mode": LogMode.EVAL},
            "acc_max": {
                "value": acc,
                "mode": LogMetricMode.MAX,
                "log_mode": LogMode.TRAIN,
                "log_freq": LogFreq.EPOCH,
            },
            "conf_matrix": {
                "value": conf_matrix,
                "mode": LogMetricMode.SUM,
                "log_mode": LogMode.EVAL,
            },
        }
        return loss, (mutable_variables, metrics)
