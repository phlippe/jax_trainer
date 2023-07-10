import jax.numpy as jnp
from datasets.utils import Batch
from experiments.base.trainer import TrainerModule, TrainState, Batch
from typing import Callable, Dict, Tuple, Any
from jax import random
import optax
import jax

from experiments.base.loggers import LogMetricMode, LogFreq, LogMode


class ImgClassifierTrainer(TrainerModule):

    def batch_to_input(self, batch: Batch) -> Any:
        return batch.input
    
    def loss_function(self, 
                      params: Any, 
                      state: TrainState, 
                      batch: Batch, 
                      rng: random.PRNGKey, 
                      train: bool = True) -> Tuple[Any, Tuple[Any, Dict]]:
        imgs = batch.input
        labels = batch.target
        logits, mutable_variables = self.model_apply(params=params, 
                                                     state=state,
                                                     input=imgs,
                                                     rng=rng,
                                                     train=train)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        preds = logits.argmax(axis=-1)
        acc = (preds == labels).mean()
        # conf_matrix = jnp.zeros((logits.shape[-1], logits.shape[-1]))
        # conf_matrix[preds, labels] += 1
        metrics = {'acc': acc, 
                   'acc_std': {'value': acc, 'mode': LogMetricMode.STD, 'log_mode': LogMode.EVAL}, 
                   'acc_max': {'value': acc, 'mode': LogMetricMode.MAX, 'log_mode': LogMode.TRAIN, 'log_freq': LogFreq.EPOCH}} 
        
                #    'conf_matrix': {'value': conf_matrix, 
                #                    'mode': 'sum'}}
        return loss, (mutable_variables, metrics)