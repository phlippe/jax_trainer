from experiments.base.trainer import TrainerModule, TrainState, Batch
from typing import Callable, Dict, Tuple, Any
from jax import random
import optax
import jax


class ImgClassifierTrainer(TrainerModule):

    def batch_to_input(self, batch: Batch) -> Any:
        return batch.input

    def create_functions(self) -> Tuple[Callable[[TrainState, Batch], Tuple[TrainState, Dict]],
                                        Callable[[TrainState, Batch], Tuple[TrainState, Dict]]]:
        """
        Creates and returns functions for the training and evaluation step. The
        functions take as input the training state and a batch from the train/
        val/test loader. Both functions are expected to return a dictionary of
        logging metrics, and the training function a new train state. This
        function needs to be overwritten by a subclass. The train_step and
        eval_step functions here are examples for the signature of the functions.
        """
        def loss_function(params, batch_stats, rng, batch, train):
            imgs = batch.input
            labels = batch.target
            rng, dropout_rng = random.split(rng)
            output = self.model.apply({'params': params},
                                      imgs,
                                      train=train,
                                      rngs={'dropout': dropout_rng})
            logits = output
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            acc = (logits.argmax(axis=-1) == labels).mean()
            metrics = {'loss': loss, 'acc': acc}
            return loss, (rng, metrics)

        def train_step(state, batch):
            loss_fn = lambda params: loss_function(params, state.batch_stats, state.rng, batch, train=True)
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, rng, metrics = ret[0], *ret[1]
            state = state.apply_gradients(grads=grads, rng=rng)
            return state, metrics

        def eval_step(state, batch):
            _, (_, metrics) = loss_function(state.params, state.batch_stats, state.rng, batch, train=False)
            return metrics

        return train_step, eval_step