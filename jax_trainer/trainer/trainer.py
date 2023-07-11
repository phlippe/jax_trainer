# Standard libraries
import importlib
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from copy import copy
from glob import glob
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

# JAX/Flax libraries
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from absl import flags, logging
from flax.training import train_state
from jax import random

# ML collections for config
from ml_collections import ConfigDict, FrozenConfigDict
from tqdm.auto import tqdm

from jax_trainer import callbacks
from jax_trainer.callbacks import ModelCheckpoint
from jax_trainer.datasets import Batch
from jax_trainer.logger import LogFreq, Logger, LogMetricMode, LogMode
from jax_trainer.optimizer import OptimizerBuilder
from jax_trainer.utils import resolve_import_from_string


class TrainState(train_state.TrainState):
    # A simple extension of TrainState to also include mutable variables
    # like batch statistics. If a model has no mutable vars, it is None
    mutable_variables: Any = None
    # You can further extend the TrainState by any additional part here
    # For example, rng to keep for init, dropout, etc.
    rng: Any = None


class TrainerModule:
    def __init__(
        self,
        trainer_config: ConfigDict,
        model_config: ConfigDict,
        optimizer_config: ConfigDict,
        exmp_input: Batch,
    ):
        """A basic Trainer module summarizing most common training functionalities like logging,
        model initialization, training loop, etc.

        Args:
          model_class: The class of the model that should be trained.
          model_hparams: A dictionary of all hyperparameters of the model. Is
            used as input to the model when created.
          optimizer_hparams: A dictionary of all hyperparameters of the optimizer.
            Used during initialization of the optimizer.
          exmp_input: Input to the model for initialization and tabulate.
          seed: Seed to initialize PRNG.
          logger_params: A dictionary containing the specification of the logger.
          enable_progress_bar: If False, no progress bar is shown.
          debug: If True, no jitting is applied. Can be helpful for debugging.
          check_val_every_n_epoch: The frequency with which the model is evaluated
            on the validation set.
        """
        super().__init__()
        self.trainer_config = trainer_config
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.exmp_input = exmp_input
        # Create empty model. Note: no parameters yet
        self.build_model(model_config)
        # Init trainer parts
        self.create_jitted_functions()
        self.init_model(exmp_input)
        self.init_logger(self.trainer_config.get("logger", ConfigDict()))
        self.init_callbacks()
        # Freeze configs since they should not be changed during training
        self.trainer_config = FrozenConfigDict(self.trainer_config)
        self.model_config = FrozenConfigDict(self.model_config)
        self.optimizer_config = FrozenConfigDict(self.optimizer_config)

    def batch_to_input(self, batch: Batch) -> Any:
        raise NotImplementedError

    def build_model(self, model_config: ConfigDict):
        """Creates the model class from the model_config.

        Args:
          model_config: A dictionary containing the model configuration.
        """
        # Create model
        model_class = resolve_import_from_string(model_config.name)
        hparams = FrozenConfigDict(model_config.hparams)
        self.model = model_class(**hparams)

    def init_logger(self, logger_config: ConfigDict):
        """Initializes a logger and creates a logging directory.

        Args:
          logger_params: A dictionary containing the specification of the logger.
        """
        full_config = ConfigDict(
            {
                "trainer": self.trainer_config,
                "model": self.model_config,
                "optimizer": self.optimizer_config,
            }
        )
        LoggerClass = logger_config.get("class", Logger)
        if isinstance(LoggerClass, str):
            LoggerClass = resolve_import_from_string(LoggerClass)
        self.logger = LoggerClass(logger_config, full_config)
        # Save config and exmp_input
        log_dir = self.logger.log_dir
        self.log_dir = log_dir
        self.trainer_config.logger.log_dir = log_dir
        os.makedirs(os.path.join(log_dir, "metrics/"), exist_ok=True)
        logging.get_absl_handler().use_absl_log_file(log_dir=log_dir, program_name="absl_logging")
        logging.set_verbosity(logger_config.get("log_file_verbosity", logging.INFO))
        logging.set_stderrthreshold(logger_config.get("stderrthreshold", "warning"))
        if not os.path.isfile(os.path.join(log_dir, "config.yaml")):
            with open(os.path.join(log_dir, "config.yaml"), "w") as f:
                yaml.dump(full_config.to_dict(), f)
        if not os.path.isfile(os.path.join(log_dir, "exmp_input.pkl")):
            with open(os.path.join(log_dir, "exmp_input.pkl"), "wb") as f:
                pickle.dump(self.exmp_input, f)
        if self.trainer_config.tabulate_model:
            tab = self.tabulate(self.exmp_input)
            logging.info("Model summary:\n" + tab)
            with open(os.path.join(log_dir, "model.txt"), "w") as f:
                f.write(tab)

    def init_callbacks(self):
        self.callbacks = []
        for name in self.trainer_config.callbacks:
            logging.info(f"Initializing callback {name}")
            try:
                callback_class = getattr(callbacks, name)
            except AttributeError:
                callback_class = resolve_import_from_string(name)
            callback_config = self.trainer_config.callbacks[name]
            self.callbacks.append(
                callback_class(config=callback_config, trainer=self, data_module=None)
            )

    def init_model(self, exmp_input: Batch):
        """Creates an initial training state with newly generated network parameters.

        Args:
          exmp_input: An input to the model with which the shapes are inferred.
        """
        # Prepare PRNG and input
        model_rng = random.PRNGKey(self.trainer_config.seed)
        model_rng, init_rng = random.split(model_rng)
        # Run model initialization
        variables = self.run_model_init(exmp_input, init_rng)
        mutable_variables, params = variables.pop("params")
        if len(mutable_variables) == 0:
            mutable_variables = None
        # Create default state. Optimizer is initialized later
        self.state = TrainState(
            step=0,
            apply_fn=self.model.apply,
            params=params,
            mutable_variables=mutable_variables,
            rng=model_rng,
            tx=None,
            opt_state=None,
        )

    def get_model_rng(self, rng: random.PRNGKey) -> Dict[str, random.PRNGKey]:
        """Returns a dictionary of PRNGKey for init and tabulate.

        Args:
          rng: The current PRNGKey.

        Returns:
          Dict of PRNG Keys
        """
        return {"params": rng}

    def run_model_init(self, exmp_input: Batch, init_rng: random.KeyArray) -> Dict:
        """The model initialization call.

        Args:
          exmp_input: An input to the model with which the shapes are inferred.
          init_rng: A jax.random.PRNGKey.

        Returns:
          The initialized variable dictionary.
        """
        rngs = self.get_model_rng(init_rng)
        exmp_input = self.batch_to_input(exmp_input)
        return self.model.init(rngs, exmp_input, train=True)

    def tabulate(self, exmp_input: Batch):
        """Prints a summary of the Module represented as table.

        Args:
          exmp_input: An input to the model with which the shapes are inferred.
        """
        rngs = self.get_model_rng(random.PRNGKey(0))
        exmp_input = self.batch_to_input(exmp_input)
        return self.model.tabulate(
            rngs, exmp_input, train=True, console_kwargs={"force_terminal": False}
        )

    def init_optimizer(self, num_epochs: int, num_train_steps_per_epoch: int):
        """Initializes the optimizer and learning rate scheduler.

        Args:
          num_epochs: Number of epochs the model will be trained for.
          num_train_steps_per_epoch: Number of training steps per epoch.
        """
        BuilderClass = self.optimizer_config.get("builder", OptimizerBuilder)
        if isinstance(BuilderClass, str):
            BuilderClass = resolve_import_from_string(BuilderClass)
        builder = BuilderClass(self.optimizer_config)
        optimizer, lr_schedule = builder.build_optimizer(
            num_epochs=num_epochs,
            num_train_steps_per_epoch=num_train_steps_per_epoch,
        )
        self.lr_schedule = lr_schedule  # Save for logging
        # Initialize training state
        self.state = TrainState.create(
            apply_fn=self.state.apply_fn,
            params=self.state.params,
            mutable_variables=self.state.mutable_variables,
            tx=optimizer,
            rng=self.state.rng,
        )

    def create_jitted_functions(self):
        """Creates jitted versions of the training and evaluation functions.

        If self.debug is True, not jitting is applied.
        """
        train_step, eval_step = self.create_functions()
        if self.trainer_config.debug:  # Skip jitting
            logging.info("Skipping jitting due to debug=True")
            self.train_step = train_step
            self.eval_step = eval_step
        else:  # Jit
            self.train_step = jax.jit(train_step)
            self.eval_step = jax.jit(eval_step)

    def loss_function(
        self, params: Any, state: TrainState, batch: Batch, rng: random.PRNGKey, train: bool = True
    ) -> Tuple[jnp.array, Tuple[Any, Dict]]:
        """The loss function that is used for training.

        This function needs to be overwritten by a subclass.
        """
        raise NotImplementedError
        # return loss, (mutable_vars, metrics)

    def model_apply(
        self,
        params: Any,
        state: TrainState,
        input: Any,
        rng: random.PRNGKey,
        train: bool = True,
        **kwargs,
    ) -> Tuple[Any, Dict]:
        """The model apply function that is used for evaluation.

        This function needs to be overwritten by a subclass.
        """
        rngs = self.get_model_rng(rng)
        variables = {"params": params}
        mutable_keys = False
        if state.mutable_variables is not None:
            variables.update(
                {k: state.mutable_variables[k] for k in state.mutable_variables.keys()}
            )
            if train:
                mutable_keys = list(state.mutable_variables.keys())
        out = state.apply_fn(
            variables, input, train=train, rngs=rngs, mutable=mutable_keys, **kwargs
        )
        if mutable_keys is not False:
            out, mutable_vars = out
        else:
            mutable_vars = None
        return out, mutable_vars

    def create_training_function(self) -> Callable[[TrainState, Batch], Tuple[TrainState, Dict]]:
        """Creates and returns a function for the training step.

        The function takes as input the training state and a batch from the train loader. The
        function is expected to return a dictionary of logging metrics, and a new train state.
        """

        def train_step(state: TrainState, batch: Batch):
            next_rng, step_rng = random.split(state.rng)
            loss_fn = lambda params: self.loss_function(params, state, batch, step_rng, train=True)
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, mutable_vars, metrics = ret[0], *ret[1]
            metrics["loss"] = loss
            state = state.apply_gradients(
                grads=grads, mutable_variables=mutable_vars, rng=next_rng
            )
            if self.trainer_config.get("log_grad_norm", False):
                grad_norm = optax.global_norm(grads)
                metrics["optimizer/grad_global_norm"] = grad_norm
                metrics["optimizer/grad_global_norm_max"] = {
                    "value": grad_norm,
                    "mode": LogMetricMode.MAX,
                    "log_freq": LogFreq.EPOCH,
                }
            return state, metrics

        return train_step

    def create_evaluation_function(self) -> Callable[[TrainState, Batch], Tuple[TrainState, Dict]]:
        """Creates and returns a function for the evaluation step.

        The function takes as input the training state and a batch from the val/test loader. The
        function is expected to return a dictionary of logging metrics, and a new train state.
        """

        def eval_step(state: TrainState, batch: Batch):
            loss, (_, metrics) = self.loss_function(
                state.params,
                state,
                batch,
                random.PRNGKey(self.trainer_config.get("seed_eval", 0)),
                train=False,
            )
            metrics["loss"] = loss
            return metrics

        return eval_step

    def create_functions(
        self,
    ) -> Tuple[
        Callable[[TrainState, Batch], Tuple[TrainState, Dict]],
        Callable[[TrainState, Batch], Tuple[TrainState, Dict]],
    ]:
        """Creates and returns functions for the training and evaluation step.

        The functions take as input the training state and a batch from the train/ val/test loader.
        Both functions are expected to return a dictionary of logging metrics, and the training
        function a new train state. This function needs to be overwritten by a subclass. The
        train_step and eval_step functions here are examples for the signature of the functions.
        """
        return self.create_training_function(), self.create_evaluation_function()

    def train_model(
        self,
        train_loader: Iterator,
        val_loader: Iterator,
        test_loader: Optional[Iterator] = None,
        num_epochs: int = 500,
    ) -> Dict[str, Any]:
        """Starts a training loop for the given number of epochs.

        Args:
          train_loader: Data loader of the training set.
          val_loader: Data loader of the validation set.
          test_loader: If given, best model will be evaluated on the test set.
          num_epochs: Number of epochs for which to train the model.

        Returns:
          A dictionary of the train, validation and evt. test metrics for the
          best model on the validation set.
        """
        # Create optimizer and the scheduler for the given number of epochs
        self.init_optimizer(num_epochs, len(train_loader))
        # Prepare training loop
        self.on_training_start()
        self.test_functions(train_loader, val_loader)
        all_eval_metrics = {}
        for epoch_idx in self.tracker(range(1, num_epochs + 1), desc="Epochs"):
            self.on_training_epoch_start(epoch_idx)
            train_metrics = self.train_epoch(train_loader, epoch_idx=epoch_idx)
            self.on_training_epoch_end(train_metrics, epoch_idx)
            # Validation every N epochs
            if epoch_idx % self.trainer_config.check_val_every_n_epoch == 0:
                eval_metrics = self.eval_model(val_loader, mode="val", epoch_idx=epoch_idx)
                all_eval_metrics[epoch_idx] = eval_metrics
                self.on_validation_epoch_end(eval_metrics, epoch_idx)
        self.on_training_end()
        # Test best model if possible
        if test_loader is not None:
            self.load_model()
            test_metrics = self.eval_model(test_loader, mode="test", epoch_idx=epoch_idx)
            self.on_test_epoch_end(test_metrics, epoch_idx)
            all_eval_metrics["test"] = test_metrics
        # Close logger
        self.logger.finalize("success")
        return all_eval_metrics

    def test_model(
        self, test_loader: Iterator, apply_callbacks: bool = False, epoch_idx: int = 0
    ) -> Dict[str, Any]:
        """Tests the model on the given test set.

        Args:
          test_loader: Data loader of the test set.
          apply_callbacks: If True, the callbacks will be applied.
          epoch_idx: The epoch index to use for the callbacks and logging.
        """
        test_metrics = self.eval_model(test_loader, mode="test", epoch_idx=epoch_idx)
        if apply_callbacks:
            self.on_test_epoch_end(test_metrics, epoch_idx=epoch_idx)
        return test_metrics

    def test_functions(self, train_loader: Iterator, val_loader: Iterator):
        """Tests the training and evaluation functions on a few batches from the train, val and
        test loader.

        This is useful to check if the functions have the correct signature and return the correct
        values.
        """
        print("Verifying training and evaluation functions...")
        train_batch = next(iter(train_loader))
        start_time = time.time()
        logging.info("Testing and compiling train_step...")
        _ = self.train_step(self.state, train_batch)
        logging.info(f"Successfully completed in {time.time() - start_time:.2f} seconds.")

        val_batch = next(iter(val_loader))
        start_time = time.time()
        logging.info("Testing and compiling eval_step...")
        _ = self.eval_step(self.state, val_batch)
        logging.info(f"Successfully completed in {time.time() - start_time:.2f} seconds.")

    def train_epoch(self, train_loader: Iterator, epoch_idx: int) -> Dict[str, Any]:
        """Trains a model for one epoch.

        Args:
          train_loader: Data loader of the training set.
          epoch_idx: Current epoch index.

        Returns:
          A dictionary of the average training metrics over all batches
          for logging.
        """
        # Train model for one epoch, and log avg loss and accuracy
        self.logger.start_epoch(epoch_idx, mode="train")
        for batch in self.tracker(train_loader, desc="Training", leave=False):
            self.state, step_metrics = self.train_step(self.state, batch)
            self.logger.log_step(step_metrics, element_count=batch.size)
        metrics = self.logger.end_epoch()
        return metrics

    def eval_model(self, data_loader: Iterator, mode: str, epoch_idx: int) -> Dict[str, Any]:
        """Evaluates the model on a dataset.

        Args:
          data_loader: Data loader of the dataset to evaluate on.
          mode: Whether 'val' or 'test'
          epoch_idx: Current epoch index.

        Returns:
          A dictionary of the evaluation metrics, averaged over data points
          in the dataset.
        """
        # Test model on all images of a data loader and return avg loss
        self.logger.start_epoch(epoch_idx, mode=mode)
        for batch in self.tracker(data_loader, desc=mode.capitalize(), leave=False):
            step_metrics = self.eval_step(self.state, batch)
            self.logger.log_step(step_metrics, element_count=batch.size)
        metrics = self.logger.end_epoch(save_metrics=True)
        return metrics

    def tracker(self, iterator: Iterator, **kwargs) -> Iterator:
        """Wraps an iterator in a progress bar tracker (tqdm) if the progress bar is enabled.

        Args:
          iterator: Iterator to wrap in tqdm.
          kwargs: Additional arguments to tqdm.

        Returns:
          Wrapped iterator if progress bar is enabled, otherwise same iterator
          as input.
        """
        if self.trainer_config.enable_progress_bar:
            return tqdm(iterator, **kwargs)
        else:
            return iterator

    def on_training_start(self):
        """Method called before training is started.

        Can be used for additional initialization operations etc.
        """
        logging.info("Starting training")
        for callback in self.callbacks:
            callback.on_training_start()

    def on_training_end(self):
        """Method called after training has finished.

        Can be used for additional logging or similar.
        """
        logging.info("Finished training")
        for callback in self.callbacks:
            callback.on_training_end()

    def on_training_epoch_start(self, epoch_idx: int):
        """Method called at the start of each training epoch. Can be used for additional logging or
        similar.

        Args:
          epoch_idx: Index of the training epoch that has started.
        """
        logging.info(f"Starting training epoch {epoch_idx}")
        for callback in self.callbacks:
            callback.on_training_epoch_start(epoch_idx)

    def on_training_epoch_end(self, train_metrics: Dict[str, Any], epoch_idx: int):
        """Method called at the end of each training epoch. Can be used for additional logging or
        similar.

        Args:
          epoch_idx: Index of the training epoch that has finished.
        """
        logging.info(f"Finished training epoch {epoch_idx}")
        for callback in self.callbacks:
            callback.on_training_epoch_end(train_metrics, epoch_idx)

    def on_validation_epoch_end(self, eval_metrics: Dict[str, Any], epoch_idx: int):
        """Method called at the end of each validation epoch. Can be used for additional logging
        and evaluation.

        Args:
          epoch_idx: Index of the training epoch at which validation was performed.
          eval_metrics: A dictionary of the validation metrics. New metrics added to
            this dictionary will be logged as well.
          val_loader: Data loader of the validation set, to support additional
            evaluation.
        """
        logging.info(f"Finished validation epoch {epoch_idx}")
        for callback in self.callbacks:
            callback.on_validation_epoch_end(eval_metrics, epoch_idx)

    def on_test_epoch_end(self, test_metrics: Dict[str, Any], epoch_idx: int):
        """Method called at the end of each test epoch. Can be used for additional logging and
        evaluation.

        Args:
          epoch_idx: Index of the training epoch at which testing was performed.
          test_metrics: A dictionary of the test metrics. New metrics added to
            this dictionary will be logged as well.
          test_loader: Data loader of the test set, to support additional
            evaluation.
        """
        logging.info(f"Finished test epoch {epoch_idx}")
        for callback in self.callbacks:
            callback.on_test_epoch_end(test_metrics, epoch_idx)

    def load_model(self, epoch_idx: int = -1):
        """Loads model parameters and batch statistics from the logging directory."""
        logging.info(f"Loading model from epoch {epoch_idx}")
        state_dict = None
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpoint):
                state_dict = callback.load_model(epoch_idx)
                break
        if state_dict is None:
            raise ValueError("No model checkpoint callback found in callbacks.")
        self.restore(state_dict)

    def restore(self, state_dict: Dict[str, Any]):
        """Restores the state of the trainer from a state dictionary.

        Args:
          state_dict: State dictionary to restore from.
        """
        logging.info("Restoring trainer state with keys " + str(state_dict.keys()))
        state_dict.pop("metrics")
        state_dict.pop("metadata")
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            # Optimizer will be overwritten when training starts
            tx=self.state.tx if self.state.tx else optax.sgd(0.1),
            rng=self.state.rng,
            **state_dict,
        )

    def bind_model(self):
        """Returns a model with parameters bound to it. Enables an easier inference access.

        Returns:
          The model with parameters and evt. batch statistics bound to it.
        """
        params = {"params": self.state.params}
        if self.state.mutable_variables is not None:
            params.update(self.state.mutable_variables)
        return self.model.bind(params)

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint: str, exmp_input: Batch = None, exmp_input_file: str = None
    ) -> Any:
        """Creates a Trainer object with same hyperparameters and loaded model from a checkpoint
        directory.

        Args:
          checkpoint: Folder in which the checkpoint and hyperparameter file is stored.
          exmp_input: An input to the model for shape inference.

        Returns:
          A Trainer object with model loaded from the checkpoint folder.
        """
        # Load config
        metadata_file = os.path.join(checkpoint, "metadata/metadata")
        assert os.path.isfile(metadata_file), "Could not find metadata file"
        with open(metadata_file, "rb") as f:
            config = ConfigDict(json.load(f))
        # Adjust log dir to where its loaded from
        adjusted_checkpoint = checkpoint.split("/")
        if adjusted_checkpoint[-1] == "":
            adjusted_checkpoint = adjusted_checkpoint[:-1]
        if len(adjusted_checkpoint) < 2:
            raise ValueError("Checkpoint path must be at least two levels deep")
        config.trainer.logger.log_dir = os.path.join(*adjusted_checkpoint[:-2])
        # Load example input
        if exmp_input is None:
            if exmp_input_file is None:
                exmp_input_file = os.path.join(checkpoint, "exmp_input.pkl")
            assert os.path.isfile(exmp_input_file), "Could not find example input file"
            with open(exmp_input_file, "rb") as f:
                exmp_input = pickle.load(f)
        # Create trainer
        trainer = cls(
            exmp_input=exmp_input,
            trainer_config=config.trainer,
            model_config=config.model,
            optimizer_config=config.optimizer,
        )
        # Load model
        trainer.load_model()
        return trainer