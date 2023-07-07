# Standard libraries
import os
import sys
from typing import Any, NamedTuple, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union
import json
import time
from tqdm.auto import tqdm
import numpy as np
from copy import copy
from glob import glob
from collections import defaultdict

# JAX/Flax libraries
import jax
from jax import random
from flax.training import train_state
import optax

# ML collections for config
from ml_collections import ConfigDict, FrozenConfigDict
import yaml
import pickle
from absl import flags, logging

from experiments.base.optimizer_constructor import build_optimizer
from experiments.base.loggers import build_logger
from datasets.utils import Batch
import experiments.callbacks as callbacks
from experiments.callbacks import ModelCheckpoint
import models
from models import SimpleEncoder


class TrainState(train_state.TrainState):
    # A simple extension of TrainState to also include batch statistics
    # If a model has no batch statistics, it is None
    batch_stats : Any = None
    # You can further extend the TrainState by any additional part here
    # For example, rng to keep for init, dropout, etc.
    rng : Any = None


class TrainerModule:

    def __init__(self,
                 trainer_config: ConfigDict,
                 model_config: ConfigDict,
                 optimizer_config: ConfigDict,
                 exmp_input: Batch):
        """
        A basic Trainer module summarizing most common training functionalities
        like logging, model initialization, training loop, etc.

        Atributes:
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
        self.init_logger(self.trainer_config.get('logger', ConfigDict()))
        self.init_callbacks()
        # Freeze configs since they should not be changed during training
        self.trainer_config = FrozenConfigDict(self.trainer_config)
        self.model_config = FrozenConfigDict(self.model_config)
        self.optimizer_config = FrozenConfigDict(self.optimizer_config)

    def batch_to_input(self, batch: Batch) -> Any:
        raise NotImplementedError

    def build_model(self,
                    model_config : ConfigDict):
        """
        Creates the model class from the model_config.

        Args:
          model_config: A dictionary containing the model configuration.
        """
        # Create model
        model_class = getattr(models, model_config.name)
        hparams = FrozenConfigDict(model_config.hparams)
        self.model = model_class(**hparams)

    def init_logger(self,
                    logger_config: ConfigDict):
        """
        Initializes a logger and creates a logging directory.

        Args:
          logger_params: A dictionary containing the specification of the logger.
        """
        full_config = ConfigDict({
            'trainer': self.trainer_config,
            'model': self.model_config,
            'optimizer': self.optimizer_config
        })
        self.logger = build_logger(logger_config, full_config)
        # Save config and exmp_input
        log_dir = self.logger.log_dir
        self.log_dir = log_dir
        os.makedirs(os.path.join(log_dir, 'metrics/'), exist_ok=True)
        logging.get_absl_handler().use_absl_log_file(log_dir=log_dir,
                                                     program_name='absl_logging')
        logging.set_verbosity(logging.INFO)
        logging.set_stderrthreshold(logger_config.get('stderrthreshold', 'warning'))
        if not os.path.isfile(os.path.join(log_dir, 'config.yaml')):
            with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
                yaml.dump(full_config.to_dict(), f)
        if not os.path.isfile(os.path.join(log_dir, 'exmp_input.pkl')):
            with open(os.path.join(log_dir, 'exmp_input.pkl'), 'wb') as f:
                pickle.dump(self.exmp_input, f)
        if self.trainer_config.tabulate_model:
            tab = self.tabulate(self.exmp_input)
            logging.info('Model summary:\n' + tab)
            with open(os.path.join(log_dir, 'model.txt'), 'w') as f:
                f.write(tab)
    
    def init_callbacks(self):
        self.callbacks = []
        for name in self.trainer_config.callbacks:
            logging.info(f'Initializing callback {name}')
            callback_class = getattr(callbacks, name)
            callback_config = self.trainer_config.callbacks[name]
            self.callbacks.append(callback_class(config=callback_config, 
                                                 trainer=self, 
                                                 data_module=None))

    def init_model(self,
                   exmp_input : Batch):
        """
        Creates an initial training state with newly generated network parameters.

        Args:
          exmp_input: An input to the model with which the shapes are inferred.
        """
        # Prepare PRNG and input
        model_rng = random.PRNGKey(self.trainer_config.seed)
        model_rng, init_rng = random.split(model_rng)
        # Run model initialization
        variables = self.run_model_init(exmp_input, init_rng)
        # Create default state. Optimizer is initialized later
        self.state = TrainState(step=0,
                                apply_fn=self.model.apply,
                                params=variables['params'],
                                batch_stats=variables.get('batch_stats'),
                                rng=model_rng,
                                tx=None,
                                opt_state=None)
                                
    def get_model_rng(self,
                      rng : random.PRNGKey) -> Dict[str, random.PRNGKey]:
        """
        Returns a dictionary of PRNGKey for init and tabulate.

        Args:
          rng: The current PRNGKey.

        Returns:
          Dict of PRNG Keys
        """
        return {'params': rng}

    def run_model_init(self,
                       exmp_input : Batch,
                       init_rng : random.KeyArray) -> Dict:
        """
        The model initialization call

        Args:
          exmp_input: An input to the model with which the shapes are inferred.
          init_rng: A jax.random.PRNGKey.

        Returns:
          The initialized variable dictionary.
        """
        rngs = self.get_model_rng(init_rng)
        exmp_input = self.batch_to_input(exmp_input)
        return self.model.init(rngs, exmp_input, train=True)

    def tabulate(self,
                 exmp_input : Batch):
        """
        Prints a summary of the Module represented as table.

        Args:
          exmp_input: An input to the model with which the shapes are inferred.
        """
        rngs = self.get_model_rng(random.PRNGKey(0))
        exmp_input = self.batch_to_input(exmp_input)
        return self.model.tabulate(rngs, exmp_input, train=True, console_kwargs={'force_terminal': False})

    def init_optimizer(self,
                       num_epochs : int,
                       num_train_steps_per_epoch : int):
        """
        Initializes the optimizer and learning rate scheduler.

        Args:
          num_epochs: Number of epochs the model will be trained for.
          num_train_steps_per_epoch: Number of training steps per epoch.
        """
        optimizer = build_optimizer(self.optimizer_config,
                                    num_epochs=num_epochs,
                                    num_train_steps_per_epoch=num_train_steps_per_epoch)
        # Initialize training state
        self.state = TrainState.create(apply_fn=self.state.apply_fn,
                                       params=self.state.params,
                                       batch_stats=self.state.batch_stats,
                                       tx=optimizer,
                                       rng=self.state.rng)

    def create_jitted_functions(self):
        """
        Creates jitted versions of the training and evaluation functions.
        If self.debug is True, not jitting is applied.
        """
        train_step, eval_step = self.create_functions()
        if self.trainer_config.debug:  # Skip jitting
            logging.info('Skipping jitting due to debug=True')
            self.train_step = train_step
            self.eval_step = eval_step
        else:  # Jit
            self.train_step = jax.jit(train_step)
            self.eval_step = jax.jit(eval_step)

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
        def train_step(state : TrainState,
                       batch : Batch):
            metrics = {}
            return state, metrics
        def eval_step(state : TrainState,
                      batch : Batch):
            metrics = {}
            return metrics
        raise NotImplementedError

    def train_model(self,
                    train_loader : Iterator,
                    val_loader : Iterator,
                    test_loader : Optional[Iterator] = None,
                    num_epochs : int = 500) -> Dict[str, Any]:
        """
        Starts a training loop for the given number of epochs.

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
        all_eval_metrics = {}
        for epoch_idx in self.tracker(range(1, num_epochs+1), desc='Epochs'):
            self.on_training_epoch_start(epoch_idx)
            train_metrics = self.train_epoch(train_loader)
            self.logger.log_metrics(train_metrics, step=epoch_idx)
            self.on_training_epoch_end(train_metrics, epoch_idx)
            # Validation every N epochs
            if epoch_idx % self.trainer_config.check_val_every_n_epoch == 0:
                eval_metrics = self.eval_model(val_loader, log_prefix='val')
                self.logger.log_metrics(eval_metrics, step=epoch_idx)
                all_eval_metrics[epoch_idx] = eval_metrics
                self.save_metrics(f'eval_epoch_{epoch_idx:04d}', eval_metrics)
                self.on_validation_epoch_end(eval_metrics, epoch_idx)
        self.on_training_end()
        # Test best model if possible
        if test_loader is not None:
            self.load_model()
            test_metrics = self.eval_model(test_loader, log_prefix='test')
            self.logger.log_metrics(test_metrics, step=epoch_idx)
            self.save_metrics('test', test_metrics)
            all_eval_metrics['test'] = test_metrics
        # Close logger
        self.logger.finalize('success')
        return all_eval_metrics

    def train_epoch(self,
                    train_loader : Iterator) -> Dict[str, Any]:
        """
        Trains a model for one epoch.

        Args:
          train_loader: Data loader of the training set.

        Returns:
          A dictionary of the average training metrics over all batches
          for logging.
        """
        # Train model for one epoch, and log avg loss and accuracy
        metrics = defaultdict(float)
        num_train_steps = len(train_loader)
        start_time = time.time()
        for batch in self.tracker(train_loader, desc='Training', leave=False):
            self.state, step_metrics = self.train_step(self.state, batch)
            for key in step_metrics:
                metrics['train/' + key] += step_metrics[key] / num_train_steps
        metrics = {key: metrics[key].item() for key in metrics}
        metrics['epoch_time'] = time.time() - start_time
        return metrics

    def eval_model(self,
                   data_loader : Iterator,
                   log_prefix : Optional[str] = '') -> Dict[str, Any]:
        """
        Evaluates the model on a dataset.

        Args:
          data_loader: Data loader of the dataset to evaluate on.
          log_prefix: Prefix to add to all metrics (e.g. 'val/' or 'test/')

        Returns:
          A dictionary of the evaluation metrics, averaged over data points
          in the dataset.
        """
        # Test model on all images of a data loader and return avg loss
        metrics = defaultdict(float)
        num_elements = 0
        start_time = time.time()
        for batch in data_loader:
            step_metrics = self.eval_step(self.state, batch)
            batch_size = batch.size
            for key in step_metrics:
                metrics[key] += step_metrics[key] * batch_size
            num_elements += batch_size
        metrics = {(log_prefix + '/' + key): (metrics[key] / num_elements).item() for key in metrics}
        metrics[f'{log_prefix}/time'] = time.time() - start_time
        return metrics

    def tracker(self,
                iterator : Iterator,
                **kwargs) -> Iterator:
        """
        Wraps an iterator in a progress bar tracker (tqdm) if the progress bar
        is enabled.

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

    def save_metrics(self,
                     filename : str,
                     metrics : Dict[str, Any]):
        """
        Saves a dictionary of metrics to file. Can be used as a textual
        representation of the validation performance for checking in the terminal.

        Args:
          filename: Name of the metrics file without folders and postfix.
          metrics: A dictionary of metrics to save in the file.
        """
        with open(os.path.join(self.log_dir, f'metrics/{filename}.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

    def on_training_start(self):
        """
        Method called before training is started. Can be used for additional
        initialization operations etc.
        """
        logging.info('Starting training')
        for callback in self.callbacks:
            callback.on_training_start()

    def on_training_end(self):
        """
        Method called after training has finished. Can be used for additional
        logging or similar.
        """
        logging.info('Finished training')
        for callback in self.callbacks:
            callback.on_training_end()

    def on_training_epoch_start(self,
                                epoch_idx : int):
        """
        Method called at the start of each training epoch. Can be used for additional
        logging or similar.
        
        Args:
          epoch_idx: Index of the training epoch that has started.
        """
        logging.info(f'Starting training epoch {epoch_idx}')
        for callback in self.callbacks:
            callback.on_training_epoch_start(epoch_idx)

    def on_training_epoch_end(self,
                              train_metrics: Dict[str, Any],
                              epoch_idx : int):
        """
        Method called at the end of each training epoch. Can be used for additional
        logging or similar.

        Args:
          epoch_idx: Index of the training epoch that has finished.
        """
        logging.info(f'Finished training epoch {epoch_idx}')
        for callback in self.callbacks:
            callback.on_training_epoch_end(train_metrics, epoch_idx)

    def on_validation_epoch_end(self,
                                eval_metrics : Dict[str, Any],
                                epoch_idx : int):
        """
        Method called at the end of each validation epoch. Can be used for additional
        logging and evaluation.

        Args:
          epoch_idx: Index of the training epoch at which validation was performed.
          eval_metrics: A dictionary of the validation metrics. New metrics added to
            this dictionary will be logged as well.
          val_loader: Data loader of the validation set, to support additional
            evaluation.
        """
        logging.info(f'Finished validation epoch {epoch_idx}')
        for callback in self.callbacks:
            callback.on_validation_epoch_end(eval_metrics, epoch_idx)
    
    def load_model(self, epoch_idx : int = -1):
        """
        Loads model parameters and batch statistics from the logging directory.
        """
        logging.info(f'Loading model from epoch {epoch_idx}')
        state_dict = None
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpoint):
                state_dict = callback.load_model(epoch_idx)
                break
        if state_dict is None:
            raise ValueError('No model checkpoint callback found in callbacks.')
        self.restore(state_dict)

    def restore(self, state_dict : Dict[str, Any]):
        """
        Restores the state of the trainer from a state dictionary.

        Args:
          state_dict: State dictionary to restore from.
        """
        logging.info('Restoring trainer state')
        state_dict.pop('metrics')
        self.state = TrainState.create(apply_fn=self.model.apply,
                                       # Optimizer will be overwritten when training starts
                                       tx=self.state.tx if self.state.tx else optax.sgd(0.1),
                                       rng=self.state.rng,
                                       **state_dict
                                      )

    def bind_model(self):
        """
        Returns a model with parameters bound to it. Enables an easier inference
        access.

        Returns:
          The model with parameters and evt. batch statistics bound to it.
        """
        params = {'params': self.state.params}
        if self.state.batch_stats:
            params['batch_stats'] = self.state.batch_stats
        return self.model.bind(params)

    @classmethod
    def load_from_checkpoint(cls,
                             checkpoint : str,
                             exmp_input : Batch = None) -> Any:
        """
        Creates a Trainer object with same hyperparameters and loaded model from
        a checkpoint directory.

        Args:
          checkpoint: Folder in which the checkpoint and hyperparameter file is stored.
          exmp_input: An input to the model for shape inference.

        Returns:
          A Trainer object with model loaded from the checkpoint folder.
        """
        hparams_file = os.path.join(checkpoint, 'config.yaml')
        assert os.path.isfile(hparams_file), 'Could not find hparams file'
        config = ConfigDict(yaml.safe_load(hparams_file))
        if exmp_input is None:
            exmp_file = os.path.join(checkpoint, 'exmp_input.pkl')
            assert os.path.isfile(exmp_file), 'Could not find example input file'
            with open(exmp_file, 'rb') as f:
                exmp_input = pickle.load(f)
        trainer = cls(exmp_input=exmp_input,
                      trainer_config=config.trainer,
                      model_config=config.model,
                      optimizer_config=config.optimizer)
        trainer.load_model()
        return trainer