# JAX-Trainer: Lightning-like API for JAX with Flax

This repository is a work in progress. The goal is to provide a Lightning-like API for JAX with Flax. The API is inspired by [PyTorch Lightning](https://github.com/Lightning-AI/lightning) and has as basic element a `TrainerModule`. This module implements common training and evaluation loops, and can be used to train a model with a few lines of code. The train loop can be extended via callbacks, which are similar to Lightning's callbacks. The API is still in flux and may change in the future.

For handling hyperparameters, this repository makes use of [ml-collections](https://ml-collections.readthedocs.io/en/latest/). This library provides a hierarchical configuration system, which is used to configure the `TrainerModule` and the callbacks.

For an example usage, see our [template repository](https://github.com/phlippe/jax_trainer_template).

## Installation

In future, the package will be available on PyPI. For now, you can install it from source:

```bash
git clone https://github.com/phlippe/jax_trainer.git
cd jax_trainer
pip install -e .
```

## Usage

In the following, we will go through the main API choices in the library. In most cases, the user will only need to implement a loss function in a subclass of `TrainerModule` for each task, besides the actual models in Flax. The training loop can be further customized via callbacks. All modules are then configured via a YAML file and can be trained with a few lines of code. For an example, see our [template repository](https://github.com/phlippe/jax_trainer_template).

### TrainerModule API

The `jax_trainer.trainer.TrainerModule` has been written with the goal to be as flexible as possible while still providing a simple API for training and evaluation. It's main functions are configurable via `ConfigDict`s and can be overwritten by the user.

The main aspects of the trainer is to:

- **Initialize the model**: The model is initialized via the `init_model` function. This function is called at the beginning of the training and evaluation. The function can be overwritten by the user to implement custom initialization logic.
- **Handling the TrainState**: The trainer keeps a `TrainState` which contains the model state, the optimizer state, the random number generator state, and any mutable variables. The `TrainState` is updated after each training step and can be used to resume training from a checkpoint.
- **Logging**: The trainer provides a simple logging interface by allowing the train and evaluation functions to return dictionaries of metrics to log.
- **Saving and loading checkpoints**: The trainer provides functions to save and load checkpoints. The checkpoints are saved as `TrainState`s and can be used to resume training or to evaluate a model. A pre-trained model can also be loaded by simply calling `TrainerModule.load_from_checkpoint`, similar to the API in Lightning.
- **Training and evaluation**: The trainer provides functions to train and evaluate a model. The training and evaluation loops can be extended via callbacks, which are called at different points during the training and evaluation.

As a user, the main function that needs to be implemented for each individual task is `loss_function(...)`. This function takes as input the model parameters and state, the batch of data, a random number generator key, and a boolean indicating whether its training or not. The function needs to return the loss, as well as a tuple of mutable variables and optional metrics. The `TrainerModule` then takes care of the rest, which includes wrapping it into a training and evaluation function, performing gradient transformations, and calling it in a loop. Additionally, to provide a unified interface with other functions like initialization, the subclass needs to implement `batch_to_input` which, given a batch, returns the input to the model. The following example shows a simple trainer module for image classification:

```python
class ImgClassifierTrainer(TrainerModule):

    def batch_to_input(self, batch: SupervisedBatch) -> Any:
        return batch.input

    def loss_function(
        self,
        params: Any,
        state: TrainState,
        batch: SupervisedBatch,
        rng: jax.Array,
        train: bool = True,
    ) -> Tuple[Any, Tuple[Any, Dict]]:
        imgs = batch.input
        labels = batch.target
        logits, mutable_variables = self.model_apply(
            params=params, state=state, input=imgs, rng=rng, train=train
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        acc = (logits.argmax(axis=-1) == labels).mean()
        metrics = {"acc": acc}
        return loss, (mutable_variables, metrics)
```

### Logging API

The `metrics` dictionary returned by the loss function is used for logging. By default, the logger supports to log values every *N* training steps and/or per epoch. For more options on the logger, see the configuration documentation below.

Further, the logging of each metric can be customized by providing additional options in the `metrics` dictionary. For each metric, the following options are available:

- `mode`: The mode of the metric describes how it should be aggregated over the epoch or batches. The different options are summarized in the `jax_trainer.logger.LogMetricMode` enum. Currently, the following modes are available:
  - `LogMetricMode.MEAN`: The mean of the metric is logged.
  - `LogMetricMode.SUM`: The sum of the metric is logged.
  - `LogMetricMode.SINGLE`: A single value of the metric is used, namely the last one logged.
  - `LogMetricMode.MAX`: The max of the metric is logged.
  - `LogMetricMode.MIN`: The min of the metric is logged.
  - `LogMetricMode.STD`: The standard deviation of the mtric is logged.
  - `LogMetricMode.CONCAT`: The values of the metric are concatenated. Note that in this case, the metric is not logged to the tool of choice (e.g. Tensorboard or WandB), but is only provided in the full metric dictionary, which can be used as input to callbacks.
- `log_freq`: The frequency of logging the metric. The options are summarized in `jax_trainer.logger.LogFreq` and are the following:
  - `LogFreq.EPOCH`: The metric is logged only once per epoch.
  - `LogFreq.STEP`: The metric is logged only per *N* steps.
  - `LogFreq.ANY`: The metric is logged both per epoch and per *N* steps.
- `log_mode`: The training mode in which the metric should be logged. This allows for different metrics to be logged during training, validation and/or testing. The options are summarized in the enum `jax_trainer.logger.LogMode` with the options:
  - `LogMode.TRAIN`: The metric is logged during training.
  - `LogMode.VAL`: The metric is logged during validation.
  - `LogMode.TEST`: The metric is logged during testing.
  - `LogMode.EVAL`: The metric is logged during both validation and testing.
  - `LogMode.ANY`: The metric is logged during any of the above modes.

### Callback API

The `TrainerModule` provides a callback API which is similar to the one in Lightning. The callbacks are called at different points during the training and evaluation. Each callback can implement the following methods:

- `on_training_start`: Called at the beginning of the training.
- `on_training_end`: Called at the end of the training.
- `on_filtered_training_epoch_start`: Called at the beginning of each training epoch.
- `on_filtered_training_epoch_end`: Called at the end of each training epoch.
- `on_filtered_validation_epoch_start`: Called at the beginning of the validation.
- `on_filtered_validation_epoch_end`: Called at the end of the validation.
- `on_test_epoch_start`: Called at the beginning of the testing.
- `on_test_epoch_end`: Called at the end of the testing.

The training and validation functions with `filtered` in the name are only called every *N* epochs, where *N* is the value of `every_n_epochs` in the callback configuration.

The following callbacks are pre-defined:

- `ModelCheckpoint`: Saves the model and optimizer state after validation. This checkpoint can be used to resume training or to evaluate the model. It is implemented using `orbax` and is similar to the `ModelCheckpoint` in Lightning.
- `LearningRateMonitor`: Logs the learning rate at the beginning of each epoch. This is similar to the `LearningRateMonitor` in Lightning.
- `ConfusionMatrixCallback`: As an example for a custom callback for classification, this callback logs the confusion matrix of a classifier after validation and testing. This callback requires the metric key `conf_matrix` to be logged.

For configuring the callbacks, also for custom callbacks, see the configuration documentation below.

### Dataset API

The dataset API abstracts the data loading with PyTorch, using numpy arrays for storage. Each dataset needs to provide a train, validation and test loader. As return type, we use `flax.struct.dataclass`es, which are similar to PyTorch's `NamedTuple`s. These dataclasses can be used in jit-compiled functions and are therefore a good fit for JAX. Additionally, each batch should define a `size` attribute, which is used for taking the correct average across batches in evaluation. For an example, see the `jax_trainer.datasets.examples` module.

## Configuration

The configuration is done via a YAML file. It consists of four main sections: `trainer`, `model`, `optimizer`, and `dataset`. The `trainer` section configures the `TrainerModule` and the callbacks. The `model` section configures the model, which is implemented by the user. The `optimizer` section configures the optimizer and the learning rate scheduler. The `dataset` section configures the dataset. The following example shows a configuration for training a simple MLP on CIFAR10:

```yaml
trainer:
  name: ImgClassifierTrainer
  train_epochs: 5
  check_val_every_n_epoch: 1
  debug: False
  enable_progress_bar: True
  tabulate_model: True
  seed: 42
  seed_eval: 0
  logger:
    log_dir: tests/checkpoints/BuildTrainerTest/
    tool: TensorBoard
    project_name: default
    log_file_verbosity: warning
  callbacks:
    ModelCheckpoint:
      monitor: val/acc
      mode: max
      save_top_k: 1
      save_optimizer_state: False
    LearningRateMonitor:
      every_n_epochs: 1
    ConfusionMatrixCallback:
      normalize: True
      cmap: Blues
      every_n_epochs: 2
model:
  name: tests.models.SimpleEncoder
  hparams:
    c_hid: 32
    latent_dim: 10
    act_fn: gelu
    batch_norm: True
optimizer:
  name: adam
  lr: 1e-3
  params:
    beta1: 0.9
  transforms:
    weight_decay: 0
    gradient_clip_norm: 10.0
  scheduler:
    name: warmup_cosine_decay
    warmup_steps: 100
dataset:
  constructor: jax_trainer.datasets.build_cifar10_datasets
  data_dir: data/
  batch_size: 128
  num_workers: 4
```

In the following, we will go through the different sections and explain the configuration options.

### Trainer

The `trainer` section configures the `TrainerModule` and the callbacks. The `TrainerModule` is configured via the following options:

- `name`: Name of the `TrainerModule` class. Currently, the following classes are available:
  - `ImgClassifierTrainer`: Trainer for image classification tasks.
  - `TrainerModule`: Base class for implementing custom trainers.
    For own-implemented trainers, the name of the class is the path to the class, e.g. `my_module.MyTrainer`.
- `train_epochs`: Number of training epochs.
- `seed`: Seed for the initialization, model state, etc.
- `check_val_every_n_epoch` (optional): Number of epochs between validation checks (default: 1). If set to `0`, no validation is performed.
- `debug` (optional): If `True`, the trainer is run in debug mode (default: False). This means that the training and validation steps are not jitted and can be easier analysed in case of an error.
- `enable_progress_bar` (optional): If True, a progress bar is shown during training and validation (default: True).
- `tabulate_model` (optional): If True, the model is tabulated and the result is printed to the logging file (default: True).
- `seed_eval` (optional): Seed for the evaluation (default: 0). This seed is used for the validation and evaluation of the model after training.
- `log_grad_norm` (optional): If True, the gradient norm is logged during training.
- `logger`: Configuration of the logger. This is optional and in case of not being provided, a default logger is created. The following options are available:
  - `class` (optional): Name of the logger class. The default logger is `Logger`, but can be overwritten by providing the path to a custom logger class, e.g. `my_module.MyLogger`.
  - `log_dir` (optional): Directory where the logging files are stored. If not provided, a default directory based on the model name and version is created.
  - `base_log_dir` (optional): Only used if `log_dir` is None or not given. Base directory where the logging files are stored. If not provided, the default directory of `checkpoints/` is used.
  - `logger_name` (optional): Name of the logger. Is appended to the `base_log_dir` with the model name if given.
  - `tool` (optional): Name of the logging tool (default: Tensorboard). Currently, the following tools are available:
    - `TensorBoard`: Logging to TensorBoard.
    - `WandB`: Logging to Weights & Biases.
  - `project_name` (optional): Name of the project. This is only used for Weights & Biases.
  - `log_steps_every` (optional): Number of training steps between logging (default: 50). If set to `0`, logging is only performed per epoch. Otherwise, both per-epoch and per-step logging is performed.
  - `log_file_verbosity` (optional): Verbosity of the logging file. Possible values are `debug`, `info`, `warning`, and `error`. By default, the verbosity is set to `info`.
  - `stderrthreshold` (optional): Verbosity of the logging to stderr. Possible values are `debug`, `info`, `warning`, and `error`. By default, the verbosity is set to `warning`.

The `callbacks` section configures the callbacks. The key of a callback is its name (if its a default one in `jax_trainer`) or arbitrary description. In case of the latter, the attrbitue `class` needs to be added, with the respective class path, e.g. `class: mymodule.MyCallback`. Each callback has its own config and parameters. The following callbacks are pre-defined:

- `ModelCheckpoint`: Saves the model and optimizer state after validation.
  - `monitor` (optional): Metric to monitor (default: `val/loss`).
  - `mode` (optional): Mode of the metric (default: `min`). Possible values are `min`, `max`, and `auto`.
  - `save_top_k` (optional): Number of best models to save (default: `1`).
  - `save_optimizer_state` (optional): If True, the optimizer state is saved as well (default: `False`).
- `LearningRateMonitor`: Logs the learning rate at the beginning of each epoch.
  - `every_n_epochs` (optional): Number of training epochs between logging (default: `1`).
- `ConfusionMatrixCallback`: Logs the confusion matrix of a classifier after validation and testing. Requires the metric key `conf_matrix` to be logged.
  - `normalize` (optional): If True, the confusion matrix is normalized (default: `True`).
  - `cmap` (optional): Colormap of the confusion matrix (default: `Blues`).
  - `figsize` (optional): Size of the figure (default: `(8, 8)`).
  - `format` (optional): Format of the in-place text in each cell (default: `'.2%'` when normalized, else `'d'`).
  - `dpi` (optional): Dots per inch of the figure (default: `100`).
  - `every_n_epochs` (optional): Number of training epochs between logging (default: `1`).

### Model

The `model` section configures the model. The following options are available:

- `name`: Path to the model class. The path is relative to the current working directory.
- `hparams` (optional): Hyperparameters of the model. These are passed to the model constructor. This is optional and can be omitted if the model does not require any hyperparameters.

### Optimizer

The `optimizer` section configures the optimizer and the learning rate scheduler. The section has three main sub-sections:

#### Optimizer class

- `name`: Name of the optimizer. Currently, the following optimizers are available:
  - `adam`: Adam optimizer.
  - `adamw`: AdamW optimizer.
  - `sgd`: SGD optimizer.

Additionally, there are some options which are specific to the optimizer that can be defined in the config option `params`:

- `beta1` (optional): Beta1 parameter of the Adam/AdamW optimizer (default: `0.9`).
- `beta2` (optional): Beta2 parameter of the Adam/AdamW optimizer (default: `0.999`).
- `eps` (optional): Epsilon parameter of the Adam/AdamW optimizer (default: `1e-8`).
- `momentum` (optional): Momentum parameter of the SGD optimizer (default: `0.0`).
- `nestorov` (optional): If True, Nesterov momentum is used in the SGD optimizer (default: `False`).

#### Learning rate scheduler

The learning rate scheduler is optional and is by default constant. The following options are available:

- `lr`: Base learning rate of the optimizer.
- `scheduler` (optional): Configuration of a learning rate scheduler. This is optional and can be omitted if no scheduler is used. The following options are available:
  - `name`: Name of the scheduler. Currently, the following schedulers are available:
    - `constant`: Constant scheduler.
    - `cosine_decay`: Cosine decay scheduler.
    - `exponential_decay`: Exponential decay scheduler.
    - `warmup_cosine_decay`: Warmup cosine decay scheduler.

For each learning rate schedule, there are some options which are specific to the scheduler:

- `alpha` (optional): Alpha parameter of the cosine decay scheduler. The minimum value of the multiplier used to adjust the learning rate (default: `0.0`).
- `decay_rate`: Decay rate of the exponential decay scheduler. Needs to be set if the scheduler is `exponential_decay`.
- `transition_steps` (optional): Factor with which to divide the step count in the exponential decay scheduler (default: `0`).
- `staircase` (optional): If True, the learning rate is decayed at discrete intervals (default: `False`).
- `warmup_steps`: Number of warmup steps of the warmup scheduler. Needs to be set if the scheduler is `warmup_cosine_decay`.
- `end_value` (optional): End value for the learning rate of the warmup scheduler (default: `0.0`).

#### Gradient Transformations

The gradient transformations are optional and can be used to transform the gradients before applying them to the model parameters. They are defined under the key `transforms`. Each can be defined by a value (e.g. float), or a dictionary with the following options:

- `value`: Value of the transformation.
- `before_optimizer`: If True, the transformation is applied before the optimizer step. If False, the transformation is applied after the optimizer step (default: `True`).

The following gradient transformations are available:

- `weight_decay` (optional): Weight decay parameter of the optimizer (default: `0.0`).
- `gradient_clip_norm` (optional): Gradient clipping norm of the optimizer (default: `None`).
- `gradient_clip_value` (optional): Gradient clipping value of the optimizer (default: `None`).

### Dataset

The `dataset` section configures the dataset and the data loading. The following options are available:

- `constructor`: Path to the dataset constructor. The path is relative to the current working directory.
- `data_dir` (optional): Directory where the dataset is stored (default: `data/`).
- `batch_size` (optional): Batch size to use during training, validation and testing (default: `128`).
- `num_workers` (optional): Number of workers to use for data loading (default: `4`).
- `seed` (optional): Seed for the data loading (default: `42`).

## Contributing

Contributions are welcome! Before contributing code, please install the pre-commit hooks with:

```bash
pip install pre-commit
pre-commit install
```

This will run the linter and formatter on every commit.

If you have any questions, feel free to open an issue or contact me directly.
