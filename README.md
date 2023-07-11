# JAX-Trainer: Lightning-like API for JAX with Flax

This repository is a work in progress. The goal is to provide a Lightning-like API for JAX with Flax. The API is inspired by [PyTorch Lightning](https://github.com/Lightning-AI/lightning) and has as basic element a `TrainerModule`. This module implements common training and evaluation loops, and can be used to train a model with a few lines of code. The train loop can be extended via callbacks, which are similar to Lightning's callbacks. The API is still in flux and may change in the future.

For handling hyperparameters, this repository makes use of [ml-collections](https://ml-collections.readthedocs.io/en/latest/). This library provides a hierarchical configuration system, which is used to configure the `TrainerModule` and the callbacks.

## Installation

In future, the package will be available on PyPI. For now, you can install it from source:

```bash
git clone https://github.com/phlippe/jax_trainer.git
cd jax_trainer
pip install -e .
```

## Usage

TODO: Write a simple example and point to the examples folder.

### TrainerModule API

### Callback API

### Dataset API

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

The `callbacks` section configures the callbacks. The key of a callback is its name (if its a default one in `jax_trainer`) or the path to an self-implemented callback. Each callback has its own config and parameters. The following callbacks are pre-defined:

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

Additionally, there are some options which are specific to the optimizer:

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

The gradient transformations are optional and can be used to transform the gradients before applying them to the model parameters. Each can be defined by a value (e.g. float), or a dictionary with the following options:

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
