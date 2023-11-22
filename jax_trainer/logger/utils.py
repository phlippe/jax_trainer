import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from absl import logging
from ml_collections import ConfigDict
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


def flatten_configdict(
    cfg: ConfigDict,
    separation_mark: str = ".",
):
    """Returns a nested OmecaConf dict as a flattened dict, merged with the separation mark.

    Example:
        With separation_mark == '.', {'data': {'this': 1, 'that': 2}} is returned as {'data.this': 1, 'data.that': 2}.

    Args:
        cfg (ConfigDict): The nested config dict.
        separation_mark (str, optional): The separation mark to use. Defaults to ".".

    Returns:
        Dict: The flattened dict.
    """
    cfgdict = dict(cfg)
    keys = list(cfgdict.keys())
    for key in keys:
        if isinstance(cfgdict[key], ConfigDict):
            flat_dict = flatten_configdict(cfgdict.pop(key), separation_mark)
            for flat_key in flat_dict.keys():
                cfgdict[separation_mark.join([key, flat_key])] = flat_dict[flat_key]
    return cfgdict


def get_logging_dir(logger_config: ConfigDict, full_config: ConfigDict):
    """Returns the logging directory and version.

    Args:
        logger_config (ConfigDict): The logger config.
        full_config (ConfigDict): The full config of the trainer. Used for getting the model name for the default logging directory.

    Returns:
        Tuple[str, str]: The logging directory and version.
    """
    # Determine logging directory
    log_dir = logger_config.get("log_dir", None)
    if log_dir == "None":
        log_dir = None
    if not log_dir:
        base_log_dir = logger_config.get("base_log_dir", "checkpoints/")
        # Prepare logging
        model_name = full_config.model.name
        if not isinstance(model_name, str):
            model_name = model_name.__name__
        log_dir = os.path.join(base_log_dir, model_name.split(".")[-1])
        if logger_config.get("logger_name", None) is not None:
            log_dir = os.path.join(log_dir, logger_config.logger_name)
        version = None
    else:
        version = ""
    return log_dir, version


def build_tool_logger(logger_config: ConfigDict, full_config: ConfigDict):
    """Builds the logger tool object, either Tensorboard or Weights and Biases.

    Args:
        logger_config (ConfigDict): The logger config.
        full_config (ConfigDict): The full config of the trainer, to be logged.

    Returns:
        The logger tool object.
    """
    # Determine logging directory
    log_dir, version = get_logging_dir(logger_config, full_config)
    # Create logger object
    logger_type = logger_config.get("tool", "TensorBoard").lower()
    if logger_type == "tensorboard":
        logger = TensorBoardLogger(save_dir=log_dir, version=version, name="")
        logger.log_hyperparams(flatten_configdict(full_config))
    elif logger_type == "wandb":
        logger = WandbLogger(
            name=logger_config.get("project_name", None),
            save_dir=log_dir,
            version=version,
            config=full_config,
        )
    else:
        raise ValueError(f"Unknown logger type {logger_type}.")
    return logger
