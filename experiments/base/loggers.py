from ml_collections import ConfigDict
import os

# Logging with Tensorboard or Weights and Biases
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


def flatten_configdict(
    cfg: ConfigDict,
    separation_mark: str = ".",
):
    """Returns a nested OmecaConf dict as a flattened dict, merged with the separation mark.
    Example:
        With separation_mark == '.', {'data':{'this': 1, 'that': 2} is returned as {'data.this': 1, 'data.that': 2}.
    """
    cfgdict = dict(cfg)
    keys = list(cfgdict.keys())
    for key in keys:
        if isinstance(cfgdict[key], ConfigDict):
            flat_dict = flatten_configdict(cfgdict.pop(key), separation_mark)
            for flat_key in flat_dict.keys():
                cfgdict[separation_mark.join([key, flat_key])] = flat_dict[flat_key]
    return cfgdict


def build_logger(logger_config: ConfigDict, full_config: ConfigDict):
    # Determine logging directory
    log_dir = logger_config.get('log_dir', None)
    if log_dir == 'None':
        log_dir = None
    if not log_dir:
        base_log_dir = logger_config.get('base_log_dir', 'checkpoints/')
        # Prepare logging
        log_dir = os.path.join(base_log_dir, full_config.model.name.split('.')[-1])
        if logger_config.get('logger_name', None) is not None:
            log_dir = os.path.join(log_dir, logger_config.logger_name)
        version = None
    else:
        version = ''
    # Create logger object
    logger_type = logger_config.get('tool', 'TensorBoard').lower()
    if logger_type == 'tensorboard':
        logger = TensorBoardLogger(save_dir=log_dir,
                                   version=version,
                                   name='')
        logger.log_hyperparams(flatten_configdict(full_config))
    elif logger_type == 'wandb':
        logger = WandbLogger(name=logger_config.get('project_name', None),
                             save_dir=log_dir,
                             version=version,
                             config=full_config)
    else:
        raise ValueError(f'Unknown logger type {logger_type}.')
    
    return logger