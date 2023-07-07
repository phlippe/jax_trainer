from ml_collections import ConfigDict
from ml_collections import config_flags
from experiments.base.optimizer_constructor import build_optimizer
from absl import app
import inspect
import os

import jax

import numpy as np
import torch
import pytorch_lightning as pl

from datasets.dataset_constructor import build_dataset_module
import experiments

_CONFIG_FILE = config_flags.DEFINE_config_file("cfg", default="cfg/default_config.py")

def main(_):
    cfg = _CONFIG_FILE.value
    cfg = cfg.unlock()  # Allow to append values to the config dict.

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    pl.seed_everything(cfg.seed)

    if cfg.get('num_gpus', -1) > -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in range(cfg.num_gpus)])
    cfg.num_gpus = cfg.get('num_gpus', jax.device_count())
    cfg.trainer.num_gpus = cfg.num_gpus
    cfg.device_information = ' | '.join([f'{x} - {x.device_kind}' for x in jax.devices()])
    print('Available GPUs:', cfg.device_information)

    cfg.dataset.seed = cfg.seed
    cfg.trainer.seed = cfg.seed

    # Build dataset
    dataset = build_dataset_module(cfg.dataset)
    exmp_input = next(iter(dataset.train_loader))
    trainer_class = getattr(experiments, cfg.trainer.name)
    trainer = trainer_class(trainer_config=cfg.trainer,
                            model_config=cfg.model,
                            optimizer_config=cfg.optimizer,
                            exmp_input=exmp_input)
    eval_metrics = trainer.train_model(train_loader=dataset.train_loader, 
                                        val_loader=dataset.val_loader, 
                                        test_loader=dataset.test_loader,
                                        num_epochs=5)


if __name__ == '__main__':
    app.run(main)