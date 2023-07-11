from typing import Any, Callable, NamedTuple, Sequence, Union

import numpy as np
import PIL
import torch
import torch.utils.data as data
from flax.struct import dataclass
from ml_collections import ConfigDict

from jax_trainer.datasets.collate import numpy_collate


def build_data_loaders(
    *datasets: Sequence[data.Dataset],
    train: Union[bool, Sequence[bool]] = True,
    collate_fn: Callable = numpy_collate,
    config: ConfigDict = ConfigDict()
):
    """Creates data loaders used in JAX for a set of datasets.

    Args:
        datasets: Datasets for which data loaders are created.
        train: Sequence indicating which datasets are used for
            training and which not. If single bool, the same value
            is used for all datasets.
        batch_size: Batch size to use in the data loaders.
        num_workers: Number of workers for each dataset.
        seed: Seed to initialize the workers and shuffling with.

    Returns:
        List of data loaders.
    """
    loaders = []
    if not isinstance(train, (list, tuple)):
        train = [train for _ in datasets]
    for dataset, is_train in zip(datasets, train):
        loader = data.DataLoader(
            dataset,
            batch_size=config.get("batch_size", 128),
            shuffle=is_train,
            drop_last=is_train,
            collate_fn=collate_fn,
            num_workers=config.get("num_workers", 4),
            persistent_workers=is_train,
            generator=torch.Generator().manual_seed(config.get("seed", 42)),
        )
        loaders.append(loader)
    return loaders
