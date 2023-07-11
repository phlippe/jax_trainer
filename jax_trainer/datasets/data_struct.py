import numpy as np
import torch.utils.data as data
from flax.struct import dataclass
from ml_collections import ConfigDict


@dataclass
class DatasetModule:
    """Data module class that holds the datasets and data loaders."""

    config: ConfigDict
    train: data.Dataset
    val: data.Dataset
    test: data.Dataset
    train_loader: data.DataLoader
    val_loader: data.DataLoader
    test_loader: data.DataLoader


@dataclass
class Batch:
    """Base class for batches.

    Attribute `size` is required and used, e.g. for logging.
    """

    size: int
    # Add any additional batch information here


@dataclass
class SupervisedBatch(Batch):
    """Extension of the base batch class for supervised learning."""

    input: np.ndarray
    target: np.ndarray
