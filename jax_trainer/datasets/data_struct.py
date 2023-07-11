import numpy as np
import torch.utils.data as data
from flax.struct import dataclass
from ml_collections import ConfigDict


@dataclass
class DatasetModule:
    config: ConfigDict
    train: data.Dataset
    val: data.Dataset
    test: data.Dataset
    train_loader: data.DataLoader
    val_loader: data.DataLoader
    test_loader: data.DataLoader


@dataclass
class Batch:
    size: int
    # Add any additional batch information here


@dataclass
class SupervisedBatch(Batch):
    input: np.ndarray
    target: np.ndarray
