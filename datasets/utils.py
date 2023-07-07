from typing import NamedTuple, Any, Union, Sequence, Callable
import numpy as np
import torch
import torch.utils.data as data
import PIL
from ml_collections import ConfigDict
from flax.struct import dataclass


class DatasetModule(NamedTuple):
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


def image_to_numpy(img : Any):
    img = np.array(img, dtype=np.float32) / 255.
    return img


def normalize_transform(mean : Union[np.ndarray, float] = 0.,
                        std : Union[np.ndarray, float] = 1.):
    return lambda x: (x - mean.astype(x.dtype)) / std.astype(x.dtype)


def numpy_collate(batch: Union[np.ndarray, Sequence[Any], Any]):
    if isinstance(batch, np.ndarray):
        return batch
    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
    

def batch_collate(tuple_class: NamedTuple, batch: Sequence[Any]):
    size = batch[0].shape[0]
    return tuple_class(size, *batch)


def build_batch_collate(tuple_class: NamedTuple):
    return lambda batch: batch_collate(tuple_class, numpy_collate(batch))



def build_data_loaders(*datasets : Sequence[data.Dataset],
                        train : Union[bool, Sequence[bool]] = True,
                        collate_fn : Callable = numpy_collate,
                        config : ConfigDict = ConfigDict()):
    """
    Creates data loaders used in JAX for a set of datasets.

    Args:
      datasets: Datasets for which data loaders are created.
      train: Sequence indicating which datasets are used for
        training and which not. If single bool, the same value
        is used for all datasets.
      batch_size: Batch size to use in the data loaders.
      num_workers: Number of workers for each dataset.
      seed: Seed to initialize the workers and shuffling with.
    """
    loaders = []
    if not isinstance(train, (list, tuple)):
        train = [train for _ in datasets]
    for dataset, is_train in zip(datasets, train):
        loader = data.DataLoader(dataset,
                                 batch_size=config.get('batch_size', 128),
                                 shuffle=is_train,
                                 drop_last=is_train,
                                 collate_fn=collate_fn,
                                 num_workers=config.get('num_workers', 4),
                                 persistent_workers=is_train,
                                 generator=torch.Generator().manual_seed(config.get('seed', 42)))
        loaders.append(loader)
    return loaders