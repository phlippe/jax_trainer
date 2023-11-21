from functools import partial
from typing import Any, Callable, NamedTuple, Sequence, Union

import numpy as np


def numpy_collate(batch: Union[np.ndarray, Sequence[Any], Any]):
    """Collate function for numpy arrays.

    This function acts as replacement to the standard PyTorch-tensor collate function in PyTorch DataLoader.

    Args:
        batch: Batch of data. Can be a numpy array, a list of numpy arrays, or nested lists of numpy arrays.

    Returns:
        Batch of data as (potential list or tuple of) numpy array(s).
    """
    if isinstance(batch, np.ndarray):
        return batch
    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def batch_collate(tuple_class: NamedTuple, batch: Sequence[Any]):
    """Transforms list of inputs to dataclass of inputs.

    Args:
        tuple_class: Batch class to be constructed. Can be a dataclass or a NamedTuple.
        batch: List of inputs.

    Returns:
        Batch object of inputs.
    """
    size = batch[0].shape[0]
    return tuple_class(size, *batch)


def numpy_batch_collate(tuple_class: NamedTuple, batch: Sequence[Any]):
    """Wrapper function to combine numpy_collate and batch_collate into a single function.

    Args:
        tuple_class: Batch class to be constructed. Can be a dataclass or a NamedTuple.
        batch: List of inputs.

    Returns:
        Batch object of inputs.
    """
    return batch_collate(tuple_class, numpy_collate(batch))


def build_batch_collate(tuple_class: NamedTuple):
    """Wrapper function to combine numpy_collate and batch_collate into a single function.

    Args:
        tuple_class: Batch class to be constructed. Can be a dataclass or a NamedTuple.

    Returns:
        Collate function to combine list of inputs to batch object of inputs.
    """
    return partial(numpy_batch_collate, tuple_class)
