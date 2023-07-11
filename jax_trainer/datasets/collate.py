from typing import Any, Callable, NamedTuple, Sequence, Union

import numpy as np


def numpy_collate(batch: Union[np.ndarray, Sequence[Any], Any]):
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
    size = batch[0].shape[0]
    return tuple_class(size, *batch)


def build_batch_collate(tuple_class: NamedTuple):
    return lambda batch: batch_collate(tuple_class, numpy_collate(batch))
