from functools import partial
from typing import Any, Union

import numpy as np


def image_to_numpy(img: Any):
    """Converts image to numpy array.

    Function used in the transformation of a PyTorch data loader. This function is used to convert the image to a numpy
    array, normalized to the range [0, 1].

    Args:
        img: Image to be converted.

    Returns:
        Numpy array of the image.
    """
    img = np.array(img, dtype=np.float32) / 255.0
    return img


def normalize_transform(mean: Union[np.ndarray, float] = 0.0, std: Union[np.ndarray, float] = 1.0):
    """Normalization transform on numpy arrays.

    Args:
        mean: Mean of the normalization.
        std: Standard deviation of the normalization.

    Returns:
        Normalization function.
    """
    if isinstance(mean, float):
        mean = np.array(mean)
    if isinstance(std, float):
        std = np.array(std)
    return partial(normalize, mean=mean, std=std)


def normalize(x: np.ndarray, mean: np.ndarray = 0.0, std: np.ndarray = 1.0):
    """Normalize numpy array.

    Args:
        x: Array to be normalized.
        mean: Mean of the normalization.
        std: Standard deviation of the normalization.

    Returns:
        Normalized array.
    """
    return (x - mean.astype(x.dtype)) / std.astype(x.dtype)
