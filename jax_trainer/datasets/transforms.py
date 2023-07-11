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
    return lambda x: (x - mean.astype(x.dtype)) / std.astype(x.dtype)
