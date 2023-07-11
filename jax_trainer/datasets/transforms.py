from typing import Any, Union

import numpy as np


def image_to_numpy(img: Any):
    img = np.array(img, dtype=np.float32) / 255.0
    return img


def normalize_transform(mean: Union[np.ndarray, float] = 0.0, std: Union[np.ndarray, float] = 1.0):
    return lambda x: (x - mean.astype(x.dtype)) / std.astype(x.dtype)
