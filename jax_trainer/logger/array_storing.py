import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class ArraySpec:
    shape: Tuple[int, ...]
    dtype: Any
    device: Any
    value: Any = 0


def array_to_spec(array: jnp.ndarray) -> ArraySpec:
    return ArraySpec(
        shape=array.shape,
        dtype=array.dtype,
        device=str(array.device()),
        value=array.reshape(-1)[0].item(),
    )


def np_array_to_spec(array: np.ndarray) -> ArraySpec:
    return ArraySpec(
        shape=array.shape, dtype=array.dtype, device="numpy", value=array.reshape(-1)[0]
    )


def spec_to_array(spec: ArraySpec) -> jnp.ndarray:
    device = spec.device
    if device == "numpy":
        return np.full(spec.shape, spec.value, dtype=spec.dtype)
    else:
        array = jnp.full(spec.shape, spec.value, dtype=spec.dtype)
        if isinstance(device, str):
            device = jax.devices(device)[0]
        array = jax.device_put(x=array, device=device)
        return array


def convert_to_array_spec(input: Any) -> Any:
    if isinstance(input, jnp.ndarray):
        return array_to_spec(input)
    elif isinstance(input, np.ndarray):
        return np_array_to_spec(input)
    else:
        return input


def convert_from_array_spec(input: Any) -> Any:
    if isinstance(input, ArraySpec):
        return spec_to_array(input)
    else:
        return input


def save_pytree(pytree: Any, path: str | Path):
    pytree = jax.tree_map(convert_to_array_spec, pytree)
    with open(path, "wb") as f:
        pickle.dump(pytree, f)


def load_pytree(path: str | Path) -> Any:
    with open(path, "rb") as f:
        pytree = pickle.load(f)
    pytree = jax.tree_map(convert_from_array_spec, pytree)
    return pytree
