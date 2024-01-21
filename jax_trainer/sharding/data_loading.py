from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from jax_trainer.datasets import Batch


def gather_batch(batch: Batch, axis_name: str) -> Batch:
    batch = jax.lax.all_gather(batch, axis_name=axis_name, tiled=True, axis=0)
    batch = batch.replace(size=batch.size.sum())
    return batch


def create_sharded_array(pytrees: Any, mesh: Mesh):
    return jax.tree_map(lambda arr: global_array_from_shard(arr, mesh), pytrees)


def create_sharded_batch(batch: Batch, mesh: Mesh) -> Batch:
    return jax.tree_map(
        lambda arr: global_array_from_shard(arr, mesh)
        if isinstance(arr, np.ndarray)
        else arr // jax.local_device_count(),
        batch,
    )


def global_array_from_shard(array: np.ndarray, mesh: Mesh):
    arrays = []
    sharding = jax.sharding.NamedSharding(mesh, P(*mesh.axis_names))
    global_shape = (
        tuple([mesh.shape[axis_name] for axis_name in mesh.axis_names]) + array.shape[1:]
    )
    slices = iter(np.split(array, jax.local_device_count()))
    for dev, s in sharding.addressable_devices_indices_map(global_shape).items():
        arrays.append(jax.device_put(next(slices)[None], device=dev))
    return jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)
