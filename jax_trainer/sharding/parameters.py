from typing import Any, Dict

import jax
from flax import struct
from flax.core import meta

PyTree = Any


class Partitioned(struct.PyTreeNode, meta.AxisMetadata):
    """A partitioned array."""

    value: jax.Array
    axis: int = struct.field(pytree_node=False)

    @property
    def shape(self):
        return self.value.shape

    @property
    def size(self):
        return self.value.size

    def unbox(self):
        return self.value

    def replace_boxed(self, val: Any) -> meta.AxisMetadata:
        return self.replace(value=val)

    def add_axis(self, index: int, params: Dict[Any, Any]):
        if index <= self.axis:
            return self.replace(axis=self.axis + 1)
        else:
            return self

    def remove_axis(self, index: int, params: Dict[Any, Any]):
        if index < self.axis:
            return self.replace(axis=self.axis - 1)
        else:
            return self


def sync_gradients(grads: PyTree, axis_name: str | None) -> PyTree:
    """Syncs gradients across the specified axis.

    Args:
        grads: The gradients to sync.
        axis_name: The axis to sync across.

    Returns:
        The synced gradients.
    """
    if axis_name is None:
        return grads

    def sync_grad(g):
        if isinstance(g, Partitioned):
            return g  # TODO: Check later if this is correct.
        else:
            return jax.lax.pmean(g, axis_name=axis_name)

    grads = jax.tree_map(sync_grad, grads, is_leaf=lambda x: isinstance(x, Partitioned))
    return grads
