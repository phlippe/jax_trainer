import functools
from typing import Any, Callable, Dict, List, Literal, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from jax_trainer.parallelism import fold_rng_over_axis
from jax_trainer.parallelism.parameters import stack_params, unstack_params

PyTree = Any
Parameter = jax.Array | nn.Partitioned
Metrics = Dict[str, Tuple[jax.Array, ...]]


def execute_pipeline_step(
    module: nn.Module,
    state: jax.Array,
    input: jax.Array,
    *args,
    model_axis_name: str,
    **kwargs,
) -> Tuple[jax.Array, jax.Array]:
    """Single micro-batch pipeline step.

    Args:
        module: Flax module representing the stage to execute.
        state: Last communicated features between stages. Used as input to the module for all stages except the first.
        input: Original micro-batch input to the pipeline stage. Used as input to the module for the first stage.
        *args: Additional arguments to the module.
        model_axis_name: Name of the model axis in the mesh/shard_map.
        **kwargs: Additional keyword arguments to the module.

    Returns:
        Tuple of the new state (after communication) and the output of the module.
    """
    num_stages = jax.lax.psum(1, model_axis_name)
    stage_index = jax.lax.axis_index(model_axis_name)
    # For the first stage, we use the microbatches as input.
    # For all other stages, we use the last state from the
    # previous stage as input.
    state = jnp.where(stage_index == 0, input, state)
    state = module(state, *args, **kwargs)
    # For the last stage, we return the state as output.
    # For all other stages, we return zeros.
    output = jnp.where(
        stage_index == num_stages - 1,
        state,
        jnp.zeros_like(state),
    )
    # Communicate the last state to the next stage.
    state = jax.lax.ppermute(
        state,
        model_axis_name,
        perm=[(i, (i + 1) % num_stages) for i in range(num_stages)],
    )
    return (state, output)


@jax.named_scope("pipeline")  # Naming scope for profiling.
def execute_pipeline(
    module: nn.Module, x: jax.Array, *args, num_microbatches: int, model_axis_name: str, **kwargs
) -> jax.Array:
    """Execute a pipeline of stages on a batch of data.

    Uses the principle of GPipe in splitting the batch into micro-batches
    and running the pipeline stages in parallel.

    Args:
        module: Flax module representing the pipeline stage to execute.
        x: Batch of input data, only needed on device of the first stage. Data will be split into micro-batches.
        *args: Additional arguments to the module.
        num_microbatches: Number of micro-batches to split the batch into.
        model_axis_name: Name of the model axis in the mesh/shard_map.
        **kwargs: Additional keyword arguments to the module.

    Returns:
        Output of the last stage of the pipeline. For devices that are not
        the last stage, the output is zeros.
    """
    num_stages = jax.lax.psum(1, model_axis_name)
    # Structure the input data into micro-batches.
    batch_size = x.shape[0]
    assert (
        batch_size % num_microbatches == 0
    ), f"Batch size {batch_size} must be divisible by number of microbatches {num_microbatches}"
    microbatch_size = batch_size // num_microbatches
    microbatches = jnp.reshape(x, (num_microbatches, microbatch_size, *x.shape[1:]))
    inputs = jnp.concatenate(  # Add zeros for unused computation blocks in first stage.
        [
            microbatches,
            jnp.zeros((num_stages - 1, *microbatches.shape[1:]), dtype=microbatches.dtype),
        ],
        axis=0,
    )
    state = jnp.zeros_like(microbatches[0])
    num_iterations = inputs.shape[0]
    # Run loop over pipeline steps.
    _, outputs = nn.scan(
        functools.partial(
            execute_pipeline_step,
            *args,
            model_axis_name=model_axis_name,
            **kwargs,
        ),
        variable_broadcast={"params": True},
        split_rngs={"params": False, "dropout": True},
        length=num_iterations,
        in_axes=0,
        out_axes=0,
    )(module, state, inputs)
    # Take last N outputs (first ones are zeros from unused computation blocks in last stage).
    outputs = jnp.concatenate(outputs[-num_microbatches:], axis=0)
    return outputs


class PipelineModule(nn.Module):
    model_axis_name: str
    num_microbatches: int
    module_fn: Callable[..., nn.Module]

    @nn.compact
    def __call__(self, *args, **kwargs):
        module = self.module_fn()
        return execute_pipeline(
            module,
            *args,
            **kwargs,
            num_microbatches=self.num_microbatches,
            model_axis_name=self.model_axis_name,
        )


class ModelParallelismWrapper(nn.Module):
    """Wrapper for adding model parallelism to a module.

    This wrapper adds sharding over the model axis to the parameters of the module
    and initializes the module with different parameters across the model axis.

    Args:
        model_axis_name: Name of the model axis to shard over.
        module_fn: Function that returns the Flax module to wrap.
        mask_except_model_idx: If not None, only the `mask_except_model_idx`-th shard will be non-zero.
        split_rngs: If True, split the random number generators across the model axis.
        module_kwargs: Additional keyword arguments to pass to the module function.
    """

    model_axis_name: str
    module_fn: Callable[..., nn.Module]
    mask_except_model_idx: int | None = None
    split_rngs: bool = True
    module_kwargs: FrozenDict[str, Any] = FrozenDict({})

    @nn.compact
    def __call__(self, *args, **kwargs):
        if self.is_initializing() and self.split_rngs:
            # Initialize each module across the model axis with different parameters.
            self.scope.rngs["params"] = self.scope.rngs["params"].replace(
                rng=fold_rng_over_axis(self.scope.rngs["params"].rng, self.model_axis_name)
            )
        # Wrap variables in nn.Partitioned objects to add sharding over the model axis.
        module = nn.map_variables(
            target=functools.partial(
                self.module_fn,
                name="sharded",
                **self.module_kwargs,
            ),
            trans_in_fn=functools.partial(unstack_params, axis_name=self.model_axis_name),
            trans_out_fn=functools.partial(
                stack_params,
                axis_name=self.model_axis_name,
                mask_except=self.mask_except_model_idx,
            ),
            mapped_collections="params",
            mutable=True,
        )()
        return module(
            *args,
            **kwargs,
        )


def scale_init(init_fn: Callable, scale_factor: float = 1.0):
    """Scales the output of the given init function by the given factor.

    Args:
        init_fn: The init function to scale.
        scale_factor: The factor to scale the output of the init function by.

    Returns:
        A new init function that scales the output of the given init function by the given factor.
    """

    def _init_fn(rng, *args, **kwargs):
        return scale_factor * init_fn(rng, *args, **kwargs)

    return _init_fn


class TPDense(nn.Module):
    """Dense layer with Tensor Parallelism support.

    This layer can be used to perform a dense layer with Tensor Parallelism support.

    Attributes:
        dense_fn: Constructor function of the dense layer to use. Needs to support the keyword argument `kernel_init`.
        model_axis_name: The name of the model axis.
        tp_mode: The Tensor Parallelism mode to use. Can be "scatter", "gather", or "none".
        skip_communication: Whether to skip communication in the Tensor Parallelism strategy. Useful for layers with custom communication or where input has been already gathered beforehand.
        kernel_init: The initializer to use for the kernel of the dense layer.
        kernel_init_adjustment: The adjustment factor to use for the kernel initializer.
        dense_name: The name of the dense layer module.
    """

    dense_fn: Any
    model_axis_name: str
    tp_mode: Literal["scatter", "gather", "none"] = "none"
    skip_communication: bool = False
    kernel_init: Callable = nn.initializers.lecun_normal()
    kernel_init_adjustment: float = 1.0
    dense_name: str = "module"

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        tp_size = jax.lax.psum(1, self.model_axis_name)
        tp_mode = self.tp_mode if tp_size > 1 else "none"
        # Wrap the dense layer in a ModelParallelismWrapper to shard the parameters.
        dense_fn = functools.partial(
            ModelParallelismWrapper,
            model_axis_name=self.model_axis_name,
            module_fn=functools.partial(
                self.dense_fn,
                kernel_init=scale_init(self.kernel_init, self.kernel_init_adjustment),
            ),
            name=self.dense_name,
        )

        if tp_mode == "none":
            # Vanilla dense layer.
            x = self.dense_fn(kernel_init=self.kernel_init)(x)
        elif tp_mode == "gather":
            # Gather strategy: communicate all the inputs to all the devices, then perform the dense layer.
            if not self.skip_communication:
                x = jax.lax.all_gather(x, self.model_axis_name, axis=-1, tiled=True)
            x = dense_fn()(x)
        elif tp_mode == "scatter":
            # Scatter strategy: perform the dense layer on each device, then communicate the outputs to all the devices.
            x = dense_fn()(x)
            if not self.skip_communication:
                x = jax.lax.psum_scatter(
                    x, axis_name=self.model_axis_name, scatter_dimension=x.ndim - 1, tiled=True
                )
        else:
            raise ValueError(f"Unknown Tensor Parallel mode: {tp_mode}")
        return x


def async_gather(x: PyTree, axis_name: str, shift_up: bool = True) -> List[PyTree]:
    """All gather using ring permutation.

    Args:
        x: The input to gather.
        axis_name: The axis name to gather along.
        shift_up: Whether to shift up (device 0 send to device 1) or down (device 1 send to device 0).

    Returns:
        List of gathered inputs.
    """
    tp_size = jax.lax.psum(1, axis_name)
    # Determine communication permutation.
    if shift_up:
        shift_perm = [(j, (j + 1) % tp_size) for j in range(tp_size)]
    else:
        shift_perm = [(j, (j - 1) % tp_size) for j in range(tp_size)]
    ps = [x]
    p = x
    # Perform all-gather using ring permutation.
    for _ in range(1, tp_size):
        p = jax.lax.ppermute(p, axis_name, perm=shift_perm)
        ps.append(p)
    return ps


def async_gather_bidirectional(
    x: jax.Array, axis_name: str, shift_up: bool = True
) -> List[jax.Array]:
    """All gather using ring permutation with bidirectional communication.

    Args:
        x: The input to gather.
        axis_name: The axis name to gather along.
        shift_up: Whether to return the order of tensors that complies with the unidrectional version of shift up (device 0 send to device 1) or down (device 1 send to device 0).

    Returns:
        List of gathered inputs.
    """
    tp_size = jax.lax.psum(1, axis_name)
    shift_up_perm = [(j, (j + 1) % tp_size) for j in range(tp_size)]
    shift_down_perm = [(j, (j - 1) % tp_size) for j in range(tp_size)]
    ps_up = []
    ps_down = []
    p_up = x
    p_down = x
    for i in range(1, tp_size):
        if i % 2 == 0:
            p_down = jax.lax.ppermute(p_down, axis_name=axis_name, perm=shift_down_perm)
            ps_down.append(p_down)
        else:
            p_up = jax.lax.ppermute(p_up, axis_name=axis_name, perm=shift_up_perm)
            ps_up.append(p_up)
    # Combine communication in both directions.
    # This list will have the same order as the unidirectional up version.
    if shift_up:
        ps = [x] + ps_up + ps_down[::-1]
    else:
        ps = [x] + ps_down + ps_up[::-1]
    return ps


def async_gather_split(x: jax.Array, axis_name: str) -> List[jax.Array]:
    """All gather using ring permutation with features split for bidirectional communication.

    Args:
        x: The input to gather.
        axis_name: The axis name to gather along.

    Returns:
        List of gathered inputs. Length is 2 * axis size - 1.
    """
    x1, x2 = jax.tree_map(lambda x: jnp.split(x, 2, axis=-1), x)
    return async_gather(x1, axis_name, shift_up=True) + async_gather(x2, axis_name, shift_up=False)


def async_scatter(xs: Sequence[PyTree], axis_name: str, shift_up: bool = True) -> PyTree:
    """Scatter sum using ring permutation.

    Args:
        xs: The inputs to scatter sum. The length of the list should match the size of the axis.
        axis_name: The axis name to scatter sum along.
        shift_up: Whether to shift up (device 0 send to device 1) or down (device 1 send to device 0).

    Returns:
        The scatter summed output.
    """
    tp_size = jax.lax.psum(1, axis_name)
    assert (
        len(xs) == tp_size
    ), f"Number of shards needs to match axis size, but got {len(xs)} with {axis_name} axis size {tp_size}."
    if shift_up:
        shift_perm = [(j, (j + 1) % tp_size) for j in range(tp_size)]
    else:
        shift_perm = [(j, (j - 1) % tp_size) for j in range(tp_size)]
    y = xs[0]
    for x in xs[1:]:
        y = jax.lax.ppermute(y, axis_name, perm=shift_perm)
        y = jax.tree_map(jnp.add, y, x)
    return y


def async_scatter_split(xs: Sequence[PyTree], axis_name: str) -> PyTree:
    """Scatter sum using ring permutation with features split for bidirectional communication.

    Args:
        xs: The inputs to scatter sum. The length of the list should match the size of the axis.
        axis_name: The axis name to scatter sum along.

    Returns:
        The scatter summed output.
    """

    def _split(x: PyTree) -> Tuple[PyTree, PyTree]:
        return (
            jax.tree_map(lambda x: x[..., : x.shape[-1] // 2], x),
            jax.tree_map(lambda x: x[..., x.shape[-1] // 2 :], x),
        )

    tp_size = jax.lax.psum(1, axis_name)
    assert (
        len(xs) == tp_size
    ), f"Number of shards needs to match axis size, but got {len(xs)} with {axis_name} axis size {tp_size}."
    shift_perm_up = [(j, (j + 1) % tp_size) for j in range(tp_size)]
    shift_perm_down = [(j, (j - 1) % tp_size) for j in range(tp_size)]
    y_up, y_down = _split(xs[0])
    for x in xs[1:]:
        y_up = jax.lax.ppermute(y_up, axis_name, perm=shift_perm_up)
        y_down = jax.lax.ppermute(y_down, axis_name, perm=shift_perm_down)
        x_up, x_down = _split(x)
        y_up = jax.tree_map(jnp.add, y_up, x_up)
        y_down = jax.tree_map(jnp.add, y_down, x_down)
    return jax.tree_map(lambda y1, y2: jnp.concatenate([y1, y2], axis=-1), y_up, y_down)


class TPAsyncDense(nn.Module):
    """Tensor-Parallel Dense Layer with Asynchronous Communication.

    This layer can be used to perform a dense layer with Tensor Parallelism support, and overlaps communication with computation whenever possible.

    Attributes:
        dense_fn: Constructor function of the dense layer to use. Needs to support the keyword argument `kernel_init`.
        model_axis_name: The name of the model axis.
        tp_mode: The Tensor Parallelism mode to use. Can be "scatter", "gather", or "none".
        kernel_init: The initializer to use for the kernel of the dense layer.
        kernel_init_adjustment: The adjustment factor to use for the kernel initializer.
        dense_name: The name of the dense layer module.
        use_bidirectional_gather: Whether to use bidirectional or unidirectional gather over the device ring for communication.
        use_bidirectional_scatter: Whether to use bidirectional or unidirectional scatter over the device ring for communication.
    """

    dense_fn: Any
    model_axis_name: str
    tp_mode: Literal["scatter", "gather", "none"] = "none"
    kernel_init: Callable = nn.initializers.lecun_normal()
    kernel_init_adjustment: float = 1.0
    dense_name: str = "module"
    use_bidirectional_gather: bool = True
    use_bidirectional_scatter: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        tp_size = jax.lax.psum(1, self.model_axis_name)
        tp_mode = self.tp_mode if tp_size > 1 else "none"

        dense_fn = functools.partial(
            ModelParallelismWrapper,
            model_axis_name=self.model_axis_name,
            module_fn=functools.partial(
                self.dense_fn,
                kernel_init=scale_init(self.kernel_init, self.kernel_init_adjustment),
            ),
            name=self.dense_name,
        )

        if tp_mode == "none":
            y = self.dense_fn(kernel_init=self.kernel_init, name="shard_0")(x)
        elif tp_mode == "gather":
            # Async gathering of all inputs.
            async_op = (
                async_gather_bidirectional if self.use_bidirectional_gather else async_gather
            )
            xs = async_op(x, axis_name=self.model_axis_name)
            # Compute output per input (scheduled as communication makes inputs available).
            ys = [
                dense_fn(
                    module_kwargs={
                        "use_bias": (i == 0)
                    },  # Only need a single per final output feature.
                    name=f"shard_{i}",
                )(x)
                for i, x in enumerate(xs)
            ]
            # Final sum of all outputs.
            y = jax.tree_map(lambda *args: sum(args), *ys)
        elif tp_mode == "scatter":
            # Calculate all outputs per device.
            ys = [
                dense_fn(
                    module_kwargs={
                        "use_bias": (i == 0)
                    },  # Only need a single per final output feature.
                    name=f"shard_{i}",
                )(x)
                for i in range(tp_size)
            ]
            # Async scatter sum of all outputs (communication already starts after first output is ready).
            async_op = async_scatter_split if self.use_bidirectional_scatter else async_scatter
            y = async_op(ys, axis_name=self.model_axis_name)
        else:
            raise ValueError(f"Unknown Tensor Parallel mode: {tp_mode}")
        return y
