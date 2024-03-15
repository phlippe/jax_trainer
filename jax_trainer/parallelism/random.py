import jax

PRNGKey = jax.Array


def fold_rng_over_axis(rng: PRNGKey, axis_name: str) -> PRNGKey:
    """Folds the random number generator over the given axis.

    This is useful for generating a different random number for each device
    across a certain axis (e.g. the model axis).

    Args:
        rng: The random number generator.
        axis_name: The axis name to fold the random number generator over.

    Returns:
        A new random number generator, different for each device index along the axis.
    """
    axis_index = jax.lax.axis_index(axis_name)
    return jax.random.fold_in(rng, axis_index)


def fold_rng_over_processes(rng: PRNGKey) -> PRNGKey:
    return jax.random.fold_in(rng, jax.process_index())
