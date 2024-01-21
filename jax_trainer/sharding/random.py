import jax

PRNGKey = jax.Array


def fold_rng_over_axis(rng: PRNGKey, axis_name: str | None) -> PRNGKey:
    if axis_name is None:
        return rng
    else:
        return jax.random.fold_in(rng, jax.lax.axis_index(axis_name))


def fold_rng_over_processes(rng: PRNGKey) -> PRNGKey:
    return jax.random.fold_in(rng, jax.process_index())
