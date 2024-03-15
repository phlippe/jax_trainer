import jax
from flax.core import FrozenDict, unfreeze

from jax_trainer.logger import StepMetrics


def sync_step_metrics(metrics: StepMetrics, axis_name: str) -> StepMetrics:
    """Sync metrics across devices.

    Args:
        metrics: Metrics to sync.
        axis_name: Axis name to sync across.

    Returns:
        Synced metrics.
    """
    for key in metrics:
        metric = metrics[key]
        if isinstance(metric, dict):
            for metric_key in ["value", "count"]:
                if metric_key in metric:
                    metric[metric_key] = jax.lax.psum(metric[metric_key], axis_name)
        else:
            metrics[key] = jax.lax.psum(metric, axis_name)
    return metrics
