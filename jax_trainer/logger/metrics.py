from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict, freeze, unfreeze

from jax_trainer.logger.enums import LogFreq, LogMetricMode, LogMode

# Immutable metrics for compilation.
ImmutableMetricElement = FrozenDict[
    str, jax.Array | int | float | LogMetricMode | LogFreq | LogMode
]
ImmutableMetrics = FrozenDict[str, ImmutableMetricElement]
# Mutable metrics for updating/editing.
MutableMetricElement = Dict[str, jax.Array | int | float | LogMetricMode | LogFreq | LogMode]
MutableMetrics = Dict[str, MutableMetricElement]
# Metrics forwarded per step.
StepMetrics = Dict[
    str,
    jax.Array
    | int
    | float
    | Dict[str, jax.Array | int | float | LogMetricMode | LogFreq | LogMode],
]
# Combined types.
MetricElement = ImmutableMetricElement | MutableMetricElement
Metrics = ImmutableMetrics | MutableMetrics
# Metrics on host (for logging).
HostMetricElement = float | int | np.ndarray
HostMetrics = Dict[str, HostMetricElement]


def update_metrics(
    global_metrics: Metrics | None,
    step_metrics: StepMetrics,
    train: bool,
    batch_size: int | jax.Array,
) -> ImmutableMetrics:
    """Update metrics with new values.

    Args:
        global_metrics: Global metrics to update. If None, a new dictionary is created.
        step_metrics: Metrics to update with.
        train: Whether the metrics are logged during training or evaluation.
        batch_size: Batch size of the current step.

    Returns:
        Updated global metrics.
    """
    if global_metrics is None:
        global_metrics = {}
    if isinstance(global_metrics, FrozenDict):
        global_metrics = unfreeze(global_metrics)
    for key in step_metrics:
        # Prepare input metric
        metric_in = step_metrics[key]
        if not isinstance(metric_in, dict):
            metric_in = {"value": metric_in}
        val = metric_in["value"]
        mode = metric_in.get("mode", LogMetricMode.MEAN)
        log_freq = metric_in.get("log_freq", LogFreq.ANY)
        log_mode = metric_in.get("log_mode", LogMode.ANY)
        count = metric_in.get("count", None)
        # Check if metric should be logged
        if (log_mode == LogMode.TRAIN and not train) or (
            log_mode not in [LogMode.TRAIN, LogMode.ANY] and train
        ):
            continue
        # Log metric in epoch and/or step, if applicable
        postfix = []
        if train:
            if log_freq in [LogFreq.ANY, LogFreq.STEP]:
                postfix.append((LogFreq.STEP, "step"))
            if log_freq in [LogFreq.ANY, LogFreq.EPOCH]:
                postfix.append((LogFreq.EPOCH, "epoch"))
        else:
            postfix.append((LogFreq.EPOCH, "epoch"))
        for sub_freq, p in postfix:
            global_metrics = _update_single_metric(
                global_metrics,
                f"{key}_{p}" if p else key,
                val,
                mode,
                sub_freq,
                log_mode,
                count,
                batch_size,
            )
    global_metrics = freeze(global_metrics)
    return global_metrics


def _update_single_metric(
    global_metrics: MutableMetrics,
    key: str,
    value: Any,
    mode: LogMetricMode,
    log_freq: LogFreq,
    log_mode: LogMode,
    count: Any,
    batch_size: int | jax.Array,
) -> MutableMetrics:
    """Update a single metric.

    Args:
        global_metrics: Global metrics to update.
        key: Key of the metric to update.
        value: Value of the metric to update.
        mode: Logging mode of the metric.
        log_freq: Logging frequency of the metric.
        log_mode: Logging mode of the metric.
        count: Count of the metric to update.
        batch_size: Batch size of the current step.

    Returns:
        Updated global metrics.
    """
    if key not in global_metrics:
        metrics_dict = {"value": 0.0, "count": 0}
    else:
        metrics_dict = global_metrics[key]
    metrics_dict["mode"] = mode
    metrics_dict["log_freq"] = log_freq
    metrics_dict["log_mode"] = log_mode
    if count is None:
        if mode == LogMetricMode.MEAN:
            count = batch_size
            value = value * batch_size
        else:
            count = 1
    metrics_dict["count"] += count
    if mode == LogMetricMode.MEAN:
        metrics_dict["value"] += value
    elif mode == LogMetricMode.SUM:
        metrics_dict["value"] += value
    elif mode == LogMetricMode.SINGLE:
        metrics_dict["value"] = value
    elif mode == LogMetricMode.MAX:
        metrics_dict["value"] = jnp.maximum(metrics_dict["value"], value)
    elif mode == LogMetricMode.MIN:
        metrics_dict["value"] = jnp.minimum(metrics_dict["value"], value)
    elif mode == LogMetricMode.STD:
        metrics_dict["value"] += value
        if "value2" not in metrics_dict:
            assert key not in global_metrics, (
                f"For metric {key} with logging mode {mode}, "
                "the second moment of the metric must be initialized "
                "if the metric is already logged."
            )
            metrics_dict["value2"] = 0.0
        metrics_dict["value2"] += value**2
    else:
        raise ValueError(f"Unknown logging mode {mode}.")
    global_metrics[key] = metrics_dict
    return global_metrics


def get_metrics(
    global_metrics: Metrics,
    log_freq: LogFreq = LogFreq.ANY,
    reset_metrics: bool = True,
) -> Tuple[ImmutableMetrics, HostMetrics]:
    """Calculates metrics to log from global metrics.

    Supports resetting the global metrics after logging. For example, if the global metrics
    are logged every epoch, the global metrics can be reset after obtaining the metrics to log
    such that the next epoch starts with empty metrics.

    Args:
        global_metrics: Global metrics to log.
        log_freq: Logging frequency of the metrics to log.
        reset_metrics: Whether to reset the metrics after logging.

    Returns:
        The updated global metrics if reset_metrics is True, otherwise the original global metrics.
        Additionally, the metrics to log on the host device are returned.
    """
    if isinstance(global_metrics, FrozenDict) and reset_metrics:
        global_metrics = unfreeze(global_metrics)
    host_metrics = jax.device_get(global_metrics)
    metrics = {}
    for key in host_metrics:
        if log_freq == LogFreq.ANY or log_freq == host_metrics[key]["log_freq"]:
            host_key = key.rsplit("_", 1)[0]  # Remove postfix of train/test.
            value = host_metrics[key]["value"]
            count = host_metrics[key]["count"]
            if host_metrics[key]["mode"] == LogMetricMode.MEAN:
                value = value / count
            elif host_metrics[key]["mode"] == LogMetricMode.STD:
                value = value / count
                value2 = host_metrics[key]["value2"] / count
                value = np.sqrt(value2 - value**2)
            metrics[host_key] = value
            if reset_metrics:
                global_metrics[key]["value"] = jnp.zeros_like(global_metrics[key]["value"])
                global_metrics[key]["count"] = jnp.zeros_like(global_metrics[key]["count"])
    if not isinstance(global_metrics, FrozenDict):
        global_metrics = freeze(global_metrics)
    return global_metrics, metrics
