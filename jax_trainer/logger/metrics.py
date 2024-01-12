from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from flax.core import FrozenDict, freeze, unfreeze

from jax_trainer.logger.enums import LogFreq, LogMetricMode, LogMode


def update_metrics(
    global_metrics: FrozenDict | Dict[str, Any] | None,
    step_metrics: Dict[str, Any],
    train: bool,
    batch_size: int | jax.Array,
) -> Dict[str, Any]:
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
            postfix.append((LogFreq.EPOCH, ""))
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
    global_metrics: Dict[str, Any],
    key: str,
    value: Any,
    mode: LogMetricMode,
    log_freq: LogFreq,
    log_mode: LogMode,
    count: Any,
    batch_size: int | jax.Array,
) -> Dict[str, Any]:
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
    global_metrics: FrozenDict | Dict[str, Any],
    log_freq: LogFreq = LogFreq.ANY,
    reset_metrics: bool = True,
) -> Tuple[FrozenDict, Dict[str, Any]]:
    if isinstance(global_metrics, FrozenDict) and reset_metrics:
        global_metrics = unfreeze(global_metrics)
    host_metrics = jax.device_get(global_metrics)
    metrics = {}
    for key in host_metrics:
        if log_freq == LogFreq.ANY or log_freq == host_metrics[key]["log_freq"]:
            value = host_metrics[key]["value"]
            count = host_metrics[key]["count"]
            if host_metrics[key]["mode"] == LogMetricMode.MEAN:
                value = value / count
            elif host_metrics[key]["mode"] == LogMetricMode.STD:
                value = value / count
                value2 = host_metrics[key]["value2"] / count
                value = jnp.sqrt(value2 - value**2)
            metrics[key] = value
            if reset_metrics:
                global_metrics[key]["value"] = jnp.zeros_like(global_metrics[key]["value"])
                global_metrics[key]["count"] = jnp.zeros_like(global_metrics[key]["count"])
    if not isinstance(global_metrics, FrozenDict):
        global_metrics = freeze(global_metrics)
    return global_metrics, metrics
