import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from absl import logging
from ml_collections import ConfigDict
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from jax_trainer.logger.enums import LogFreq, LogMetricMode, LogMode
from jax_trainer.logger.utils import build_tool_logger


class Logger:
    """Logger class to log metrics, images, etc.

    to Tensorboard or Weights and Biases.
    """

    def __init__(self, config: ConfigDict, full_config: ConfigDict):
        """Logger class to log metrics, images, etc. to Tensorboard or Weights and Biases.

        Args:
            config (ConfigDict): The logger config.
            full_config (ConfigDict): The full config of the trainer, to be logged.
        """
        self.config = config
        self.full_config = full_config
        self.logger = build_tool_logger(config, full_config)
        self.logging_mode = "train"
        self.step_metrics = self._get_new_metrics_dict()
        self.step_count = 0
        self.full_step_counter = 0
        self.log_steps_every = config.get("log_steps_every", 50)
        self.epoch_metrics = self._get_new_metrics_dict()
        self.epoch_idx = 0
        self.epoch_element_count = 0
        self.epoch_step_count = 0
        self.epoch_log_prefix = ""
        self.epoch_start_time = None

    def _get_new_metrics_dict(self):
        """
        Returns a default dict for metrics, with the following structure:
        {
            'METRIC_KEY': {
                'value': float | jnp.ndarray,
                'mode': LogMetricMode
            }
        }

        Returns:
            Dict[str, Dict[str, Any]]: The default dict for metrics.
        """
        return defaultdict(lambda: {"value": 0, "mode": "mean"})

    def log_metrics(self, metrics: Dict[str, Any], step: int, log_postfix: str = ""):
        """Logs a dict of metrics to the tool of choice (e.g. Tensorboard/Wandb).

        Args:
            metrics (Dict[str, Any]): The metrics that should be logged.
            step (int): The step at which the metrics should be logged.
            log_postfix (str, optional): Postfix to append to the log key. Defaults to "".
        """
        metrics_to_log = {}
        for metric_key in metrics:
            metric_value = metrics[metric_key]
            if isinstance(metric_value, jnp.ndarray):
                if metric_value.size == 1:
                    metric_value = metric_value.item()
                else:
                    continue
            save_key = metric_key
            if len(log_postfix) > 0:
                save_key += f"_{log_postfix}"
            metrics_to_log[save_key] = metric_value
        if len(metrics_to_log) > 0:
            self.logger.log_metrics(metrics_to_log, step)

    def log_scalar(
        self,
        metric_key: str,
        metric_value: Union[float, jnp.ndarray],
        step: int,
        log_postfix: str = "",
    ):
        """Logs a single scalar metric to the tool of choice (e.g. Tensorboard/Wandb).

        Args:
            metric_key (str): The key of the metric to log.
            metric_value (Union[float, jnp.ndarray]): The value of the metric to log.
            step (int): The step at which the metric should be logged.
            log_postfix (str, optional): Postfix to append to the log key. Defaults to "".
        """
        self.log_metrics({metric_key: metric_value}, step, log_postfix)

    def finalize(self, status: str):
        """Closes the logger.

        Args:
            status (str): The status of the training run (e.g. success, failure).
        """
        self.logger.finalize(status)

    def start_epoch(self, epoch: int, mode: str = "train"):
        """Starts a new epoch.

        Args:
            epoch (int): The index of the epoch.
            mode (str, optional): The logging mode. Should be in ["train", "val", "test"]. Defaults to "train".
        """
        assert mode in ["train", "val", "test"], f"Unknown logging mode {mode}."
        self.logging_mode = mode
        self.epoch_idx = epoch
        self._reset_epoch_metrics()

    def log_step(self, metrics: Dict[str, Any], element_count: int = 1):
        """Log metrics for a single step.

        Args:
            metrics (Dict[str, Any]): The metrics that should be logged in this step.
                If a metric is a dict, it should have the following structure:
                {
                    'value': float | jnp.ndarray,
                    'mode': LogMetricMode,
                    'log_freq': LogFreq,
                    'log_mode': LogMode
                }
                The value is the actual value to log for the given metric key.
                The mode determines how the metric is aggregated over the epoch.
                The log_freq determines whether the metric is logged in every step,
                at the end of the epoch, or both. The log_mode determines whether the
                metric is logged during training, validation and/or testing. 'eval'
                represents both validation and testing.
            element_count (int, optional): The number of elements that were processed in this step.
                Used for averaging metrics with batches of unequal size. Defaults to 1.
        """
        self.epoch_element_count += element_count
        self.epoch_step_count += 1
        for key in metrics:
            # Prepare input metric
            metric_in = metrics[key]
            if not isinstance(metric_in, dict):
                metric_in = {"value": metric_in}
            val = metric_in["value"]
            mode = metric_in.get("mode", LogMetricMode.MEAN)
            log_freq = metric_in.get("log_freq", LogFreq.ANY)
            log_mode = metric_in.get("log_mode", LogMode.ANY)
            # Check if metric should be logged
            if (
                (log_mode == LogMode.TRAIN and self.logging_mode != "train")
                or (log_mode == LogMode.VAL and self.logging_mode != "val")
                or (log_mode == LogMode.TEST and self.logging_mode != "test")
                or (log_mode == LogMode.EVAL and self.logging_mode == "train")
            ):
                continue
            # Log metric in epoch and/or step, if applicable
            for metrics_dict, dict_key in [
                (self.step_metrics, "step"),
                (self.epoch_metrics, "epoch"),
            ]:
                if dict_key == "step" and self.logging_mode != "train":
                    continue
                if (log_freq == LogFreq.EPOCH and dict_key != "epoch") or (
                    log_freq == LogFreq.STEP and dict_key != "step"
                ):
                    continue
                metrics_dict[key]["mode"] = mode
                if mode == LogMetricMode.MEAN:
                    metrics_dict[key]["value"] += val * (
                        element_count if dict_key == "epoch" else 1
                    )
                elif mode == LogMetricMode.SUM:
                    metrics_dict[key]["value"] += val
                elif mode == LogMetricMode.SINGLE:
                    metrics_dict[key]["value"] = val
                elif mode == LogMetricMode.MAX:
                    metrics_dict[key]["value"] = max(metrics_dict[key]["value"], val)
                elif mode == LogMetricMode.MIN:
                    metrics_dict[key]["value"] = min(metrics_dict[key]["value"], val)
                elif mode in [LogMetricMode.STD, LogMetricMode.CONCAT]:
                    if not isinstance(metrics_dict[key]["value"], list):
                        metrics_dict[key]["value"] = []
                    metrics_dict[key]["value"].append(val)
                else:
                    raise ValueError(f"Unknown logging mode {mode}.")
        # Log step metrics if applicable
        if self.logging_mode == "train" and self.log_steps_every > 0:
            self.step_count += 1
            self.full_step_counter += 1
            if self.step_count >= self.log_steps_every:
                if self.step_count > self.log_steps_every:
                    logging.warning(
                        f"Logging step count is {self.step_count} but should be {self.log_steps_every}."
                    )
                final_step_metrics = self._finalize_metrics(
                    metrics=self.step_metrics, element_count=self.step_count
                )
                self.log_metrics(
                    final_step_metrics, step=self.full_step_counter, log_postfix="step"
                )
                self._reset_step_metrics()

    def _reset_step_metrics(self):
        """Resets the metric dict and step count for the current step."""
        self.step_metrics = self._get_new_metrics_dict()
        self.step_count = 0

    def _reset_epoch_metrics(self):
        """Resets the metric dict and step count for the current epoch."""
        self.epoch_metrics = self._get_new_metrics_dict()
        self.epoch_element_count = 0
        self.epoch_step_count = 0
        self.epoch_start_time = time.time()

    def log_epoch_scalar(self, key: str, value: Union[float, int, jnp.ndarray]):
        """Logs a single scalar metric in the metric dict of the current epoch.

        Args:
            key (str): The key of the metric to log.
            value (Union[float, int, jnp.ndarray]): The value of the metric to log.
        """
        self.epoch_metrics[key]["value"] = value
        self.epoch_metrics[key]["mode"] = LogMetricMode.SINGLE

    def _finalize_metrics(self, metrics: Dict[str, Dict[str, Any]], element_count: int = 1):
        """Finalizes the metrics of the current epoch by aggregating them over the epoch,
        corresponding to their selected mode.

        Args:
            metrics (Dict[str, Dict[str, Any]]): The metrics to finalize.
            element_count (int, optional): The number of elements that were processed in this epoch.
        """
        final_metrics = {}
        for key in metrics:
            val = metrics[key]["value"]
            if metrics[key]["mode"] == LogMetricMode.MEAN:
                val /= element_count
            elif metrics[key]["mode"] == LogMetricMode.STD:
                val = jnp.std(jnp.array(val), axis=0)
            elif metrics[key]["mode"] == LogMetricMode.CONCAT:
                val = jnp.concatenate(val, axis=0)
            if isinstance(val, jnp.ndarray) and val.size == 1:
                val = val.item()

            if "/" not in key:
                save_key = f"{self.logging_mode}/{key}"
            else:
                save_key = key
            final_metrics[save_key] = val
        return final_metrics

    def end_epoch(self, save_metrics: bool = False):
        """Ends the current epoch and logs the epoch metrics.

        Args:
            save_metrics (bool, optional): Whether to save the metrics to a file. Defaults to False.
        """
        self.log_epoch_scalar("time", time.time() - self.epoch_start_time)
        final_epoch_metrics = self._finalize_metrics(
            metrics=self.epoch_metrics, element_count=self.epoch_element_count
        )
        self.log_metrics(
            final_epoch_metrics,
            step=self.epoch_idx,
            log_postfix="epoch" if self.logging_mode == "train" else "",
        )
        if save_metrics:
            self.save_metrics(
                filename=f"{self.logging_mode}_epoch_{self.epoch_idx:04d}",
                metrics=final_epoch_metrics,
            )
        if (
            self.logging_mode == "train"
            and self.log_steps_every > 0
            and self.epoch_step_count < self.log_steps_every
        ):
            logging.info(
                "Training epoch has fewer steps than the logging frequency. Resetting step metrics."
            )
            self._reset_step_metrics()
        self._reset_epoch_metrics()
        return final_epoch_metrics

    def save_metrics(self, filename: str, metrics: Dict[str, Any]):
        """Saves a dictionary of metrics to file. Can be used as a textual representation of the
        validation performance for checking in the terminal.

        Args:
          filename: Name of the metrics file without folders and postfix.
          metrics: A dictionary of metrics to save in the file.
        """
        metrics = {
            k: metrics[k] for k in metrics if isinstance(metrics[k], (int, float, str, bool))
        }
        with open(os.path.join(self.log_dir, f"metrics/{filename}.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    def log_image(self, key: str, image: jnp.ndarray, step: int = None, log_postfix: str = ""):
        """Logs an image to the tool of choice (e.g. Tensorboard/Wandb).

        Args:
          key: Name of the image.
          image: Image to log.
          step: Step to log the image at.
          log_postfix: Postfix to append to the log key.
        """
        if step is None:
            step = self.full_step_counter
        if isinstance(image, jnp.ndarray):
            image = jax.device_get(image)
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_image(
                tag=f"{self.logging_mode}/{key}{log_postfix}",
                img_tensor=image,
                global_step=step,
                dataformats="HWC",
            )
        elif isinstance(self.logger, WandbLogger):
            self.logger.log_image(
                key=f"{self.logging_mode}/{key}{log_postfix}", image=image, step=step
            )
        else:
            raise ValueError(f"Unknown logger {self.logger}.")

    def log_figure(self, key: str, figure: plt.Figure, step: int = None, log_postfix: str = ""):
        """Logs a matplotlib figure to the tool of choice (e.g. Tensorboard/Wandb).

        Args:
            key: Name of the figure.
            figure: Figure to log.
            step: Step to log the image at.
            log_postfix: Postfix to append to the log key.
        """
        if step is None:
            step = self.full_step_counter
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_figure(
                tag=f"{self.logging_mode}/{key}{log_postfix}", figure=figure, global_step=step
            )
        elif isinstance(self.logger, WandbLogger):
            self.logger.experiment.log(
                {f"{self.logging_mode}/{key}{log_postfix}": figure}, step=step
            )
        else:
            raise ValueError(f"Unknown logger {self.logger}.")

    @property
    def log_dir(self):
        """Returns the logging directory of the logger."""
        return self.logger.log_dir
