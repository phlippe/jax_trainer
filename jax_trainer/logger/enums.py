from enum import IntEnum


class LogMetricMode(IntEnum):
    """Enum to determine how metrics are aggregated over the epoch.

    MEAN: The metric is averaged over the epoch.
    SUM: The metric is summed over the epoch.
    SINGLE: The metric is logged as a single value.
    MAX: The maximum value of the metric is logged.
    MIN: The minimum value of the metric is logged.
    STD: The standard deviation of the metric is logged.
    """

    MEAN = 1
    SUM = 2
    SINGLE = 3
    MAX = 4
    MIN = 5
    STD = 6


class LogMode(IntEnum):
    """Enum to determine when metrics are logged.

    ANY: The metric is logged in any logging mode.
    TRAIN: The metric is logged during training.
    VAL: The metric is logged during validation.
    TEST: The metric is logged during testing.
    EVAL: The metric is logged during both validation and testing.
    """

    ANY = 0
    TRAIN = 1
    VAL = 2
    TEST = 3
    EVAL = 4


class LogFreq(IntEnum):
    """Enum to determine how often metrics are logged.

    ANY: The metric is logged in any logging frequency.
    STEP: The metric is logged at the end of every N steps.
    EPOCH: The metric is logged at the end of every epoch.
    """

    ANY = 0
    STEP = 1
    EPOCH = 2
