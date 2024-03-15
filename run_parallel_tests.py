import os

from absl import logging
from absl.testing import absltest

from tests.trainer.test_parallel_trainer import TestBuildParallelTrainer

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    flags = os.environ.get("XLA_FLAGS", "")
    flags += " --xla_force_host_platform_device_count=8"  # Simulate 8 devices
    os.environ["XLA_FLAGS"] = flags
    logging.set_verbosity(logging.WARNING)
    absltest.main()
