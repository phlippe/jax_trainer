import os

from absl import logging
from absl.testing import absltest

from tests.trainer.test_parallel_trainer import TestBuildParallelTrainer

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    logging.set_verbosity(logging.WARNING)
    absltest.main()
