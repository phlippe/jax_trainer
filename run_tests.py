import os

from absl import logging
from absl.testing import absltest

from tests.test_datasets import TestBuildDatasets
from tests.test_optimizer import TestBuildOptimizer
from tests.test_trainer import TestBuildTrainer

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    logging.set_verbosity(logging.WARNING)
    absltest.main()
