import os

from absl import logging
from absl.testing import absltest

# from tests.datasets.test_datasets import TestBuildDatasets
# from tests.logger.test_logger import TestLogger
# from tests.optimizer.test_optimizer import TestBuildOptimizer
from tests.trainer.test_trainer import TestBuildTrainer

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    logging.set_verbosity(logging.WARNING)
    absltest.main()
