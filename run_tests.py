import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from absl.testing import absltest
from absl import logging
from tests.test_optimizer import *
from tests.test_datasets import *
from tests.test_trainer import *

if __name__ == '__main__':
    logging.set_verbosity(logging.WARNING)
    absltest.main()