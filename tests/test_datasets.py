from datasets import build_dataset_module, DatasetModule
import optax
from ml_collections import ConfigDict
from absl.testing import absltest


class TestBuildDatasets(absltest.TestCase):
    # Test if constructing various optimizers work

    def test_build_cifar10(self):
        dataset_config = {
            'name': 'cifar10',
            'batch_size': 128,
            'num_workers': 4,
            'data_dir': 'data/'
        }
        dataset_config = ConfigDict(dataset_config)
        dataset_module = build_dataset_module(dataset_config)
        self.assertTrue(isinstance(dataset_module, DatasetModule))

        for loaders in [dataset_module.train_loader, dataset_module.val_loader, dataset_module.test_loader]:
            batch = next(iter(loaders))
            self.assertEqual(batch.size, 128)
            self.assertEqual(batch.input.shape, (128, 32, 32, 3))
            self.assertEqual(batch.target.shape, (128,))
            self.assertEqual(batch.input.dtype, 'float32')
            self.assertEqual(batch.target.dtype, 'int64')