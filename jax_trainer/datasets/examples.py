import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from ml_collections import ConfigDict
from torchvision.datasets import CIFAR10, MNIST

from jax_trainer.datasets.collate import build_batch_collate
from jax_trainer.datasets.data_struct import DatasetModule, SupervisedBatch
from jax_trainer.datasets.transforms import image_to_numpy, normalize_transform
from jax_trainer.datasets.utils import build_data_loaders


def build_cifar10_datasets(dataset_config: ConfigDict):
    """Builds CIFAR10 datasets.

    Args:
      dataset_config: Configuration for the dataset.

    Returns:
      train_dataset: CIFAR10 training dataset.
      test_dataset: CIFAR10 test dataset.
    """
    normalize = dataset_config.get("normalize", True)
    transform = transforms.Compose(
        [
            image_to_numpy,
            normalize_transform(
                mean=np.array([0.4914, 0.4822, 0.4465]), std=np.array([0.2023, 0.1994, 0.2010])
            )
            if normalize
            else transforms.Lambda(lambda x: x),
        ]
    )
    # Loading the training/validation set
    train_dataset = CIFAR10(
        root=dataset_config.data_dir, train=True, transform=transform, download=True
    )
    val_size = dataset_config.get("val_size", 5000)
    split_seed = dataset_config.get("split_seed", 42)
    train_set, val_set = data.random_split(
        train_dataset,
        [50000 - val_size, val_size],
        generator=torch.Generator().manual_seed(split_seed),
    )
    # Loading the test set
    test_set = CIFAR10(
        root=dataset_config.data_dir, train=False, transform=transform, download=True
    )

    train_loader, val_loader, test_loader = build_data_loaders(
        train_set,
        val_set,
        test_set,
        train=[True, False, False],
        collate_fn=build_batch_collate(SupervisedBatch),
        config=dataset_config,
    )

    return DatasetModule(
        dataset_config, train_set, val_set, test_set, train_loader, val_loader, test_loader
    )


def build_mnist_datasets(dataset_config: ConfigDict):
    """Builds MNIST datasets.

    Args:
      dataset_config: Configuration for the dataset.
    """
    normalize = dataset_config.get("normalize", True)
    transform = transforms.Compose(
        [
            image_to_numpy,
            normalize_transform(mean=np.array([0.1307]), std=np.array([0.3081]))
            if normalize
            else transforms.Lambda(lambda x: x),
        ]
    )

    # Loading the training/validation set
    train_dataset = MNIST(
        root=dataset_config.data_dir, train=True, transform=transform, download=True
    )
    val_size = dataset_config.get("val_size", 5000)
    split_seed = dataset_config.get("split_seed", 42)
    train_set, val_set = data.random_split(
        train_dataset,
        [60000 - val_size, val_size],
        generator=torch.Generator().manual_seed(split_seed),
    )
    # Loading the test set
    test_set = MNIST(root=dataset_config.data_dir, train=False, transform=transform, download=True)

    train_loader, val_loader, test_loader = build_data_loaders(
        train_set,
        val_set,
        test_set,
        train=[True, False, False],
        collate_fn=build_batch_collate(SupervisedBatch),
        config=dataset_config,
    )

    return DatasetModule(
        dataset_config, train_set, val_set, test_set, train_loader, val_loader, test_loader
    )
