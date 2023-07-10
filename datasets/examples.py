import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from ml_collections import ConfigDict
from torchvision.datasets import CIFAR10

from datasets.utils import (
    DatasetModule,
    SupervisedBatch,
    build_batch_collate,
    build_data_loaders,
    image_to_numpy,
    normalize_transform,
)


def build_cifar10_datasets(dataset_config: ConfigDict):
    """Builds CIFAR10 datasets.

    Args:
      dataset_config: Configuration for the dataset.

    Returns:
      train_dataset: CIFAR10 training dataset.
      test_dataset: CIFAR10 test dataset.
    """
    transform = transforms.Compose(
        [
            image_to_numpy,
            normalize_transform(
                mean=np.array([0.4914, 0.4822, 0.4465]), std=np.array([0.2023, 0.1994, 0.2010])
            ),
        ]
    )
    # Loading the training/validation set
    train_dataset = CIFAR10(
        root=dataset_config.data_dir, train=True, transform=transform, download=True
    )
    train_set, val_set = data.random_split(
        train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42)
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
        collate_fn=build_batch_collate(
            SupervisedBatch
        ),  # lambda s, i, t: SupervisedBatch(size=s, input=i, target=t)),
        config=dataset_config,
    )

    return DatasetModule(
        dataset_config, train_set, val_set, test_set, train_loader, val_loader, test_loader
    )
