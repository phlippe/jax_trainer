from ml_collections import ConfigDict

from datasets.examples import build_cifar10_datasets
from datasets.utils import DatasetModule


def build_dataset_module(dataset_config: ConfigDict) -> DatasetModule:
    """Builds the dataset module.

    Args:
      dataset_config: Configuration for the dataset.

    Returns:
      dataset_module: Dataset module.
    """
    name = dataset_config.name.lower()
    if name == "cifar10":
        return build_cifar10_datasets(dataset_config)
    else:
        raise ValueError(f"Unknown dataset {name}.")
