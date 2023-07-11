from ml_collections import ConfigDict

from jax_trainer.datasets.data_struct import DatasetModule
from jax_trainer.utils import resolve_import_from_string


def build_dataset_module(dataset_config: ConfigDict) -> DatasetModule:
    """Builds the dataset module.

    Args:
      dataset_config: Configuration for the dataset.

    Returns:
      dataset_module: Dataset module.
    """
    constructor = dataset_config.constructor.lower()
    constructor = resolve_import_from_string(constructor)
    dataset_module = constructor(dataset_config)
    return dataset_module
