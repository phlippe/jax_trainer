import pathlib

import yaml
from ml_collections import config_dict


def get_config():
    # Load the yaml config file as a dict and use it to build a ConfigDict
    default_cfg = yaml.safe_load(pathlib.Path("cfg/default_config.yaml").read_text())
    default_cfg = config_dict.ConfigDict(default_cfg)
    return default_cfg