import pandas as pd
from ml_collections import config_dict


def flatten_configdict(
    cfg: config_dict.ConfigDict,
    separation_mark: str = ".",
):
    """Returns a nested OmecaConf dict as a flattened dict, merged with the separation mark.
    Example:
        With separation_mark == '.', {'data':{'this': 1, 'that': 2} is returned as {'data.this': 1, 'data.that': 2}.
    """
    cfgdict = dict(cfg)
    cfgdict = pd.json_normalize(cfgdict, sep=separation_mark)
    return cfgdict.to_dict(orient="records")[0]