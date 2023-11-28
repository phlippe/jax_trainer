from typing import Any, Dict

from flax.core import FrozenDict


def flatten_dict(d: Dict) -> Dict:
    """Flattens a nested dictionary."""
    flat_dict = {}
    for k, v in d.items():
        if isinstance(v, (dict, FrozenDict)):
            flat_dict.update({f"{k}.{k2}": v2 for k2, v2 in flatten_dict(v).items()})
        else:
            flat_dict[k] = v
    return flat_dict
