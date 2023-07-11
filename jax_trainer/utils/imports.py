import importlib
import sys
from typing import Any

from absl import logging


def resolve_import_from_string(import_string: str) -> Any:
    """Resolves an import from a string."""
    if "." in import_string:
        module_path, class_name = import_string.rsplit(".", 1)
        module = importlib.import_module(module_path)
        resolved_class = getattr(module, class_name)
    else:
        resolved_class = getattr(sys.modules[__name__], import_string)
    return resolved_class
