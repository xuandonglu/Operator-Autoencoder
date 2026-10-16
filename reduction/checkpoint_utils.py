"""Load legacy reduction checkpoints without depending on old source folders."""

import importlib
import pickle
from pathlib import Path
from types import ModuleType

import torch


def load_reduction_checkpoint(path, case, method, device=None):
    """Resolve the historical ``nn`` and ``__main__`` model class names."""
    if case not in {"wave", "kdv", "ns", "cylinder"}:
        raise ValueError(f"Unknown case: {case}")
    if method not in {"AE", "OAE", "FOAE"}:
        raise ValueError(f"Unknown method: {method}")
    module_name = f"nn_ae_{case}" if method == "AE" else f"nn_{case}"
    package = f"{__package__}." if __package__ else ""
    definitions = importlib.import_module(package + module_name)

    class LegacyUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module in {"nn", "__main__"} and hasattr(definitions, name):
                return getattr(definitions, name)
            return super().find_class(module, name)

    legacy_pickle = ModuleType("legacy_pickle")
    legacy_pickle.__dict__.update(pickle.__dict__)
    legacy_pickle.Unpickler = LegacyUnpickler
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(
        Path(path), map_location=device, pickle_module=legacy_pickle, weights_only=False
    )
    return model.to(device)
