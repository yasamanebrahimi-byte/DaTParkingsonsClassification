"""Reproducibility controls."""

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def derive_seed(base_seed: int, *components: int) -> int:
    """Derive a stable non-negative seed without relying on Python's hash randomization."""

    value = int(base_seed) & 0x7FFFFFFF
    for component in components:
        value = (value * 1_000_003 + int(component) * 97_409 + 17) & 0x7FFFFFFF
    return value
