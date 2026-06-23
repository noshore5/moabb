"""Reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np
import torch

from .configs import SeedConfig


def set_reproducible_seed(cfg: SeedConfig) -> None:
    """Set Python, NumPy, CPU, and CUDA seeds before module construction."""

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    torch.backends.cudnn.benchmark = bool(cfg.benchmark)
    torch.backends.cudnn.deterministic = bool(cfg.deterministic)
    torch.use_deterministic_algorithms(bool(cfg.deterministic), warn_only=True)
