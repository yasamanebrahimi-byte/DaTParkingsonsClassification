"""Small framework-independent optimizer scheduling helpers."""

from __future__ import annotations

import math


def gradient_accumulation_optimizer_steps(num_batches: int, accumulation_steps: int) -> int:
    """Return the number of optimizer updates in one epoch."""
    num_batches = int(num_batches)
    accumulation_steps = int(accumulation_steps)
    if num_batches < 0:
        raise ValueError("num_batches must be non-negative")
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be at least 1")
    return int(math.ceil(num_batches / accumulation_steps)) if num_batches else 0
