"""Explicit room placement. Edges are given, not estimated from non-overlapping photos."""

from __future__ import annotations

import numpy as np


def eye4() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def as_T(value) -> np.ndarray | None:
    if value is None:
        return None
    T = np.asarray(value, dtype=np.float64).reshape(4, 4)
    return T


def T_to_json(T: np.ndarray | None):
    if T is None:
        return None
    return np.asarray(T, dtype=np.float64).reshape(4, 4).tolist()
