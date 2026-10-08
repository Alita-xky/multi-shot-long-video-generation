"""Reconstruction backends return points and cameras. They do not read nominal poses."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from scene_memory.types import Reconstruction


class ReconstructionBackend(Protocol):
    name: str

    def reconstruct(self, image_paths: list[Path], work_dir: Path | None) -> Reconstruction:
        """Register one overlap group. Image order is not a camera trajectory."""
