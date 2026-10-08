"""Feed-forward reconstructors are named, not runnable, until weights and a device exist."""

from __future__ import annotations

from pathlib import Path

from scene_memory.types import Reconstruction


class UnavailableBackend:
    def __init__(self, name: str, reason: str):
        self.name = name
        self.reason = reason

    def reconstruct(self, image_paths: list[Path], work_dir: Path | None) -> Reconstruction:
        raise RuntimeError(
            f"{self.name} is not a runnable backend here ({self.reason}). Use ColmapBackend."
        )


VGGT = UnavailableBackend("vggt", "no CUDA build and no weights in this workspace")
DUST3R = UnavailableBackend("dust3r", "no CUDA build and no weights in this workspace")
MAST3R = UnavailableBackend("mast3r", "no CUDA build and no weights in this workspace")
WORLDMIRROR = UnavailableBackend(
    "worldmirror",
    "the LightX2V example targets CUDA and is not the first backend",
)
