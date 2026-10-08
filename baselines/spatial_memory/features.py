"""Global image features for GEN3C-lite spatial memory.

Note:
Features label pseudo-geometry from nominal SE3 + monocular depth.
They are not a learned 3D encoder. Lazy-load DINOv2-small; fall back to
the existing CLIP extractor in memory_bank.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


ImageLike = Union[Image.Image, np.ndarray, str, Path]
DINO_ID = "facebook/dinov2-small"


def _to_pil(image: ImageLike) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr[..., :3], mode="RGB")


class FeatureExtractor:
    def __init__(self):
        self.backend: str | None = None
        self._model = None
        self._processor = None
        self._clip = None

    def _load_dino(self) -> bool:
        try:
            from transformers import AutoImageProcessor, AutoModel
            import torch

            print(f"[Feature] Loading {DINO_ID}", flush=True)
            self._processor = AutoImageProcessor.from_pretrained(DINO_ID)
            self._model = AutoModel.from_pretrained(DINO_ID)
            self._model.eval()
            self.torch = torch
            self.backend = "dinov2-small"
            print("[Feature] Using dinov2-small backend", flush=True)
            return True
        except Exception as exc:
            print(f"[Feature] DINOv2 unavailable ({exc!r})", flush=True)
            return False

    def _load_clip(self) -> None:
        import sys

        pkg = Path(__file__).resolve().parents[1]
        if str(pkg) not in sys.path:
            sys.path.insert(0, str(pkg))
        from memory_bank import VisualEncoder

        self._clip = VisualEncoder()
        self.backend = f"clip-fallback:{self._clip.backend}"
        print(f"[Feature] Using CLIP fallback ({self._clip.backend})", flush=True)

    def _ensure(self) -> None:
        if self.backend is not None:
            return
        if not self._load_dino():
            self._load_clip()

    def extract(self, image: ImageLike) -> np.ndarray:
        """Global L2-normalized feature vector."""
        self._ensure()
        if self.backend and self.backend.startswith("dinov2"):
            torch = self.torch
            pil = _to_pil(image)
            inputs = self._processor(images=pil, return_tensors="pt")
            with torch.inference_mode():
                out = self._model(**inputs)
                feat = out.last_hidden_state[:, 0].reshape(-1)
                feat = feat / feat.norm().clamp(min=1e-8)
            return feat.cpu().numpy().astype(np.float32)
        path = image if isinstance(image, (str, Path)) else None
        if path is None:
            tmp = Path("/tmp/_feat_query.png")
            _to_pil(image).save(tmp)
            path = tmp
        return self._clip.embed(Path(path)).astype(np.float32)
