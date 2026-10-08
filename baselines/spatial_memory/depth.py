"""Monocular depth for GEN3C-lite spatial memory.

Note:
H3 has no real camera extrinsics. This module only produces a monocular
depth estimate. Downstream camera_pose comes from CameraCtrl waypoint
interpolation (nominal SE3). Any later point cloud is trajectory-aligned
pseudo geometry, not COLMAP/NeRF reconstruction.

Never fails closed: Depth Anything V2 if available, else mock depth.
Does not load the model at import time.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


ImageLike = Union[Image.Image, np.ndarray, str, Path]

DA_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def _to_pil(image: ImageLike) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr[..., :3], mode="RGB")


def _pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "npu") and torch.npu.is_available():
            # HF depth pipeline is not validated on Ascend; stay on CPU.
            return "cpu"
    except Exception:
        pass
    return "cpu"


def mock_depth(image: ImageLike) -> np.ndarray:
    """Smooth pseudo-depth so back-projection can be tested without a model.

    Near is high, far is low, in [0, 1]. Not a constant map.
    """
    rgb = np.asarray(_to_pil(image), dtype=np.float32) / 255.0
    h, w = rgb.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = xs / max(w - 1, 1)
    yn = ys / max(h - 1, 1)
    # Horizontal gradient: left closer than right, plus a mild vertical bowl.
    base = 0.72 - 0.35 * xn - 0.12 * ((yn - 0.5) ** 2)
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    # Brighter regions slightly closer (interior lighting bias).
    base = base + 0.18 * luma
    # Low-frequency noise so regions differ even on flat walls.
    noise = (
        0.06 * np.sin(2 * np.pi * xn * 3.0) * np.cos(2 * np.pi * yn * 2.0)
        + 0.03 * np.sin(2 * np.pi * (xn + yn) * 5.0)
    )
    depth = np.clip(base + noise, 0.0, 1.0).astype(np.float32)
    return depth


def save_depth(depth: np.ndarray, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = np.asarray(depth, dtype=np.float32)
    d = np.clip(d, 0.0, 1.0)
    vis = (d * 255.0).astype(np.uint8)
    Image.fromarray(vis, mode="L").save(path)
    return path


class DepthEstimator:
    """image -> HxW float32 depth. Backend is auto-detected on first use."""

    def __init__(self):
        self.backend: str | None = None
        self.device: str | None = None
        self._pipe = None
        self._last_infer_s: float | None = None

    def _try_load_depth_anything(self) -> bool:
        device = _pick_device()
        try:
            from transformers import pipeline

            print(
                f"[Depth] Loading Depth Anything V2 small on {device}: {DA_MODEL_ID}",
                flush=True,
            )
            self._pipe = pipeline(
                task="depth-estimation",
                model=DA_MODEL_ID,
                device=0 if device == "cuda" else -1,
            )
            self.device = device
            self.backend = "depth_anything_v2"
            print("[Depth] Using depth_anything_v2 backend", flush=True)
            return True
        except Exception as exc:
            print(f"[Depth] Depth Anything V2 unavailable ({exc!r})", flush=True)
            self._pipe = None
            return False

    def _ensure_backend(self) -> None:
        if self.backend is not None:
            return
        if self._try_load_depth_anything():
            return
        self.backend = "mock"
        self.device = "cpu"
        print("[Depth] Using mock depth backend", flush=True)

    def estimate_depth_raw(self, image: ImageLike) -> np.ndarray:
        """Raw monocular disparity (larger = nearer). No per-frame [0,1] stretch."""
        self._ensure_backend()
        pil = _to_pil(image)
        if self.backend == "depth_anything_v2" and self._pipe is not None:
            try:
                out = self._pipe(pil)
                raw = out["depth"]
                arr = np.asarray(raw, dtype=np.float32)
                if arr.shape[:2] != (pil.height, pil.width):
                    arr = np.asarray(
                        Image.fromarray(arr).resize(pil.size, Image.BILINEAR),
                        dtype=np.float32,
                    )
                return np.ascontiguousarray(arr, dtype=np.float32)
            except Exception as exc:
                print(f"[Depth] Raw inference failed ({exc!r}); mock", flush=True)
                self.backend = "mock"
                self._pipe = None
        return mock_depth(pil)

    def estimate_depth(self, image: ImageLike) -> np.ndarray:
        """Return HxW float32 depth in [0, 1], near=high, far=low."""
        self._ensure_backend()
        pil = _to_pil(image)
        t0 = time.perf_counter()
        if self.backend == "depth_anything_v2" and self._pipe is not None:
            try:
                out = self._pipe(pil)
                raw = out["depth"]
                if isinstance(raw, Image.Image):
                    arr = np.asarray(raw, dtype=np.float32)
                else:
                    arr = np.asarray(raw, dtype=np.float32)
                if arr.max() > arr.min():
                    arr = (arr - arr.min()) / (arr.max() - arr.min())
                else:
                    arr = np.zeros_like(arr, dtype=np.float32)
                # HF Depth Anything: larger = farther. Flip so near=high.
                depth = (1.0 - arr).astype(np.float32)
                if depth.shape[:2] != (pil.height, pil.width):
                    depth = np.asarray(
                        Image.fromarray((depth * 255).astype(np.uint8), mode="L").resize(
                            pil.size, Image.BILINEAR
                        ),
                        dtype=np.float32,
                    ) / 255.0
            except Exception as exc:
                print(
                    f"[Depth] Inference failed ({exc!r}); falling back to mock",
                    flush=True,
                )
                self.backend = "mock"
                self._pipe = None
                depth = mock_depth(pil)
        else:
            depth = mock_depth(pil)
        self._last_infer_s = time.perf_counter() - t0
        return np.ascontiguousarray(depth, dtype=np.float32)
