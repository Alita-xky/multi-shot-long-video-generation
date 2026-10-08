"""Frame-level spatial memory bank (Context-as-Memory, inference-only)."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class MemoryFrame:
    clip_id: str
    frame_idx: int
    t_sec: float
    path: str
    camera_yaw: float
    camera_x: float
    camera_y: float


CLIP_PHASES = {
    "clip_01": (0.0, math.pi / 2),
    "clip_02": (math.pi / 2, math.pi),
    "clip_03": (math.pi, 2 * math.pi),
}


def interpolate_pose(clip_id: str, alpha: float, radius: float = 1.0) -> tuple[float, float, float]:
    start, end = CLIP_PHASES[clip_id]
    yaw = start + (end - start) * max(0.0, min(1.0, alpha))
    x = radius * math.sin(yaw)
    y = -radius * math.cos(yaw)
    return yaw, x, y


def probe_nb_frames(video_path: Path) -> int:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-of", "default=nw=1:nk=1", str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    if out.isdigit():
        return int(out)
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames", "-of", "default=nw=1:nk=1",
        str(video_path),
    ]
    return int(subprocess.check_output(cmd, text=True).strip())


class VisualEncoder:
    """CLIP image encoder with ResNet18 fallback. CPU-only so NPUs stay free."""

    def __init__(self):
        import torch

        self.torch = torch
        self.backend = "resnet18"
        self._cache: dict[str, np.ndarray] = {}
        self.model = None
        self.processor = None
        self.transform = None
        try:
            from transformers import CLIPModel, CLIPProcessor

            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.model.eval()
            self.backend = "clip-vit-b32"
        except Exception as exc:
            print(f"CLIP unavailable ({exc}); falling back to ResNet18", flush=True)
            import torchvision.models as models

            weights = models.ResNet18_Weights.DEFAULT
            backbone = models.resnet18(weights=weights)
            self.model = torch.nn.Sequential(*(list(backbone.children())[:-1]))
            self.model.eval()
            self.transform = weights.transforms()
            self.backend = "resnet18"

    def embed(self, path: Path) -> np.ndarray:
        key = str(Path(path).resolve())
        if key in self._cache:
            return self._cache[key]
        torch = self.torch
        img = Image.open(path).convert("RGB")
        with torch.inference_mode():
            if self.backend.startswith("clip"):
                inputs = self.processor(images=img, return_tensors="pt")
                out = self.model.get_image_features(**inputs)
                if torch.is_tensor(out):
                    feat = out
                elif getattr(out, "image_embeds", None) is not None:
                    feat = out.image_embeds
                elif getattr(out, "pooler_output", None) is not None:
                    feat = out.pooler_output
                else:
                    feat = out.last_hidden_state[:, 0]
                feat = feat.reshape(-1)
            else:
                feat = self.model(self.transform(img).unsqueeze(0)).flatten()
            feat = feat / feat.norm().clamp(min=1e-8)
        arr = feat.cpu().numpy()
        self._cache[key] = arr
        return arr


class MemoryBank:
    def __init__(self, root: Path, stride: int = 24, fps: float = 24.0):
        self.root = Path(root)
        self.frames_dir = self.root / "frames"
        self.vis_dir = self.root / "visualizations"
        self.index_path = self.root / "index.jsonl"
        self.stride = stride
        self.fps = fps
        self.frames: list[MemoryFrame] = []
        self.encoder = VisualEncoder()
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        if self.index_path.is_file():
            self._load()

    def _load(self) -> None:
        self.frames = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                self.frames.append(MemoryFrame(**json.loads(line)))

    def _save(self) -> None:
        with self.index_path.open("w", encoding="utf-8") as f:
            for fr in self.frames:
                f.write(json.dumps(asdict(fr)) + "\n")

    def add_clip(self, clip_id: str, video_path: Path) -> list[MemoryFrame]:
        video_path = Path(video_path).resolve()
        n = probe_nb_frames(video_path)
        if n < 1:
            raise RuntimeError(f"no frames in {video_path}")
        indices = sorted(set(list(range(0, n, self.stride)) + [0, n - 1]))
        added: list[MemoryFrame] = []
        clip_dir = self.frames_dir / clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        for idx in indices:
            out = clip_dir / f"f{idx:05d}.png"
            if not out.is_file():
                subprocess.check_call(
                    [
                        "ffmpeg", "-nostdin", "-v", "error", "-y",
                        "-i", str(video_path),
                        "-vf", f"select=eq(n\\,{idx})",
                        "-fps_mode", "vfr", "-frames:v", "1",
                        str(out),
                    ]
                )
            alpha = idx / max(1, n - 1)
            yaw, x, y = interpolate_pose(clip_id, alpha)
            added.append(
                MemoryFrame(
                    clip_id=clip_id,
                    frame_idx=idx,
                    t_sec=idx / self.fps,
                    path=str(out.resolve()),
                    camera_yaw=yaw,
                    camera_x=x,
                    camera_y=y,
                )
            )
        self.frames = [f for f in self.frames if f.clip_id != clip_id] + added
        self._save()
        return added

    def retrieve_memory(
        self,
        query_path: Path,
        k: int,
        exclude_paths: set[str] | None = None,
    ) -> list[tuple[MemoryFrame, float]]:
        """CLIP/DINO-style cosine retrieval vs the current continuation frame.

        Temporal sparsity: at most one frame per stride bin, and skip near-
        duplicates of the query so the bank is not just last-frame copies.
        """
        exclude_paths = {str(Path(p).resolve()) for p in (exclude_paths or set())}
        exclude_paths.add(str(Path(query_path).resolve()))
        q = self.encoder.embed(Path(query_path))
        scored: list[tuple[MemoryFrame, float]] = []
        for fr in self.frames:
            if str(Path(fr.path).resolve()) in exclude_paths:
                continue
            score = float(np.dot(q, self.encoder.embed(Path(fr.path))))
            if score >= 0.995:
                continue
            scored.append((fr, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        selected: list[tuple[MemoryFrame, float]] = []
        used_bins: set[tuple[str, int]] = set()
        for fr, score in scored:
            bin_key = (fr.clip_id, fr.frame_idx // self.stride)
            if bin_key in used_bins:
                continue
            used_bins.add(bin_key)
            selected.append((fr, score))
            if len(selected) >= k:
                break
        if len(selected) < k:
            for fr, score in scored:
                if any(fr.path == s[0].path for s in selected):
                    continue
                selected.append((fr, score))
                if len(selected) >= k:
                    break
        return selected[:k]

    def visualize_memory(
        self,
        query_path: Path,
        retrieved: list[tuple[MemoryFrame, float]],
        out_path: Path,
        title: str = "",
    ) -> Path:
        """Contact sheet: query + retrieved frames with clip/id/score labels."""
        tiles: list[tuple[str, Path]] = [("query", Path(query_path))]
        for fr, score in retrieved:
            label = f"{fr.clip_id} f{fr.frame_idx} {score:.3f}"
            tiles.append((label, Path(fr.path)))
        cell_w, cell_h = 320, 200
        cols = min(3, max(1, len(tiles)))
        rows = math.ceil(len(tiles) / cols)
        header = 36
        canvas = Image.new("RGB", (cols * cell_w, header + rows * (cell_h + 28)), (18, 18, 18))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        draw.text((8, 10), title or f"memory vis ({self.encoder.backend})", fill=(240, 240, 240), font=font)
        for i, (label, path) in enumerate(tiles):
            r, c = divmod(i, cols)
            x, y = c * cell_w, header + r * (cell_h + 28)
            im = Image.open(path).convert("RGB")
            im.thumbnail((cell_w - 8, cell_h - 8))
            ox = x + (cell_w - im.width) // 2
            oy = y + (cell_h - im.height) // 2
            canvas.paste(im, (ox, oy))
            draw.text((x + 6, y + cell_h - 4), label[:42], fill=(220, 220, 220), font=font)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)
        return out_path
