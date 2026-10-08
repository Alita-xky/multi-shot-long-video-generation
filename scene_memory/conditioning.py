"""Hand a rendered memory view to an existing generator as one more image reference.

This does not load a video model, does not change weights, and does not add a pose encoder.
The mask is written beside the render so holes stay inspectable. It is not a reference input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from scene_memory.render import MemoryView


MAX_REFERENCES = 9
MEMORY_NOTE = (
    "<Picture 2> is a render of the persistent 3D scene memory from the requested viewpoint. "
    "Black pixels are unobserved. They are not a new room, and they must not be filled by inventing layout."
)


def write_memory_view(memory, camera_pose: dict, out_dir: Path | str) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    view = memory.query(camera_pose)
    rgb_path = out_dir / "memory_view.png"
    mask_path = out_dir / "memory_mask.png"
    _save_view(view, rgb_path, mask_path)
    return {
        "rgb": str(rgb_path),
        "mask": str(mask_path),
        "valid_pixels": int(view.valid_pixels),
        "changes_weights": False,
    }


def generation_condition(prompt: str, reference_image, memory_view=None, mask_path=None, extra_references=None) -> dict:
    text = prompt if memory_view is None else prompt.rstrip() + "\n\n" + MEMORY_NOTE
    references = [("image", str(reference_image))]
    if memory_view is not None:
        references.append(("image", str(memory_view)))
    for item in extra_references or []:
        references.append((str(item[0]), str(item[1])))
    if len(references) > MAX_REFERENCES:
        raise ValueError(f"Ref2VA image cap is {MAX_REFERENCES}, got {len(references)}")
    argv = ["--prompt", text]
    for kind, path in references:
        argv.extend(["--reference", f"{kind}:{path}"])
    return {
        "prompt": text,
        "references": references,
        "argv": argv,
        "uses_memory_view": memory_view is not None,
        "changes_weights": False,
        "mask_path": None if mask_path is None else str(mask_path),
    }


def ablation(prompt: str, reference_image, memory_view, mask_path=None) -> dict:
    return {
        "baseline": generation_condition(prompt, reference_image, None),
        "proposed": generation_condition(prompt, reference_image, memory_view, mask_path=mask_path),
    }


def _save_view(view: MemoryView, rgb_path: Path, mask_path: Path) -> None:
    rgb = np.asarray(view.rgb, dtype=np.uint8)
    mask = (np.asarray(view.mask, dtype=np.uint8) * 255)
    Image.fromarray(rgb).save(rgb_path)
    Image.fromarray(mask).save(mask_path)
    saved = np.asarray(Image.open(rgb_path).convert("RGB"))
    if not np.array_equal(saved, rgb):
        raise RuntimeError("memory view png does not match the render")
    holes = ~np.asarray(view.mask)
    if holes.any() and int(saved[holes].max()) != 0:
        raise RuntimeError("empty memory pixels were not left black")
