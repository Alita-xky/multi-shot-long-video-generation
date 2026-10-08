#!/usr/bin/env python3
"""Evaluate a saved SceneMemory. Does not run the video generator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scene_memory.evaluate import object_persistence, return_report, spatial_report  # noqa: E402
from scene_memory.memory import SceneMemory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatial, return, and object metrics for a SceneMemory directory.")
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--return-a", default=None)
    parser.add_argument("--return-b", default=None)
    parser.add_argument("--objects", type=Path, default=None, help="JSON object tracks: {name: [{camera,u,v}, ...]}.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    memory = SceneMemory.load(args.memory)
    report = {"spatial": spatial_report(memory)}
    if args.return_a and args.return_b:
        report["return"] = return_report(memory, args.return_a, args.return_b)
    tracks = None
    if args.objects is not None:
        tracks = json.loads(args.objects.read_text(encoding="utf-8"))
    report["objects"] = object_persistence(memory, tracks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"registration_ratio": report["spatial"]["registration_ratio"], "num_points": report["spatial"]["num_points"]}))


if __name__ == "__main__":
    main()
