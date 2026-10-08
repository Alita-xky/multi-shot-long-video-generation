# Spatial geometry diagnosis

No H3 run. No splat. No hole filling.

Fixed before this real-data pass: `interpolate_clip_poses` was slerping R_w2c and lerping t_w2c. That moved the camera center off the waypoint eye by up to 0.322 m. It now interpolates eye / look_at / up and rebuilds the look-at pose. `look_at_w2c` now normalizes with a true unit vector so R is orthonormal.

## Results

| Test | Metric | Value | Pass/Fail | Diagnosis |
|------|--------|-------|-----------|-----------|
| Synthetic project/unproject | max xyz / max px | 1.998e-15 / 4.547e-12 | PASS | Formulas invert each other at numerical noise. |
| SE3 inverse | max |T_c2w T_w2c - I| | 2.220e-16 | PASS | w2c and c2w are inverses after exact normalization. |
| Synthetic two-view fusion | plane |Z-3| max / landmark max | 4.441e-16 / 7.448e-16 | PASS | Exact depth from two poses lands on the same plane. NN p90 is the non-overlap fringe, not a transform bug. |
| Real single-frame round-trip | median px / max px | 2.842e-14 / 3.411e-13 | PASS | Same K and same w2c project back to the source pixel. |
| Real two-frame fusion | median NN m / normalized | 0.205 / 0.084 | FAIL | Production per-frame anchor with Z clip. |
| Center-anchor scale | median NN m | 0.206 | see ablation | overlap linear fit on raw disparity is not a clear improvement over per-frame anchoring |
| Overlap scale/shift | median NN raw / inv | 0.304 / 0.220 | see ablation | s_raw=-0.06059 s_inv=152.4 |
| Nominal pose validation | rot direct deg / Sampson est vs nom | 14.48 / 1.201e-07 vs 2.872e-05 | FAIL | Estimated essential vs nominal essential on the same inliers. |
| 5-frame fusion | median NN of farthest listed frame | 0.590 (f96) | FAIL | Unclipped per-frame anchor, colored by frame. |
| Orbit closure | median NN / normalized / camera return m | 0.346 / 0.142 / 0.0000 | FAIL | C_trajectory clip_01 f0 vs clip_03 f242, production Z |

## Root cause

Classification: **G**, dominated by **F**. A and B are ruled out.

- A/B do not explain the streaks. Synthetic project/unproject, SE3 inverse, synthetic two-view fusion, and the real single-frame round-trip all pass at numerical noise (max pixel error 3.4e-13).
- F is the cross-frame failure. On frames 0 and 24, 206 essential-matrix inliers. Median Sampson error is 1.20e-7 for the estimated E and 2.87e-5 for the nominal CameraCtrl E (about 240x larger). Relative rotation differs by 14.5 deg. Translation direction differs by 25.7 deg after taking the essential-matrix sign flip (154 deg before the flip). A 14 deg pose error at a 3–5 m surface moves points by several tens of centimetres, which matches the two-frame median NN of 0.205 m and p90 of 0.71 m. Nominal orbit closure is exact (camera centers 0.000 m apart) while the reconstructed clouds still have median NN 0.346 m, so the waypoint path closes and the video does not follow it.
- D is real and secondary. Frame 0 region means: near floor 231, table 79, far window 31, so raw is disparity (near=high). `Z = k / raw` with raw min 0 produces one spike at 2.3e5 m, but only 0.0015% of pixels have Z>30 m. 11.5% have Z>10 m and production clips those to the Z=10 shell. That shell is not the 0.20 m two-frame error.
- E is real and not sufficient. Center-anchor k is 231 on frame 0 and 197 on frame 24. Replacing it with one global k, or with an overlap fit `s*raw+t` (s=-0.061, residual 1.85 m → 1.38 m), makes the nearest-neighbor error worse (0.206 m → 0.304 m). An overlap fit on `1/raw` also does not beat per-frame anchoring (0.220 m). So stopping center anchoring does not put the two frames in the same world while the nominal pose is wrong.

## Depth orientation (frame 0, measured)

raw value is larger when closer. Treat raw as disparity (near=high), not as metric depth.

```
{
  "region_mean_raw": {
    "near_bottom_center": 230.72682189941406,
    "table_center": 78.88230895996094,
    "far_top_center": 31.173471450805664
  },
  "unclipped_Z": {
    "min": 0.9064508868679698,
    "p1": 0.943448882250336,
    "p5": 1.270027341490837,
    "p50": 4.71724441125168,
    "p95": 11.557248807566616,
    "p99": 15.409665076755488,
    "max": 231144.97615133232
  },
  "fraction_Z_above_10m": 0.11461971507352942
}
```

## Files

- tests/spatial_geometry_debug/coordinate_convention.txt
- tests/spatial_geometry_debug/synthetic_results.json
- tests/spatial_geometry_debug/synthetic_two_view_top.png
- tests/spatial_geometry_debug/synthetic_two_view_side.png
- tests/spatial_geometry_debug/rgb.png
- tests/spatial_geometry_debug/raw_depth.png
- tests/spatial_geometry_debug/single_frame_roundtrip.json
- tests/spatial_geometry_debug/two_frame_top.png
- tests/spatial_geometry_debug/two_frame_side.png
- tests/spatial_geometry_debug/two_frame_3d.png
- tests/spatial_geometry_debug/two_frame_alignment.json
- tests/spatial_geometry_debug/depth_alignment_comparison.png
- tests/spatial_geometry_debug/depth_alignment_metrics.json
- tests/spatial_geometry_debug/pose_matches.png
- tests/spatial_geometry_debug/pose_validation.json
- tests/spatial_geometry_debug/fusion_02frames.png
- tests/spatial_geometry_debug/fusion_03frames.png
- tests/spatial_geometry_debug/fusion_05frames.png
- tests/spatial_geometry_debug/fusion_10frames.png
- tests/spatial_geometry_debug/orbit_closure_top.png
- tests/spatial_geometry_debug/orbit_closure_side.png
