# COLMAP sparse gate

No H3 run. Nominal CameraCtrl poses were not used inside feature extraction, matching, or mapping. They are only compared after the sparse model exists. No DAV2 depth, no guessed HFOV, no dense stereo.

## 1. COLMAP environment

- Binary: `/home/ma-user/miniconda3/envs/colmap_cpu/bin/colmap`
- Version: COLMAP 4.2.1, conda-forge `cpu_h5b89731_0`, banner says `without GPU support`
- The package did not pull OpenImageIO 3.1. `libopenimageio=3.1.17.0` was installed so `libOpenImageIO.so.3.1` resolves. No CUDA build.
- Camera model: `SIMPLE_RADIAL`, one shared camera (`ImageReader.single_camera 1`)
- Focal prior left at COLMAP's own default (`default_focal_length_factor=1.2`, prior 1152 px). `camera_params` was not set.
- Bundle adjustment refined focal length and the radial term. Principal point was left at the default (image center); `ba_refine_principal_point` stays 0.
- Matching: exhaustive, `FeatureMatching.use_gpu 0`, CPU SIFT brute force.

## 2. Input video / keyframes

The previous pair tests used `exp_runs/10_06_memory_cameractrl/clip_01_traj.mp4` (243 frames). That clip does not return to the start. Orbit closure needs the full generated orbit from the same CameraCtrl run:

- Path: `exp_runs/10_06_memory_cameractrl/C_trajectory/final.mp4`
- 960 x 544, 24 fps, 729 frames, 30.407 s
- It is clip_01 + clip_02 + clip_03. Frame 0 and frame 728 are the pair previously treated as the orbit start and the returning view.

Keyframes: stride 24, plus frame 728. 32 images. List is `frames.json` (kept outside `frames/` so COLMAP does not try to read it).

## 3. Sparse reconstruction result

One model was kept (`sparse/0`). Later initialization attempts were discarded by COLMAP and were not used.

| metric | value |
|---|---|
| keyframes | 32 |
| registered | 15 |
| registration ratio | 0.469 |
| sparse points | 831 |
| observations | 2977 |
| mean reprojection error | 0.782 px |
| median reprojection error | 0.662 px |
| camera model | SIMPLE_RADIAL |
| focal length | 649.09 px |
| principal point | (480, 272) |
| radial k | -0.0149 |
| implied HFOV | 73.0 deg |

Registered frame indices: 0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 384, 408, 432.

Frames 0–240 are a contiguous stride-24 chain. After frame 240, registration skips 288–360 and everything from 456 through 728, including the returning frame.

Of 496 exhaustive pairs, 102 have verified inliers. 100 of those are COLMAP config 3 (uncalibrated fundamental matrix), not a calibrated essential model. The strongest inlier counts are sequential: 216↔240 has 374, 0↔24 has 312. Frame 0↔728 has 0 verified inliers.

## 4. Orbit closure

The returning frame 728 is not registered, and it has 0 verified matches to frame 0. A COLMAP orbit closure of the generated start and end does not exist.

On the registered endpoints that do exist (frame 0 and frame 432):

- camera-center distance: 9.25 (COLMAP units)
- median camera distance to the trajectory centroid: 3.23
- normalized closure: 2.86
- relative rotation: 156.4 deg
- verified inliers between those two frames: 0

The trajectory plot shows frames 0–240 on one smooth arc, then 264/384/408/432 leave that arc. Frame 432 is not a return to frame 0.

## 5. COLMAP pose vs CameraCtrl nominal pose

Sampson error uses the same COLMAP-verified inliers and the same estimated K (focal 649 px, radial term removed by iteration). Nominal extrinsics come from the waypoint poses. This comparison happens after mapping.

| pair | matches | COLMAP error | nominal error | ratio nominal/COLMAP | rotation disagreement |
|---|---|---|---|---|---|
| 0→24 | 312 | 2.97e-7 | 9.82e-5 | 331 | 15.2 deg |
| 24→48 | 185 | 4.02e-7 | 2.52e-4 | 628 | 21.7 deg |
| 48→72 | 170 | 3.79e-7 | 9.11e-5 | 241 | 25.5 deg |
| 72→96 | 219 | 3.60e-7 | 2.15e-4 | 598 | 26.1 deg |
| 0→48 | 94 | 6.62e-7 | 7.60e-4 | 1149 | 36.9 deg |
| 0→728 | 0 | — | — | — | frame 728 not registered |

Inside the registered prefix, the COLMAP pose explains the correspondences two to three orders of magnitude better than the nominal CameraCtrl pose. The rotation gap grows as the baseline grows.

## 6. Sim(3) held-out test

Sim(3) maps nominal camera centers into the COLMAP world. Fit: frames 0, 24, 48, 72, 96, 120, 144. Test: 168, 192, 216, 240, 264, 384, 408, 432. Scale s = 2.27. Errors are in COLMAP units. The registered trajectory's median radius is 3.23, so a 0.39 error is about 12% of that radius.

| split | position median | position p90 | position RMSE | rotation median | rotation p90 |
|---|---|---|---|---|---|
| FIT | 0.393 | 0.466 | 0.361 | 178.8 deg | 179.4 deg |
| TEST | 0.817 | 3.772 | 2.220 | 174.9 deg | 179.4 deg |

By quartile of the registered sequence, position median goes 0.39, 0.34, 0.64, 2.69. The last quartile is the frames that leave the arc. Rotation error stays near 180 deg on every quartile, including the fit set. A single similarity aligns the early camera-center arc only roughly, and it does not align viewing directions.

## 7. Diagnosis

**CASE B. Local reconstruction works. The long-horizon video does not form one closed rigid orbit.**

- Frames 0–240 register as one smooth arc, with mean reprojection 0.78 px. On those pairs, COLMAP poses match the images and nominal CameraCtrl poses do not.
- The supposed return (frame 728) has no verified correspondence with frame 0 and is not in the model. The registered path does not close.
- A global Sim(3) fit on the first half does not predict the held-out tail, and the viewing-direction error is about 180 deg even on the fit frames. That is not a small fixed calibration bias.

Not CASE A: only 15/32 keyframes register, and the return view is absent.

Not CASE C: adjacent frames in the first 10 seconds do support a rigid pinhole model. Reprojection on that subset is sub-pixel, and the camera centers form a smooth arc.

## 8. DAV2 sparse-depth alignment

Not started. The sparse gate is CASE B.

## 9. Dense reconstruction comparison

Not started.

## 10. Recommendation

Stop. Do not build a dense cloud from nominal poses, and do not treat this COLMAP model as a full-orbit calibration.

A later experiment, not done here, would be chunk-level reconstruction on the locally rigid prefix, then an explicit alignment across chunks. The return frame cannot be tied to the start from these features: there is no verified match between frame 0 and frame 728.

## Figures

- `visualizations/colmap_sparse_top.png`
- `visualizations/colmap_sparse_side.png`
- `visualizations/colmap_sparse_3d.png`
- `visualizations/colmap_trajectory.png`
- `visualizations/orbit_closure.png`
- `visualizations/trajectory_nominal_vs_colmap.png`
