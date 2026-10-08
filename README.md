# 多镜头长视频里的空间记忆

从一台没有 NVIDIA GPU 的实验机上整理出来的、换到 CUDA 机器后还能接着用的部分。研究问题是给长视频一块可解释的外部三维场景记忆，而不是把名义相机轨迹或单目深度当成度量几何。

视频、模型权重、生成结果没有放进来。

## 现在能用的

`scene_memory/` 是持久场景记忆，接口是 build / query / update / save / load。它不依赖视频生成器。

- 重建后端目前只有 COLMAP。VGGT、DUSt3R、MASt3R、WorldMirror 只留了名字，没有权重时会直接报错。
- 单个重建组放在单位位姿上。没有显式放置的多个组不会被硬塞进同一个刚体。
- 更新只在有验证重叠时写入。来源是名义位姿、waypoint 或 CameraCtrl 时，更新会在调用后端之前被拒绝。
- 记忆视图是一张额外参考图，交给现有生成器（MiniMax-H3 Ref2VA 的 `--reference image:path`），不改权重。掩码单独保存，不作为参考图。黑色像素是没观测到的区域。

COLMAP 调用保持稀疏门的设置：`SIMPLE_RADIAL`、共享相机、不传 `camera_params`、不跑 `pose_prior_mapper`、主点不优化。特征提取和匹配目前写死 `use_gpu 0`。二进制用 `COLMAP_BIN`，否则用 `PATH` 里的 `colmap`。

```bash
pip install -r requirements.txt
PYTHONPATH=. python -m unittest discover -s tests
python run_orbit_prefix.py --video /path/to/video.mp4
```

轨道前缀的关键帧已经在 `evidence/orbit_prefix/images/`（帧 0, 24, …, 240）。

## 已经确认的几何结论

1. 投影和反投影公式是对的。错的是名义 CameraCtrl 位姿。相邻帧上，COLMAP 的 Sampson 误差比名义位姿好两到三个数量级。详见 `docs/spatial_geometry.md` 和 `docs/colmap_sparse_gate.md`。
2. CPU COLMAP 在轨道前缀上 11/11 注册，663 个稀疏点，平均重投影约 0.72 px，`pose_prior_count` 为 0。稀疏点投回第一帧，落在灯、沙发、桌子和门上。图在 `evidence/orbit_prefix/visualizations/`。
3. 整段生成轨道没有闭合。帧 728 与帧 0 没有验证匹配，返回帧没有注册。不要把这段当成一个闭合刚体房间。
4. 在图像上对稀疏深度做线性插值，再把 11 个视角融进同一个点云，得到的稠密结果不是餐厅。原图颜色贴回去只能说明像素被盖住，不能说明三维里有桌面和沙发的体积。那团点云没有放进仓库。
5. 这台机器是 aarch64，只有 Ascend NPU。ViPE 和 GPU COLMAP 都没跑成，见 `docs/vipe_not_deployed.md`。

## 换到 CUDA 机器上

先复现轨道前缀的稀疏结果，再做真正的多视角重建。候选是带 CUDA 的 COLMAP，或 VGGT / MASt3R。不要把名义 waypoint 传进 mapper，也不要用 Depth Anything 当度量深度。

稠密结果要同时看侧视和从别的相机重投影。只在原相机上把照片贴回去，不算重建成功。

`baselines/spatial_memory/` 是更早的 GEN3C 式记忆（名义位姿加单目深度）。它说明了失败模式，不是度量场景模型。`baselines/h3_prompt/` 是提示词层面的房间锚点和参考图接口，不包含三维点。
