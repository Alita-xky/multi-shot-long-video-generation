# 证据

`orbit_prefix/` 是帧 0 到 240、步长 24 的 COLMAP 稀疏结果。`memory/points.npz` 是 663 个三角化点。`visualizations/frame0_overlay.png` 是这些点投回第一帧。`visualizations/cloud_cropped_by_frame.png` 是裁掉离群点之后的相机弧。记忆视图只有几百个有效像素，其余是黑的。

`colmap_sparse_gate/` 是更早的 32 关键帧稀疏门：15/32 注册，831 点，返回帧没有进来。`workspace/database.db` 里 `pose_priors` 为空。报告在 `docs/colmap_sparse_gate.md`。

这里没有稠密融合点云。那个结果不能当成房间模型。
