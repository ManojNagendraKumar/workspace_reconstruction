"""Run the unchanged C++ segmentation pipeline on one real RGBD frame.

The frame is the public example from GraspNet 1Billion (Fang et al., CVPR 2020): a RealSense camera mounted on a
robot arm looking at about ten everyday objects. It is downloaded from the graspnet-baseline GitHub repository, not
stored here. The example has no camera pose, so the table plane is fitted first and used to define a table frame
(z up, table at z = 0), which stands in for the robot base frame the pipeline expects. No pipeline parameter is
changed for real data.

The frame has no per point labels, so this is a qualitative check: it shows what the pipeline does on real sensor
data and how fast it runs. The annotated object positions from the frame's metadata are drawn for reference.

Output: results/real_frame_segmentation.png, results/real_frame_check.json
"""
import json
import subprocess
from collections import Counter
import sys
import urllib.request
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from evaluate import read_pcd_xyzl  # noqa: E402
from generate_views import write_pcd  # noqa: E402

URL = "https://raw.githubusercontent.com/graspnet/graspnet-baseline/main/doc/example_data/"
FILES = ("color.png", "depth.png", "workspace_mask.png", "meta.mat")
BIN = ROOT / "cpp/build/workspace_segmentation"
NOISE = 4294967295
TABLE_ORIGIN = np.array([0.6, 0.0, 0.0])  # where the table frame origin sits, inside the pipeline's crop box


def download(dst):
    dst.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if not (dst / f).exists():
            urllib.request.urlretrieve(URL + f, dst / f)


def fit_plane(pts, rng, iters=400, thresh=0.005):
    """RANSAC plane fit; returns unit normal n and offset d with n.p + d = 0, plus the inlier mask."""
    best, best_n, best_d = None, None, None
    sample = pts[rng.choice(len(pts), min(30000, len(pts)), replace=False)]
    for _ in range(iters):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        d = -n @ a
        inl = np.abs(sample @ n + d) < thresh
        if best is None or inl.sum() > best.sum():
            best, best_n, best_d = inl, n, d
    inliers = np.abs(pts @ best_n + best_d) < thresh
    # refine with least squares on all inliers
    q = pts[inliers]
    c = q.mean(0)
    n = np.linalg.svd(q - c, full_matrices=False)[2][-1]
    return n, -n @ c, np.abs(pts @ n - n @ c) < thresh


def main():
    data = ROOT / "data/real_example"
    download(data)
    rgb = np.array(Image.open(data / "color.png").convert("RGB"))
    meta = sio.loadmat(data / "meta.mat")
    K = meta["intrinsic_matrix"]
    depth = np.array(Image.open(data / "depth.png")).astype(np.float64) / float(meta["factor_depth"][0, 0])
    ws = np.array(Image.open(data / "workspace_mask.png")) > 0

    v, u = np.nonzero(ws & (depth > 0))
    z = depth[v, u]
    pts_cam = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)

    rng = np.random.default_rng(0)
    n, d, table_inl = fit_plane(pts_cam, rng)
    if n @ (-pts_cam[table_inl].mean(0)) < 0:      # make the normal point toward the camera (up)
        n, d = -n, -d
    x = np.array([1.0, 0.0, 0.0]) - n[0] * n          # camera x axis projected onto the table
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(n, x), n])                # rows: table frame axes in camera coordinates
    o = pts_cam[table_inl].mean(0)
    o = o - (n @ o + d) * n                             # table frame origin, on the plane
    pts_tab = (pts_cam - o) @ R.T + TABLE_ORIGIN

    pcd = data / "real_frame.pcd"
    write_pcd(pcd, pts_tab, np.zeros(len(pts_tab), np.uint32))
    out = ROOT / "results/runs/real_frame"
    subprocess.run([str(BIN), str(pcd), str(out), "20"], check=True, capture_output=True)
    timing = json.loads((out / "timing.json").read_text())
    q, cl = read_pcd_xyzl(out / "segmented.pcd")
    ids = sorted(set(cl.tolist()) - {0, NOISE})

    # annotated object positions (model origins) from the metadata, in the table frame. Each annotated object is
    # assigned the cluster that holds most output points within 3 cm of it (above the table), which shows whether
    # objects come out on their own or merged with neighbours.
    gt_cam = meta["poses"][:, 3, :].T
    gt_tab = (gt_cam - o) @ R.T + TABLE_ORIGIN
    owner = []
    for g in gt_tab:
        near = (np.linalg.norm(q[:, :2] - g[:2], axis=1) < 0.03) & (q[:, 2] > 0.008)
        labs = cl[near]
        labs = labs[(labs != 0) & (labs != NOISE)]
        owner.append(int(np.bincount(labs).argmax()) if len(labs) else None)
    share = Counter(c for c in owner if c is not None)
    separated = sum(1 for c in owner if c is not None and share[c] == 1)
    merged = sum(1 for c in owner if c is not None and share[c] > 1)
    missed = sum(c is None for c in owner)
    table_level = [c for c in ids if q[cl == c, 2].max() < 0.005]   # clusters that never rise above the table

    # project every output point back into the image to draw the result on the real photo
    back = (q - TABLE_ORIGIN) @ R + o
    pu = (K[0, 0] * back[:, 0] / back[:, 2] + K[0, 2]).astype(int)
    pv = (K[1, 1] * back[:, 1] / back[:, 2] + K[1, 2]).astype(int)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#17becf", "#e377c2", "#bcbd22", "#8c564b"]
    col = {c: palette[i % len(palette)] for i, c in enumerate(ids)}
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.16))
    ax[0].imshow(rgb)
    for c in ids:
        m = cl == c
        ax[0].scatter(pu[m], pv[m], s=2.5, color=col[c], alpha=0.8, linewidths=0)
    ax[0].set_title("Real RealSense frame (GraspNet), pipeline output drawn back onto the photo", fontsize=11)
    ax[0].axis("off")
    t = cl == 0
    ax[1].scatter(q[t, 0], q[t, 1], s=0.3, color="0.82", linewidths=0)
    for c in ids:
        m = cl == c
        ax[1].scatter(q[m, 0], q[m, 1], s=0.8, color=col[c], linewidths=0)
    nz = cl == NOISE
    ax[1].scatter(q[nz, 0], q[nz, 1], s=0.8, color="k", linewidths=0)
    ax[1].scatter(gt_tab[:, 0], gt_tab[:, 1], marker="x", s=60, color="crimson", linewidths=2)
    ax[1].set_aspect("equal")          # x runs with the image columns and y with the image rows, so the view
    ax[1].set_xlabel("x in table frame (m)")   # matches the photo
    ax[1].set_ylabel("y in table frame (m)")
    ax[1].set_title(f"Top view: {separated} of {len(owner)} objects on their own, {merged} merged into one "
                    f"cluster,\n{len(table_level)} false clusters at table level ({timing['mean_ms']['total']:.0f} ms)",
                    fontsize=11)
    from matplotlib.lines import Line2D
    ax[1].legend(handles=[Line2D([], [], ls="", marker="o", color="0.82", label="table"),
                          Line2D([], [], ls="", marker="o", color="tab:blue", label="clusters (one colour each)"),
                          Line2D([], [], ls="", marker="o", color="k", label="unclustered"),
                          Line2D([], [], ls="", marker="x", color="crimson", mew=2, label="annotated object position")],
                 loc="lower left", fontsize=8, framealpha=0.9)
    fig.savefig(ROOT / "results/real_frame_segmentation.png", dpi=110, bbox_inches="tight")

    result = {
        "source": "GraspNet 1Billion example frame (graspnet-baseline repository), RealSense camera on a robot arm",
        "image_size": list(rgb.shape[:2]),
        "workspace_points": int(len(pts_cam)),
        "missing_depth_fraction_in_workspace": float(np.mean(depth[ws] == 0)),
        "table_plane_inlier_fraction": float(table_inl.mean()),
        "pipeline_input_points": timing["input_points"],
        "pipeline_filtered_points": timing["filtered_points"],
        "clusters": len(ids),
        "cluster_points": [int(np.sum(cl == c)) for c in ids],
        "unclustered_points": int(nz.sum()),
        "annotated_objects": int(gt_tab.shape[0]),
        "annotated_object_ids": meta["cls_indexes"].ravel().tolist(),
        "cluster_of_each_annotated_object": owner,
        "objects_on_their_own": separated,
        "objects_merged_with_others": merged,
        "objects_without_a_cluster": missed,
        "false_clusters_at_table_level": len(table_level),
        "timing_ms": timing["mean_ms"],
        "notes": "No per point labels exist for this frame, so there is no point level score. Each annotated object "
                 "is assigned the cluster holding most points within 3 cm of its annotated position.",
    }
    (ROOT / "results/real_frame_check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("notes",)}, indent=1))


if __name__ == "__main__":
    main()
