"""Run the C++ pipeline on every scene and score it against the simulator's ground truth labels.

Metrics (per configuration, over all scenes):
  object_recall     share of ground truth objects matched by a cluster with point IoU >= 0.5
  cluster_precision share of clusters that match a ground truth object
  point_accuracy    share of object points that end up in the cluster matched to their own object
  table_removed     share of table points labelled as table by RANSAC
  merges            clusters that contain two or more objects (each contributing >= 30% of its points)
Configurations: single view vs five fused views, and baseline vs optimized stage order (timing).

Writes results/segmentation.json, results/timing.json and figures.
"""
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "cpp/build/workspace_segmentation"
NOISE = 4294967295
MIN_GT_POINTS = 30


def read_pcd_xyzl(path):
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline().decode("ascii").strip()
            k, *v = line.split()
            header[k] = v
            if k == "DATA":
                break
        n = int(header["POINTS"][0])
        if header["DATA"][0] == "binary":
            fields = header["FIELDS"]
            types = {"F": "f", "U": "u", "I": "i"}
            dt = np.dtype([(name, f"<{types[t]}{s}") for name, t, s in
                           zip(fields, header["TYPE"], header["SIZE"])])
            rec = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
        else:
            raise ValueError("only binary PCD is supported here")
    return np.c_[rec["x"], rec["y"], rec["z"]].astype(float), rec["label"].astype(np.int64)


def score(scene_dir, out_dir, cloud_name):
    xyz_in, lab_in = read_pcd_xyzl(scene_dir / cloud_name)
    xyz, cl = read_pcd_xyzl(out_dir / "segmented.pcd")
    gt = lab_in[cKDTree(xyz_in).query(xyz)[1]]          # ground truth label of each output point
    names = json.loads((scene_dir / "labels.json").read_text())
    gt_ids = [k for k in np.unique(gt) if k > 0 and (gt == k).sum() >= MIN_GT_POINTS]
    cl_ids = [c for c in np.unique(cl) if c not in (0, NOISE)]
    iou = np.zeros((len(gt_ids), len(cl_ids)))
    for i, g in enumerate(gt_ids):
        gm = gt == g
        for j, c in enumerate(cl_ids):
            cm = cl == c
            inter = np.sum(gm & cm)
            if inter:
                iou[i, j] = inter / np.sum(gm | cm)
    rows, cols = linear_sum_assignment(-iou) if iou.size else ([], [])
    matched = {gt_ids[r]: cl_ids[c] for r, c in zip(rows, cols) if iou[r, c] >= 0.5}
    obj_pts = gt > 0
    correct = sum(np.sum((gt == g) & (cl == c)) for g, c in matched.items())
    merges = 0
    for c in cl_ids:
        cm = cl == c
        parts = [g for g in gt_ids if np.sum(cm & (gt == g)) >= 0.3 * np.sum(gt == g)]
        merges += len(parts) >= 2
    per_class = defaultdict(lambda: [0, 0])
    for g in gt_ids:
        per_class[names.get(str(g), "unknown")][0] += 1
        per_class[names.get(str(g), "unknown")][1] += g in matched

    # diagnostics: where do the wrongly assigned object points go, and why is each missed object missed?
    table_z = float(np.median(xyz[gt == 0, 2])) if np.any(gt == 0) else 0.0
    ok = np.zeros(len(gt), bool)
    for g, c in matched.items():
        ok |= (gt == g) & (cl == c)
    wrong = obj_pts & ~ok
    as_table = wrong & (cl == 0)
    near_table = (xyz[:, 2] - table_z) < 0.006
    point_errors = {"wrong": int(wrong.sum()), "as_table": int(as_table.sum()),
                    "as_table_within_6mm_of_table": int((as_table & near_table).sum()),
                    "as_noise": int((wrong & (cl == NOISE)).sum())}
    point_errors["in_other_cluster"] = point_errors["wrong"] - point_errors["as_table"] - point_errors["as_noise"]
    misses = []
    for g in gt_ids:
        if g in matched:
            continue
        gm = gt == g
        n = gm.sum()
        lost = np.sum(gm & ((cl == 0) | (cl == NOISE))) / n
        shares = {c: np.sum(gm & (cl == c)) / n for c in cl_ids}
        merged = any(sh >= 0.3 and any(h != g and np.sum((gt == h) & (cl == c)) >= 0.3 * np.sum(gt == h)
                                       for h in gt_ids) for c, sh in shares.items())
        reason = "absorbed by table or noise" if lost >= 0.5 else ("merged" if merged else "split")
        misses.append({"class": names.get(str(g), "unknown"), "reason": reason,
                       "height_mm": float(1000 * (xyz[gm, 2].max() - table_z))})
    return {"gt_objects": len(gt_ids), "clusters": len(cl_ids), "matched": len(matched),
            "object_points": int(obj_pts.sum()), "correct_points": int(correct),
            "table_points": int((gt == 0).sum()), "table_removed": int(np.sum((gt == 0) & (cl == 0))),
            "merges": int(merges), "per_class": dict(per_class), "point_errors": point_errors,
            "misses": misses, "xyz": xyz, "cl": cl, "gt": gt}


def summarize(tot, per_class, n_scenes):
    return {
        "scenes": n_scenes, "gt_objects": int(tot["gt_objects"]),
        "object_recall": tot["matched"] / tot["gt_objects"],
        "cluster_precision": tot["matched"] / max(tot["clusters"], 1),
        "point_accuracy": tot["correct_points"] / tot["object_points"],
        "table_removed": tot["table_removed"] / tot["table_points"],
        "merged_clusters": int(tot["merges"]),
        "recall_per_class": {c: {"objects": n, "recall": m / n} for c, (n, m) in sorted(per_class.items())},
    }


def main():
    scenes = sorted((ROOT / "data").glob("scene_*"))
    configs = {"single_view": "view0.pcd", "fused_5_views": "fused.pcd"}
    results, timing = {}, {"baseline": defaultdict(list), "optimized": defaultdict(list)}
    example = None
    keys = ("gt_objects", "clusters", "matched", "object_points", "correct_points", "table_points",
            "table_removed", "merges")
    for cfg, cloud in configs.items():
        tot = {m: defaultdict(float) for m in ("baseline", "optimized")}
        per_class = {m: defaultdict(lambda: [0, 0]) for m in ("baseline", "optimized")}
        point_errors, misses = defaultdict(int), []
        for s in scenes:
            out = ROOT / "results/runs" / cfg / s.name
            out.mkdir(parents=True, exist_ok=True)
            for mode in ("baseline", "optimized"):
                subprocess.run([str(BIN), str(s / cloud), str(out), "10", mode], check=True,
                               env={"OMP_NUM_THREADS": "1"}, capture_output=True)
                if cfg == "fused_5_views":
                    t = json.loads((out / "timing.json").read_text())
                    for k, v in t["mean_ms"].items():
                        timing[mode][k].append(v)
                    timing[mode]["input_points"].append(t["input_points"])
                sc = score(s, out, cloud)             # both stage orders are scored
                for k in keys:
                    tot[mode][k] += sc[k]
                for c, (n, m) in sc["per_class"].items():
                    per_class[mode][c][0] += n
                    per_class[mode][c][1] += m
                if mode == "optimized":
                    for k, v in sc["point_errors"].items():
                        point_errors[k] += v
                    misses += sc["misses"]
                    if cfg == "fused_5_views" and example is None:
                        example = (s.name, sc)
        results[cfg] = summarize(tot["optimized"], per_class["optimized"], len(scenes))
        results[cfg]["baseline_order"] = summarize(tot["baseline"], per_class["baseline"], len(scenes))
        del results[cfg]["baseline_order"]["recall_per_class"]
        results[cfg]["wrong_object_points"] = dict(point_errors)
        reasons = defaultdict(int)
        for m in misses:
            reasons[m["reason"]] += 1
        results[cfg]["missed_objects"] = {"by_reason": dict(reasons), "list": misses}
        print(cfg, json.dumps({k: v for k, v in results[cfg].items() if k not in ("recall_per_class", "missed_objects")}),
              json.dumps(dict(reasons)), flush=True)

    tm = {mode: {k: float(np.mean(v)) for k, v in d.items()} for mode, d in timing.items()}
    for mode in tm:
        tm[mode]["hz"] = 1000.0 / tm[mode]["total"]
    tm["note"] = "mean over all fused scenes, 10 repeats each, one CPU thread"
    (ROOT / "results/segmentation.json").write_text(json.dumps(results, indent=2))
    (ROOT / "results/timing.json").write_text(json.dumps(tm, indent=2))
    print(json.dumps(tm, indent=2))

    # figure: timing per stage, baseline vs optimized
    stages = ["1_crop", "2_voxel", "3_outliers", "4_plane", "5_clusters"]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = np.arange(len(stages))
    ax.bar(x - 0.2, [tm["baseline"].get(s, 0) for s in stages], 0.4, label="baseline order")
    ax.bar(x + 0.2, [tm["optimized"].get(s, 0) for s in stages], 0.4, label="optimized order")
    ax.set_xticks(x, ["crop", "voxel 5 mm", "outlier removal", "RANSAC plane", "clustering"])
    ax.set_ylabel("ms per frame (one CPU thread)")
    ax.set_title(f"Pipeline time: {tm['baseline']['total']:.0f} ms baseline vs "
                 f"{tm['optimized']['total']:.0f} ms optimized")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(ROOT / "results/timing_per_stage.png", dpi=130, bbox_inches="tight")

    # figure: top view of one fused scene, ground truth vs clusters
    name, sc = example
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
    for a, lab, title in ((ax[0], sc["gt"], "ground truth objects"), (ax[1], sc["cl"], "PCL clusters")):
        m = (lab != 0) & (lab != NOISE)
        a.scatter(sc["xyz"][~m, 1], sc["xyz"][~m, 0], s=0.3, c="0.85")
        a.scatter(sc["xyz"][m, 1], sc["xyz"][m, 0], s=1.0, c=lab[m] % 10, cmap="tab10")
        a.set_title(title)
        a.set_xlabel("y in robot base frame (m)")
        a.set_aspect("equal")
    ax[0].set_ylabel("x in robot base frame (m)")
    fig.suptitle(f"{name}: top view of the fused cloud (grey = table)")
    fig.tight_layout()
    fig.savefig(ROOT / "results/segmentation_example.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    sys.exit(main())
