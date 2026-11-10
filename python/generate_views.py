"""Capture several depth views of a cluttered table with a wrist (eye in hand) camera and fuse them.

For each scene the simulated Panda moves its wrist camera to five viewpoints. Every depth image is back
projected and moved into the robot base frame with
    T_base_cam = T_base_ee (forward kinematics at capture time) * T_ee_cam (hand eye calibration)
where T_ee_cam carries a realistic calibration error (2 mm, 0.3 degrees), as it would on a real robot.

Output per scene (data/scene_XXX/):
  fused.pcd   all views fused, base frame, fields x y z label (0 = table/background, k > 0 = object k)
  view0.pcd   the first view alone, for the single frame vs fused comparison
Points that belong to the robot itself are dropped (on a real robot: a URDF based self filter).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pybullet as p

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from geometry import depth_to_points, transform  # noqa: E402
from sim_scene import CameraConfig, Workcell, add_depth_noise  # noqa: E402

N_SCENES = 30
VIEW_POSITIONS = [(0.56, 0.00, 0.46), (0.42, 0.20, 0.40), (0.42, -0.20, 0.40), (0.62, 0.25, 0.44),
                  (0.62, -0.25, 0.44)]
PIXEL_STRIDE = 2                                # use every second pixel row and column (320 x 240 effective)
LOOK_AT = np.array([0.58, 0.0, 0.0])           # workspace centre on the table, base frame


def look_at_rotation(pos, target):
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.c_[x, y, z]


def mat_to_quat(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()       # (x, y, z, w), PyBullet order


def small_error_transform(rng, trans_mm=2.0, rot_deg=0.3):
    from scipy.spatial.transform import Rotation
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    E = np.eye(4)
    E[:3, :3] = Rotation.from_rotvec(axis * np.radians(rot_deg)).as_matrix()
    E[:3, 3] = d * trans_mm / 1000.0
    return E


def write_pcd(path, pts, labels):
    """Binary PCD with fields x y z label, readable by pcl::io::loadPCDFile<pcl::PointXYZL>."""
    rec = np.zeros(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("label", "<u4")])
    rec["x"], rec["y"], rec["z"], rec["label"] = pts[:, 0], pts[:, 1], pts[:, 2], labels
    header = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z label\n"
              "SIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n"
              f"WIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(pts)}\nDATA binary\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes())


def render_from(wc, T_world_cam, cam):
    """Render with an arbitrary OpenCV camera pose (z forward, y down)."""
    eye = T_world_cam[:3, 3]
    view = p.computeViewMatrix(eye, eye + T_world_cam[:3, 2], -T_world_cam[:3, 1], physicsClientId=wc.cid)
    proj = p.computeProjectionMatrixFOV(cam.fov_v_deg, cam.width / cam.height, cam.near, cam.far,
                                        physicsClientId=wc.cid)
    _, _, _, zbuf, seg = p.getCameraImage(cam.width, cam.height, view, proj, renderer=p.ER_TINY_RENDERER,
                                          physicsClientId=wc.cid)
    zbuf = np.reshape(zbuf, (cam.height, cam.width)).astype(np.float64)
    depth = cam.far * cam.near / (cam.far - (cam.far - cam.near) * zbuf)
    depth[zbuf >= 1.0] = 0.0
    return depth.astype(np.float32), np.reshape(seg, (cam.height, cam.width)).astype(np.int32)


def main():
    out_root = HERE.parent / "data"
    cam = CameraConfig(fov_v_deg=55.0)              # wrist camera, short range
    T_ee_cam = np.eye(4)
    T_ee_cam[:3, 3] = [0.0, 0.06, -0.10]            # mounted beside and behind the grasp point
    for s in range(N_SCENES):
        wc = Workcell(seed=3000 + s)
        rng = wc.rng
        wc.spawn_random_objects(4, 8)
        wc.add_distractors(2)
        obj_uids = list(wc.objects) + list(getattr(wc, "distractors", []))
        label_of = {uid: k + 1 for k, uid in enumerate(obj_uids)}
        T_ee_cam_est = small_error_transform(rng) @ T_ee_cam      # what calibration would give us
        all_pts, all_lab = [], []
        for v, pos in enumerate(VIEW_POSITIONS):
            pos = np.array(pos)
            T_base_ee = wc.move_ee_to(pos, mat_to_quat(look_at_rotation(pos, LOOK_AT)))
            T_world_cam_true = wc.T_world_base @ T_base_ee @ T_ee_cam
            depth, seg = render_from(wc, T_world_cam_true, cam)
            depth = add_depth_noise(depth, rng, f_px=cam.K[0, 0])
            sub = np.zeros_like(depth, dtype=bool)
            sub[::PIXEL_STRIDE, ::PIXEL_STRIDE] = True
            depth[~sub] = 0.0
            pts_cam, (vv, uu) = depth_to_points(depth, cam.K)
            seg_px = seg[vv, uu]
            keep = seg_px != wc.robot                                # robot self filter
            pts_base = transform(T_base_ee @ T_ee_cam_est, pts_cam[keep])
            lab = np.array([label_of.get(int(i), 0) for i in seg_px[keep]])
            all_pts.append(pts_base)
            all_lab.append(lab)
            if v == 0:
                d = out_root / f"scene_{s:03d}"
                d.mkdir(parents=True, exist_ok=True)
                write_pcd(d / "view0.pcd", pts_base, lab)
        write_pcd(out_root / f"scene_{s:03d}" / "fused.pcd", np.vstack(all_pts), np.concatenate(all_lab))
        names = {**{uid: cls for uid, cls in wc.objects.items()},
                 **{uid: "distractor" for uid in getattr(wc, "distractors", [])}}
        (out_root / f"scene_{s:03d}" / "labels.json").write_text(
            json.dumps({str(label_of[uid]): names[uid] for uid in obj_uids}))
        wc.close()
        print("scene", s, "points", sum(len(a) for a in all_pts), flush=True)


if __name__ == "__main__":
    main()
