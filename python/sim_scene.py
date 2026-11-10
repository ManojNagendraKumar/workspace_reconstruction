"""Simulated tabletop workcell: a Franka Panda arm, a table, graspable objects and an RGBD camera.

Everything runs headless in PyBullet. The scene gives us what a real cell cannot:
exact ground truth for every pixel (object id), every pose, and the camera extrinsics,
so each stage of the perception pipeline can be measured on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pybullet as p
import pybullet_data

# Object classes the detector learns. Scale is chosen so all objects are 5 to 13 cm,
# which is the size range a cobot gripper typically handles.
CLASSES = {
    "mug": ("objects/mug.urdf", 0.8),
    "duck": ("duck_vhacd.urdf", 1.0),
    "teddy": ("teddy_vhacd.urdf", 1.0),
    "lego_brick": ("lego/lego.urdf", 1.8),
    "jenga_block": ("jenga/jenga.urdf", 1.0),
}
CLASS_NAMES = list(CLASSES.keys())

TABLE_POS = (0.5, 0.0, 0.0)
ROBOT_BASE_POS = (0.0, 0.0, 0.626)   # Panda sits on the table edge
WORKSPACE_X = (0.38, 0.78)           # object area in front of the robot (world frame)
WORKSPACE_Y = (-0.30, 0.30)
EE_LINK = 11                         # panda_grasptarget


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fov_v_deg: float = 50.0
    near: float = 0.05
    far: float = 3.0
    eye: tuple = (1.02, 0.0, 1.12)
    target: tuple = (0.56, 0.0, 0.63)
    up: tuple = (0.0, 0.0, 1.0)

    @property
    def K(self) -> np.ndarray:
        fy = self.height / (2.0 * np.tan(np.deg2rad(self.fov_v_deg) / 2.0))
        fx = fy  # square pixels: aspect ratio is folded into the image size
        cx, cy = (self.width - 1) / 2.0, (self.height - 1) / 2.0
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


@dataclass
class Frame:
    rgb: np.ndarray            # (H, W, 3) uint8
    depth: np.ndarray          # (H, W) float32, metres along the optical axis (OpenCV Z)
    seg: np.ndarray            # (H, W) int32, PyBullet body id per pixel (-1 = empty)
    K: np.ndarray              # 3x3 intrinsics
    T_world_cam: np.ndarray    # 4x4, OpenCV camera convention (x right, y down, z forward)
    objects: list = field(default_factory=list)  # list of dicts: id, cls, T_world_obj, aabb


def view_to_T_world_cam(view_matrix) -> np.ndarray:
    """PyBullet gives an OpenGL view matrix (column major, camera looks along -z, y up).
    Convert to the OpenCV camera convention used everywhere else in this project."""
    V = np.array(view_matrix, dtype=float).reshape(4, 4).T     # world -> GL camera
    T_world_glcam = np.linalg.inv(V)
    gl_to_cv = np.diag([1.0, -1.0, -1.0, 1.0])
    return T_world_glcam @ gl_to_cv


def pose_to_T(pos, quat) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    T[:3, 3] = pos
    return T


class Workcell:
    def __init__(self, seed: int = 0, gui: bool = False):
        self.rng = np.random.default_rng(seed)
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.cid)
        self.table = p.loadURDF("table/table.urdf", TABLE_POS, physicsClientId=self.cid)
        self.table_top_z = p.getAABB(self.table, physicsClientId=self.cid)[1][2]
        self.robot = p.loadURDF("franka_panda/panda.urdf", ROBOT_BASE_POS, useFixedBase=True,
                                physicsClientId=self.cid)
        self.T_world_base = pose_to_T(ROBOT_BASE_POS, (0, 0, 0, 1))
        self.home_joints = [0.0, -0.6, 0.0, -2.4, 0.0, 1.9, 0.785]
        self.set_joints(self.home_joints)
        self.objects: dict[int, str] = {}
        self.camera = CameraConfig()

    # ---------------- robot ----------------
    def set_joints(self, q):
        for j, v in enumerate(q):
            p.resetJointState(self.robot, j, v, physicsClientId=self.cid)
        for j in (9, 10):
            p.resetJointState(self.robot, j, 0.04, physicsClientId=self.cid)

    def move_ee_to(self, pos_base, quat_base):
        """Inverse kinematics to an end effector pose given in the robot base frame."""
        pos_w = (self.T_world_base @ np.r_[pos_base, 1.0])[:3]
        q = p.calculateInverseKinematics(self.robot, EE_LINK, pos_w, quat_base, maxNumIterations=200,
                                         residualThreshold=1e-5, physicsClientId=self.cid)
        self.set_joints(q[:7])
        return self.T_base_ee()

    def T_base_ee(self) -> np.ndarray:
        """Forward kinematics: end effector pose in the robot base frame."""
        st = p.getLinkState(self.robot, EE_LINK, computeForwardKinematics=True, physicsClientId=self.cid)
        T_world_ee = pose_to_T(st[4], st[5])
        return np.linalg.inv(self.T_world_base) @ T_world_ee

    # ---------------- objects ----------------
    def clear_objects(self):
        for uid in list(self.objects):
            p.removeBody(uid, physicsClientId=self.cid)
        self.objects.clear()

    def spawn_random_objects(self, n_min=3, n_max=7, settle_steps=240):
        self.clear_objects()
        n = int(self.rng.integers(n_min, n_max + 1))
        placed = []
        for _ in range(n):
            cls = str(self.rng.choice(CLASS_NAMES))
            for _attempt in range(30):  # rejection sampling so objects do not start inside each other
                x = self.rng.uniform(*WORKSPACE_X)
                y = self.rng.uniform(*WORKSPACE_Y)
                if all(np.hypot(x - px, y - py) > 0.10 for px, py in placed):
                    break
            placed.append((x, y))
            yaw = self.rng.uniform(-np.pi, np.pi)
            self.add_object(cls, (x, y, self.table_top_z + 0.08), yaw)
        for _ in range(settle_steps):
            p.stepSimulation(physicsClientId=self.cid)

    def add_distractors(self, n_max=2):
        """Unlabelled clutter (procedurally generated shapes) so the detector learns what NOT to fire on."""
        self.distractors = getattr(self, "distractors", [])
        for uid in self.distractors:
            p.removeBody(uid, physicsClientId=self.cid)
        self.distractors = []
        for _ in range(int(self.rng.integers(0, n_max + 1))):
            k = int(self.rng.integers(0, 1000))
            x, y = self.rng.uniform(*WORKSPACE_X), self.rng.uniform(*WORKSPACE_Y)
            uid = p.loadURDF(f"random_urdfs/{k:03d}/{k:03d}.urdf", (x, y, self.table_top_z + 0.1),
                             globalScaling=0.7, physicsClientId=self.cid)
            self.distractors.append(uid)
        for _ in range(120):
            p.stepSimulation(physicsClientId=self.cid)

    def randomize_appearance(self, recolor_prob=0.4):
        """Domain randomization: table tint and, sometimes, object colours, so the detector cannot
        rely on colour alone (a red mug in sim is not a guarantee of a red mug in the cell)."""
        tint = list(self.rng.uniform(0.55, 1.0, 3)) + [1.0]
        p.changeVisualShape(self.table, -1, rgbaColor=tint, physicsClientId=self.cid)
        for uid in self.objects:
            if self.rng.random() < recolor_prob:
                col = list(self.rng.uniform(0.15, 1.0, 3)) + [1.0]
                p.changeVisualShape(uid, -1, rgbaColor=col, physicsClientId=self.cid)

    def add_object(self, cls, pos_world, yaw=0.0):
        path, scale = CLASSES[cls]
        quat = p.getQuaternionFromEuler([0, 0, yaw])
        uid = p.loadURDF(path, pos_world, quat, globalScaling=scale, physicsClientId=self.cid)
        p.changeDynamics(uid, -1, lateralFriction=0.9, physicsClientId=self.cid)
        self.objects[uid] = cls
        return uid

    def object_states(self):
        out = []
        for uid, cls in self.objects.items():
            pos, quat = p.getBasePositionAndOrientation(uid, physicsClientId=self.cid)
            lo, hi = p.getAABB(uid, physicsClientId=self.cid)
            out.append({"id": uid, "cls": cls, "T_world_obj": pose_to_T(pos, quat),
                        "aabb_center_world": (np.array(lo) + np.array(hi)) / 2.0,
                        "aabb_size": np.array(hi) - np.array(lo)})
        return out

    # ---------------- camera ----------------
    def randomize_camera(self, jitter_pos=0.06, jitter_target=0.04):
        c = CameraConfig()
        c.eye = tuple(np.array(c.eye) + self.rng.uniform(-jitter_pos, jitter_pos, 3))
        c.target = tuple(np.array(c.target) + self.rng.uniform(-jitter_target, jitter_target, 3))
        self.camera = c

    def sample_lighting(self) -> dict:
        d = self.rng.normal(size=3)
        d[2] = abs(d[2]) + 0.5           # light comes from above
        return dict(lightDirection=list(d / np.linalg.norm(d) * 3.0),
                    lightColor=list(self.rng.uniform(0.7, 1.0, 3)),
                    lightAmbientCoeff=float(self.rng.uniform(0.25, 0.7)),
                    lightDiffuseCoeff=float(self.rng.uniform(0.3, 0.8)),
                    lightSpecularCoeff=float(self.rng.uniform(0.0, 0.4)),
                    shadow=int(self.rng.integers(0, 2)))

    def render(self, lighting="random", camera: CameraConfig | None = None) -> Frame:
        cam = camera or self.camera
        view = p.computeViewMatrix(cam.eye, cam.target, cam.up, physicsClientId=self.cid)
        proj = p.computeProjectionMatrixFOV(cam.fov_v_deg, cam.width / cam.height, cam.near, cam.far,
                                            physicsClientId=self.cid)
        if lighting == "random":
            kw = self.sample_lighting()
        elif isinstance(lighting, dict):
            kw = lighting
        else:
            kw = {}
        w, h, rgba, zbuf, seg = p.getCameraImage(cam.width, cam.height, view, proj,
                                                 renderer=p.ER_TINY_RENDERER, physicsClientId=self.cid, **kw)
        rgb = np.reshape(rgba, (h, w, 4))[:, :, :3].astype(np.uint8)
        zbuf = np.reshape(zbuf, (h, w)).astype(np.float64)
        # OpenGL depth buffer is non linear; convert to metric depth along the optical axis
        depth = cam.far * cam.near / (cam.far - (cam.far - cam.near) * zbuf)
        depth[zbuf >= 1.0] = 0.0  # nothing hit: invalid, like a real sensor's zero depth
        seg = np.reshape(seg, (h, w)).astype(np.int32)
        return Frame(rgb=rgb, depth=depth.astype(np.float32), seg=seg, K=cam.K,
                     T_world_cam=view_to_T_world_cam(view), objects=self.object_states())

    def close(self):
        p.disconnect(physicsClientId=self.cid)


def add_depth_noise(depth: np.ndarray, rng, f_px: float, baseline_m=0.05, disparity_sigma_px=0.08,
                    dropout=0.01) -> np.ndarray:
    """Stereo style depth noise: error grows with Z squared (dZ = Z^2 / (f B) * d_disparity),
    plus random invalid pixels, as on a real active stereo sensor."""
    d = depth.copy()
    valid = d > 0
    sigma = d ** 2 / (f_px * baseline_m) * disparity_sigma_px
    d[valid] += rng.normal(size=valid.sum()) * sigma[valid]
    d[rng.random(d.shape) < dropout] = 0.0
    return d
