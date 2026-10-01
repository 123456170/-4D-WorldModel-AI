import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
import io
import json
import math
import time
import base64
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict, Any
from collections import deque

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except Exception:
    IMAGEIO_AVAILABLE = False


st.set_page_config(
    page_title="4D WorldModel AI",
    page_icon="🌍",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top:18px;}
    .small-muted {font-size:0.85em; opacity:0.75;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Configuration and data structures
# ============================================================

@dataclass
class DemoConfig:
    mode: str
    duration_sec: float
    fps: int
    width: int
    height: int
    num_objects: int
    noise: float
    min_conf: float
    low_conf_threshold: float
    max_range: float
    seed: int
    prediction_horizon: float
    show_gt: bool
    simulate_faults: bool
    eval_match_gate: float = 2.5


@dataclass
class SimObject:
    obj_id: int
    label: str
    shape: str
    color: Tuple[int, int, int]
    center: np.ndarray
    base_height: float
    vel: np.ndarray
    size: np.ndarray
    phase: float
    current_vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))


# ============================================================
# Geometry helpers
# ============================================================

def make_camera(width: int, height: int) -> Dict[str, float]:
    fx = float(width) * 1.15
    fy = float(width) * 1.15
    return {
        "fx": fx,
        "fy": fy,
        "cx": float(width) / 2.0,
        "cy": float(height) * 0.45,
        "height": 1.7,
    }


def project_point(point: np.ndarray, cam: Dict[str, float]) -> Optional[Tuple[float, float, float]]:
    point = np.asarray(point, dtype=np.float32)
    x, y, z = float(point[0]), float(point[1]), float(point[2])
    if z <= 0.2:
        return None
    y_cam = cam["height"] - y
    u = cam["fx"] * x / z + cam["cx"]
    v = cam["fy"] * y_cam / z + cam["cy"]
    return float(u), float(v), float(z)


# ============================================================
# Synthetic 4D demo scene
# ============================================================

def create_objects(num_objects: int, rng: np.random.Generator) -> List[SimObject]:
    objects = []
    base_colors = [
        (48, 132, 201),
        (62, 62, 205),
        (80, 180, 80),
        (190, 120, 220),
        (60, 200, 240),
        (90, 160, 250),
    ]

    for i in range(num_objects):
        shape = str(rng.choice(["box", "cylinder", "drone"], p=[0.50, 0.25, 0.25]))
        label = "drone" if shape == "drone" else str(rng.choice(["crate", "barrel", "cart"]))

        x = float(rng.uniform(-6.0, 6.0))
        z = float(rng.uniform(8.0, 20.0))
        base_height = float(rng.uniform(1.2, 1.8)) if shape == "drone" else float(rng.uniform(0.45, 0.85))

        if shape == "drone":
            size = np.array([0.75, 0.30, 0.75], dtype=np.float32)
        else:
            size = np.array(
                [
                    float(rng.uniform(0.45, 1.20)),
                    float(rng.uniform(0.45, 1.45)),
                    float(rng.uniform(0.40, 1.10)),
                ],
                dtype=np.float32,
            )

        vel = np.array(
            [
                float(rng.uniform(-1.35, 1.35)),
                0.0,
                float(rng.uniform(-0.85, 0.85)),
            ],
            dtype=np.float32,
        )

        if abs(float(vel[0])) < 0.2:
            vel[0] = 0.65 * (1.0 if float(rng.standard_normal()) >= 0 else -1.0)
        if abs(float(vel[2])) < 0.2:
            vel[2] = 0.45 * (1.0 if float(rng.standard_normal()) >= 0 else -1.0)

        base = base_colors[i % len(base_colors)]
        color = tuple(int(np.clip(c + int(rng.integers(-28, 28)), 0, 255)) for c in base)

        objects.append(
            SimObject(
                obj_id=i,
                label=label,
                shape=shape,
                color=color,
                center=np.array([x, base_height, z], dtype=np.float32),
                base_height=base_height,
                vel=vel,
                size=size,
                phase=float(rng.uniform(0.0, 2.0 * math.pi)),
            )
        )
    return objects


def step_objects(objects: List[SimObject], dt: float, t: float) -> None:
    for o in objects:
        o.center[0] += o.vel[0] * dt
        o.center[2] += o.vel[2] * dt

        if o.center[0] < -8.0:
            o.center[0] = -8.0
            o.vel[0] *= -1.0
        if o.center[0] > 8.0:
            o.center[0] = 8.0
            o.vel[0] *= -1.0

        if o.center[2] < 6.0:
            o.center[2] = 6.0
            o.vel[2] *= -1.0
        if o.center[2] > 22.0:
            o.center[2] = 22.0
            o.vel[2] *= -1.0

        if o.shape == "drone":
            o.center[1] = o.base_height + 0.45 * math.sin(1.6 * t + o.phase)
            vy = 0.45 * 1.6 * math.cos(1.6 * t + o.phase)
        else:
            o.center[1] = o.base_height
            vy = 0.0

        o.current_vel = np.array([o.vel[0], vy, o.vel[2]], dtype=np.float32)


def render_frame(
    objects: List[SimObject],
    t: float,
    frame_id: int,
    cam: Dict[str, float],
    cfg: DemoConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    w, h = cfg.width, cfg.height
    horizon = int(cam["cy"] + 4.0 * math.sin(0.7 * t))
    horizon = max(5, min(h - 5, horizon))

    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), np.inf, dtype=np.float32)

    # Sky
    if horizon > 0:
        sky_top = np.array([185, 155, 105], dtype=np.float32)
        sky_bottom = np.array([225, 195, 155], dtype=np.float32)
        for v in range(horizon):
            alpha = float(v) / max(1.0, float(horizon - 1))
            bgr[v, :] = (sky_top * (1.0 - alpha) + sky_bottom * alpha).astype(np.uint8)

    # Ground depth and color
    if horizon < h:
        rows = np.arange(horizon, h, dtype=np.float32)
        denom = rows - float(horizon) + 1e-3
        z = (cam["fy"] * cam["height"] / denom).astype(np.float32)
        depth[horizon:, :] = z[:, None]

        intensity = np.clip(1.0 - z / max(cfg.max_range, 1e-3), 0.0, 1.0)
        ground = np.zeros((len(rows), w, 3), dtype=np.uint8)
        ground[:, :, 0] = np.clip(55 + 55 * intensity[:, None], 0, 255).astype(np.uint8)
        ground[:, :, 1] = np.clip(85 + 60 * intensity[:, None], 0, 255).astype(np.uint8)
        ground[:, :, 2] = np.clip(55 + 45 * intensity[:, None], 0, 255).astype(np.uint8)
        bgr[horizon:, :] = ground

    # Horizon and simple lane cues
    cv2.line(bgr, (0, horizon), (w, horizon), (105, 145, 165), 1)
    for gx in range(-6, 7, 3):
        p1 = project_point(np.array([gx, 0.0, 6.0], dtype=np.float32), cam)
        p2 = project_point(np.array([gx, 0.0, 22.0], dtype=np.float32), cam)
        if p1 is not None and p2 is not None:
            cv2.line(
                bgr,
                (int(p1[0]), int(p1[1])),
                (int(p2[0]), int(p2[1])),
                (80, 110, 95),
                1,
            )

    gt = []
    sorted_objs = sorted(objects, key=lambda o: float(o.center[2]), reverse=True)

    for o in sorted_objs:
        proj = project_point(o.center, cam)
        if proj is None:
            gt.append(
                {
                    "gt_id": int(o.obj_id),
                    "label": o.label,
                    "bbox": [0, 0, 0, 0],
                    "center": o.center.astype(np.float32).tolist(),
                    "vel": o.current_vel.astype(np.float32).tolist(),
                    "visible": False,
                    "depth": 0.0,
                }
            )
            continue

        u, v, Z = proj
        if Z <= 0.5 or Z > cfg.max_range + 4.0:
            gt.append(
                {
                    "gt_id": int(o.obj_id),
                    "label": o.label,
                    "bbox": [0, 0, 0, 0],
                    "center": o.center.astype(np.float32).tolist(),
                    "vel": o.current_vel.astype(np.float32).tolist(),
                    "visible": False,
                    "depth": float(Z),
                }
            )
            continue

        # Shadow
        ground_proj = project_point(np.array([o.center[0], 0.0, o.center[2]], dtype=np.float32), cam)
        if ground_proj is not None:
            gu, gv, gZ = ground_proj
            shadow_w = int(max(4, cam["fx"] * float(o.size[0]) / max(gZ, 0.5)))
            shadow_h = int(max(2, shadow_w * 0.28))
            if 0 <= int(gu) < w and 0 <= int(gv) < h:
                cv2.ellipse(bgr, (int(gu), int(gv)), (shadow_w, shadow_h), 0, 0, 360, (45, 45, 45), -1)

        w_px = int(max(4, cam["fx"] * float(o.size[0]) / Z))
        h_px = int(max(4, cam["fy"] * float(o.size[1]) / Z))
        front_depth = max(0.3, float(Z) - float(o.size[2]) * 0.5)

        bbox = [0, 0, 0, 0]
        visible = False

        if o.shape in ["box", "cylinder"]:
            x1 = int(u - w_px / 2)
            y1 = int(v - h_px / 2)
            x2 = int(u + w_px / 2)
            y2 = int(v + h_px / 2)

            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)

            if x2c > x1c and y2c > y1c:
                depth[y1c:y2c, x1c:x2c] = front_depth
                cv2.rectangle(bgr, (x1c, y1c), (x2c, y2c), o.color, -1)

                if o.shape == "cylinder":
                    axes = (max(1, int(w_px / 2)), max(1, int(h_px / 8)))
                    lighten = tuple(int(min(255, c + 45)) for c in o.color)
                    cv2.ellipse(
                        bgr,
                        (int(u), max(0, y1c)),
                        axes,
                        0,
                        0,
                        360,
                        lighten,
                        -1,
                    )

                cv2.rectangle(bgr, (x1c, y1c), (x2c, y2c), (25, 25, 25), 1)
                bbox = [x1c, y1c, x2c, y2c]
                area = (x2c - x1c) * (y2c - y1c)
                visible = area > 120 and (x2c - x1c) > 4 and (y2c - y1c) > 4

        elif o.shape == "drone":
            rad = int(max(3, min(w_px, h_px) / 2))
            center = (int(u), int(v))
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, center, rad, 255, -1)
            idx = mask > 0
            bgr[idx] = o.color
            depth[idx] = front_depth

            cv2.circle(bgr, center, rad, (20, 20, 20), 1)
            cv2.line(
                bgr,
                (center[0] - rad * 2, center[1]),
                (center[0] + rad * 2, center[1]),
                (30, 30, 30),
                2,
            )
            cv2.circle(bgr, center, max(1, rad // 4), (225, 225, 225), -1)

            x1c = max(0, int(u - rad))
            y1c = max(0, int(v - rad))
            x2c = min(w, int(u + rad))
            y2c = min(h, int(v + rad))
            bbox = [x1c, y1c, x2c, y2c]
            area = int(math.pi * rad * rad)
            visible = area > 100 and rad > 3

        gt.append(
            {
                "gt_id": int(o.obj_id),
                "label": o.label,
                "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                "center": o.center.astype(np.float32).tolist(),
                "vel": o.current_vel.astype(np.float32).tolist(),
                "visible": bool(visible),
                "depth": float(Z),
            }
        )

    # Sensor noise
    if cfg.noise > 0:
        noise_img = rng.normal(0.0, 7.0 * cfg.noise, bgr.shape).astype(np.int16)
        bgr = np.clip(bgr.astype(np.int16) + noise_img, 0, 255).astype(np.uint8)

        valid_noise = np.isfinite(depth) & (depth > 0.2) & (depth < cfg.max_range)
        count = int(np.count_nonzero(valid_noise))
        if count > 0:
            depth_noise = rng.normal(0.0, 0.015 * cfg.noise, size=count).astype(np.float32)
            depth[valid_noise] += depth_noise * depth[valid_noise]

    # Deterministic simulated sensor fault
    if cfg.simulate_faults and frame_id % 151 == 137:
        y0 = int(rng.integers(0, max(1, h // 2)))
        y1 = min(h, y0 + max(5, h // 3))
        if y1 > y0:
            depth[y0:y1, :] = np.inf
            bgr[y0:y1, :] = rng.integers(0, 30, size=(y1 - y0, w, 3), dtype=np.uint8)

    return bgr, depth, gt


# ============================================================
# Perception: depth-based object extraction
# ============================================================

def detect_objects(
    bgr: np.ndarray,
    depth: np.ndarray,
    cam: Dict[str, float],
    cfg: DemoConfig,
) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, Any]]:
    h, w = depth.shape
    if h < 60 or w < 80:
        raise ValueError("Insufficient resolution for reliable 4D perception.")

    valid = np.isfinite(depth) & (depth > 0.2) & (depth < cfg.max_range)
    valid_ratio = float(np.mean(valid))

    if valid_ratio < 0.05:
        return [], np.zeros((h, w), dtype=np.uint8), {"sensor_fault": True, "valid_ratio": valid_ratio}

    depth_filled = np.where(valid, depth, cfg.max_range)
    row_ground = np.percentile(depth_filled, 75, axis=1).astype(np.float32)
    margin = np.maximum(0.15 * row_ground, 0.25)
    threshold = (row_ground - margin)[:, None]

    fg = valid & (depth < threshold)
    fg_u8 = (fg.astype(np.uint8)) * 255

    kernel = np.ones((3, 3), dtype=np.uint8)
    fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_u8, connectivity=8)
    detections = []
    min_area = max(25, int(0.0007 * w * h))

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])

        if bw < 3 or bh < 3:
            continue

        mask = (labels[y : y + bh, x : x + bw] == i).astype(np.uint8)
        depth_roi = depth[y : y + bh, x : x + bw]
        vals = depth_roi[mask > 0]
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            continue

        z_med = float(np.median(vals))
        if not np.isfinite(z_med) or z_med <= 0.2:
            continue

        z_std = float(np.std(vals))
        u0, v0 = float(centroids[i][0]), float(centroids[i][1])

        X = (u0 - cam["cx"]) * z_med / cam["fx"]
        Ycam = (v0 - cam["cy"]) * z_med / cam["fy"]
        world_y = cam["height"] - Ycam

        width_m = bw * z_med / cam["fx"]
        height_m = bh * z_med / cam["fy"]

        area_score = min(1.0, area / max(1.0, 0.004 * w * h))
        depth_score = max(0.0, 1.0 - z_std / (0.20 * z_med + 1e-3))
        border_score = 0.65 if (x < 2 or y < 2 or x + bw >= w - 2 or y + bh >= h - 2) else 1.0
        res_score = min(1.0, (bw * bh) / 500.0)
        range_score = 1.0 if z_med < cfg.max_range * 0.85 else 0.55

        conf = float(
            np.clip(
                0.32 * area_score
                + 0.28 * depth_score
                + 0.12 * border_score
                + 0.16 * res_score
                + 0.12 * range_score,
                0.0,
                1.0,
            )
        )

        detections.append(
            {
                "det_id": len(detections),
                "bbox": [x, y, x + bw, y + bh],
                "center": np.array([X, world_y, z_med], dtype=np.float32),
                "dims": np.array([width_m, height_m, max(0.25, 0.5 * width_m)], dtype=np.float32),
                "confidence": conf,
                "mean_depth": z_med,
                "depth_std": z_std,
                "area": area,
            }
        )

    return detections, fg_u8, {"sensor_fault": False, "valid_ratio": valid_ratio}


# ============================================================
# Temporal tracker
# ============================================================

class SimpleTracker:
    def __init__(self, max_missed: int = 8, dist_gate: float = 3.5):
        self.tracks: List[Dict[str, Any]] = []
        self.next_id = 1
        self.max_missed = max_missed
        self.dist_gate = dist_gate

    def update(
        self,
        detections: List[Dict[str, Any]],
        dt: float,
        timestamp: float,
        horizon_sec: float = 1.0,
    ) -> List[Dict[str, Any]]:
        for tr in self.tracks:
            tr["prev_pos"] = tr["pos"].copy()
            tr["pos"] = tr["pos"] + tr["vel"] * dt
            tr["age"] += dt
            tr["missed"] += 1
            tr["last_update"] = timestamp

        costs = []
        for ti, tr in enumerate(self.tracks):
            for di, det in enumerate(detections):
                d = float(np.linalg.norm(tr["pos"] - det["center"]))
                if d < self.dist_gate:
                    costs.append((d, ti, di))

        costs.sort(key=lambda x: x[0])
        matched_t = set()
        matched_d = set()

        for d, ti, di in costs:
            if ti in matched_t or di in matched_d:
                continue

            tr = self.tracks[ti]
            det = detections[di]

            matched_t.add(ti)
            matched_d.add(di)

            meas_vel = (det["center"] - tr["prev_pos"]) / max(dt, 1e-4)
            tr["vel"] = (0.65 * tr["vel"] + 0.35 * meas_vel).astype(np.float32)
            tr["pos"] = (0.45 * tr["pos"] + 0.55 * det["center"]).astype(np.float32)
            tr["missed"] = 0
            tr["hits"] += 1
            tr["confidence"] = float(np.clip(0.70 * tr["confidence"] + 0.30 * det["confidence"], 0.0, 1.0))
            tr["bbox"] = det["bbox"]
            tr["dims"] = det["dims"]
            tr["area"] = det["area"]
            tr["history"].append((timestamp, tr["pos"].copy()))
            if len(tr["history"]) > 120:
                tr["history"].pop(0)

        for di, det in enumerate(detections):
            if di in matched_d:
                continue

            tr = {
                "id": self.next_id,
                "pos": det["center"].copy(),
                "prev_pos": det["center"].copy(),
                "vel": np.zeros(3, dtype=np.float32),
                "confidence": float(det["confidence"]),
                "bbox": det["bbox"],
                "dims": det["dims"],
                "area": det["area"],
                "missed": 0,
                "hits": 1,
                "age": 0.0,
                "birth": timestamp,
                "last_update": timestamp,
                "history": [(timestamp, det["center"].copy())],
            }
            self.next_id += 1
            self.tracks.append(tr)

        self.tracks = [tr for tr in self.tracks if tr["missed"] <= self.max_missed]

        steps = max(1, int(horizon_sec / 0.25))
        active = []
        for tr in self.tracks:
            tr["future"] = [tr["pos"] + tr["vel"] * (k * 0.25) for k in range(1, steps + 1)]
            if tr["missed"] <= 1:
                active.append(tr)

        return active


# ============================================================
# 4D world model: sparse point memory + BEV
# ============================================================

class WorldModel:
    def __init__(self, max_points: int = 8000, ttl_sec: float = 4.0, rng: Optional[np.random.Generator] = None):
        self.points = deque()
        self.max_points = max_points
        self.ttl = ttl_sec
        self.rng = rng if rng is not None else np.random.default_rng()

    def update(self, depth: np.ndarray, fg_u8: np.ndarray, cam: Dict[str, float], timestamp: float) -> None:
        if fg_u8 is not None:
            pts = cv2.findNonZero(fg_u8)
            if pts is not None and len(pts) > 0:
                max_add = 160
                if len(pts) > max_add:
                    idx = self.rng.choice(len(pts), size=max_add, replace=False)
                    pts = pts[idx]

                for p in pts:
                    u = int(p[0][0])
                    v = int(p[0][1])
                    z = float(depth[v, u])
                    if not np.isfinite(z) or z <= 0.2:
                        continue

                    X = (u - cam["cx"]) * z / cam["fx"]
                    Ycam = (v - cam["cy"]) * z / cam["fy"]
                    world_y = cam["height"] - Ycam
                    self.points.append((float(X), float(world_y), float(z), float(timestamp)))

        while self.points and timestamp - self.points[0][3] > self.ttl:
            self.points.popleft()

        while len(self.points) > self.max_points:
            self.points.popleft()

    def get_bev(
        self,
        tracks: List[Dict[str, Any]],
        timestamp: float,
        low_thresh: float = 0.45,
        w: int = 180,
        h: int = 130,
        x_range: Tuple[float, float] = (-9.0, 9.0),
        z_range: Tuple[float, float] = (4.0, 24.0),
    ) -> np.ndarray:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (90, 90, 90), 1)

        for z_grid in range(int(z_range[0]), int(z_range[1]) + 1, 4):
            y = int(h - (z_grid - z_range[0]) / (z_range[1] - z_range[0]) * h)
            cv2.line(img, (0, y), (w, y), (40, 40, 40), 1)

        def to_bev(pos: np.ndarray) -> Optional[Tuple[int, int]]:
            x = float(pos[0])
            z = float(pos[2])
            if x < x_range[0] or x > x_range[1] or z < z_range[0] or z > z_range[1]:
                return None
            px = int((x - x_range[0]) / (x_range[1] - x_range[0]) * (w - 1))
            py = int(h - (z - z_range[0]) / (z_range[1] - z_range[0]) * (h - 1))
            return px, py

        for x, yw, z, t in self.points:
            if x < x_range[0] or x > x_range[1] or z < z_range[0] or z > z_range[1]:
                continue
            age = max(0.0, min(1.0, (timestamp - t) / max(self.ttl, 1e-3)))
            brightness = int(255 * (1.0 - age))
            px = int((x - x_range[0]) / (x_range[1] - x_range[0]) * (w - 1))
            py = int(h - (z - z_range[0]) / (z_range[1] - z_range[0]) * (h - 1))
            cv2.circle(img, (px, py), 1, (0, brightness // 3, brightness), -1)

        for tr in tracks:
            p = to_bev(tr["pos"])
            if p is None:
                continue

            conf = float(tr.get("confidence", 0.0))
            if conf < low_thresh:
                color = (0, 0, 255)
            elif conf < max(0.60, low_thresh + 0.15):
                color = (0, 165, 255)
            else:
                color = (0, 220, 0)

            cv2.circle(img, p, 3, color, -1)

            prev = p
            for fp in tr.get("future", []):
                fp_p = to_bev(fp)
                if fp_p is not None and prev is not None:
                    cv2.line(img, prev, fp_p, (0, 255, 255), 1)
                prev = fp_p

            cv2.putText(
                img,
                str(tr.get("id", "?")),
                (p[0] + 4, p[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.circle(img, (w // 2, h - 6), 3, (0, 255, 255), -1)
        cv2.putText(img, "BEV", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
        return img


# ============================================================
# Video recorder
# ============================================================

class VideoRecorder:
    def __init__(self, fps: int, width: int, height: int):
        self.width = width
        self.height = height
        self.fps = max(1, int(fps))
        self.tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        self.path = self.tmp.name
        self.tmp.close()

        self.mode = None
        self.writer = None
        self.mime = "video/mp4"

        if IMAGEIO_AVAILABLE:
            try:
                self.writer = imageio.get_writer(
                    self.path,
                    fps=self.fps,
                    codec="libx264",
                    quality=8,
                    macro_block_size=None,
                )
                self.mode = "imageio"
                self.mime = "video/mp4"
                return
            except Exception:
                self.writer = None

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.path, fourcc, self.fps, (self.width, self.height))
        if self.writer.isOpened():
            self.mode = "cv2"
            self.mime = "video/mp4"
        else:
            self.path = self.path.replace(".mp4", ".avi")
            self.writer = cv2.VideoWriter(
                self.path,
                cv2.VideoWriter_fourcc(*"MJPG"),
                self.fps,
                (self.width, self.height),
            )
            self.mode = "cv2"
            self.mime = "video/avi"

    def add(self, bgr: np.ndarray) -> None:
        if bgr.shape[1] != self.width or bgr.shape[0] != self.height:
            bgr = cv2.resize(bgr, (self.width, self.height))

        if self.mode == "imageio":
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.writer.append_data(rgb)
        else:
            self.writer.write(bgr)

    def close(self) -> bytes:
        if self.mode == "imageio":
            self.writer.close()
        elif self.writer is not None:
            self.writer.release()

        with open(self.path, "rb") as f:
            data = f.read()

        try:
            os.unlink(self.path)
        except Exception:
            pass

        return data


# ============================================================
# Evaluation
# ============================================================

def iou_2d(boxA: List[int], boxB: List[int]) -> float:
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter

    return float(inter / union) if union > 0 else 0.0


def evaluate_tracks(
    gt_objects: List[Dict[str, Any]],
    tracks: List[Dict[str, Any]],
    prev_gt_map: Dict[int, int],
    cfg: DemoConfig,
) -> Tuple[Dict[str, Any], Dict[int, int], int]:
    gt_vis = [g for g in gt_objects if g.get("visible", False)]
    costs = []

    for gi, g in enumerate(gt_vis):
        gc = np.array(g["center"], dtype=np.float32)
        for ti, tr in enumerate(tracks):
            d = float(np.linalg.norm(gc - tr["pos"]))
            if d < cfg.eval_match_gate:
                costs.append((d, gi, ti))

    costs.sort(key=lambda x: x[0])
    matched_g = set()
    matched_t = set()
    matches = []

    for d, gi, ti in costs:
        if gi in matched_g or ti in matched_t:
            continue
        matched_g.add(gi)
        matched_t.add(ti)
        matches.append((gi, ti, d))

    tp = len(matches)
    fp = len(tracks) - tp
    fn = len(gt_vis) - tp

    center_errors = []
    depth_errors = []
    vel_errors = []
    ious = []
    new_gt_map = {}
    idsw = 0

    for gi, ti, d in matches:
        g = gt_vis[gi]
        tr = tracks[ti]

        center_errors.append(float(d))
        depth_errors.append(abs(float(g["center"][2]) - float(tr["pos"][2])))

        gv = np.array(g["vel"], dtype=np.float32)
        tv = np.asarray(tr["vel"], dtype=np.float32)
        vel_errors.append(float(np.linalg.norm(gv - tv)))

        if g.get("bbox") is not None and tr.get("bbox") is not None:
            ious.append(iou_2d(g["bbox"], tr["bbox"]))

        prev = prev_gt_map.get(int(g["gt_id"]))
        if prev is not None and prev != int(tr["id"]):
            idsw += 1
        new_gt_map[int(g["gt_id"])] = int(tr["id"])

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)

    metrics = {
        "gt_count": len(gt_vis),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "center_errors": center_errors,
        "depth_errors": depth_errors,
        "vel_errors": vel_errors,
        "ious": ious,
    }

    return metrics, new_gt_map, idsw


# ============================================================
# Annotation / visualization
# ============================================================

def put_picture(frame: np.ndarray, img: np.ndarray, x: int, y: int, alpha: float = 0.70) -> None:
    ih, iw = img.shape[:2]
    fh, fw = frame.shape[:2]
    if y + ih > fh or x + iw > fw:
        return

    roi = frame[y : y + ih, x : x + iw]
    if roi.shape[:2] != (ih, iw):
        return

    blended = cv2.addWeighted(img, alpha, roi, 1.0 - alpha, 0)
    frame[y : y + ih, x : x + iw] = blended


def annotate_demo(
    bgr: np.ndarray,
    depth: np.ndarray,
    gt: List[Dict[str, Any]],
    detections: List[Dict[str, Any]],
    tracks: List[Dict[str, Any]],
    world: WorldModel,
    cam: Dict[str, float],
    cfg: DemoConfig,
    metric: Dict[str, Any],
    timestamp: float,
    frame_id: int,
) -> np.ndarray:
    frame = bgr.copy()
    h, w = frame.shape[:2]

    # Sparse reconstructed point overlay
    recent_points = list(world.points)[-220:]
    for x, yw, z, t in recent_points:
        pr = project_point(np.array([x, yw, z], dtype=np.float32), cam)
        if pr is None:
            continue
        u, v, Z = pr
        ui, vi = int(u), int(v)
        if 0 <= ui < w and 0 <= vi < h:
            age = max(0.0, min(1.0, (timestamp - t) / max(world.ttl, 1e-3)))
            g = int(70 + 160 * (1.0 - age))
            b = int(90 + 165 * (1.0 - age))
            frame[vi, ui] = (0, g, b)

    # Optional ground-truth overlay
    if cfg.show_gt:
        for g in gt:
            if not g.get("visible", False):
                continue
            x1, y1, x2, y2 = map(int, g["bbox"])
            if x2 > x1 and y2 > y1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                cv2.putText(
                    frame,
                    f"GT{g['gt_id']}",
                    (x1, max(10, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.30,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    # Track overlay
    for tr in tracks:
        conf = float(tr.get("confidence", 0.0))

        if conf < cfg.low_conf_threshold:
            color = (0, 0, 255)
            tag = "LOW"
        elif conf < max(0.60, cfg.low_conf_threshold + 0.15):
            color = (0, 165, 255)
            tag = "MED"
        else:
            color = (0, 220, 0)
            tag = "OK"

        bbox = tr.get("bbox")
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            if x2 > x1 and y2 > y1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID{tr['id']} {conf:.2f} {tag}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.34,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        p1 = project_point(tr["pos"], cam)
        p2 = project_point(tr["pos"] + tr["vel"] * 0.5, cam)
        if p1 is not None and p2 is not None:
            cv2.arrowedLine(
                frame,
                (int(p1[0]), int(p1[1])),
                (int(p2[0]), int(p2[1])),
                (255, 255, 0),
                1,
                tipLength=0.22,
            )

        prev = p1
        for fp in tr.get("future", []):
            pp = project_point(fp, cam)
            if pp is not None and prev is not None:
                cv2.line(
                    frame,
                    (int(prev[0]), int(prev[1])),
                    (int(pp[0]), int(pp[1])),
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            prev = pp

    # BEV picture-in-picture
    bev = world.get_bev(tracks, timestamp, cfg.low_conf_threshold)
    put_picture(frame, bev, 8, 8, 0.78)

    # HUD
    hud_h = 48
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - hud_h), (w, h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    line1 = (
        f"t={timestamp:5.2f}s  frame={frame_id:04d}  "
        f"FPS={metric.get('wall_fps', 0.0):4.1f}  latency={metric.get('latency_ms', 0.0):4.0f}ms"
    )
    line2 = (
        f"tracks={metric.get('active_tracks', 0):02d}  low={metric.get('low_conf_count', 0):02d}  "
        f"depth_mae={metric.get('depth_mae', 0.0):.2f}  MOTA={metric.get('mota', 0.0):.2f}"
    )

    cv2.putText(frame, line1, (8, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, line2, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    if metric.get("sensor_fault", False):
        cv2.putText(
            frame,
            "SENSOR FAULT - LOW DEPTH COVERAGE",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return frame


def annotate_uploaded(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    tracks: List[Dict[str, Any]],
    cfg: DemoConfig,
    metric: Dict[str, Any],
    timestamp: float,
    frame_id: int,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    for tr in tracks:
        conf = float(tr.get("confidence", 0.0))
        if conf < cfg.low_conf_threshold:
            color = (0, 0, 255)
            tag = "LOW"
        elif conf < max(0.60, cfg.low_conf_threshold + 0.15):
            color = (0, 165, 255)
            tag = "MED"
        else:
            color = (0, 220, 0)
            tag = "OK"

        bbox = tr.get("bbox")
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            if x2 > x1 and y2 > y1:
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                label = f"ID{tr['id']} {conf:.2f} {tag}"
                cv2.putText(
                    out,
                    label,
                    (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    hud_h = 48
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - hud_h), (w, h), (20, 20, 20), -1)
    out = cv2.addWeighted(overlay, 0.65, out, 0.35, 0)

    line1 = (
        f"t={timestamp:5.2f}s  frame={frame_id:04d}  "
        f"FPS={metric.get('wall_fps', 0.0):4.1f}  latency={metric.get('latency_ms', 0.0):4.0f}ms"
    )
    line2 = (
        f"tracks={metric.get('active_tracks', 0):02d}  dets={metric.get('detections', 0):02d}  "
        f"low={metric.get('low_conf_count', 0):02d}  MONOCULAR/PSEUDO-DEPTH"
    )

    cv2.putText(out, line1, (8, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, line2, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        out,
        "LOCAL PROCESSING - NO IDENTITY RECOGNITION",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )

    return out


# ============================================================
# Main deterministic demo pipeline
# ============================================================

def run_demo_pipeline(cfg: DemoConfig, progress_callback=None) -> Dict[str, Any]:
    run_id = uuid.uuid4().hex[:10]
    started = datetime.now(timezone.utc).isoformat()

    rng = np.random.default_rng(cfg.seed)
    objects = create_objects(cfg.num_objects, rng)
    cam = make_camera(cfg.width, cfg.height)
    tracker = SimpleTracker()
    world = WorldModel(rng=np.random.default_rng(cfg.seed + 1))

    recorder = VideoRecorder(cfg.fps, cfg.width, cfg.height)

    metrics = []
    events = []
    track_rows = []
    det_rows = []
    low_rows = []

    total_gt = 0
    total_fp = 0
    total_fn = 0
    total_idsw = 0
    gt_track_map: Dict[int, int] = {}
    prev_track_ids = set()

    num_frames = max(1, int(cfg.duration_sec * cfg.fps))
    dt = 1.0 / float(cfg.fps)
    last_wall = time.perf_counter()

    for frame_id in range(num_frames):
        timestamp = frame_id * dt

        try:
            step_objects(objects, dt, timestamp)
            bgr, depth, gt = render_frame(objects, timestamp, frame_id, cam, cfg, rng)

            start = time.perf_counter()
            raw_dets, fg, meta = detect_objects(bgr, depth, cam, cfg)

            dets = []
            suppressed_low = 0

            for det in raw_dets:
                bbox = det["bbox"]
                bbox_str = f"{bbox[0]}:{bbox[1]}:{bbox[2]}:{bbox[3]}"
                status = "accepted" if det["confidence"] >= cfg.min_conf else "suppressed"

                det_rows.append(
                    {
                        "timestamp": timestamp,
                        "frame_id": frame_id,
                        "confidence": float(det["confidence"]),
                        "mean_depth": float(det["mean_depth"]),
                        "center_x": float(det["center"][0]),
                        "center_y": float(det["center"][1]),
                        "center_z": float(det["center"][2]),
                        "area": int(det["area"]),
                        "bbox": bbox_str,
                        "status": status,
                        "evidence": f"frame_{frame_id:04d}",
                    }
                )

                if det["confidence"] < cfg.min_conf:
                    if det["confidence"] < cfg.low_conf_threshold:
                        suppressed_low += 1
                        low_rows.append(
                            {
                                "timestamp": timestamp,
                                "frame_id": frame_id,
                                "confidence": float(det["confidence"]),
                                "depth": float(det["mean_depth"]),
                                "center_x": float(det["center"][0]),
                                "center_y": float(det["center"][1]),
                                "center_z": float(det["center"][2]),
                                "status": "suppressed_below_min_conf",
                                "evidence": f"frame_{frame_id:04d}",
                            }
                        )
                        if len(events) < 3000:
                            events.append(
                                {
                                    "timestamp": timestamp,
                                    "type": "low_confidence_suppressed",
                                    "message": f"Detection suppressed, conf={det['confidence']:.2f}",
                                    "evidence": f"frame_{frame_id:04d}",
                                }
                            )
                    continue

                dets.append(det)

            tracks = tracker.update(dets, dt, timestamp, cfg.prediction_horizon)
            world.update(depth, fg, cam, timestamp)
            latency = time.perf_counter() - start

            evalm, gt_track_map, idsw = evaluate_tracks(gt, tracks, gt_track_map, cfg)
            total_gt += evalm["gt_count"]
            total_fp += evalm["fp"]
            total_fn += evalm["fn"]
            total_idsw += idsw

            mota = 1.0 - (total_fn + total_fp + total_idsw) / max(1, total_gt)
            mota = float(np.clip(mota, -1.0, 1.0))

            low_conf_count = sum(1 for d in dets if d["confidence"] < cfg.low_conf_threshold)

            for det in dets:
                if det["confidence"] < cfg.low_conf_threshold:
                    low_rows.append(
                        {
                            "timestamp": timestamp,
                            "frame_id": frame_id,
                            "confidence": float(det["confidence"]),
                            "depth": float(det["mean_depth"]),
                            "center_x": float(det["center"][0]),
                            "center_y": float(det["center"][1]),
                            "center_z": float(det["center"][2]),
                            "status": "routed_for_review",
                            "evidence": f"frame_{frame_id:04d}",
                        }
                    )
                    if len(events) < 3000:
                        events.append(
                            {
                                "timestamp": timestamp,
                                "type": "low_confidence_review",
                                "message": f"Low-confidence detection conf={det['confidence']:.2f} routed for review",
                                "evidence": f"frame_{frame_id:04d}",
                            }
                        )

            current_ids = {int(tr["id"]) for tr in tracks}
            new_ids = current_ids - prev_track_ids
            lost_ids = prev_track_ids - current_ids

            for nid in new_ids:
                if len(events) < 3000:
                    events.append(
                        {
                            "timestamp": timestamp,
                            "type": "new_track",
                            "message": f"Track {nid} created",
                            "evidence": f"frame_{frame_id:04d}",
                        }
                    )

            for lid in lost_ids:
                if len(events) < 3000:
                    events.append(
                        {
                            "timestamp": timestamp,
                            "type": "track_lost",
                            "message": f"Track {lid} lost",
                            "evidence": f"frame_{frame_id:04d}",
                        }
                    )

            prev_track_ids = current_ids

            for tr in tracks:
                bbox = tr.get("bbox")
                bbox_str = f"{bbox[0]}:{bbox[1]}:{bbox[2]}:{bbox[3]}" if bbox is not None else ""
                track_rows.append(
                    {
                        "timestamp": timestamp,
                        "frame_id": frame_id,
                        "track_id": int(tr["id"]),
                        "x": float(tr["pos"][0]),
                        "y": float(tr["pos"][1]),
                        "z": float(tr["pos"][2]),
                        "vx": float(tr["vel"][0]),
                        "vy": float(tr["vel"][1]),
                        "vz": float(tr["vel"][2]),
                        "confidence": float(tr["confidence"]),
                        "bbox": bbox_str,
                        "evidence": f"frame_{frame_id:04d}",
                    }
                )

            wall_dt = time.perf_counter() - last_wall
            last_wall = time.perf_counter()

            metric = {
                "timestamp": timestamp,
                "frame_id": frame_id,
                "latency_ms": float(latency * 1000.0),
                "processing_fps": float(1.0 / max(latency, 1e-6)),
                "wall_fps": float(1.0 / max(wall_dt, 1e-6)),
                "active_tracks": len(tracks),
                "detections": len(dets),
                "low_conf_count": int(low_conf_count),
                "sensor_fault": bool(meta.get("sensor_fault", False)),
                "precision": float(evalm["precision"]),
                "recall": float(evalm["recall"]),
                "depth_mae": float(np.mean(evalm["depth_errors"])) if evalm["depth_errors"] else 0.0,
                "center_mae": float(np.mean(evalm["center_errors"])) if evalm["center_errors"] else 0.0,
                "vel_mae": float(np.mean(evalm["vel_errors"])) if evalm["vel_errors"] else 0.0,
                "mean_iou": float(np.mean(evalm["ious"])) if evalm["ious"] else 0.0,
                "mota": mota,
            }
            metrics.append(metric)

            if meta.get("sensor_fault", False) and len(events) < 3000:
                events.append(
                    {
                        "timestamp": timestamp,
                        "type": "sensor_fault",
                        "message": "Depth coverage below reliability threshold",
                        "evidence": f"frame_{frame_id:04d}",
                    }
                )

            annotated = annotate_demo(
                bgr=bgr,
                depth=depth,
                gt=gt,
                detections=dets,
                tracks=tracks,
                world=world,
                cam=cam,
                cfg=cfg,
                metric=metric,
                timestamp=timestamp,
                frame_id=frame_id,
            )
            recorder.add(annotated)

            if progress_callback is not None and frame_id % max(1, num_frames // 50) == 0:
                progress_callback(float(frame_id) / float(num_frames))

        except Exception as exc:
            if len(events) < 3000:
                events.append(
                    {
                        "timestamp": timestamp,
                        "type": "pipeline_error",
                        "message": str(exc),
                        "evidence": f"frame_{frame_id:04d}",
                    }
                )

            black = np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8)
            cv2.putText(
                black,
                f"PIPELINE ERROR @ frame {frame_id}",
                (10, cfg.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            recorder.add(black)

    if progress_callback is not None:
        progress_callback(1.0)

    video_bytes = recorder.close()

    metrics_df = pd.DataFrame(metrics)
    events_df = pd.DataFrame(events)
    track_df = pd.DataFrame(track_rows)
    low_df = pd.DataFrame(low_rows)
    det_df = pd.DataFrame(det_rows)

    if metrics_df.empty:
        metrics_df = pd.DataFrame(
            columns=[
                "timestamp",
                "frame_id",
                "latency_ms",
                "processing_fps",
                "wall_fps",
                "active_tracks",
                "detections",
                "low_conf_count",
                "sensor_fault",
                "precision",
                "recall",
                "depth_mae",
                "center_mae",
                "vel_mae",
                "mean_iou",
                "mota",
            ]
        )

    def df_mean(col: str) -> float:
        if not metrics_df.empty and col in metrics_df.columns:
            return float(metrics_df[col].mean())
        return 0.0

    summary = {
        "run_id": run_id,
        "mode": "Deterministic demo",
        "started_at": started,
        "duration_sec": float(cfg.duration_sec),
        "frames": int(len(metrics_df)),
        "objects_simulated": int(cfg.num_objects),
        "avg_latency_ms": df_mean("latency_ms"),
        "avg_processing_fps": df_mean("processing_fps"),
        "avg_wall_fps": df_mean("wall_fps"),
        "final_mota": float(metrics_df["mota"].iloc[-1]) if not metrics_df.empty and "mota" in metrics_df else 0.0,
        "mean_depth_mae": df_mean("depth_mae"),
        "mean_center_mae": df_mean("center_mae"),
        "mean_vel_mae": df_mean("vel_mae"),
        "mean_iou": df_mean("mean_iou"),
        "low_conf_reviews": int(len(low_df)),
        "events": int(len(events_df)),
        "tracks_seen": int(track_df["track_id"].nunique()) if not track_df.empty else 0,
    }

    return {
        "video_bytes": video_bytes,
        "mime": recorder.mime,
        "run_id": run_id,
        "started_at": started,
        "summary": summary,
        "metrics_df": metrics_df,
        "events_df": events_df,
        "track_df": track_df,
        "low_conf_df": low_df,
        "det_df": det_df,
    }


# ============================================================
# Uploaded video / webcam pipeline
# ============================================================

def detect_from_mask(mask: np.ndarray, cam: Dict[str, float], cfg: DemoConfig) -> List[Dict[str, Any]]:
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    min_area = max(25, int(0.0008 * w * h))

    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 4 or bh < 4:
            continue

        # Pseudo-depth from assumed object height.
        z = float(np.clip(cam["fy"] * 1.35 / max(bh, 8), 1.0, cfg.max_range))
        u = float(x + bw / 2)
        X = (u - cam["cx"]) * z / cam["fx"]
        world_y = 0.75

        area_score = min(1.0, area / max(1.0, 0.004 * w * h))
        res_score = min(1.0, (bw * bh) / 500.0)
        aspect_score = min(1.0, float(bh) / max(1.0, float(bw)))
        conf = float(np.clip(0.55 * area_score + 0.25 * res_score + 0.20 * aspect_score, 0.0, 1.0))

        detections.append(
            {
                "det_id": len(detections),
                "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
                "center": np.array([X, world_y, z], dtype=np.float32),
                "dims": np.array([bw * z / cam["fx"], bh * z / cam["fy"], 0.5], dtype=np.float32),
                "confidence": conf,
                "mean_depth": z,
                "depth_std": 0.0,
                "area": int(area),
            }
        )

    return detections


def run_cv_source(
    cfg: DemoConfig,
    source: Any,
    source_name: str,
    progress_callback=None,
) -> Dict[str, Any]:
    run_id = uuid.uuid4().hex[:10]
    started = datetime.now(timezone.utc).isoformat()

    cam = make_camera(cfg.width, cfg.height)
    tracker = SimpleTracker()
    recorder = VideoRecorder(cfg.fps, cfg.width, cfg.height)

    if source == 0:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"{source_name} could not be opened. Check camera/video availability.")

    bg = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=24, detectShadows=False)
    kernel = np.ones((3, 3), dtype=np.uint8)

    metrics = []
    events = []
    track_rows = []
    low_rows = []
    det_rows = []

    num_frames = max(1, int(cfg.duration_sec * cfg.fps))
    dt = 1.0 / float(cfg.fps)
    frame_id = 0
    last_wall = time.perf_counter()
    live = source == 0

    while frame_id < num_frames:
        loop_start = time.perf_counter()
        ok, frame = cap.read()

        if not ok:
            if frame_id == 0:
                cap.release()
                raise RuntimeError(f"No frames received from {source_name}.")
            if len(events) < 3000:
                events.append(
                    {
                        "timestamp": frame_id * dt,
                        "type": "frame_error",
                        "message": "Could not read frame; stopping source capture",
                        "evidence": f"frame_{frame_id:04d}",
                    }
                )
            break

        frame = cv2.resize(frame, (cfg.width, cfg.height))

        start = time.perf_counter()
        fg_mask = bg.apply(frame)
        fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        raw_dets = detect_from_mask(fg_mask, cam, cfg)
        dets = []

        for det in raw_dets:
            bbox = det["bbox"]
            bbox_str = f"{bbox[0]}:{bbox[1]}:{bbox[2]}:{bbox[3]}"
            status = "accepted" if det["confidence"] >= cfg.min_conf else "suppressed"

            det_rows.append(
                {
                    "timestamp": frame_id * dt,
                    "frame_id": frame_id,
                    "confidence": float(det["confidence"]),
                    "mean_depth": float(det["mean_depth"]),
                    "center_x": float(det["center"][0]),
                    "center_y": float(det["center"][1]),
                    "center_z": float(det["center"][2]),
                    "area": int(det["area"]),
                    "bbox": bbox_str,
                    "status": status,
                    "evidence": f"frame_{frame_id:04d}",
                }
            )

            if det["confidence"] < cfg.min_conf:
                if det["confidence"] < cfg.low_conf_threshold:
                    low_rows.append(
                        {
                            "timestamp": frame_id * dt,
                            "frame_id": frame_id,
                            "confidence": float(det["confidence"]),
                            "depth": float(det["mean_depth"]),
                            "center_x": float(det["center"][0]),
                            "center_y": float(det["center"][1]),
                            "center_z": float(det["center"][2]),
                            "status": "suppressed_below_min_conf",
                            "evidence": f"frame_{frame_id:04d}",
                        }
                    )
                continue

            dets.append(det)

        tracks = tracker.update(dets, dt, frame_id * dt, cfg.prediction_horizon)
        latency = time.perf_counter() - start

        low_conf_count = sum(1 for d in dets if d["confidence"] < cfg.low_conf_threshold)

        for det in dets:
            if det["confidence"] < cfg.low_conf_threshold:
                low_rows.append(
                    {
                        "timestamp": frame_id * dt,
                        "frame_id": frame_id,
                        "confidence": float(det["confidence"]),
                        "depth": float(det["mean_depth"]),
                        "center_x": float(det["center"][0]),
                        "center_y": float(det["center"][1]),
                        "center_z": float(det["center"][2]),
                        "status": "routed_for_review",
                        "evidence": f"frame_{frame_id:04d}",
                    }
                )
                if len(events) < 3000:
                    events.append(
                        {
                            "timestamp": frame_id * dt,
                            "type": "low_confidence_review",
                            "message": f"Low-confidence detection conf={det['confidence']:.2f} routed for review",
                            "evidence": f"frame_{frame_id:04d}",
                        }
                    )

        for tr in tracks:
            bbox = tr.get("bbox")
            bbox_str = f"{bbox[0]}:{bbox[1]}:{bbox[2]}:{bbox[3]}" if bbox is not None else ""
            track_rows.append(
                {
                    "timestamp": frame_id * dt,
                    "frame_id": frame_id,
                    "track_id": int(tr["id"]),
                    "x": float(tr["pos"][0]),
                    "y": float(tr["pos"][1]),
                    "z": float(tr["pos"][2]),
                    "vx": float(tr["vel"][0]),
                    "vy": float(tr["vel"][1]),
                    "vz": float(tr["vel"][2]),
                    "confidence": float(tr["confidence"]),
                    "bbox": bbox_str,
                    "evidence": f"frame_{frame_id:04d}",
                }
            )

        wall_dt = time.perf_counter() - last_wall
        last_wall = time.perf_counter()

        metric = {
            "timestamp": frame_id * dt,
            "frame_id": frame_id,
            "latency_ms": float(latency * 1000.0),
            "processing_fps": float(1.0 / max(latency, 1e-6)),
            "wall_fps": float(1.0 / max(wall_dt, 1e-6)),
            "active_tracks": len(tracks),
            "detections": len(dets),
            "low_conf_count": int(low_conf_count),
            "sensor_fault": False,
            "precision": 0.0,
            "recall": 0.0,
            "depth_mae": 0.0,
            "center_mae": 0.0,
            "vel_mae": 0.0,
            "mean_iou": 0.0,
            "mota": 0.0,
        }
        metrics.append(metric)

        annotated = annotate_uploaded(
            frame=frame,
            detections=dets,
            tracks=tracks,
            cfg=cfg,
            metric=metric,
            timestamp=frame_id * dt,
            frame_id=frame_id,
        )
        recorder.add(annotated)

        frame_id += 1

        if live:
            elapsed = time.perf_counter() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

        if progress_callback is not None and frame_id % 5 == 0:
            progress_callback(float(frame_id) / float(num_frames))

    cap.release()
    video_bytes = recorder.close()

    metrics_df = pd.DataFrame(metrics)
    events_df = pd.DataFrame(events)
    track_df = pd.DataFrame(track_rows)
    low_df = pd.DataFrame(low_rows)
    det_df = pd.DataFrame(det_rows)

    def df_mean(col: str) -> float:
        if not metrics_df.empty and col in metrics_df.columns:
            return float(metrics_df[col].mean())
        return 0.0

    summary = {
        "run_id": run_id,
        "mode": source_name,
        "started_at": started,
        "duration_sec": float(frame_id / max(1, cfg.fps)),
        "frames": int(len(metrics_df)),
        "objects_simulated": 0,
        "avg_latency_ms": df_mean("latency_ms"),
        "avg_processing_fps": df_mean("processing_fps"),
        "avg_wall_fps": df_mean("wall_fps"),
        "final_mota": 0.0,
        "mean_depth_mae": 0.0,
        "mean_center_mae": 0.0,
        "mean_vel_mae": 0.0,
        "mean_iou": 0.0,
        "low_conf_reviews": int(len(low_df)),
        "events": int(len(events_df)),
        "tracks_seen": int(track_df["track_id"].nunique()) if not track_df.empty else 0,
    }

    return {
        "video_bytes": video_bytes,
        "mime": recorder.mime,
        "run_id": run_id,
        "started_at": started,
        "summary": summary,
        "metrics_df": metrics_df,
        "events_df": events_df,
        "track_df": track_df,
        "low_conf_df": low_df,
        "det_df": det_df,
    }


def process_pipeline(cfg: DemoConfig, uploaded_file, progress_callback=None) -> Dict[str, Any]:
    if cfg.mode == "Deterministic demo":
        return run_demo_pipeline(cfg, progress_callback)

    if cfg.mode == "Uploaded video":
        if uploaded_file is None:
            raise RuntimeError("Upload a video file first.")

        suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_file.getvalue())
        tmp.close()

        try:
            return run_cv_source(cfg, tmp.name, "Uploaded video", progress_callback)
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

    if cfg.mode == "Webcam":
        return run_cv_source(cfg, 0, "Webcam", progress_callback)

    raise RuntimeError("Unknown input mode.")


# ============================================================
# Streamlit UI
# ============================================================

st.title("🌍 4D WorldModel AI")
st.caption(
    "Spatiotemporal vision prototype: evolving 3D scene reconstruction, object tracking, motion prediction, "
    "uncertainty routing, and evaluation. No API key required."
)

with st.sidebar:
    st.subheader("⚙️ Control Plane")

    mode = st.selectbox(
        "Input mode",
        ["Deterministic demo", "Uploaded video", "Webcam"],
        help="Deterministic demo auto-generates a synthetic 4D scene. Uploaded video/webcam use monocular pseudo-depth.",
    )

    duration_sec = st.slider("Demo / export duration (s)", 3, 60, 8, 1)
    fps = st.select_slider("Frame rate (FPS)", options=[10, 15, 20, 24, 30], value=15)
    res = st.selectbox(
        "Resolution",
        [(320, 240), (480, 360), (640, 480)],
        format_func=lambda v: f"{v[0]}x{v[1]}",
    )

    num_objects = st.slider("Simulated objects", 1, 12, 5)
    noise = st.slider("Sensor noise", 0.0, 2.0, 0.4, 0.1)
    min_conf = st.slider("Minimum detection confidence", 0.0, 0.9, 0.30, 0.01)
    low_conf_threshold = st.slider("Low-confidence review threshold", 0.0, 0.9, 0.45, 0.01)
    max_range = st.slider("Max depth range (m)", 5, 40, 24)
    prediction_horizon = st.slider("Motion prediction horizon (s)", 0.5, 3.0, 1.5, 0.25)
    seed = st.number_input("Deterministic seed", 0, 99999, 42)

    show_gt = st.checkbox("Overlay ground truth", True)
    simulate_faults = st.checkbox("Simulate sensor faults", True)

    uploaded = None
    if mode == "Uploaded video":
        uploaded = st.file_uploader(
            "Upload video",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            help="Processed locally only. No upload to external services.",
        )

    st.markdown("---")
    run_button = st.button("🚀 Run / Refresh pipeline", type="primary", use_container_width=True)
    st.caption("Deterministic demo loads automatically on first launch.")
    st.caption("No external paid API key is required.")

cfg = DemoConfig(
    mode=mode,
    duration_sec=float(duration_sec),
    fps=int(fps),
    width=int(res[0]),
    height=int(res[1]),
    num_objects=int(num_objects),
    noise=float(noise),
    min_conf=float(min_conf),
    low_conf_threshold=float(low_conf_threshold),
    max_range=float(max_range),
    seed=int(seed),
    prediction_horizon=float(prediction_horizon),
    show_gt=bool(show_gt),
    simulate_faults=bool(simulate_faults),
)

if "result" not in st.session_state:
    st.session_state.result = None

if "pending_cfg" not in st.session_state:
    st.session_state.pending_cfg = cfg
    st.session_state.pending_upload = uploaded

if run_button:
    st.session_state.pending_cfg = cfg
    st.session_state.pending_upload = uploaded
    st.session_state.result = None

if st.session_state.pending_cfg is not None:
    cfg_run = st.session_state.pending_cfg
    upload_run = st.session_state.pending_upload

    if cfg_run.mode == "Uploaded video" and upload_run is None:
        st.warning("Uploaded video mode requires a video file. Switch to Deterministic demo or upload a file.")
        st.session_state.pending_cfg = None
        st.session_state.pending_upload = None
    else:
        progress = st.progress(0.0, text="Starting 4D spatiotemporal pipeline...")
        try:
            def progress_cb(p: float):
                try:
                    progress.progress(float(np.clip(p, 0.0, 1.0)), text=f"4D world model rendering: {int(p * 100)}%")
                except Exception:
                    pass

            result = process_pipeline(cfg_run, upload_run, progress_cb)
            st.session_state.result = result
            st.session_state.last_cfg = cfg_run
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            st.info("Tip: use Deterministic demo for a guaranteed local demo, or check camera/video availability.")
        finally:
            progress.empty()
            st.session_state.pending_cfg = None
            st.session_state.pending_upload = None

if st.session_state.result is not None:
    result = st.session_state.result
    summary = result.get("summary", {})

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Avg latency", f"{summary.get('avg_latency_ms', 0.0):.0f} ms")
    m2.metric("Avg FPS", f"{summary.get('avg_wall_fps', 0.0):.1f}")
    m3.metric("Active tracks seen", f"{summary.get('tracks_seen', 0)}")
    m4.metric("Low-confidence reviews", f"{summary.get('low_conf_reviews', 0)}")
    m5.metric("MOTA / quality", f"{summary.get('final_mota', 0.0):.2f}")

    tab_live, tab_analytics, tab_evaluation, tab_evidence, tab_events, tab_privacy = st.tabs(
        ["🎥 Live Demo", "📊 Analytics", "🧪 Evaluation", "🗂 Evidence", "🚨 Events", "🔒 Privacy"]
    )

    with tab_live:
        st.subheader("Live processed recording")
        st.caption(
            "This video is generated locally by the running pipeline. It includes reconstructed object tracks, "
            "motion vectors, prediction trails, BEV world model, confidence state, and system telemetry."
        )

        mime = result.get("mime", "video/mp4")
        video_bytes = result.get("video_bytes", b"")

        if len(video_bytes) > 0:
            if summary.get("duration_sec", 0.0) <= 15.0:
                b64 = base64.b64encode(video_bytes).decode()
                st.components.v1.html(
                    f"""
                    <video width="100%" autoplay loop muted controls>
                        <source src="data:{mime};base64,{b64}" type="{mime}">
                        Your browser does not support embedded video.
                    </video>
                    """,
                    height=520,
                )
            else:
                st.video(video_bytes)

            ext = "mp4" if mime.endswith("mp4") else "avi"
            st.download_button(
                "⬇️ Download live recording",
                data=video_bytes,
                file_name=f"4DWorldModel_{result.get('run_id', 'demo')}.{ext}",
                mime=mime,
                type="primary",
            )

            if summary.get("low_conf_reviews", 0) > 0:
                st.warning(
                    f"{summary.get('low_conf_reviews', 0)} low-confidence result(s) were visibly flagged and routed "
                    "to the Evidence tab for review."
                )
            else:
                st.success("No low-confidence review items were generated in this run.")
        else:
            st.error("No video bytes were produced.")

    with tab_analytics:
        metrics_df = result.get("metrics_df", pd.DataFrame())

        if not metrics_df.empty:
            st.line_chart(metrics_df.set_index("timestamp")[["active_tracks", "detections", "low_conf_count"]])
            st.line_chart(metrics_df.set_index("timestamp")[["latency_ms", "processing_fps", "wall_fps"]])

            if {"precision", "recall", "mean_iou"}.issubset(metrics_df.columns):
                st.line_chart(metrics_df.set_index("timestamp")[["precision", "recall", "mean_iou"]])

            st.dataframe(metrics_df.tail(50), use_container_width=True)
        else:
            st.info("No analytics were produced.")

    with tab_evaluation:
        st.subheader("Task metrics")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Depth MAE", f"{summary.get('mean_depth_mae', 0.0):.2f} m")
        e2.metric("3D center MAE", f"{summary.get('mean_center_mae', 0.0):.2f} m")
        e3.metric("Velocity MAE", f"{summary.get('mean_vel_mae', 0.0):.2f} m/s")
        e4.metric("Mean IoU", f"{summary.get('mean_iou', 0.0):.2f}")

        if summary.get("mode", "") == "Deterministic demo":
            st.success(
                "Deterministic mode includes synthetic ground truth, enabling MOTA-style tracking metrics, "
                "depth error, 3D center error, velocity error, and IoU."
            )
        else:
            st.info(
                "Uploaded/webcam mode uses monocular pseudo-depth and does not include ground truth. "
                "It reports latency, FPS, track counts, and low-confidence routing."
            )

    with tab_evidence:
        track_df = result.get("track_df", pd.DataFrame())
        low_df = result.get("low_conf_df", pd.DataFrame())

        st.subheader("Structured evidence")
        st.caption("Timestamped object/track records include IDs, confidence, measurements, and frame evidence references.")

        if not track_df.empty:
            st.markdown("**Latest track records**")
            st.dataframe(track_df.sort_values("timestamp", ascending=False).head(200), use_container_width=True)
            st.download_button(
                "Download tracks CSV",
                data=track_df.to_csv(index=False).encode("utf-8"),
                file_name=f"tracks_{result.get('run_id', 'demo')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No track records were produced.")

        st.markdown("---")

        if not low_df.empty:
            st.markdown("**Low-confidence review queue**")
            st.dataframe(low_df.sort_values("timestamp", ascending=False).head(200), use_container_width=True)
            st.download_button(
                "Download low-confidence CSV",
                data=low_df.to_csv(index=False).encode("utf-8"),
                file_name=f"low_conf_{result.get('run_id', 'demo')}.csv",
                mime="text/csv",
            )
        else:
            st.success("Low-confidence review queue is empty.")

        json_payload = {
            "summary": summary,
            "metrics": result.get("metrics_df", pd.DataFrame()).to_dict(orient="records"),
            "tracks": track_df.to_dict(orient="records"),
            "low_conf": low_df.to_dict(orient="records"),
            "events": result.get("events_df", pd.DataFrame()).to_dict(orient="records"),
        }
        json_bytes = json.dumps(json_payload, default=str, indent=2).encode("utf-8")
        st.download_button(
            "Download full JSON report",
            data=json_bytes,
            file_name=f"4DWorldModel_report_{result.get('run_id', 'demo')}.json",
            mime="application/json",
        )

    with tab_events:
        events_df = result.get("events_df", pd.DataFrame())
        if not events_df.empty:
            st.dataframe(events_df.sort_values("timestamp", ascending=False).head(300), use_container_width=True)
        else:
            st.success("No notable events were logged.")

    with tab_privacy:
        st.subheader("Privacy and authorization")
        st.write(
            """
            - The deterministic demo uses synthetic objects only.
            - Uploaded videos and webcam frames are processed locally in this app.
            - No identity recognition, face embedding, or biometric inference is performed.
            - No external paid API is used.
            - Stored evidence consists of numeric measurements, timestamps, and frame references.
            - For production deployment, add explicit user consent, camera authorization, and retention controls.
            """
        )

else:
    st.info("Preparing automatic demo. If it does not start, click **Run / Refresh pipeline** in the sidebar.")