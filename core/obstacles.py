"""Monocular obstacle detection for things that are not in the tag map.

Needs no map and no depth: anything that grows in the image faster than the rest of
the scene is closing in, whether it is a static unmapped object or a moving one.
Sparse LK optical flow gives the radial expansion of tracked features about the image
center; the global translation (yaw/strafe) is subtracted first. Features on AprilTags
are masked out so mapped course markers never count as obstacles.

    det = ObstacleDetector()
    obs = det.update(frame_gray, tag_corners, time.time())
    # {"rate": [left, center, right] expansion 1/s, "level": 0..1+, "side": -1|0|1, "t": ...}
"""

from __future__ import annotations

import cv2
import numpy as np

# Expansion rate (1/s) at which an obstacle is treated as level 1.0 (needs braking).
# Uncalibrated starting point: tune against recordings/.
BRAKE_RATE = 0.8
MIN_POINTS = 8
MIN_RADIUS_FRAC = 0.08
MAX_FEATURES = 120
SMOOTH = 0.5  # EMA weight of the newest measurement


class ObstacleDetector:
    def __init__(self) -> None:
        self._prev: np.ndarray | None = None
        self._prev_t = 0.0
        self._rate = np.zeros(3)

    def reset(self) -> None:
        self._prev = None
        self._rate = np.zeros(3)

    def update(self, gray: np.ndarray, tag_corners=(), t: float = 0.0) -> dict | None:
        """Feed one grayscale frame. Returns None until a previous frame exists."""
        prev, prev_t = self._prev, self._prev_t
        self._prev, self._prev_t = gray, t
        if prev is None or prev.shape != gray.shape or t <= prev_t:
            return None
        dt = t - prev_t
        if dt > 0.5:  # stale pair: flow would be meaningless
            return None

        h, w = gray.shape
        mask = np.full((h, w), 255, np.uint8)
        for corners in tag_corners:
            quad = np.asarray(corners, np.int32).reshape(-1, 1, 2)
            hull = cv2.convexHull(quad)
            cv2.fillConvexPoly(mask, hull, 0)
            x, y, bw, bh = cv2.boundingRect(hull)
            pad = max(bw, bh) // 4
            cv2.rectangle(mask, (x - pad, y - pad), (x + bw + pad, y + bh + pad), 0, -1)

        pts = cv2.goodFeaturesToTrack(prev, MAX_FEATURES, 0.01, 10, mask=mask)
        if pts is None or len(pts) < MIN_POINTS:
            return self._result(t, measured=None)
        nxt, ok, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None, winSize=(21, 21), maxLevel=3)
        ok = ok.reshape(-1).astype(bool)
        if ok.sum() < MIN_POINTS:
            return self._result(t, measured=None)

        p0 = pts.reshape(-1, 2)[ok]
        flow = nxt.reshape(-1, 2)[ok] - p0
        flow = flow - np.median(flow, axis=0)  # remove global translation (yaw, strafe)

        rel = p0 - np.array([w / 2.0, h / 2.0])
        radius = np.linalg.norm(rel, axis=1)
        keep = radius > MIN_RADIUS_FRAC * w
        rel, flow, radius, p0 = rel[keep], flow[keep], radius[keep], p0[keep]
        if len(rel) < MIN_POINTS:
            return self._result(t, measured=None)

        # Per-feature expansion: radial flow / radius, per second.
        expansion = np.einsum("ij,ij->i", flow, rel) / (radius**2) / dt
        zone = np.clip((p0[:, 0] / w * 3).astype(int), 0, 2)
        measured = np.zeros(3)
        for z in range(3):
            sel = expansion[zone == z]
            if len(sel) >= MIN_POINTS // 2:
                measured[z] = max(0.0, float(np.median(sel)))
        return self._result(t, measured)

    def _result(self, t: float, measured: np.ndarray | None) -> dict:
        if measured is None:
            measured = np.zeros(3)
        self._rate = SMOOTH * measured + (1 - SMOOTH) * self._rate
        worst = int(np.argmax(self._rate))
        level = float(self._rate[worst] / BRAKE_RATE)
        side = 0 if worst == 1 else (-1 if worst == 0 else 1)  # side of the frame the threat is on
        return {"rate": [round(float(r), 3) for r in self._rate], "level": round(level, 3),
                "side": side, "t": t}
