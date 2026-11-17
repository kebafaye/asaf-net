"""Extraction offline RGB + depth + poses depuis rosbag2 VIGS-Fusion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from depth_fusion.vigs.poses import pose_msg_to_tum_row, save_tum

TOPIC_RGB = "/camera/color/image_raw/compressed"
TOPIC_DEPTH = "/camera/depth/image_rect_raw"
TOPIC_POSE = "/Darknav/pose"
TOPIC_CAM_INFO = "/camera/color/camera_info"


def _nearest_idx(query_ns: int, stamps: np.ndarray, max_delta_ns: int) -> Optional[int]:
    if stamps.size == 0:
        return None
    i = int(np.argmin(np.abs(stamps - query_ns)))
    if abs(int(stamps[i]) - query_ns) > max_delta_ns:
        return None
    return i


def _decode_depth_msg(depth_msg) -> np.ndarray:
    """Decode ROS Image depth (uint16 mm) en respectant le stride (step)."""
    h, w = int(depth_msg.height), int(depth_msg.width)
    row_bytes = w * 2
    step = int(getattr(depth_msg, "step", row_bytes) or row_bytes)
    buf = np.frombuffer(depth_msg.data, dtype=np.uint8)
    if step == row_bytes:
        return buf.view(np.uint16).reshape(h, w)
    depth = np.empty((h, w), dtype=np.uint16)
    for i in range(h):
        row = buf[i * step : i * step + row_bytes]
        depth[i] = row.view(np.uint16)
    return depth


def extract_bag(
    bag_path: Path,
    output_dir: Path,
    *,
    max_frames: int = 0,
    sync_tolerance_ms: float = 50.0,
    skip_existing: bool = True,
) -> Dict[str, int]:
    """
    Extrait rgb/, depth_sensor/, poses_gt.tum, intrinsics, frames.csv.

    Synchronisation : une frame par image RGB ; depth et pose les plus proches en temps.
    """
    bag_path = Path(bag_path)
    output_dir = Path(output_dir)
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth_sensor"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing and (output_dir / "frames.csv").exists():
        df = pd.read_csv(output_dir / "frames.csv")
        return {"frames": len(df), "skipped": 1}

    max_delta_ns = int(sync_tolerance_ms * 1e6)
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    rgb_list: List[Tuple[int, object]] = []
    depth_list: List[Tuple[int, object]] = []
    pose_list: List[Tuple[int, object]] = []
    cam_info = None

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        for conn, ts, raw in reader.messages():
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == TOPIC_RGB:
                rgb_list.append((ts, msg))
            elif conn.topic == TOPIC_DEPTH:
                depth_list.append((ts, msg))
            elif conn.topic == TOPIC_POSE:
                pose_list.append((ts, msg))
            elif conn.topic == TOPIC_CAM_INFO and cam_info is None:
                cam_info = msg

    if not rgb_list:
        raise RuntimeError(f"Aucune image RGB dans {bag_path}")

    depth_ts = np.array([t for t, _ in depth_list], dtype=np.int64)
    pose_ts = np.array([t for t, _ in pose_list], dtype=np.int64)

    if cam_info is not None:
        k = np.array(cam_info.k, dtype=np.float64).reshape(3, 3).tolist()
        with (output_dir / "intrinsics_rgb.json").open("w", encoding="utf-8") as f:
            json.dump(k, f, indent=2)

    tum_rows = []
    csv_rows = []
    n_limit = max_frames if max_frames > 0 else len(rgb_list)

    for frame_id, (ts_rgb, rgb_msg) in enumerate(rgb_list[:n_limit]):
        depth_idx = _nearest_idx(ts_rgb, depth_ts, max_delta_ns)
        if depth_idx is None:
            continue

        _, depth_msg = depth_list[depth_idx]
        arr = np.frombuffer(rgb_msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            continue

        depth = _decode_depth_msg(depth_msg)

        stem = f"{frame_id:04d}"
        rgb_path = rgb_dir / f"image_{stem}.png"
        depth_path = depth_dir / f"depth_{stem}.npy"
        cv2.imwrite(str(rgb_path), bgr)
        np.save(depth_path, depth)

        pose_idx = _nearest_idx(ts_rgb, pose_ts, max_delta_ns * 4)
        pose_ts_sec = None
        if pose_idx is not None:
            ts_p, pose_msg = pose_list[pose_idx]
            stamp_sec = ts_p * 1e-9
            tum_rows.append(
                pose_msg_to_tum_row(stamp_sec, pose_msg.pose.position, pose_msg.pose.orientation)
            )
            pose_ts_sec = stamp_sec

        csv_rows.append(
            {
                "frame_id": frame_id,
                "timestamp_ns": ts_rgb,
                "rgb": str(rgb_path.resolve()),
                "depth_sensor": str(depth_path.resolve()),
                "depth_gt": str(depth_path.resolve()),
                "pose_timestamp": pose_ts_sec,
            }
        )

    if not csv_rows:
        raise RuntimeError("Aucune frame synchronisée (vérifier sync_tolerance_ms)")

    pd.DataFrame(csv_rows).to_csv(output_dir / "frames.csv", index=False)
    save_tum(output_dir / "poses_gt.tum", tum_rows)

    return {
        "frames": len(csv_rows),
        "rgb_total": len(rgb_list),
        "depth_total": len(depth_list),
        "pose_total": len(pose_list),
    }


def _resolve_depth_output_path(depth_path_str: str, output_dir: Path, depth_dir: Path) -> Path:
    """Remappe les chemins frames.csv (ex. machine locale) vers output_dir/depth_sensor/."""
    p = Path(depth_path_str)
    try:
        p.resolve().relative_to(output_dir.resolve())
        return p
    except ValueError:
        return depth_dir / p.name


def restore_depth_sensor(
    bag_path: Path,
    output_dir: Path,
    *,
    sync_tolerance_ms: float = 50.0,
    skip_existing: bool = True,
) -> Dict[str, int]:
    """
    Réécrit uniquement depth_sensor/*.npy à partir d'un rosbag et du frames.csv existant.
    Préserve rgb/, da3_scaled/ et frames.csv (ré-extraction légère après cleanup disque).
    """
    bag_path = Path(bag_path)
    output_dir = Path(output_dir)
    frames_csv = output_dir / "frames.csv"
    depth_dir = output_dir / "depth_sensor"
    if not frames_csv.is_file():
        raise FileNotFoundError(f"frames.csv manquant: {frames_csv}")

    depth_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(frames_csv)
    max_delta_ns = int(sync_tolerance_ms * 1e6)
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    depth_list: List[Tuple[int, object]] = []
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        for conn, ts, raw in reader.messages():
            if conn.topic != TOPIC_DEPTH:
                continue
            depth_list.append((ts, reader.deserialize(raw, conn.msgtype)))

    if not depth_list:
        raise RuntimeError(f"Aucune depth dans {bag_path}")

    depth_ts = np.array([t for t, _ in depth_list], dtype=np.int64)
    written = 0
    skipped = 0
    missing = 0

    for _, row in df.iterrows():
        depth_path = _resolve_depth_output_path(str(row["depth_sensor"]), output_dir, depth_dir)
        if skip_existing and depth_path.is_file():
            try:
                d = np.load(depth_path)
                if d.ndim == 2 and d.size > 0:
                    skipped += 1
                    continue
            except (OSError, ValueError, EOFError):
                pass

        ts_rgb = int(row["timestamp_ns"])
        depth_idx = _nearest_idx(ts_rgb, depth_ts, max_delta_ns)
        if depth_idx is None:
            missing += 1
            continue

        _, depth_msg = depth_list[depth_idx]
        depth = _decode_depth_msg(depth_msg)
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(depth_path, depth)
        written += 1

    return {
        "frames_csv": len(df),
        "written": written,
        "skipped": skipped,
        "missing_sync": missing,
    }
