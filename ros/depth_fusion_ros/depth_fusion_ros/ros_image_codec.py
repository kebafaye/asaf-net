"""
Encodage/décodage sensor_msgs Image ↔ numpy SANS cv_bridge.

cv_bridge (binaire compilé contre NumPy 1.x) segfaulte avec un venv NumPy 2.x.
Ces conversions sont triviales pour nos encodages (16UC1, 32FC1, rgb8, jpeg),
donc on les fait à la main — portable local/pod, aucun binaire ROS requis.
"""

from __future__ import annotations

import numpy as np
from sensor_msgs.msg import CompressedImage, Image


def decode_compressed_rgb(msg: CompressedImage) -> np.ndarray:
    """CompressedImage (jpeg/png) → RGB uint8 (H,W,3)."""
    import cv2

    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("imdecode a échoué")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_image(msg: Image) -> np.ndarray:
    """Image → numpy. Supporte 16UC1/mono16, 32FC1, rgb8/bgr8."""
    h, w = int(msg.height), int(msg.width)
    data = bytes(msg.data)
    enc = msg.encoding

    if enc in ("16UC1", "mono16"):
        dt = np.dtype(np.uint16).newbyteorder(">" if msg.is_bigendian else "<")
        arr = np.frombuffer(data, dtype=dt)
        row = msg.step // 2 if msg.step else w
        return np.ascontiguousarray(arr.reshape(h, row)[:, :w])
    if enc == "32FC1":
        dt = np.dtype(np.float32).newbyteorder(">" if msg.is_bigendian else "<")
        arr = np.frombuffer(data, dtype=dt)
        row = msg.step // 4 if msg.step else w
        return np.ascontiguousarray(arr.reshape(h, row)[:, :w])
    if enc in ("rgb8", "bgr8"):
        arr = np.frombuffer(data, dtype=np.uint8)
        row = msg.step if msg.step else w * 3
        img = np.ascontiguousarray(arr.reshape(h, row)[:, : w * 3].reshape(h, w, 3))
        if enc == "bgr8":
            img = img[:, :, ::-1].copy()
        return img
    raise ValueError(f"Encodage non supporté: {enc}")


def encode_image(arr: np.ndarray, encoding: str, header) -> Image:
    """numpy → Image. Supporte 16UC1 (uint16) et 32FC1 (float32)."""
    msg = Image()
    msg.header = header
    msg.height, msg.width = int(arr.shape[0]), int(arr.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = 0
    if encoding == "16UC1":
        arr = np.ascontiguousarray(arr.astype(np.uint16))
        msg.step = msg.width * 2
    elif encoding == "32FC1":
        arr = np.ascontiguousarray(arr.astype(np.float32))
        msg.step = msg.width * 4
    else:
        raise ValueError(f"Encodage non supporté: {encoding}")
    msg.data = arr.tobytes()
    return msg
