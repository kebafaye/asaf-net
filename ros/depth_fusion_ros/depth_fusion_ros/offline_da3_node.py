"""
Publie les profondeurs DA3 pré-calculées (dataset_fusion/da3_scaled) sur /camera/depth_any.

Évite de charger Depth-Anything-3 en live (lent + nombreuses dépendances).
Synchronisation par compteur de frames : le bag Line doit être lu depuis le début.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image

from depth_fusion_ros.ros_image_codec import encode_image


class OfflineDA3Node(Node):
    def __init__(self) -> None:
        super().__init__("offline_da3_node")

        self.declare_parameter("da3_dir", "")
        self.declare_parameter("rgb_compressed_topic", "/camera/color/image_raw/compressed")
        self.declare_parameter("output_topic", "/camera/depth_any")
        self.declare_parameter("start_index", 0)
        self.declare_parameter("filename_pattern", "depth_{:04d}.npy")

        da3_dir = self.get_parameter("da3_dir").get_parameter_value().string_value
        if not da3_dir:
            da3_dir = str(
                Path(__file__).resolve().parents[3]
                / "dataset_fusion"
                / "da3_scaled"
            )
        self.da3_dir = Path(da3_dir)
        if not self.da3_dir.is_dir():
            raise RuntimeError(f"da3_dir introuvable: {self.da3_dir}")

        topic = self.get_parameter("rgb_compressed_topic").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.start_index = int(self.get_parameter("start_index").get_parameter_value().integer_value)
        self.pattern = self.get_parameter("filename_pattern").get_parameter_value().string_value

        self.frame_idx = self.start_index
        self.pub = self.create_publisher(Image, self.output_topic, 10)
        self.sub = self.create_subscription(CompressedImage, topic, self.callback, 10)
        self.get_logger().info(f"DA3 offline: {self.da3_dir} (index départ {self.start_index})")

    def callback(self, rgb_msg: CompressedImage) -> None:
        path = self.da3_dir / self.pattern.format(self.frame_idx)
        if not path.exists():
            self.get_logger().warn(f"Fichier absent: {path}")
            return

        depth = np.load(path).astype(np.float32)
        # Publication en MÈTRES : les .npy DA3 peuvent être en mm selon le
        # pipeline de précalcul — auto-détection par la médiane des valides.
        v = depth[np.isfinite(depth) & (depth > 0)]
        if v.size and float(np.median(v)) > 100.0:
            depth = depth / 1000.0
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        self.pub.publish(encode_image(depth, "32FC1", rgb_msg.header))
        self.frame_idx += 1


def main() -> None:
    rclpy.init()
    node = OfflineDA3Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
