"""
Convertisseur profondeur → 32FC1 mètres pour VIGS-Fusion.

VIGS charge la profondeur en CV_32FC1 sans conversion (le nœud
cuda_depth_register de l'auteur, non publié, faisait cette étape).
Ce nœud le remplace : 16UC1/mono16 (mm) ou 32FC1 (mm auto-détectés)
→ 32FC1 en mètres, header conservé pour la synchro message_filters.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from depth_fusion_ros.ros_image_codec import decode_image, encode_image


class DepthToMetricNode(Node):
    def __init__(self) -> None:
        super().__init__("depth_to_metric")
        self.declare_parameter("input_topic", "/camera/depth/image_rect_raw")
        self.declare_parameter("output_topic", "/camera/depth_metric")
        inp = self.get_parameter("input_topic").get_parameter_value().string_value
        out = self.get_parameter("output_topic").get_parameter_value().string_value
        self.pub = self.create_publisher(Image, out, 10)
        self.sub = self.create_subscription(Image, inp, self.callback, 10)
        self.get_logger().info(f"depth_to_metric: {inp} -> {out} (32FC1 m)")

    def callback(self, msg: Image) -> None:
        try:
            d = decode_image(msg).astype(np.float32)
        except Exception as e:
            self.get_logger().error(f"décodage ({msg.encoding}): {e}")
            return
        if msg.encoding in ("16UC1", "mono16"):
            d = d / 1000.0
        else:
            v = d[np.isfinite(d) & (d > 0)]
            if v.size and float(np.median(v)) > 100.0:  # mm probables
                d = d / 1000.0
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        self.pub.publish(encode_image(d, "32FC1", msg.header))


def main() -> None:
    rclpy.init()
    node = DepthToMetricNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
