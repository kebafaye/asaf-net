import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from depth_fusion_ros.ros_image_codec import decode_image, encode_image


class DepthVizNode(Node):
    def __init__(self):
        super().__init__("depth_viz_node")

        self.declare_parameter("input_topic", "/camera/depth_fused")
        self.declare_parameter("output_topic", "/camera/depth_fused_viz")
        self.declare_parameter("min_depth", 0.1)
        self.declare_parameter("max_depth", 10.0)
        self.declare_parameter("auto_range", True)

        input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.min_depth = float(self.get_parameter("min_depth").get_parameter_value().double_value)
        self.max_depth = float(self.get_parameter("max_depth").get_parameter_value().double_value)
        self.auto_range = bool(self.get_parameter("auto_range").get_parameter_value().bool_value)

        self.pub = self.create_publisher(Image, self.output_topic, 10)
        self.sub = self.create_subscription(Image, input_topic, self.callback, 10)
        self._logged = False

    def callback(self, msg: Image):
        try:
            raw = decode_image(msg).astype(np.float32)
            depth = raw / 1000.0 if msg.encoding in ("16UC1", "mono16") else raw
        except Exception as e:
            self.get_logger().error(f"Erreur décodage ({msg.encoding}): {e}")
            return

        depth = depth.astype(np.float32)
        valid = np.isfinite(depth)
        if self.auto_range and np.any(valid):
            v = depth[valid]
            d_min = float(np.percentile(v, 2))
            d_max = float(np.percentile(v, 98))
        else:
            d_min, d_max = self.min_depth, self.max_depth

        if d_max <= d_min or not np.any(valid):
            self.get_logger().warn("Profondeur vide ou invalide, frame ignorée")
            return

        depth_clip = np.clip(depth, d_min, d_max)
        depth_norm = (depth_clip - d_min) / (d_max - d_min)
        depth_u8 = (depth_norm * 255.0).astype(np.uint8)

        # Palette pour visualisation
        color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
        out_msg = Image()
        out_msg.header = msg.header
        out_msg.height, out_msg.width = color.shape[0], color.shape[1]
        out_msg.encoding = "bgr8"
        out_msg.is_bigendian = 0
        out_msg.step = color.shape[1] * 3
        out_msg.data = np.ascontiguousarray(color).tobytes()
        self.pub.publish(out_msg)
        if not self._logged:
            self.get_logger().info(
                f"Publication {self.output_topic} ({depth.shape[1]}x{depth.shape[0]}, "
                f"range {d_min:.2f}-{d_max:.2f} m)"
            )
            self._logged = True


def main():
    rclpy.init()
    node = DepthVizNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
