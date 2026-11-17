import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class DepthAnythingNode(Node):
    def __init__(self):
        super().__init__("depth_anything_node")
        self.bridge = CvBridge()

        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("output_topic", "/camera/depth_any")
        self.declare_parameter("repo_path", "")
        self.declare_parameter("weights_path", "")
        self.declare_parameter("encoder", "vitl")
        self.declare_parameter("image_size", 518)
        self.declare_parameter("use_cuda", True)

        rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        repo_path = self.get_parameter("repo_path").get_parameter_value().string_value
        weights_path = self.get_parameter("weights_path").get_parameter_value().string_value
        self.encoder = self.get_parameter("encoder").get_parameter_value().string_value
        self.image_size = int(self.get_parameter("image_size").get_parameter_value().integer_value)
        use_cuda = bool(self.get_parameter("use_cuda").get_parameter_value().bool_value)

        if not repo_path or not os.path.isdir(repo_path):
            raise RuntimeError("repo_path invalide pour Depth-Anything")
        if not weights_path or not os.path.isfile(weights_path):
            raise RuntimeError("weights_path invalide pour Depth-Anything")

        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"DepthAnything device: {self.device}")

        sys.path.insert(0, repo_path)
        try:
            from depth_anything.dpt import DepthAnything
            from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
        except Exception as e:
            raise RuntimeError(f"Impossible d'importer Depth-Anything: {e}")

        self.transform = [
            Resize(
                width=self.image_size,
                height=self.image_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]

        # s'assurer que torch.hub trouve le répertoire "torchhub" local
        prev_cwd = os.getcwd()
        os.chdir(repo_path)
        try:
            config = {
                "encoder": self.encoder,
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
                "use_bn": False,
                "use_clstoken": False,
                "localhub": True,
            }
            self.model = DepthAnything(config).to(self.device).eval()
            ckpt = torch.load(weights_path, map_location=self.device)
            state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
            self.model.load_state_dict(state, strict=False)
        finally:
            os.chdir(prev_cwd)

        self.pub = self.create_publisher(Image, self.output_topic, 10)
        self.sub = self.create_subscription(Image, rgb_topic, self.callback, 10)

    def _apply_transform(self, image):
        data = {"image": image}
        for t in self.transform:
            data = t(data)
        return data["image"]

    def callback(self, rgb_msg):
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        except Exception as e:
            self.get_logger().error(f"Erreur cv_bridge: {e}")
            return

        image = (rgb.astype(np.float32) / 255.0)
        h, w = image.shape[:2]
        image = self._apply_transform(image)
        image_t = torch.from_numpy(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            depth = self.model(image_t)

        depth = F.interpolate(depth[None], (h, w), mode="bilinear", align_corners=False)[0, 0]
        depth_np = depth.cpu().numpy().astype(np.float32)

        out_msg = self.bridge.cv2_to_imgmsg(depth_np, encoding="32FC1")
        out_msg.header = rgb_msg.header
        self.pub.publish(out_msg)


def main():
    rclpy.init()
    node = DepthAnythingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
