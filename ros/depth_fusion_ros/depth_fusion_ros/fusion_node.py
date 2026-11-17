import numpy as np
import cv2
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber

from depth_fusion_ros.models import ResNetUNet


class DepthFusionNode(Node):
    def __init__(self):
        super().__init__("depth_fusion_node")
        self.bridge = CvBridge()

        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_any_topic", "/camera/depth_any")
        self.declare_parameter("depth_cam_topic", "/camera/depth_registered/image_rect_raw")
        self.declare_parameter("output_topic", "/camera/depth_fused")
        self.declare_parameter("model_path", "")
        self.declare_parameter("image_size", 256)
        self.declare_parameter("model_in_channels", 4)
        self.declare_parameter("use_cam_as_input", False)
        self.declare_parameter("use_depth_scale", True)
        self.declare_parameter("scale_min_depth", 0.1)
        self.declare_parameter("scale_max_depth", 10.0)
        self.declare_parameter("use_cuda", True)

        rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        depth_any_topic = self.get_parameter("depth_any_topic").get_parameter_value().string_value
        depth_cam_topic = self.get_parameter("depth_cam_topic").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.image_size = int(self.get_parameter("image_size").get_parameter_value().integer_value)
        self.model_in_channels = int(
            self.get_parameter("model_in_channels").get_parameter_value().integer_value
        )
        self.use_cam_as_input = bool(
            self.get_parameter("use_cam_as_input").get_parameter_value().bool_value
        )
        self.use_depth_scale = bool(self.get_parameter("use_depth_scale").get_parameter_value().bool_value)
        self.scale_min_depth = float(self.get_parameter("scale_min_depth").get_parameter_value().double_value)
        self.scale_max_depth = float(self.get_parameter("scale_max_depth").get_parameter_value().double_value)
        use_cuda = bool(self.get_parameter("use_cuda").get_parameter_value().bool_value)

        if not model_path:
            self.get_logger().error("model_path est vide. Le noeud ne peut pas démarrer.")
            raise RuntimeError("model_path manquant")

        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Device: {self.device}")

        self.model = ResNetUNet(
            backbone="resnet34",
            pretrained=False,
            in_channels=self.model_in_channels,
            out_channels=1,
        )
        ckpt = torch.load(model_path, map_location=self.device)
        state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
        self.model.load_state_dict(state)
        self.model.eval()
        self.model.to(self.device)

        self.pub = self.create_publisher(Image, self.output_topic, 10)

        self.sub_rgb = Subscriber(self, Image, rgb_topic)
        self.sub_any = Subscriber(self, Image, depth_any_topic)

        self.use_depth_cam = self.use_cam_as_input or self.use_depth_scale
        if self.use_depth_cam:
            self.sub_cam = Subscriber(self, Image, depth_cam_topic)
            self.ts = ApproximateTimeSynchronizer(
                [self.sub_rgb, self.sub_any, self.sub_cam], queue_size=10, slop=0.05
            )
            self.ts.registerCallback(self.callback_with_cam)
        else:
            self.ts = ApproximateTimeSynchronizer(
                [self.sub_rgb, self.sub_any], queue_size=10, slop=0.05
            )
            self.ts.registerCallback(self.callback_no_cam)

    def _normalize_depth_any(self, depth_any, depth_cam=None):
        max_d = max(float(np.max(depth_any)), 1.0)
        if depth_cam is not None:
            max_d = max(max_d, float(np.max(depth_cam)), 1.0)
        return depth_any / max_d, max_d

    def _scale_to_cam(self, pred, depth_cam):
        valid = np.isfinite(depth_cam)
        valid &= depth_cam > self.scale_min_depth
        valid &= depth_cam < self.scale_max_depth
        if not np.any(valid):
            return pred
        pred_valid = pred[valid]
        cam_valid = depth_cam[valid]
        pred_med = np.median(pred_valid)
        cam_med = np.median(cam_valid)
        if pred_med <= 1e-6:
            return pred
        scale = cam_med / pred_med
        return pred * scale

    def _prepare_input(self, rgb_msg, depth_any_msg, depth_cam_msg=None):
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            depth_any = self.bridge.imgmsg_to_cv2(depth_any_msg, desired_encoding="32FC1")
            depth_cam = (
                self.bridge.imgmsg_to_cv2(depth_cam_msg, desired_encoding="32FC1")
                if depth_cam_msg is not None
                else None
            )
        except Exception as e:
            self.get_logger().error(f"Erreur cv_bridge: {e}")
            return None, None, None, None

        rgb_res = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        depth_any_res = cv2.resize(
            depth_any, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )

        rgb_t = torch.from_numpy(rgb_res.astype(np.float32) / 255.0).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        rgb_t = (rgb_t - mean) / std

        depth_any_f = depth_any_res.astype(np.float32)
        depth_cam_f = depth_cam.astype(np.float32) if depth_cam is not None else None
        depth_any_norm, max_d = self._normalize_depth_any(
            depth_any_f, depth_cam_f if self.use_cam_as_input else None
        )
        depth_any_t = torch.from_numpy(depth_any_norm).unsqueeze(0)

        if self.model_in_channels == 5 and self.use_cam_as_input and depth_cam_f is not None:
            depth_cam_norm = depth_cam_f / max_d
            depth_cam_res = cv2.resize(
                depth_cam_norm, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
            )
            depth_cam_t = torch.from_numpy(depth_cam_res).unsqueeze(0)
            inp = torch.cat([rgb_t, depth_any_t, depth_cam_t], dim=0)
        else:
            inp = torch.cat([rgb_t, depth_any_t], dim=0)

        inp = inp.unsqueeze(0).to(self.device)
        return rgb, depth_cam, inp, rgb_msg

    def callback_with_cam(self, rgb_msg, depth_any_msg, depth_cam_msg):
        rgb, depth_cam, inp, header_msg = self._prepare_input(rgb_msg, depth_any_msg, depth_cam_msg)
        if inp is None:
            return

        with torch.no_grad():
            pred = self.model(inp).squeeze().cpu().numpy().astype(np.float32)

        if self.use_depth_scale and depth_cam is not None:
            pred = self._scale_to_cam(pred, depth_cam)

        pred_resized = cv2.resize(pred, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        out_msg = self.bridge.cv2_to_imgmsg(pred_resized, encoding="32FC1")
        out_msg.header = header_msg.header
        self.pub.publish(out_msg)

    def callback_no_cam(self, rgb_msg, depth_any_msg):
        rgb, _, inp, header_msg = self._prepare_input(rgb_msg, depth_any_msg, None)
        if inp is None:
            return
        with torch.no_grad():
            pred = self.model(inp).squeeze().cpu().numpy().astype(np.float32)

        pred_resized = cv2.resize(pred, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        out_msg = self.bridge.cv2_to_imgmsg(pred_resized, encoding="32FC1")
        out_msg.header = header_msg.header
        self.pub.publish(out_msg)


def main():
    rclpy.init()
    node = DepthFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
