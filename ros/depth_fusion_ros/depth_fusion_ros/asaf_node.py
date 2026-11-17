"""
Nœud ROS2 : fusion ASAF-Net (RGB + DA3 + profondeur capteur) pour SLAM / VIGS-Fusion.

Publie une carte de profondeur 16UC1 en millimètres, compatible avec VIGS-Fusion
(voir cuda_depth_register du dépôt officiel).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
import torch
import torchvision.transforms as T
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image

from depth_fusion_ros.ros_image_codec import decode_compressed_rgb, decode_image, encode_image


def _add_project_to_path(project_root: str) -> None:
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise RuntimeError(f"project_root invalide: {project_root}")
    p = str(root)
    if p not in sys.path:
        sys.path.insert(0, p)


class ASAFNode(Node):
    def __init__(self) -> None:
        super().__init__("asaf_node")

        self.declare_parameter("project_root", "")
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("rgb_compressed_topic", "/camera/color/image_raw/compressed")
        self.declare_parameter("use_compressed_rgb", True)
        self.declare_parameter("mono_topic", "/camera/depth_any")
        self.declare_parameter("sensor_topic", "/camera/depth/image_rect_raw")
        self.declare_parameter("output_topic", "/camera/depth_fused")
        self.declare_parameter("image_size", 256)
        self.declare_parameter("depth_scale_mm", 1.0)
        self.declare_parameter("use_cuda", False)
        # "v1" (legacy, normalisation par max) ou "v2" (métrique, masque validité)
        self.declare_parameter("model_version", "v1")
        self.declare_parameter("confidence_topic", "/camera/depth_fused_confidence")
        # "16UC1" (mm, visualisation) ou "32FC1" (mètres — attendu par VIGS,
        # qui charge la profondeur en CV_32FC1 sans conversion).
        self.declare_parameter("output_encoding", "16UC1")

        project_root = self.get_parameter("project_root").get_parameter_value().string_value
        checkpoint = self.get_parameter("checkpoint_path").get_parameter_value().string_value
        self.use_compressed = bool(
            self.get_parameter("use_compressed_rgb").get_parameter_value().bool_value
        )
        rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        rgb_compressed = (
            self.get_parameter("rgb_compressed_topic").get_parameter_value().string_value
        )
        mono_topic = self.get_parameter("mono_topic").get_parameter_value().string_value
        sensor_topic = self.get_parameter("sensor_topic").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.image_size = int(self.get_parameter("image_size").get_parameter_value().integer_value)
        self.depth_scale_mm = float(
            self.get_parameter("depth_scale_mm").get_parameter_value().double_value
        )
        use_cuda = bool(self.get_parameter("use_cuda").get_parameter_value().bool_value)

        if not project_root or not checkpoint:
            raise RuntimeError("project_root et checkpoint_path sont requis")

        _add_project_to_path(project_root)

        from depth_fusion.models.asaf_net import (
            ASAFNet,
            ASAFNetConfig,
            ASAFNetV2,
            ASAFNetV2Config,
        )
        from depth_fusion.utils.checkpoint import load_checkpoint

        self.model_version = (
            self.get_parameter("model_version").get_parameter_value().string_value or "v1"
        )
        confidence_topic = (
            self.get_parameter("confidence_topic").get_parameter_value().string_value
        )

        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"ASAF device: {self.device} | version: {self.model_version}")

        if self.model_version == "v2":
            self.model = ASAFNetV2(ASAFNetV2Config(pretrained=False)).to(self.device)
        else:
            self.model = ASAFNet(
                ASAFNetConfig(
                    use_sensor_depth=True, use_confidence_head=True, residual_refinement=True
                )
            ).to(self.device)
        load_checkpoint(checkpoint, self.model, device=str(self.device))
        self.model.eval()

        self.pub_conf = (
            self.create_publisher(Image, confidence_topic, 10)
            if self.model_version == "v2"
            else None
        )
        self.output_encoding = (
            self.get_parameter("output_encoding").get_parameter_value().string_value or "16UC1"
        )

        # NB : _publish_depth était définie ICI, au milieu du constructeur.
        # Tout ce qui suit — transformations, publisher, les trois abonnements,
        # le synchroniseur et son callback — se retrouvait donc dans le corps de
        # cette méthode, jamais appelée à l'initialisation. Le nœud démarrait,
        # chargeait son modèle, et restait inerte sans créer un seul abonnement :
        # /camera/depth_fused muet, aucune erreur, 0 pose côté VIGS.
        # La méthode a été déplacée après la fin réelle du constructeur.
        self.rgb_transform = T.Compose(
            [
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.pub = self.create_publisher(Image, self.output_topic, 10)

        rgb_type = CompressedImage if self.use_compressed else Image
        rgb_sub_topic = rgb_compressed if self.use_compressed else rgb_topic
        self.sub_rgb = Subscriber(self, rgb_type, rgb_sub_topic)
        self.sub_mono = Subscriber(self, Image, mono_topic)
        self.sub_sensor = Subscriber(self, Image, sensor_topic)

        # Paramétrables : si couleur et profondeur du bag sont décalées de plus
        # de `slop`, l'appariement est IMPOSSIBLE et le nœud reste muet sans
        # lever la moindre erreur. Pouvoir les régler depuis le script évite un
        # cycle édition + recompilation pour un simple essai.
        self.declare_parameter("sync_slop", 0.1)
        self.declare_parameter("sync_queue_size", 10)
        slop = float(self.get_parameter("sync_slop").value)
        qsize = int(self.get_parameter("sync_queue_size").value)
        self.ts = ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_mono, self.sub_sensor],
            queue_size=qsize, slop=slop,
        )
        self.ts.registerCallback(self.callback)
        self.get_logger().info(f"synchro: slop={slop}s queue={qsize}")

        # --- instrumentation de la synchronisation -------------------------
        # Sans elle, un nœud qui reçoit tout mais n'apparie rien est
        # indiscernable d'un nœud qui ne reçoit rien : dans les deux cas
        # /camera/depth_fused reste muet et aucune erreur n'est levée.
        # On compte séparément les arrivées par entrée et les appariements,
        # avec le dernier horodatage vu — un décalage entre entrées saute
        # alors aux yeux.
        self._n = {"rgb": 0, "mono": 0, "sensor": 0, "sync": 0}
        self._last_stamp = {"rgb": 0.0, "mono": 0.0, "sensor": 0.0}

        def _tally(key):
            def _cb(msg):
                self._n[key] += 1
                self._last_stamp[key] = (
                    msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                )
            return _cb

        self.sub_rgb.registerCallback(_tally("rgb"))
        self.sub_mono.registerCallback(_tally("mono"))
        self.sub_sensor.registerCallback(_tally("sensor"))
        # Horloge SYSTÈME explicite : le nœud tourne avec use_sim_time:=true,
        # et un timer sur l'horloge simulée reste figé tant que /clock n'a pas
        # été correctement rattaché — ce qui, dans cet environnement, ne se
        # produit pas. Un diagnostic muet est pire qu'aucun diagnostic.
        from rclpy.clock import Clock, ClockType

        self.create_timer(
            5.0, self._report, clock=Clock(clock_type=ClockType.SYSTEM_TIME)
        )

    def _publish_depth(self, pred_m: np.ndarray, header) -> None:
        """Publie la profondeur fusionnée (mètres) selon l'encodage configuré."""
        if self.output_encoding == "32FC1":
            self.pub.publish(encode_image(pred_m.astype(np.float32), "32FC1", header))
        else:
            depth_mm = np.clip(pred_m * 1000.0, 0, 65535).astype(np.uint16)
            self.pub.publish(encode_image(depth_mm, "16UC1", header))

    def _report(self) -> None:
        n, s = self._n, self._last_stamp
        spread = max(s.values()) - min(s.values()) if all(s.values()) else float("nan")
        self.get_logger().info(
            f"reçus rgb={n['rgb']} mono={n['mono']} sensor={n['sensor']} "
            f"| appariés={n['sync']} | écart max des derniers stamps={spread:.3f}s"
        )
        if n["sync"] == 0 and min(n["rgb"], n["mono"], n["sensor"]) > 0:
            self.get_logger().warn(
                "les trois entrées arrivent mais AUCUN appariement : "
                f"slop=0.1s insuffisant (écart {spread:.3f}s) ou files saturées."
            )

    def _decode_rgb(self, msg) -> np.ndarray | None:
        try:
            if self.use_compressed:
                return decode_compressed_rgb(msg)
            return decode_image(msg)
        except Exception as e:
            self.get_logger().error(f"Décodage RGB: {e}")
            return None

    def _decode_depth_sensor(self, msg: Image) -> np.ndarray | None:
        try:
            raw = decode_image(msg)
            if msg.encoding in ("16UC1", "mono16"):
                return raw.astype(np.float32) * self.depth_scale_mm / 1000.0
            return raw.astype(np.float32)
        except Exception as e:
            self.get_logger().error(f"Décodage capteur ({msg.encoding}): {e}")
            return None

    def _decode_mono(self, msg: Image) -> np.ndarray | None:
        try:
            return decode_image(msg).astype(np.float32)
        except Exception as e:
            self.get_logger().error(f"Décodage mono: {e}")
            return None

    def _resize_depth(self, depth: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
        h, w = size
        arr = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(arr.astype(np.float32))[None, None]

    def callback(self, rgb_msg: Image, mono_msg: Image, sensor_msg: Image) -> None:
        self._n["sync"] += 1
        rgb = self._decode_rgb(rgb_msg)
        mono = self._decode_mono(mono_msg)
        sensor = self._decode_depth_sensor(sensor_msg)
        if rgb is None or mono is None or sensor is None:
            return

        h, w = rgb.shape[:2]
        from PIL import Image as PILImage

        rgb_t = self.rgb_transform(PILImage.fromarray(rgb)).unsqueeze(0).to(self.device)

        if self.model_version == "v2":
            self._callback_v2(rgb_t, mono, sensor, (w, h), sensor_msg)
            return

        d_mono_t = self._resize_depth(mono, (self.image_size, self.image_size)).to(self.device)
        d_sensor_t = self._resize_depth(sensor, (self.image_size, self.image_size)).to(self.device)

        scale = max(d_mono_t.max().item(), d_sensor_t.max().item(), 1e-6)
        d_mono_n = d_mono_t / scale
        d_sensor_n = d_sensor_t / scale

        with torch.no_grad():
            out = self.model(rgb_t, d_mono_n, d_sensor_n)
            pred = out["depth"]

        pred_np = pred.squeeze().cpu().numpy()
        pred_np = cv2.resize(pred_np, (w, h), interpolation=cv2.INTER_LINEAR)
        pred_m = pred_np * scale

        self._publish_depth(pred_m, sensor_msg.header)

    def _callback_v2(
        self,
        rgb_t: torch.Tensor,
        mono: np.ndarray,
        sensor: np.ndarray,
        out_size: "tuple[int, int]",
        sensor_msg: Image,
    ) -> None:
        """Inférence V2 : métrique de bout en bout, resize NEAREST, masque validité."""
        w, h = out_size
        s = self.image_size

        # mono peut arriver en mm (offline_da3_node) — heuristique médiane
        mono_m = mono.astype(np.float32)
        v = mono_m[np.isfinite(mono_m) & (mono_m > 0)]
        if v.size and float(np.median(v)) > 100.0:
            mono_m = mono_m / 1000.0

        mono_r = cv2.resize(mono_m, (s, s), interpolation=cv2.INTER_NEAREST)
        sensor_r = cv2.resize(sensor.astype(np.float32), (s, s), interpolation=cv2.INTER_NEAREST)
        valid = ((sensor_r > 0.05) & (sensor_r < 20.0) & np.isfinite(sensor_r)).astype(np.float32)

        mono_t = torch.from_numpy(np.nan_to_num(mono_r))[None, None].to(self.device)
        sensor_t = torch.from_numpy(np.nan_to_num(sensor_r) * valid)[None, None].to(self.device)
        valid_t = torch.from_numpy(valid)[None, None].to(self.device)

        with torch.no_grad():
            out = self.model(rgb_t, mono_t, sensor_t, valid_t)

        pred_m = out["depth"].squeeze().cpu().numpy()
        pred_m = cv2.resize(pred_m, (w, h), interpolation=cv2.INTER_LINEAR)
        self._publish_depth(pred_m, sensor_msg.header)

        if self.pub_conf is not None:
            conf = out["confidence"].squeeze().cpu().numpy().astype(np.float32)
            conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_LINEAR)
            self.pub_conf.publish(encode_image(conf, "32FC1", sensor_msg.header))


def main() -> None:
    rclpy.init()
    node = ASAFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
