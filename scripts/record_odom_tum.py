#!/usr/bin/env python3
"""Enregistre /odom (nav_msgs/Odometry) en fichier TUM pendant un run SLAM."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rosgraph_msgs.msg import Clock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from depth_fusion.vigs.poses import pose_msg_to_tum_row, save_tum


class OdomRecorder(Node):
    def __init__(self, topic: str, rows: list) -> None:
        super().__init__("odom_tum_recorder")
        self.rows = rows
        self.n_fallback = 0
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=200,
        )
        self.create_subscription(Odometry, topic, self.callback, qos)

        # Abonnement EXPLICITE à /clock plutôt que use_sim_time.
        # use_sim_time via « --ros-args -p » n'a pas pris effet ici (horodatages
        # restés en heure murale malgré le paramètre), et déboguer la plomberie
        # sim-time de rclpy coûtait plus cher que de lire l'horloge directement.
        # rosbag2 publie /clock en RELIABLE ; on souscrit en BEST_EFFORT, qui
        # est compatible avec un éditeur RELIABLE (l'inverse ne l'est pas).
        self.sim_time: float | None = None
        self.create_subscription(
            Clock, "/clock", self._on_clock,
            QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
            ),
        )
        self.get_logger().info(f"Enregistrement {topic} → TUM")

    def _on_clock(self, msg: Clock) -> None:
        self.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

    def callback(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0.0:
            # VIGS-Fusion publie /odom_gs avec un horodatage NUL : la ligne qui
            # le renseignait (GaussianSplattingSlamNode.cpp:230) se trouve dans
            # un bloc commenté, et le publish de la ligne 268 envoie un message
            # dont le header n'a jamais été rempli.
            #
            # Le repli DOIT être l'horloge du nœud, pas time.time(). Avec
            # use_sim_time et le /clock publié par `ros2 bag play --clock`, elle
            # rend le temps du BAG. time.time() rendait l'heure murale : à 0.3x
            # les trajectoires duraient 385 s au lieu de 115 s, avec 47 millions
            # de secondes d'écart avec la GT — aucune association possible dans
            # evo, donc aucun ATE calculable.
            if self.sim_time is not None:
                stamp = self.sim_time
            else:
                # Aucun /clock reçu : on écrit tout de même, mais la trajectoire
                # sera inassociable avec la GT — l'avertissement de fin le dira.
                stamp = time.time()
            self.n_fallback += 1
        self.rows.append(pose_msg_to_tum_row(stamp, p, q))
        if len(self.rows) == 1 or len(self.rows) % 20 == 0:
            self.get_logger().info(f"{len(self.rows)} poses reçues")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--topic", default="/odom")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=7200.0)
    # parse_known_args (et non parse_args) pour laisser passer « --ros-args
    # -p use_sim_time:=true » : sans use_sim_time, le repli d'horodatage rend
    # l'heure murale au lieu du temps du bag.
    args, ros_args = p.parse_known_args()

    rclpy.init(args=sys.argv)
    rows: list = []
    node = OdomRecorder(args.topic, rows)
    deadline = time.time() + args.timeout
    try:
        while rclpy.ok() and time.time() < deadline:
            try:
                rclpy.spin_once(node, timeout_sec=0.1)
            except ExternalShutdownException:
                break
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_tum(args.output, rows)
        print(f"Écrit {len(rows)} poses → {args.output}", flush=True)
        # Garde-fou : si le repli d'horodatage a servi ET que le premier stamp
        # est proche de l'heure courante, c'est que use_sim_time n'a pas pris
        # et que l'on vient d'écrire une trajectoire en heure murale —
        # inassociable avec la GT du bag. Le signaler ICI évite de découvrir le
        # problème après coup, quand evo ne trouve aucune correspondance.
        if getattr(node, "n_fallback", 0) and rows:
            first = float(rows[0][0])
            if abs(first - time.time()) < 86400.0:
                print(
                    "ATTENTION : horodatages en HEURE MURALE "
                    f"(premier stamp {first:.3f}). VIGS publie /odom_gs sans "
                    "horodatage et AUCUN message /clock n'a été reçu — la "
                    "trajectoire ne pourra pas être associée à la GT. "
                    "Vérifier que le bag est rejoué avec `--clock`.",
                    flush=True,
                )


if __name__ == "__main__":
    main()
