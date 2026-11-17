"""
Lancement VIGS-Fusion + pipeline ASAF sur le bag ROS2 Line.

Prérequis :
  - colcon build dans ~/ros2_ws (vigs_fusion, depth_fusion_ros, depth_anything_da3_ros)
  - GPU NVIDIA + CUDA >= 12 (VIGS-Fusion)
  - libunwind-dev pour compiler glog/vigs_fusion

Usage :
  ros2 launch <racine du dépôt>/launch/vigs_fusion_line.launch.py

Dans un autre terminal :
  ros2 bag play <racine du dépôt>/data/Line/Line --clock
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    project_root = os.environ.get(
        "DEPTH_FUSION_PROJECT",
        "/workspace/depth_fusion_project"
        if os.path.isdir("/workspace/depth_fusion_project")
        else os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    )
    checkpoint = os.path.join(
        project_root,
        "checkpoints/experiments/asaf_net_vigs_multiseq/best_model.pth",
    )
    da3_repo = os.path.join(project_root, "memoire_upgrade_da3/Depth-Anything-3")

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("use_cuda", default_value="false"),
            DeclareLaunchArgument("run_vigs", default_value="true"),
            DeclareLaunchArgument(
                "checkpoint_path",
                default_value=checkpoint,
            ),
            DeclareLaunchArgument(
                "model_version",
                default_value="v1",
                description="v1 (legacy) ou v2 (métrique, checkpoint asaf_net_v2_vigs)",
            ),
            DeclareLaunchArgument(
                "vigs_depth_topic",
                default_value="/camera/depth_fused",
                description="Topic profondeur consommé par VIGS "
                "(/camera/depth/image_rect_raw pour le mode sensor)",
            ),
            DeclareLaunchArgument(
                "run_asaf",
                default_value="true",
                description="Désactiver (false) en mode sensor pur",
            ),
            DeclareLaunchArgument(
                "da3_dir",
                default_value=os.path.join(project_root, "dataset_fusion", "da3_scaled"),
            ),
            DeclareLaunchArgument(
                "use_offline_da3",
                default_value="true",
                description="Publier da3_scaled pré-calculé (recommandé pour bag Line)",
            ),
            DeclareLaunchArgument("run_depth_viz", default_value="true"),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                output="screen",
                arguments=[
                    "0.0456823",
                    "0.0360948",
                    "-0.0745264",
                    "-0.00910219",
                    "0.0119687",
                    "-0.0126518",
                    "0.999804",
                    "xsens_imu_link",
                    "camera_link",
                ],
                parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
            ),
            Node(
                package="depth_fusion_ros",
                executable="offline_da3_node",
                name="offline_da3",
                condition=IfCondition(LaunchConfiguration("use_offline_da3")),
                parameters=[
                    {
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "da3_dir": LaunchConfiguration("da3_dir"),
                        "rgb_compressed_topic": "/camera/color/image_raw/compressed",
                        "output_topic": "/camera/depth_any",
                        "start_index": 0,
                    }
                ],
            ),
            # DA3 online uniquement si ASAF actif ET pas de cartes offline
            # (évite le crash numpy/cv_bridge en mode sensor)
            Node(
                package="depth_anything_da3_ros",
                executable="da3_node",
                name="depth_anything3",
                condition=IfCondition(
                    PythonExpression([
                        "'",
                        LaunchConfiguration("use_offline_da3"),
                        "' == 'false' and '",
                        LaunchConfiguration("run_asaf"),
                        "' == 'true'",
                    ])
                ),
                parameters=[
                    {
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "rgb_topic": "/camera/color/image_raw/compressed",
                        "output_topic": "/camera/depth_any",
                        "repo_path": da3_repo,
                        "use_cuda": LaunchConfiguration("use_cuda"),
                        "use_compressed_rgb": True,
                    }
                ],
            ),
            # Visualisations profondeur (BGR8 pour Foxglove / RViz)
            Node(
                package="depth_fusion_ros",
                executable="depth_viz_node",
                name="depth_sensor_viz",
                condition=IfCondition(LaunchConfiguration("run_depth_viz")),
                parameters=[
                    {
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "input_topic": "/camera/depth/image_rect_raw",
                        "output_topic": "/camera/depth_sensor_viz",
                        "auto_range": True,
                    }
                ],
            ),
            Node(
                package="depth_fusion_ros",
                executable="depth_viz_node",
                name="depth_metric_viz",
                condition=IfCondition(
                    PythonExpression([
                        "'",
                        LaunchConfiguration("run_depth_viz"),
                        "' == 'true' and '",
                        LaunchConfiguration("run_asaf"),
                        "' == 'false'",
                    ])
                ),
                parameters=[
                    {
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "input_topic": "/camera/depth_metric",
                        "output_topic": "/camera/depth_metric_viz",
                        "auto_range": True,
                    }
                ],
            ),
            Node(
                package="depth_fusion_ros",
                executable="depth_viz_node",
                name="depth_fused_viz",
                condition=IfCondition(
                    PythonExpression([
                        "'",
                        LaunchConfiguration("run_depth_viz"),
                        "' == 'true' and '",
                        LaunchConfiguration("run_asaf"),
                        "' == 'true'",
                    ])
                ),
                parameters=[
                    {
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "input_topic": "/camera/depth_fused",
                        "output_topic": "/camera/depth_fused_viz",
                        "auto_range": True,
                    }
                ],
            ),
            Node(
                package="depth_fusion_ros",
                executable="depth_viz_node",
                name="depth_da3_viz",
                condition=IfCondition(
                    PythonExpression([
                        "'",
                        LaunchConfiguration("run_depth_viz"),
                        "' == 'true' and '",
                        LaunchConfiguration("run_asaf"),
                        "' == 'true'",
                    ])
                ),
                parameters=[
                    {
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "input_topic": "/camera/depth_any",
                        "output_topic": "/camera/depth_any_viz",
                        "auto_range": True,
                    }
                ],
            ),
            # Mode sensor : VIGS attend du 32FC1 en mètres (il charge la
            # profondeur sans conversion) — conversion du 16UC1 mm du bag.
            Node(
                package="depth_fusion_ros",
                executable="depth_to_metric_node",
                name="depth_to_metric",
                condition=UnlessCondition(LaunchConfiguration("run_asaf")),
                parameters=[
                    {
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "input_topic": "/camera/depth/image_rect_raw",
                        "output_topic": "/camera/depth_metric",
                    }
                ],
            ),
            Node(
                package="depth_fusion_ros",
                executable="asaf_node",
                name="asaf_fusion",
                condition=IfCondition(
                    PythonExpression([
                        "'",
                        LaunchConfiguration("run_asaf"),
                        "' == 'true'",
                    ])
                ),
                parameters=[
                    {
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "project_root": project_root,
                        "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                        "use_compressed_rgb": True,
                        "mono_topic": "/camera/depth_any",
                        "sensor_topic": "/camera/depth/image_rect_raw",
                        "output_topic": "/camera/depth_fused",
                        "use_cuda": LaunchConfiguration("use_cuda"),
                        "model_version": LaunchConfiguration("model_version"),
                        # VIGS charge la profondeur en CV_32FC1 (mètres)
                        "output_encoding": "32FC1",
                    }
                ],
            ),
            Node(
                package="vigs_fusion",
                executable="gaussian_splatting_slam_node",
                output="screen",
                condition=IfCondition(
                    PythonExpression([
                        "'",
                        LaunchConfiguration("run_vigs"),
                        "' == 'true'",
                    ])
                ),
                parameters=[
                    {
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "color_transport": "compressed",
                        "viewer": False,
                        "downsample": True,
                        "pose_iterations": 12,
                        "update_iterations": 8,
                    }
                ],
                remappings=[
                    ("camera_info", "/camera/color/camera_info"),
                    ("image_color/compressed", "/camera/color/image_raw/compressed"),
                    ("image_depth", LaunchConfiguration("vigs_depth_topic")),
                    ("imu", "/filtered_imu_xsens"),
                    ("imu_preint", "/filtered_imu_xsens"),
                ],
            ),
        ]
    )
