from setuptools import setup

package_name = "depth_fusion_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@todo.todo",
    description="ROS2 node for RGB + Depth-Anything fusion using a PyTorch model.",
    license="MIT",
        entry_points={
        "console_scripts": [
            "fusion_node = depth_fusion_ros.fusion_node:main",
            "depth_anything_node = depth_fusion_ros.depth_anything_node:main",
            "depth_viz_node = depth_fusion_ros.depth_viz_node:main",
            "asaf_node = depth_fusion_ros.asaf_node:main",
            "offline_da3_node = depth_fusion_ros.offline_da3_node:main",
            "depth_to_metric_node = depth_fusion_ros.depth_to_metric_node:main",
        ],
    },
)
