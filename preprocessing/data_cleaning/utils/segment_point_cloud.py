# -*- coding: utf-8 -*-
# """
# segment_point_cloud.py
# Created on Oct Sept 02, 2024
# """

from typing import Any

import numpy as np
import open3d as o3d


# Segment point cloud
def segment_point_cloud(normalized_points, mesh_points, tolerance=0.01) -> np.ndarray:
    """
    Segment the point cloud based on proximity to the mesh points.

    Parameters:
    - normalized_points: numpy array of normalized point cloud data.
    - mesh_points: numpy array of points generated within the mesh.
    - tolerance: distance tolerance to consider a point as part of a plane.

    Returns:
    - labels: numpy array of segmentation labels (2 for points near a plane, 1 otherwise).
    """
    labels: np.ndarray = np.zeros(len(normalized_points), dtype=int)

    for i, point in enumerate(normalized_points):
        distances: np.ndarray = np.linalg.norm(mesh_points - point, axis=1)
        if np.any(distances < tolerance):
            labels[i] = 2
        else:
            labels[i] = 1

    return labels


# Save segmentation labels
def save_segmentation_labels(labels, file_path) -> None:
    """
    Save the segmentation labels to a .sep file.

    Parameters:
    - labels: numpy array of segmentation labels.
    - file_path: path to save the .sep file.
    """
    np.savetxt(file_path, labels, fmt="%d")


def count_labels(file_path) -> tuple[Any, Any]:
    """
    Count the number of 1's and 0's in the .sep file.

    Parameters:
    - file_path: path to the .sep file.

    Returns:
    - count_ones: number of 2's (Roof Top) in the file.
    - count_zeros: number of 1's (Others) in the file.
    """
    labels: np.ndarray = np.loadtxt(file_path, dtype=int)
    count_two: int = np.sum(labels == 2)
    count_ones: int = np.sum(labels == 1)
    return count_ones, count_two


def visualize_segmentation(normalized_points, labels, width=1280, height=720) -> None:
    """
    Visualize the point cloud with segmentation labels.

    Parameters:
    - normalized_points: numpy array of normalized point cloud data.
    - labels: numpy array of segmentation labels (1 for points near a plane, 0 otherwise).
    - width: width of the visualization window.
    - height: height of the visualization window.
    """
    # Create a point cloud object
    pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(normalized_points)

    # Assign colors based on labels
    colors: np.ndarray = np.zeros((normalized_points.shape[0], 3))
    colors[labels == 2] = [1, 0, 0]  # Red for points labeled as 2
    colors[labels == 1] = [0.5, 0.5, 0.5]  # Gray for points labeled as 1
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create a visualizer object
    vis: o3d.visualization.Visualizer = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Point Cloud Segmentation Visualization", width=width, height=height
    )
    vis.add_geometry(pcd)

    # Set render options
    opt: o3d.visualization.RenderOption = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])  # Set background to white
    opt.point_size = 2.0  # Set point size

    # Run the visualizer
    vis.run()
    vis.destroy_window()


# # usage:
# # `normalized_points` is point cloud data and `mesh_points` is the list of meshes from `process_planes`
# labels = segment_point_cloud(normalized_points, plane_meshes, tolerance=0.05)
# save_segmentation_labels(labels, "segmentation_labels.sep")

# # Usage:
# file_path = "segmentation_labels.sep"
# count_ones, count_two = count_labels(file_path)
# print(f"Number of 2's: {count_two}")
# print(f"Number of 1's: {count_ones}")
# print(f"Total points: {count_ones + count_two}")

# # Usage:
# # normalized_points` is normalized point cloud data and `labels` is the segmentation labels
# labels = np.loadtxt("segmentation_labels.sep", dtype=int)
# visualize_segmentation(normalized_points, labels, width=1280, height=720)
