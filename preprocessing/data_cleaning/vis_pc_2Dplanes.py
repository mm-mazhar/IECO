# -*- coding: utf-8 -*-
# """
# vis_pc_2Dplanes.py
# Created on Oct Sept 02, 2024
# """


import json
import os
import random
from typing import Any

import numpy as np
import open3d as o3d

from utils.basic_utils import cent_norm, load_planes, load_point_cloud
from utils.generatePointsInMesh import generate_points_in_mesh
from utils.processPlanes import process_planes
from utils.segment_point_cloud import (
    count_labels,
    save_segmentation_labels,
    segment_point_cloud,
    visualize_segmentation,
)
from utils.visualize_2Dplanes import visualize_2Dplanes_only
from utils.visualize_point_cloud_2Dplanes import visualize_point_cloud_and_planes

# Paths to your files
dsm_file = (
    "../../Dataset 2024-09-23/omdena/f286d7eab2542b54d5172a577ebf630321172681/dsm.json"
)
planes_file = "../../Dataset 2024-09-23/omdena/f286d7eab2542b54d5172a577ebf630321172681/planes.json"

# Load data
points: np.ndarray = load_point_cloud(dsm_file)
planes: list[dict[str, Any]] = load_planes(planes_file)

# Get minimum z-value of the point cloud
min_z: float = np.min(points[:, 2])

# Define a reference point for translation (e.g., the centroid of the UTM coordinates)
reference_point: np.ndarray = np.mean(points[:, :2], axis=0)
# print(f"Reference Point: {reference_point}")

# Process planes with translation
plane_meshes: list = process_planes(planes, min_z, reference_point)

# Combine point cloud and plane vertices
plane_vertices: np.ndarray = np.vstack(
    [np.asarray(mesh.vertices) for mesh in plane_meshes]
)

# Combine all points for unified normalization
all_points: np.ndarray = np.vstack((points, plane_vertices))

# Compute centroid and max_distance based on all points
# centroid = np.mean(all_points, axis=0)
# max_distance = np.max(np.linalg.norm(all_points - centroid, axis=1))

# Normalize point cloud
# normalized_points = (points - centroid) / max_distance

normalized_points, centroid, max_distance = cent_norm(all_points)

# Normalize plane meshes
for mesh in plane_meshes:
    vertices: np.ndarray = np.asarray(mesh.vertices)
    normalized_vertices: np.ndarray = (vertices - centroid) / max_distance
    mesh.vertices = o3d.utility.Vector3dVector(normalized_vertices)

# print("*" * 60)
# print("Centroid:", centroid)
# print("Max Distance:", max_distance)
# print("Min Z:", min_z)
# print("*" * 60)

# print("*" * 60)
# print("Plane Meshes:", plane_meshes)
# print(f"Type: {type(plane_meshes)}")
# print("*" * 60)

# Visualize 2D planes only
visualize_2Dplanes_only(plane_meshes, width=1280, height=720)

# Visualize point cloud and planes
visualize_point_cloud_and_planes(
    normalized_points, plane_meshes, width=1280, height=720
)

# mesh_points = generate_points_within_polygon(plane_meshes, num_points=100)
mesh_points: np.ndarray = generate_points_in_mesh(
    plane_meshes,
    generate_additional_points=True,
    num_points_per_mesh=2000,
    width=1280,
    height=720,
    show_visualization=False,
)

# Segment point cloud
labels: np.ndarray = segment_point_cloud(normalized_points, mesh_points, tolerance=0.05)
# Save segmentation labels
# sub_folder_name: str = os.path.basename(os.path.dirname(dsm_file))
# file_path: str = os.path.join(sub_folder_name, "seg")
file_path = "segmentation_labels.seg"
save_segmentation_labels(labels, file_path)

# Count labels
count_ones, count_two = count_labels(file_path)
print(f"Number of 2's (Roof Top): {count_two}")
print(f"Number of 1's (Others): {count_ones}")
print(f"Total points: {count_ones + count_two}")

# # normalized_points` is normalized point cloud data and `labels` is the segmentation labels
labels = np.loadtxt(file_path, dtype=int)
visualize_segmentation(normalized_points, labels, width=1280, height=720)
