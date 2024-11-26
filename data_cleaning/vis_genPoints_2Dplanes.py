# -*- coding: utf-8 -*-
# """
# vis_genPoints_2Dplanes.py
# Created on Oct Sept 02, 2024
# """


import numpy as np
import open3d as o3d
from utils.basic_utils import cent_norm, load_planes, load_point_cloud
from utils.generatePointsInMesh import generate_points_in_mesh
from utils.processPlanes import process_planes

# Paths to your files
dsm_file = (
    "../Dataset 2024-09-23/omdena/0bfa84a5d6cc8177dd3bcc3d74588271ae4fd108/dsm.json"
)
planes_file = (
    "../Dataset 2024-09-23/omdena/0bfa84a5d6cc8177dd3bcc3d74588271ae4fd108/planes.json"
)

# Load data
points: np.ndarray = load_point_cloud(dsm_file)
planes: list = load_planes(planes_file)

# Get minimum z-value of the point cloud
min_z: float = np.min(points[:, 2])

# Normalize point cloud and get transformation parameters
normalized_points, centroid, max_distance = cent_norm(points)

# Process planes to get meshes
plane_meshes: list = process_planes(planes, centroid, max_distance, min_z)

# `plane_meshes` is the list of meshes from `process_planes`
generate_points_in_mesh(
    plane_meshes,
    generate_additional_points=True,
    num_points_per_mesh=1000,
    width=1280,
    height=720,
    show_visualization=True,
)
