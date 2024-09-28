# -*- coding: utf-8 -*-
# """
# json_npy.py
# Created on Wed Sept 28, 2024, 13:55:22
# @author: Mazhar
# """


import json
import numpy as np
import open3d as o3d
from typing import Any

# Step 1: Load the JSON file containing the XYZ coordinates
def load_json(json_file) -> np.ndarray:
    with open(json_file, 'r') as f:
        data: Any = json.load(f)
    return np.array(data)  # Convert the list of lists into a NumPy array

# Step 2: Compute normals using Open3D
def compute_normals(xyz_points) -> np.ndarray:
    # Create an Open3D PointCloud object
    pcd: Any = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_points)

    # Estimate the normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Retrieve the normals
    normals: np.ndarray[Any, np.dtype[Any]] = np.asarray(pcd.normals)
    return normals

# Step 3: Save as .npy in the (N, 6) pc_normal format
def save_as_npy(xyz_points, normals, npy_file):
    # Combine the XYZ coordinates with their corresponding normals
    pc_normal_data = np.hstack((xyz_points, normals))

    # Save the array to an .npy file
    np.save(npy_file, pc_normal_data)
    print(f"Point cloud with normals saved to {npy_file}")

# # Usage example
# json_file_path = 'path/to/your/point_cloud.json'  # Replace with the actual path to your JSON file
# npy_output_path = 'point_cloud_pc_normal.npy'  # Output .npy file

# # Load XYZ points from the JSON file
# xyz_points = load_json(json_file_path)

# # Compute the normals for the point cloud
# normals = compute_normals(xyz_points)

# # Save the combined XYZ and normal data as an .npy file
# save_as_npy(xyz_points, normals, npy_output_path)
