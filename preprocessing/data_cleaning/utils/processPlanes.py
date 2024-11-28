# -*- coding: utf-8 -*-
# """
# processPlanes.py
# Created on Oct Sept 02, 2024
# ""

import random

import numpy as np
import open3d as o3d


def create_mesh_from_polygon(points) -> o3d.geometry.TriangleMesh:
    """
    Create a TriangleMesh from a list of 3D points defining a polygon.
    """
    mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    # Create triangles
    triangles: list[list[int]] = []
    num_points: int = len(points)
    for i in range(1, num_points - 1):
        triangles.append([0, i, i + 1])
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh


def process_planes(planes, min_z, reference_point) -> list:
    """
    Process planes to get 3D meshes for visualization.
    """
    plane_meshes: list = []

    for plane in planes:
        azimuth: float = plane["azimuth"]
        tilt: float = plane["tilt"]
        height: float = plane["height"]
        perimeter: list[dict[str, float]] = plane["perimeter"]

        # Adjust azimuth based on conventions
        # adjusted_azimuth = 90 - azimuth
        adjusted_azimuth: float = (90 - azimuth) % 360

        # Adjust tilt based on convention
        # adjusted_tilt = 90 - tilt
        adjusted_tilt: float = tilt

        # Convert to radians
        # azimuth_rad = np.radians(azimuth)
        # tilt_rad = np.radians(tilt)
        azimuth_rad: float = np.radians(adjusted_azimuth)
        tilt_rad: float = np.radians(adjusted_tilt)

        # Compute normal vector
        n_x: float = np.sin(tilt_rad) * np.cos(azimuth_rad)
        n_y: float = np.sin(tilt_rad) * np.sin(azimuth_rad)
        n_z: float = np.cos(tilt_rad)

        # If normal vector is pointing in the opposite direction, invert it:
        # n_x = -n_x
        # n_y = -n_y
        # n_z = -n_z

        # print("*" * 60)
        # print("normal vectors")
        # print("n_x:", n_x)
        # print("n_y:", n_y)
        # print("n_z:", n_z)
        # print("*" * 60)

        # print("*" * 60)
        # print(f"Tilt Rad: {tilt_rad}, Azimuth Rad: {azimuth_rad}")
        # print(f"Adjusted Azimuth: {adjusted_azimuth}, Adjusted Tilt: {adjusted_tilt}")
        # print("*" * 60)

        # Transform plane coordinates into UTM coordinate system
        x_list: np.ndarray = np.array([p["x"] for p in perimeter]) + reference_point[0]
        y_list: np.ndarray = np.array([p["y"] for p in perimeter]) + reference_point[1]

        # print(f"x List: {x_list}")
        # print(f"y List: {y_list}")

        # Adjust height (assuming planes' heights are relative to min_z)
        z0: float = height + min_z

        # Compute centroid of the plane
        x0: float = np.mean(x_list)
        y0: float = np.mean(y_list)

        # print(f"x0: {x0}, y0: {y0}")
        # print(f"z0: {z0}")

        # For each perimeter point, compute z
        points_3d: list[list[float]] = []
        for x, y in zip(x_list, y_list):
            z: float = z0 - (n_x * (x - x0) + n_y * (y - y0)) / n_z
            points_3d.append([x, y, z])

        points_3d = np.array(points_3d)

        # Create mesh
        mesh: o3d.geometry.TriangleMesh = create_mesh_from_polygon(points_3d)
        mesh.paint_uniform_color([random.random(), random.random(), random.random()])
        plane_meshes.append(mesh)
        # print(f"Mesh: {mesh}")

    return plane_meshes
