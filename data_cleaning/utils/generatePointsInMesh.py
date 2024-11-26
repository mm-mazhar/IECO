# -*- coding: utf-8 -*-
# """
# generatePointsInMesh.py
# Created on Oct Sept 02, 2024
# """

import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay


def generate_points_in_mesh(
    plane_meshes,
    generate_additional_points=False,
    num_points_per_mesh=100,
    width=1024,
    height=768,
    show_visualization=False,  # Parameter to control visualization
) -> np.ndarray:
    """
    Visualize the mesh points as a point cloud, optionally generating additional points within the planes.

    Parameters:
    - plane_meshes: list of open3d.geometry.TriangleMesh objects.
    - generate_additional_points: flag to generate additional points within the planes.
    - num_points_per_mesh: number of additional points to generate within each mesh (if flag is True).
    - width: width of the visualization window.
    - height: height of the visualization window.
    - show_visualization: flag to control whether to display the visualization.

    Returns:
    - mesh_points: numpy array of shape (total_points, 3) with all mesh points.
    """

    def generate_points_within_polygon(vertices, num_points=100) -> np.ndarray:
        """
        Generate points within the polygon defined by the vertices.

        Parameters:
        - vertices: numpy array of shape (n, 3) defining the polygon vertices.
        - num_points: number of points to generate within the polygon.

        Returns:
        - points: numpy array of shape (num_points, 3) with generated points.
        """
        vertices: np.ndarray = np.asarray(vertices)  # Ensure vertices is a NumPy array
        tri = Delaunay(vertices[:, :2])
        min_x, min_y = np.min(vertices[:, :2], axis=0)
        max_x, max_y = np.max(vertices[:, :2], axis=0)
        points: list[list[float]] = []

        while len(points) < num_points:
            random_points = np.random.rand(num_points, 2) * (
                max_x - min_x,
                max_y - min_y,
            ) + (min_x, min_y)
            mask = tri.find_simplex(random_points) >= 0
            valid_points: list[list[float]] = random_points[mask]
            points.extend(valid_points[: num_points - len(points)])

        points = np.array(points)
        z_values: np.ndarray = np.interp(points[:, 0], vertices[:, 0], vertices[:, 2])
        points = np.hstack((points, z_values[:, np.newaxis]))

        return points

    # Create a point cloud object
    mesh_points: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for mesh in plane_meshes:
        vertices: np.ndarray = np.asarray(mesh.vertices)
        mesh_points.append(vertices)
        colors.append(
            np.tile([0, 0, 1], (vertices.shape[0], 1))
        )  # Original points in blue
        if generate_additional_points:
            additional_points: np.ndarray = generate_points_within_polygon(
                vertices, num_points=num_points_per_mesh
            )
            mesh_points.append(additional_points)
            colors.append(
                np.tile([1, 0, 0], (additional_points.shape[0], 1))
            )  # Additional points in red
    mesh_points = np.vstack(mesh_points)
    colors = np.vstack(colors)

    pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mesh_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if show_visualization:  # Check if visualization should be shown
        # Create a visualizer object
        vis: o3d.visualization.Visualizer = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="2D Plane Mesh Points", width=width, height=height
        )
        vis.add_geometry(pcd)

        # Set render options
        opt: o3d.visualization.RenderOption = vis.get_render_option()
        opt.background_color = np.array([1, 1, 1])  # Set background to white
        opt.point_size = 2.0  # Set point size

        # Run the visualizer
        vis.run()
        vis.destroy_window()

    return mesh_points  # Return the mesh points


# Usage:
# `plane_meshes` is the list of meshes from `process_planes`
# mesh_points = generate_points_in_mesh(plane_meshes, generate_additional_points=True, num_points_per_mesh=1000, width=1280, height=720, show_visualization=True)
