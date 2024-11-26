# -*- coding: utf-8 -*-
# """
# visualize_point_cloud_2Dplanes.py
# Created on Oct Sept 02, 2024
# """

import numpy as np
import open3d as o3d


def visualize_point_cloud_and_planes(
    points, plane_meshes, width=1024, height=768
) -> None:
    pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Set point cloud color to gray

    vis: o3d.visualization.Visualizer = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Point Cloud with 2D Planes", width=width, height=height
    )
    vis.add_geometry(pcd)
    for mesh in plane_meshes:
        vis.add_geometry(mesh)
    opt: o3d.visualization.RenderOption = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])  # Set background to white
    # Disable backface culling
    opt.mesh_show_back_face = True
    vis.run()
    vis.destroy_window()
