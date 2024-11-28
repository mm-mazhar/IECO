# -*- coding: utf-8 -*-
# """
# visualize_2Dplanes.py
# Created on Oct Sept 02, 2024
# """

import numpy as np
import open3d as o3d


def visualize_2Dplanes_only(plane_meshes, width=1024, height=768) -> None:
    vis: o3d.visualization.Visualizer = o3d.visualization.Visualizer()
    vis.create_window(window_name="2D Planes Only", width=width, height=height)
    for mesh in plane_meshes:
        vis.add_geometry(mesh)
    opt: o3d.visualization.RenderOption = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])  # Set background to white
    # Disable backface culling
    opt.mesh_show_back_face = True
    vis.run()
    vis.destroy_window()


# Visualize 2D planes only
# visualize_2Dplanes_only(plane_meshes, width=1280, height=720)
