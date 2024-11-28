# -*- coding: utf-8 -*-
# """
# dataset_preprocessor.py
# Created on Oct Sept 02, 2024
# """

import json
import os
import shutil
from typing import Any

import numpy as np
import open3d as o3d
from tqdm import tqdm
from utils.basic_utils import cent_norm, load_planes, load_point_cloud
from utils.generatePointsInMesh import generate_points_in_mesh
from utils.processPlanes import process_planes
from utils.segment_point_cloud import (
    count_labels,
    save_segmentation_labels,
    segment_point_cloud,
)


class DatasetPreprocessor:
    def __init__(
        self,
        dataset_folder_path: str,
        output_points_folder_path: str,
        output_label_folder_path: str,
    ):
        self.dataset_folder_path: str = dataset_folder_path
        self.output_points_folder_path: str = output_points_folder_path
        self.output_label_folder_path: str = output_label_folder_path

        self._prepare_folders()

    def _prepare_folders(self) -> None:
        # Prepare output folders
        if os.path.exists(self.output_label_folder_path):
            shutil.rmtree(self.output_label_folder_path)
        os.makedirs(self.output_label_folder_path)

        if os.path.exists(self.output_points_folder_path):
            shutil.rmtree(self.output_points_folder_path)
        os.makedirs(self.output_points_folder_path)

    def convert_json_to_pts(self, pc_file_path: str, pts_file_path: str) -> None:
        with open(pc_file_path, "r") as json_file:
            coordinates: list[list[float]] = json.load(json_file)  # Load the JSON data

        with open(pts_file_path, "w") as pts_file:
            for coordinate_set in coordinates:
                # Each coordinate set should be a list of [x, y, z]
                pts_file.write(
                    f"{coordinate_set[0]} {coordinate_set[1]} {coordinate_set[2]}\n"
                )

    def process_pts_files(self) -> None:
        print(f"Processing '.pts' files...")
        error_folders: list[str] = []
        for subdir in tqdm(os.listdir(self.dataset_folder_path)):
            subdir_path: str = os.path.join(self.dataset_folder_path, subdir)
            subdir_name: str = os.path.basename(subdir_path)

            if os.path.isdir(subdir_path):  # Ensure it's a sub-folder
                try:
                    pc_file_path: str = os.path.join(subdir_path, "dsm.json")
                    pc_file_path = pc_file_path.replace("\\", "/")
                    if os.path.exists(
                        pc_file_path
                    ):  # Check if dsm.json exists in the sub-folder
                        pts_file_path: str = os.path.join(
                            self.output_points_folder_path, f"{subdir_name}.pts"
                        )
                        self.convert_json_to_pts(pc_file_path, pts_file_path)
                except Exception as e:
                    # print(f"Error processing {subdir_name}: {e}")
                    error_folders.append(subdir_name)

        # print("Processing '.pts' files...Completed!")
        if error_folders:
            print("Errors occurred in the following folders during .pts processing:")
            for folder in error_folders:
                print(folder)

    def process_labels(self) -> None:
        error_folders: list[str] = []
        label_counts: list[tuple[str, str]] = []

        for subdir in tqdm(
            os.listdir(self.dataset_folder_path), desc="Processing Labels..."
        ):
            subdir_path: str = os.path.join(self.dataset_folder_path, subdir)
            subdir_name: str = os.path.basename(subdir_path)

            if os.path.isdir(subdir_path):  # Ensure it's a sub-folder
                try:
                    dsm_file: str = os.path.join(subdir_path, "dsm.json")
                    dsm_file = dsm_file.replace("\\", "/")
                    planes_file: str = os.path.join(subdir_path, "planes.json")
                    planes_file: str = planes_file.replace("\\", "/")

                    if os.path.exists(dsm_file) and os.path.exists(planes_file):
                        # Load data
                        points: np.ndarray = load_point_cloud(dsm_file)
                        planes: list[dict[str, Any]] = load_planes(planes_file)

                        # Get minimum z-value of the point cloud
                        min_z: float = np.min(points[:, 2])

                        # Define a reference point for translation (e.g., the centroid of the UTM coordinates)
                        reference_point: np.ndarray = np.mean(points[:, :2], axis=0)
                        # print(f"Reference Point: {reference_point}")

                        # Process planes to get meshes
                        plane_meshes: list = process_planes(
                            planes, min_z, reference_point
                        )

                        # Combine point cloud and plane vertices
                        plane_vertices: np.ndarray = np.vstack(
                            [np.asarray(mesh.vertices) for mesh in plane_meshes]
                        )

                        # Combine all points for unified normalization
                        all_points: np.ndarray = np.vstack((points, plane_vertices))

                        # Normalize All points
                        normalized_points, centroid, max_distance = cent_norm(all_points)

                        # Normalize plane meshes
                        for mesh in plane_meshes:
                            vertices: np.ndarray = np.asarray(mesh.vertices)
                            normalized_vertices: np.ndarray = (
                                vertices - centroid
                            ) / max_distance
                            mesh.vertices = o3d.utility.Vector3dVector(
                                normalized_vertices
                            )

                        # Generate points within the mesh
                        mesh_points: np.ndarray = generate_points_in_mesh(
                            plane_meshes,
                            generate_additional_points=True,
                            num_points_per_mesh=2000,
                            width=1280,
                            height=720,
                            show_visualization=False,
                        )

                        # Segment point cloud
                        labels: np.ndarray = segment_point_cloud(
                            normalized_points, mesh_points, tolerance=0.05
                        )

                        # Save segmentation labels
                        label_file_path: str = os.path.join(
                            self.output_label_folder_path, f"{subdir_name}.seg"
                        )
                        save_segmentation_labels(labels, label_file_path)

                        # Store label file path for summary
                        label_counts.append((subdir_name, label_file_path))
                except Exception as e:
                    # print(f"Error processing {subdir_name}: {e}")
                    error_folders.append(subdir_name)

        # Summary of label counts
        print("Label counts summary:")
        for subdir_name, label_file_path in label_counts:
            try:
                count_ones, count_two = count_labels(label_file_path)
                print(
                    f"Subdir: {subdir_name} - Number of 2's (Roof Top): {count_two}, Number of 1's (Others): {count_ones}, Total points: {count_ones + count_two}"
                )
            except Exception as e:
                print(f"Error counting labels for {subdir_name}: {e}")
                error_folders.append(subdir_name)

        # When there are errors in processing plane files
        if error_folders:
            print(
                "Errors occurred in the following folders during plane files processing:"
            )
            for folder in error_folders:
                print(folder)


# Usage
if __name__ == "__main__":
    DATASET_FOLDER_PATH = "../../Dataset_v1/"
    OUTPUT_POINTS_FOLDER_PATH = "../../Dataset_v1/data/points/"
    OUTPUT_LABEL_FOLDER_PATH = "../../Dataset_v1/data/expert_verified/points_label/"

    preprocessor = DatasetPreprocessor(
        DATASET_FOLDER_PATH, OUTPUT_POINTS_FOLDER_PATH, OUTPUT_LABEL_FOLDER_PATH
    )
    preprocessor.process_pts_files()
    preprocessor.process_labels()
