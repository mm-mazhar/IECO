# -*- coding: utf-8 -*-
# """
# basic_utils.py
# Created on Oct Sept 02, 2024
# """

import json
from typing import Any

import numpy as np


def load_point_cloud(dsm_file) -> np.ndarray:
    with open(dsm_file, "r") as f:
        points: list[list[float]] = json.load(f)
    return np.array(points)


def load_planes(planes_file) -> list[dict[str, Any]]:
    with open(planes_file, "r") as f:
        planes: list[dict[str, Any]] = json.load(f)
    return planes


def cent_norm(verts) -> tuple[Any, Any, Any]:
    """
    Center and normalize the point cloud data.
    """
    centroid: np.ndarray = np.mean(verts, axis=0)
    verts_centered: np.ndarray = verts - centroid
    max_distance: float = np.max(np.sqrt(np.sum(verts_centered**2, axis=1)))
    verts_normalized: np.ndarray = verts_centered / max_distance
    return verts_normalized, centroid, max_distance
