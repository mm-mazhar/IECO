# 3D Point Clouds and 2D Planes Visualization and Labelling

This repository provides tools for visualizing and processing 3D point clouds and 2D plane data. It allows users to load, transform, visualize, and segment spatial datasets to extract meaningful insights. Two key scripts drive the main functionality:

- vis_pc_2Dplanes.py - For visualizing point clouds and 2D planes.
- dataset_preprocessor.py - For preprocessing and segmenting point clouds based on proximity to planes.

### Features

- Translation and alignment: Aligns 2D planes with 3D point clouds.
- Normalization: Normalizes the point cloud data to fit within a unit sphere, making it easier to process and visualize.
- 3D Mesh Generation: Converts 2D planes into 3D mesh objects, which can be visualized alongside the point cloud.
- Segmentation: Segments the point cloud based on proximity to planes, labeling points as part of a plane or other structure.

### Key Python Files:

1. **`basic_utils.py`**: Contains utility functions for centering, normalizing, and loading the point cloud data.
2. **`generatePointsInMesh.py`**: Contains methods for generating points within mesh polygons.
3. **`processPlanes.py`**: Processes plane data into mesh objects.
4. **`segment_point_cloud.py`**: Functions for segmenting the point cloud based on plane meshes.
5. **`visualize_2Dplanes.py`**: Visualizes the processed planes in a 2D projection.
6. **`visualize_point_cloud_2Dplanes.py`**: Visualizes both the point cloud and the plane meshes together.

---

## Setup Instructions

### Prerequisites

- Python 3.8+
- The following Python packages are required:
  - `numpy`
  - `open3d`
  - `scipy`
  - Any additional dependencies from `utils/` scripts.

### Installation

1. Clone the repository and navigate to the directory.

   ```bash
   git clone https://github.com/OmdenaAI/IECO.git

   cd IECO

   git checkout maz_dataset_visualization_labeling

   ```

2. Install the required Python libraries.

   ```
   conda env create -f ./requirements/requirements_local.yml

   conda activate ieco
   ```

### File Paths

Make sure that the paths to `dsm.json` and `planes.json` are correctly specified in the `vis_pc_2Dplanes.py` script.

```
   e.g.

   dsm_file = "../Dataset 2024-09-23/omdena/e97b4319719e573215ea3c6751f0211a194c14bd/dsm.json"

   planes_file = "../Dataset 2024-09-23/omdena/e97b4319719e573215ea3c6751f0211a194c14bd/planes.json"
```

also, Make sure that the paths to dataset folder and output folders are correctly specified in the `dataset_preprocessor.py` script.

```
e.g.

   DATASET_FOLDER_PATH = "../../Dataset_v1/"
   OUTPUT_POINTS_FOLDER_PATH = "../../Dataset_v1/data/points/"
   OUTPUT_LABEL_FOLDER_PATH = "../../Dataset_v1/data/expert_verified/points_label/"
```

---

### Usage

`vis_pc_2Dplanes.py` | Visualizing 3D Point Clouds and 2D Planes

This script handles the visualization of point clouds and 2D planes, transforming the planes into 3D meshes for display.

#### Key Steps:

- Load point cloud and planes: Load spatial data from dsm.json (point cloud) and planes.json (2D planes).
- Translation: Align plane data with the point cloud.
- Normalization
  - `cent_norm()` function in `basic_utils.py` file
  - Center and normalize the point cloud so it fits within a unit sphere.
- 3D Mesh Creation: Convert 2D planes into 3D mesh objects using the `process_planes` function in `processPlanes.py` file.
- Visualization:
  -Use the `visualize_point_cloud_and_planes` function in `visualize_point_cloud_2Dplanes.py` file to render the point cloud and planes together in a 3D environment.

#### Commands

To run the visualization script, execute:

```
python ./vis_pc_2Dplanes.py
```

This will run the entire pipeline, from loading the point cloud and planes to segmenting the point cloud and visualization.

- 2D visualization of the planes.
- 3D visualization of the point cloud and plane meshes

---

`dataset_preprocessor.py` | Preparing Data for Segmentation | PointNet Model

This script preprocesses the point cloud and plane data for segmentation based on proximity. It converts the point cloud from JSON format into .pts, aligns the planes, and generates segmentation labels.

#### Key Steps:

- Convert `dsm.json` to `.pts`
  - Transform the point cloud data into a more manageable .pts format using `convert_json_to_pts` method.
- Translation: Align the 2D plane data to the point cloud.
- Plane Processing:
  - `processPlanes.py`
  - Generate 3D meshes from the 2D planes for further segmentation.
- Point Generation in Meshes:
  - `generatePointsInMesh.py`
  - Generate points within plane meshes to help in the segmentation process.
- Segmentation:
  - `segment_point_cloud.py`
  - Segment the point cloud based on the proximity to plane meshes using and saves lebels in `.seg` file format.

#### Commands

To run the `dataset_preprocessor.py` file, execute:

```
python ./dataset_preprocessor.py
```

---

## Future Work

- Improve the mesh generation algorithm to support more complex plane shapes.
- Refine the point cloud segmentation process for greater accuracy. (Right now it needs seperate tolerence levels of each point cloud instance)
- Add more visualization options for better understanding of the results.
