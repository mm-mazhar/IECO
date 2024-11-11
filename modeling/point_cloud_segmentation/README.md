# PointNetSeg - Point Cloud Segmentation

This repository implements the PointNet-based segmentation model (PointNetSeg) for point cloud data. The model is designed to classify each point in a point cloud into a predefined set of classes, leveraging a combination of feature and spatial transformations.

### Model Overview

#### T-Net

The T-Net module is used for both input and feature transformation. It helps align the point cloud in a canonical space by predicting affine transformations.

#### Transform Module

The Transform module includes the T-Net for both the input and feature space. This module is crucial for applying spatial transformations to the point cloud data before extracting features.

#### PointNetSeg

The PointNetSeg model processes the transformed features to perform point-wise segmentation. The model’s final layer predicts the class of each point.

- Input: Point cloud of shape (batch_size, num_points, 3), where each point has 3 coordinates (x, y, z).
- Output: Class scores for each point in the point cloud, with a shape of (batch_size, num_classes, num_points).

#### Dataset

Point cloud data and corresponding segmentation labels are loaded using a custom Dataset class. The dataset is stored in .pts files (for point clouds) and .seg files (for labels).
for dataset preparation, [check my other repo](https://github.com/OmdenaAI/IECO/tree/maz_dataset_visualization_labeling) or download the dataset from [here](https://drive.google.com/drive/folders/14PhmCEIb0fK62fkJn06ID1iaxNKGDsR1?usp=drive_link)

Each point cloud is normalized and augmented with random rotations and noise. The point clouds are sampled to 2000 points.

#### Data Processing Functions

- cent_norm: Centers and normalizes the point cloud.
- rotation_z: Rotates the point cloud around the z-axis.
- add_noise: Adds random Gaussian noise to the points.
- sample_2000: Randomly samples 2000 points from the point cloud for training.

#### Training

The model is trained using a standard cross-entropy loss (NLLLoss), with optional regularization to penalize transformations that deviate from identity matrices (using Frobenius norm). The training script supports validation, model saving, and batch-wise training with data augmentation.

#### Key Components:

- Input Transformation: The input transformation network aligns the input point cloud in a canonical space.
- Feature Transformation: The feature transformation network aligns the feature space for point-wise predictions.
- Loss Function: The loss function includes classification loss and regularization based on transformation matrices.
