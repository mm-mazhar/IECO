# 3D Roof Reconstruction with Computer Vision for Solar Energy Optimization

## Modeling Approaches

### Aerial Image Segmentation

Model: SAMv2  
Objective: Identify roof planes from aerial images

**Technical Walkthrough**
- Extract 2D information from planes.json
- Generate masks for each plane
- Mark keypoints, required by SAM
- Train mask and prompt encoder
- Generate segmentation map for inference

### Mesh Construction

Model: Mesh Anything v2  
Objective: Reconstruct 3D building from its point cloud

**Technical Walkthrough**
- Compute point cloud normals
- Execute command line interface

### Point Cloud Segmentation

Model: PointNet
Objective: Identify roof planes from point clouds

**Technical Walkthrough**
- Highlight point clouds associated with roof planes
- Use these labels (roof/non-roof) to train pointnet

### Plane Attribute Estimation

Model: EfficientNet/ResNet50
Objective: Estimate plane parameters (tilt, azimuth, height) from point clouds and aerial images

**Technical Walkthrough**
- Perform data augmentation and standard preprocessing
- Use CNN backbone as feature extraction from aerial images concatenated with point cloud
- Support custom loss weights to predict plane attributes
