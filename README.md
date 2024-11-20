# 3D Roof Reconstruction with Computer Vision for Solar Energy Optimization

## Proposed Solution
**Image Masking**: Use the roof polygons as masks on the aerial image. This will highlight only the roof areas, allowing the model to learn features solely within these regions.  
**Point Cloud Extraction**: Once the roof areas are segmented in the 2D image, you can map these masked regions to your 3D point cloud data. Assuming you have spatial alignment (e.g., GPS coordinates) between the aerial image and the point cloud, extract the points within the 2D boundaries of each roof polygon.  
**3D Reconstruction**: After extracting the point cloud data within each masked area, you’ll have the spatial points associated with each roof. These can be processed for reconstruction, even accounting for height, since the 3D coordinates are preserved in the point cloud.  

## Modeling Experiments

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
- Annotate point clouds overlapping with roof planes
- Augment points with random rotation and noise
- Train PointNet using T-module and cross entropy loss

### Plane Attribute Estimation

Model: EfficientNet/ResNet50  
Objective: Estimate plane parameters (tilt, azimuth, height) from point clouds and aerial images  

**Technical Walkthrough**
- Perform data augmentation and standard preprocessing
- Use CNN backbone as feature extraction from aerial images concatenated with point cloud
- Support custom loss weights to predict plane attributes
