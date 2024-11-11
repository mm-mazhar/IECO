## Mesh Anything from Point Clouds to Simplified Mesh

source: https://github.com/buaacyw/MeshAnythingV2

### Aerial Image

![aerial image](https://github.com/OmdenaAI/IECO/blob/louis/src/tools/mesh-anything/images/input.png)

### Mesh Anything Output

![mesh object](https://github.com/OmdenaAI/IECO/blob/louis/src/tools/mesh-anything/images/mesh-anything.png)

### Poisson surface construction

based on https://github.com/OmdenaAI/IECO/tree/louis/src/tools/point_cloud_visualizer

![poisson surface](https://github.com/OmdenaAI/IECO/blob/louis/src/tools/mesh-anything/images/poisson-construction.png)


### Challenges

1. requires A100 GPU on google-colab
2. it requires at least 4096 points to construct mesh
3. point cloud data is not clean (roof object not isolated)
