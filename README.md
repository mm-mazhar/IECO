<p align="center">
  <h3 align="center"><strong>MeshAnything V2:<br> Artist-Created Mesh Generation<br>With Adjacent Mesh Tokenization</strong></h3>

<div align="center">

<a href='https://arxiv.org/abs/2408.02555'><img src='https://img.shields.io/badge/arXiv-2408.02555-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://buaacyw.github.io/meshanything-v2/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/Yiwen-ntu/MeshAnythingV2/tree/main"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-HF-orange"></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/spaces/Yiwen-ntu/MeshAnythingV2"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-HF-orange"></a>

</div>


<p align="center">
    <img src="demo/demo_video.gif" alt="Demo GIF" width="512px" />
</p>


## Running point_cloud_interface.ipynb on Google Colab
Works with CUDA 11.8 or higher and with A800 GPUs or higher.

- Upload `point_cloud_inference.ipynb` to Google Drive
- Make sure paths are set properly in `jupyter notebook`


### Point Cloud Command line inference
```
# Note: if you want to use your own point cloud, please make sure the normal is included.
# The file format should be a .npy file with shape (N, 6), where N is the number of points. The first 3 columns are the coordinates, and the last 3 columns are the normal.

# inference for folder
python main.py --input_dir pc_examples --out_dir pc_output --input_type pc_normal

# inference for single file
python main.py --input_path pc_examples/grenade.npy --out_dir pc_output --input_type pc_normal
```

## BibTeX
```
@misc{chen2024meshanythingv2artistcreatedmesh,
      title={MeshAnything V2: Artist-Created Mesh Generation With Adjacent Mesh Tokenization}, 
      author={Yiwen Chen and Yikai Wang and Yihao Luo and Zhengyi Wang and Zilong Chen and Jun Zhu and Chi Zhang and Guosheng Lin},
      year={2024},
      eprint={2408.02555},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.02555}, 
}
```
