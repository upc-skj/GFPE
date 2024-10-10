# GFPE: Enhanced Camera Pose Estimation via 3D Gaussian Splatting and Feature Matching

\*\***notice:** The current upload contains only part of the code. The complete code is being organized, and we will upload it to this repository as soon as it is ready.\*\*

## Installation

Create environment through conda:

```
conda create -n GFPE python=3.8
conda activate GFPE
```

Install compatible versions of PyTorch and CUDA. The PyTorch version used in the experiment is 1.12.0, and the CUDA version is 11.3:

```
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

Clone the repository and install dependencies:

```
git clone https://github.com/upc-skj/GFPE
cd GFPE
pip install -r requirements.txt
```

## Acknowledgments

The code in this repository is based on or references the following code, and we appreciate their contributions:

3D Gaussian Splatting: https://github.com/graphdeco-inria/gaussian-splatting
```
@Article{kerbl3Dgaussians,
   author    = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
   title     = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
   journal   = {ACM Transactions on Graphics},
   number    = {4},
   volume    = {42},
   month     = {July},
   year      = {2023},
   url       = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```
iComMa: https://github.com/YuanSun-XJTU/iComMa

```
@Article{sun2023icomma,
  title      = {icomma: Inverting 3d gaussians splatting for camera pose estimation via comparing and matching},
  author     = {Sun, Yuan and Wang, Xuan and Zhang, Yunfan and Zhang, Jie and Jiang, Caigui and Guo, Yu and Wang, Fei},
  journal    = {arXiv preprint arXiv:2312.09031},
  year       = {2023}
}
```

LoFTR: https://github.com/zju3dv/LoFTR

```
@article{sun2021loftr,
  title      = {LoFTR: Detector-Free Local Feature Matching with Transformers},
  author     = {Sun, Jiaming and Shen, Zehong and Wang, Yuang and Bao, Hujun and Zhou, Xiaowei},
  journal    = {CVPR},
  year       = {2021}
}
```
