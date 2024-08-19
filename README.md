# TGMformer: Transferability Guided Mask Transformer for Segmentation Domain Adaptation

Welcome to the official code repository for **TGMformer**, the Transferability Guided Mask Transformer, designed for segmentation domain adaptation. This repository contains all the necessary code and instructions to reproduce the experiments presented in our paper, "TGMformer: Transferability Guided Mask Transformer for Segmentation Domain Adaptation."

## Installation

### Requirements

- **Operating System**: Linux or macOS
- **Python**: Version ≥ 3.6
- **PyTorch**: Version ≥ 1.9 along with the corresponding version of `torchvision`. It's recommended to install them together from [pytorch.org](https://pytorch.org/) to ensure compatibility. Please verify that the PyTorch version aligns with the version required by Detectron2.
- **Detectron2**: Follow the official [Detectron2 installation instructions](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md) to install.
- **OpenCV** (Optional): Required for running demos and visualizations.
- **Other Dependencies**: Install additional requirements by running:

  ```bash
  pip install -r requirements.txt
  ```

### CUDA Kernel for MSDeformAttn

After setting up the required environment, compile the CUDA kernel for MSDeformAttn by running the following commands:

```bash
# Navigate to the directory containing the CUDA setup script
cd hgformer/modeling/pixel_decoder/ops

# Compile the CUDA kernel
python setup.py build install
```

Make sure that the `CUDA_HOME` environment variable is defined and points to the directory of your installed CUDA toolkit.

### Building on Another System

If you're building on a system without a GPU but with CUDA drivers installed, you can use the following command:

```bash
TORCH_CUDA_ARCH_LIST='8.0' FORCE_CUDA=1 python setup.py build install
```

### Example Conda Environment Setup

Below is an example of setting up the environment using Conda:

```bash
# Create a new conda environment with Python 3.8
conda create --name hgformer python=3.8 -y
conda activate hgformer

# Install PyTorch, torchvision, and CUDA toolkit
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia

# Optionally, install OpenCV for visualization
pip install -U opencv-python

# Install Detectron2 under your working directory
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

# Install Cityscapes scripts from the official repository
pip install git+https://github.com/mcordts/cityscapesScripts.git
```
