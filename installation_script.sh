# install tensorflow in linux  with nvidia
conda create -n tf_env -y
conda activate tf_env
conda install python=3.9 -y
conda install -c conda-forge cudatoolkit=12.2.0 cudnn=8.1.0 -y
pip install --upgrade pip
pip install tensorflow
pip install numpy matplotlib seaborn scipy plotly scikit-learn openpyxl langchain streamlit openai huggingface_hub transformers ipykernel
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# install pytorch-gpu in linux with nvidia
conda create -n torch_gpu -y
conda activate torch_gpu
conda install python=3.11 -y
conda install -c nvidia cuda-nvcc -y
conda install -c "nvidia/label/cuda-12.2.0" cuda-nvcc -y
conda install -c anaconda cudatoolkit -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install numpy matplotlib seaborn scipy plotly scikit-learn openpyxl langchain streamlit openai huggingface_hub transformers ipykernel


# links
https://medium.com/nerd-for-tech/installing-tensorflow-with-gpu-acceleration-on-linux-f3f55dd15a9
https://medium.com/@jeanpierre_lv/installing-pytorch-with-gpu-support-on-ubuntu-a-step-by-step-guide-38dcf3f8f266
https://pytorch.org/get-started/locally/


# To install in Macbook m1
##########################-----------------Tensorflow installation on mac-----------------##########################
# install mambaforge and then
mamba create -n tf_env
mamba activate tf_env
conda config --env --set subdir osx-arm64
mamba install apple::tensorflow-deps==2.5.0 -y
mamba install -y python=3.9
mamba install -y jupyter pandas scipy plotly scikit-learn numpy seaborn matplotlib openpyxl pillow tqdm h5py ipykernel
pip install tensorflow-macos
pip install tensorflow-metal    
pip install Pyarrow   


##########################-----------------pytorch installation on mac-----------------##########################

# install mambaforge and then
# pytorch installation on mac M1
mamba create -n torch_gpu
mamba activate torch_gpu
mamba install python=3.8
mamba install pytorch::pytorch torchvision torchaudio -c pytorch -y
mamba install jupyter pandas numpy matplotlib scikit-learn tqdm -y
pip insytall ipykernel

#test pytorch
import torch
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
print(f"PyTorch version: {torch.__version__}")
# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")
# Set the device      
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
import torch
# Set the device
device = "mps" if torch.backends.mps.is_available() else "cpu"
# Create data and send it to the device
x = torch.rand(size=(3, 4)).to(device)
x