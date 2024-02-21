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