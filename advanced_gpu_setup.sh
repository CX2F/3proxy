
#!/bin/bash

cd gpt2-llm.c

# Install cuDNN if possible
echo "Attempting to install cuDNN..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcudnn9-dev-cuda-12

# Clone cuDNN frontend
echo "Setting up cuDNN frontend..."
git clone https://github.com/NVIDIA/cudnn-frontend.git

# Compile with cuDNN support
echo "Compiling with cuDNN support..."
make train_gpt2cu USE_CUDNN=1

echo "Setup complete. Run ./train_gpt2cu to train with mixed precision and Flash Attention."
