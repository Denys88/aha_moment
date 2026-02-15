#!/usr/bin/env bash
set -euo pipefail

# install_deps.sh — Install system-level dependencies for GRPO training
# Tested on Ubuntu 24.04 (WSL2) with RTX 5090

echo "=== Installing CUDA Toolkit 12.8 ==="

# Add NVIDIA package repo
if [ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ] && ! dpkg -s cuda-keyring &>/dev/null; then
    KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
    wget -q "$KEYRING_URL" -O /tmp/cuda-keyring.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    rm /tmp/cuda-keyring.deb
    sudo apt-get update
fi

# Install CUDA toolkit and compatible gcc
sudo apt-get install -y cuda-toolkit-12-8 gcc-12 g++-12

echo ""
echo "=== Verifying installation ==="
/usr/local/cuda-12.8/bin/nvcc --version

echo ""
echo "=== Add these to your shell profile (.bashrc / .zshrc): ==="
cat <<'ENVBLOCK'

# CUDA 12.8
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:/usr/local/cuda-12.8/nvvm/bin:$PATH

# Force nvcc to use gcc-12 (CUDA 12.x requires gcc <= 12)
export NVCC_PREPEND_FLAGS="--compiler-bindir=/usr/bin/gcc-12"

ENVBLOCK

echo "=== Done. Source your profile or export the variables above, then: ==="
echo "    python3 -m venv venv"
echo "    source venv/bin/activate"
echo "    pip install -r requirements_pretrained.txt"
echo "    python train_pretrained.py"
