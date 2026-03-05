#!/bin/bash
# Setup training environment for EfficientNet-B3 on a GPU machine (WSL2/Linux).
# Requires: NVIDIA GPU with CUDA support, ~100GB free disk space.
# Run as: bash scripts/setup_training_env.sh
set -euo pipefail

echo "=== EfficientNet-B3 Training Environment Setup ==="

# 1. System packages
echo ">>> Installing system packages..."
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip \
  wget curl git parallel tabix samtools bcftools

# 2. Create virtualenv with Python 3.10
echo ">>> Creating Python virtualenv..."
python3.10 -m venv ~/dv-train-env
source ~/dv-train-env/bin/activate

# 3. Install TensorFlow with GPU support (2.13.1 to match DeepVariant)
echo ">>> Installing TensorFlow 2.13.1 with GPU support..."
pip install --upgrade pip
pip install tensorflow[and-cuda]==2.13.1

# 4. Install DeepVariant training dependencies
echo ">>> Installing training dependencies..."
pip install ml-collections clu tf-models-official==2.13.2 protobuf==3.20.3

# 5. Verify GPU detection
echo ">>> Verifying GPU..."
python3 -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs detected: {len(gpus)}')
for g in gpus:
    print(f'  {g}')
if not gpus:
    print('WARNING: No GPU detected! Training will be very slow.')
    print('Make sure NVIDIA drivers are installed on the Windows host.')
"

# 6. Create data directories
echo ">>> Creating data directories..."
mkdir -p ~/dv-training-data/{reference,bam,truth,training_examples,tuning_examples,output}

echo ""
echo "=== Setup complete ==="
echo "Activate the environment with: source ~/dv-train-env/bin/activate"
echo "Next step: bash scripts/download_training_data.sh"
