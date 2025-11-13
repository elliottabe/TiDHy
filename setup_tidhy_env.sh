#!/bin/bash
# setup_tidhy_env.sh
# Complete setup script for TiDHy environment
# Creates conda environment and installs all Python packages with UV
#
# Usage:
#   bash setup_tidhy_env.sh

set -e  # Exit on error

ENV_NAME="TiDHy"
ENV_FILE="environment.yaml"

echo "========================================================"
echo "TiDHy Environment Setup"
echo "========================================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed or not in PATH!"
    echo "Please install Miniconda or Anaconda first."
    exit 1
fi

# Check if environment file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found!"
    echo "Please run this script from the TiDHy repository root."
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "WARNING: Environment '$ENV_NAME' already exists!"
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
    else
        echo "Keeping existing environment. Skipping conda env creation."
        echo "Will attempt to install/update packages with uv..."
        SKIP_ENV_CREATE=1
    fi
fi

# Create conda environment
if [ -z "$SKIP_ENV_CREATE" ]; then
    echo ""
    echo "Step 1/3: Creating conda environment from $ENV_FILE"
    echo "This will install: Python, RAPIDS, numpy, scipy, matplotlib, etc."
    echo "--------------------------------------------------------"
    conda env create -f "$ENV_FILE"
    echo ""
    echo "✓ Conda environment created successfully!"
fi

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Activate environment
echo ""
echo "Step 2/3: Activating environment '$ENV_NAME'"
echo "--------------------------------------------------------"
conda activate "$ENV_NAME"

if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo "ERROR: Failed to activate environment!"
    exit 1
fi

echo "✓ Environment activated"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not installed in the environment!"
    echo "This should have been installed by conda. Something went wrong."
    exit 1
fi

# Install Python packages with uv
echo ""
echo "Step 3/3: Installing Python packages with UV (fast!)"
echo "This will install: JAX, Flax, Optax, Dynamax, Hydra, etc."
echo "--------------------------------------------------------"

uv pip install \
  --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
  'jax[cuda12]>=0.4.20' \
  'jaxlib>=0.4.20' \
  'flax>=0.8.0' \
  'optax>=0.1.7' \
  'orbax-checkpoint>=0.4.0' \
  'chex>=0.1.8' \
  'dynamax>=1.0.0' \
  'scikit-learn>=1.3.0' \
  'hydra-core>=1.3.0' \
  'omegaconf>=2.3.0' \
  'wandb' \
  'tqdm>=4.65.0' \
  'natsort' \
  -e .

echo ""
echo "✓ All packages installed successfully!"

# Verify installation
echo ""
echo "========================================================"
echo "Verifying Installation"
echo "========================================================"

python -c "
import sys
try:
    import jax
    print(f'✓ JAX version: {jax.__version__}')

    devices = jax.devices()
    print(f'✓ JAX devices: {devices}')

    if any('cuda' in str(d).lower() or 'gpu' in str(d).lower() for d in devices):
        print('✓ CUDA/GPU support: ENABLED')
    else:
        print('⚠ CUDA/GPU support: NOT DETECTED (CPU only)')

    import flax
    print(f'✓ Flax version: {flax.__version__}')

    import dynamax
    print(f'✓ Dynamax version: {dynamax.__version__}')

    import TiDHy
    print(f'✓ TiDHy package: INSTALLED')

    print('')
    print('All core packages verified successfully!')

except Exception as e:
    print(f'✗ Verification failed: {e}', file=sys.stderr)
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================"
    echo "Setup Complete!"
    echo "========================================================"
    echo ""
    echo "To activate the environment, run:"
    echo "  conda activate $ENV_NAME"
    echo ""
    echo "To test GPU support:"
    echo "  python -c 'import jax; print(jax.devices())'"
    echo ""
    echo "To start training:"
    echo "  python Run_TiDHy_NNX_vmap.py"
    echo ""
else
    echo ""
    echo "========================================================"
    echo "Setup completed with warnings"
    echo "========================================================"
    echo "Please check the verification output above."
    exit 1
fi
