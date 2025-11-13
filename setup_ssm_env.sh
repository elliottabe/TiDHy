#!/bin/bash
# setup_ssm_env.sh
# Complete setup script for SSM baseline environment
# Creates conda environment and installs all Python packages with UV
#
# Usage:
#   bash setup_ssm_env.sh

set -e  # Exit on error

ENV_NAME="ssm"
ENV_FILE="ssm_environment.yml"

echo "========================================================"
echo "SSM Baseline Environment Setup"
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
    echo "This will install: Python, numpy, scipy, scikit-learn, etc."
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
echo "This will install: JAX, SSM, Dynamax, TensorFlow Probability, etc."
echo "--------------------------------------------------------"

uv pip install \
  --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
  'jax[cuda12]>=0.4.20' \
  'jaxlib>=0.4.20' \
  'optax>=0.2.0' \
  'chex>=0.1.8' \
  'jaxtyping>=0.3.0' \
  'dm-tree>=0.1.8' \
  'git+https://github.com/lindermanlab/ssm@main' \
  'dynamax>=1.0.0' \
  'tensorflow-probability>=0.24.0' \
  'hydra-core>=1.3.0' \
  'omegaconf>=2.3.0' \
  'tensorboardX>=2.6.0' \
  'wandb' \
  'dm-env>=1.6' \
  'gym-notices>=0.0.8' \
  'PyOpenGL>=3.1.7' \
  'pytinyrenderer>=0.0.14' \
  'trimesh>=4.4.0' \
  'autograd>=1.7.0' \
  'cloudpickle>=3.0.0' \
  'ml-collections>=0.1.1' \
  'fastprogress>=1.0.0' \
  'rich>=13.0.0' \
  'tqdm>=4.65.0' \
  'natsort' \
  'click>=8.1.0'

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

    import ssm
    print(f'✓ SSM (lindermanlab) version: {ssm.__version__}')

    import dynamax
    print(f'✓ Dynamax version: {dynamax.__version__}')

    import tensorflow_probability as tfp
    print(f'✓ TensorFlow Probability version: {tfp.__version__}')

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
    echo "To run SSM baselines:"
    echo "  python Run_ARHMM.py"
    echo "  python Run_SSM.py"
    echo ""
else
    echo ""
    echo "========================================================"
    echo "Setup completed with warnings"
    echo "========================================================"
    echo "Please check the verification output above."
    exit 1
fi
