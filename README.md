# TiDHy: Timescale Demixing via Hypernetworks

## Installation

TiDHy provides two conda environments optimized for different use cases:

### Quick Start: TiDHy Environment (Main Development)

The TiDHy environment includes JAX/Flax, RAPIDS for GPU-accelerated operations, and all necessary dependencies.

**One-command setup:**
```bash
bash setup_tidhy_env.sh
```

This script will:
1. Create the conda environment with Python 3.13, RAPIDS 25.10, and CUDA 12.x support
2. Install all Python packages using UV (fast dependency resolver)
3. Install TiDHy as an editable package
4. Verify the installation and check GPU/CUDA availability

**Activate the environment:**
```bash
conda activate TiDHy
```

### SSM Baseline Environment (Optional)

For running SSM baseline comparisons (ARHMM, SLDS, etc.):

```bash
bash setup_ssm_env.sh
```

**Activate the environment:**
```bash
conda activate ssm
```

### Manual Installation (Advanced)

If you prefer manual setup:

1. **Create conda environment:**
   ```bash
   conda env create -f environment.yaml  # For TiDHy
   # OR
   conda env create -f ssm_environment.yml  # For SSM baselines
   ```

2. **Activate environment:**
   ```bash
   conda activate TiDHy  # or 'ssm'
   ```

3. **Install Python packages with UV:**
   ```bash
   uv pip install --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
     'jax[cuda12]>=0.4.20' 'jaxlib>=0.4.20' 'flax>=0.8.0' \
     'optax>=0.1.7' 'orbax-checkpoint>=0.4.0' 'chex>=0.1.8' \
     'dynamax>=1.0.0' 'scikit-learn>=1.3.0' \
     'hydra-core>=1.3.0' 'omegaconf>=2.3.0' 'wandb' \
     'tqdm>=4.65.0' 'natsort' -e .
   ```

### Requirements

- **Conda/Miniconda**: Required for environment management
- **CUDA 12.x**: For GPU acceleration (check with `nvidia-smi`)
- **Python 3.13**: Installed automatically by conda

### Compatibility Notes

- **TensorFlow Probability + JAX 0.8+**: TFP 0.25.0 requires a compatibility patch for JAX 0.8+. The patch is automatically applied in all entry point scripts (`Run_TiDHy_NNX_vmap.py`, etc.) via `TiDHy.utils.tfp_jax_patch.apply_tfp_jax_patch()`. If you import TiDHy modules directly, apply the patch before importing.
- **JAX**: Version 0.8+ recommended for Python 3.13 support
- **RAPIDS**: Version 25.10 for latest features and Python 3.13 support

### Verify Installation

Check if JAX can detect your GPU:
```bash
python -c "import jax; print(jax.devices())"
```

Expected output should show CUDA/GPU devices if properly configured.


## Usage

### Training TiDHy Models

Run the main training script with Hydra configuration overrides:

```bash
python Run_TiDHy_NNX_vmap.py dataset=SLDS model=sparsity
```

Available datasets: `SLDS`, `SSM`, `Rossler`, `AnymalTerrain`, `CalMS21`

Available model configs: `default_model`, `sparsity`, `r2_sparse`, 

### Running Baseline Models

**SSM baseline (requires ssm environment):**
```bash
conda activate ssm
python Run_SSM.py dataset=SLDS
```


## Custom Datasets
To add a custom dataset you can load data in any way you want. The final formatting should follow the convention of:  
- train_data: `(time x features)`  
- val_data:   `(time x features)`  
- test_data:  `(time x features)`  

The data can then be stacked with overlapping windows using the `stack_data` function:  
`train_inputs = stack_data(train_inputs,cfg.train.sequence_length,overlap=cfg.train.sequence_length//cfg.train.overlap_factor)`
