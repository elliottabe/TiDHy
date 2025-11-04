# Standard library imports
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0
os.environ["JAX_CAPTURED_CONSTANTS_REPORT_FRAMES"]="-1"
from pathlib import Path
import jax 
jax.config.update("jax_compilation_cache_dir", (Path.cwd() / "tmp/jax_cache").as_posix())
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
try:
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
except AttributeError:
    pass  # Skip if not available in this JAX version

import logging
from typing import Optional, Tuple, Dict

# Third-party imports
import hydra
import jax.numpy as jnp
import wandb
from flax import nnx
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

# Local imports
from TiDHy.datasets.load_data import load_data, stack_data
from TiDHy.models.TiDHy_nnx_vmap import TiDHy
from TiDHy.models.TiDHy_nnx_vmap_training import (
    train_model_with_checkpointing,
    evaluate_record,
    load_checkpoint,
    create_multi_lr_optimizer,
    create_optimizer
)
from TiDHy.utils.path_utils import convert_dict_to_path, convert_dict_to_string
import TiDHy.utils.io_dict_to_hdf5 as ioh5

def setup_config_and_paths(cfg: DictConfig) -> Tuple[DictConfig, str]:
    """
    Handle config loading, run_id setup, and path conversion.

    Args:
        cfg: Hydra configuration object

    Returns:
        Tuple of (updated config, run_id)
    """
    # Handle load_jobid for HPC job requeuing
    if (
        ("load_jobid" in cfg)
        and (cfg["load_jobid"] is not None)
        and (cfg["load_jobid"] != "")
    ):
        run_id = cfg.load_jobid
        load_cfg_path = (
            Path(cfg.paths.base_dir) / f"run_id={run_id}/logs/run_config.yaml"
        )
        cfg = OmegaConf.load(load_cfg_path)
        logging.info(f"Loading job config from: {load_cfg_path}")
    else:
        run_id = cfg.run_id

    # Convert path strings to Path objects
    cfg.paths = convert_dict_to_path(cfg.paths)

    return cfg, run_id


def discover_checkpoint(cfg: DictConfig, run_id: str) -> Optional[Path]:
    """
    Auto-discover latest checkpoint or use explicit checkpoint path.

    Args:
        cfg: Configuration object
        run_id: Run identifier

    Returns:
        Path to checkpoint directory if found, None otherwise
    """
    restore_checkpoint = None

    # Auto-discover from checkpoint directory
    if cfg.paths.ckpt_dir.exists() and any(cfg.paths.ckpt_dir.iterdir()):
        ckpt_files = natsorted([
            Path(f.path) for f in os.scandir(cfg.paths.ckpt_dir) if f.is_dir()
        ])
        if len(ckpt_files) > 0:
            restore_checkpoint = ckpt_files[-1]  # Latest checkpoint

            # Try to load config from checkpoint run
            config_path = Path(cfg.paths.base_dir) / f"run_id={run_id}/logs/run_config.yaml"
            if config_path.exists():
                logging.info(f"Loading job config from: {config_path}")
            logging.info(f"Found checkpoint to resume from: {restore_checkpoint}")

    # Fallback to explicit restore checkpoint if specified
    if restore_checkpoint is None and hasattr(cfg, 'restore_checkpoint') and cfg.restore_checkpoint != "":
        restore_checkpoint = Path(cfg.restore_checkpoint)
        logging.info(f"Using explicit restore checkpoint: {restore_checkpoint}")

    return restore_checkpoint


def prepare_data(cfg: DictConfig) -> Tuple[jax.Array, jax.Array, int]:
    """
    Load and prepare training/validation data.

    Args:
        cfg: Configuration object

    Returns:
        Tuple of (train_inputs, val_inputs, input_dim)
    """
    # Load dataset
    data_dict = load_data(cfg)

    # Convert to JAX arrays
    train_inputs = jnp.array(data_dict['inputs_train'], dtype=jnp.float32)
    val_inputs = jnp.array(data_dict['inputs_val'], dtype=jnp.float32)

    input_dim = train_inputs.shape[-1]

    # Stack or reshape training data
    if cfg.train.stack_inputs:
        train_inputs = stack_data(
            train_inputs,
            cfg.train.sequence_length,
            overlap=cfg.train.sequence_length // cfg.train.overlap_factor
        )
    else:
        train_inputs = train_inputs.reshape(-1, cfg.train.sequence_length, input_dim)

    # Reshape validation data
    val_inputs = val_inputs.reshape(-1, cfg.train.sequence_length, input_dim)

    logging.info(f'Data prepared - train shape: {train_inputs.shape}, val shape: {val_inputs.shape}')

    return train_inputs, val_inputs, input_dim


def create_optimizer_from_config(cfg: DictConfig, model: TiDHy) -> nnx.Optimizer:
    """
    Create optimizer with proper learning rate schedules from config.

    Args:
        cfg: Configuration object
        model: TiDHy model instance

    Returns:
        NNX Optimizer
    """
    # Check if multi-LR setup is configured
    if hasattr(cfg.train, 'learning_rate_s') and cfg.train.learning_rate_s is not None:
        # Multi-LR optimizer with schedules
        return create_multi_lr_optimizer(
            model,
            learning_rate_s=cfg.train.learning_rate_s,
            learning_rate_t=getattr(cfg.train, 'learning_rate_t', cfg.train.learning_rate),
            learning_rate_h=getattr(cfg.train, 'learning_rate_h', cfg.train.learning_rate),
            weight_decay=getattr(cfg.train, 'weight_decay', 1e-4),
            use_schedule=getattr(cfg.train, 'use_schedule', False),
            schedule_transition_steps=getattr(cfg.train, 'schedule_transition_steps', 200),
            schedule_decay_rate=getattr(cfg.train, 'schedule_decay', 0.96)
        )
    else:
        # Single LR optimizer
        optimizer_tx = create_optimizer(
            cfg.train.learning_rate,
            getattr(cfg.train, 'weight_decay', 1e-4)
        )
        return nnx.Optimizer(model=model, tx=optimizer_tx, wrt=nnx.Param)


def setup_wandb(cfg: DictConfig, run_id: str, model: TiDHy, data_info: Dict) -> None:
    """
    Initialize Weights & Biases logging with model and data info.

    Args:
        cfg: Configuration object
        run_id: Run identifier
        model: TiDHy model instance
        data_info: Dictionary containing data shape information
    """
    wandb.init(
        dir=cfg.paths.log_dir,
        project=cfg.train.wandb_project,
        config=OmegaConf.to_container(cfg, resolve=True),
        notes=getattr(cfg, 'note', ''),
        id=f"{run_id}",
        resume="allow",
        name=f"{cfg.dataset.name}_{run_id}",
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training pipeline using NNX implementation with wandb logging.

    Args:
        cfg: Hydra configuration object
    """
    # ===== 1. Setup Configuration and Paths =====
    cfg, run_id = setup_config_and_paths(cfg)
    logging.info(f"Starting run: {run_id}")

    # ===== 2. Load and Prepare Data =====
    logging.info("Loading and preparing data...")
    train_inputs, val_inputs, input_dim = prepare_data(cfg)
    cfg.model.input_dim = input_dim

    # ===== 3. Create Model =====
    logging.info("Creating TiDHy model...")
    rngs = nnx.Rngs(cfg.seed)
    model_params = OmegaConf.to_container(cfg.model, resolve=True)
    model_params['input_dim'] = input_dim
    model = TiDHy(**model_params, rngs=rngs)
    logging.info('Model created successfully')

    # ===== 4. Handle Checkpoint Loading =====
    checkpoint_path = discover_checkpoint(cfg, run_id)
    loaded_optimizer = None
    start_epoch = 0

    if checkpoint_path is not None:
        logging.info(f"Attempting to load checkpoint from: {checkpoint_path}")
        try:
            # Create optimizer structure for loading
            loaded_optimizer = create_optimizer_from_config(cfg, model)

            # Load checkpoint
            model, loaded_optimizer, start_epoch = load_checkpoint(
                model, loaded_optimizer, checkpoint_path
            )
            logging.info(f"Successfully loaded checkpoint from epoch {start_epoch}")

        except Exception as e:
            logging.warning(f"Failed to load checkpoint: {e}")
            logging.info("Continuing with fresh model and optimizer...")
            loaded_optimizer = None
            start_epoch = 0

    # ===== 5. Setup Logging =====
    logging.info("Initializing wandb...")
    data_info = {
        'train_shape': train_inputs.shape,
        'val_shape': val_inputs.shape,
        'input_dim': input_dim
    }
    setup_wandb(cfg, run_id, model, data_info)

    # Save configuration
    temp_cfg = cfg.copy()
    temp_cfg.paths = convert_dict_to_string(temp_cfg.paths)
    logging.info(f"Configuration:\n{OmegaConf.to_yaml(temp_cfg)}")
    OmegaConf.save(temp_cfg, cfg.paths.log_dir / 'run_config.yaml')

    # ===== 6. Training =====
    logging.info('Starting training...')
    checkpoint_dir = cfg.paths.ckpt_dir
    checkpoint_dir.mkdir(exist_ok=True)

    trained_model, history = train_model_with_checkpointing(
        model=model,
        train_data=train_inputs,
        config=cfg,
        val_data=val_inputs,
        checkpoint_dir=str(checkpoint_dir),
        start_epoch=start_epoch,
        optimizer=loaded_optimizer,
        verbose=True,
        checkpoint_every=getattr(cfg.train, 'save_summary_steps', 1),
        # Enable wandb logging (wandb already initialized above)
        use_wandb=True,
        log_params_every=getattr(cfg.train, 'log_params_every', 10),
        log_sparsity_every=getattr(cfg.train, 'log_sparsity_every', 5)
    )
    ioh5.save(str(cfg.paths.log_dir / 'training_history.h5'), history)
    # ===== 7. Final Evaluation =====
    logging.info('Running final evaluation...')
    spatial_loss_rhat, spatial_loss_rbar, temp_loss, _ = evaluate_record(
        trained_model,
        val_inputs,
    )

    final_val_loss = spatial_loss_rhat + spatial_loss_rbar + temp_loss
    logging.info(f'Final validation loss: {final_val_loss:.4f}')

    # Log final metrics
    wandb.log({
        'final/val_spatial_loss_rhat': float(spatial_loss_rhat),
        'final/val_spatial_loss_rbar': float(spatial_loss_rbar),
        'final/val_temp_loss': float(temp_loss),
        'final/val_total_loss': float(final_val_loss)
    })

    # ===== 8. Cleanup =====
    wandb.finish()
    logging.info('Training completed successfully!')


if __name__ == "__main__":
    main()