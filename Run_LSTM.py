#!/usr/bin/env python3
"""
LSTM Baseline Training Script

A simple LSTM baseline for comparison with TiDHy.
Uses the same data loading, configuration, and logging infrastructure.
"""

# Apply TFP compatibility patch BEFORE importing JAX-dependent modules
from TiDHy.utils.tfp_jax_patch import apply_tfp_jax_patch

apply_tfp_jax_patch()

import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp
from flax import nnx
import wandb
from pathlib import Path
import time
import contextlib
from tqdm import tqdm

# TiDHy imports
from TiDHy.utils.path_utils import convert_dict_to_path, convert_dict_to_string
from TiDHy.datasets.load_data import load_data, stack_data
from TiDHy.models.LSTM_baseline import LSTMBaseline
from TiDHy.models.LSTM_training import (
    create_optimizer,
    train_epoch,
    evaluate,
    checkpoint_lstm_model,
    load_lstm_checkpoint,
    discover_lstm_checkpoint,
    evaluate_lstm_record,
)
import orbax.checkpoint as ocp
import TiDHy.utils.io_dict_to_hdf5 as ioh5
import os
import logging


def setup_device(cfg: DictConfig) -> jax.Device:
    """
    Configure GPU/device selection programmatically using JAX.

    This function uses JAX's programmatic device selection, which works reliably
    on clusters without needing to set CUDA_VISIBLE_DEVICES before import.

    Args:
        cfg: Configuration object with train.device_id parameter
            - If device_id is None/null: Use default device (typically GPU 0)
            - If device_id is an integer: Use that specific GPU ID

    Returns:
        The selected JAX device, or None to use JAX's default device
    """
    try:
        all_devices = jax.devices()
        print(f"JAX detected devices: {all_devices}")

        if hasattr(cfg.train, "device_id") and cfg.train.device_id is not None:
            device_id = int(cfg.train.device_id)

            # Get GPU devices only
            gpu_devices = [d for d in all_devices if d.platform == "gpu"]

            if not gpu_devices:
                print("Warning: No GPU devices found. Falling back to default device.")
                return None

            if device_id >= len(gpu_devices):
                print(
                    f"Warning: Requested GPU {device_id} but only {len(gpu_devices)} GPUs available. "
                    f"Using GPU {len(gpu_devices) - 1} instead."
                )
                device_id = len(gpu_devices) - 1

            selected_device = gpu_devices[device_id]
            print(f"GPU selection: Using {selected_device}")
            return selected_device
        else:
            print("GPU selection: Using JAX default device (no device_id specified)")
            return None

    except Exception as e:
        print(f"Warning: Could not configure device: {e}. Using JAX default.")
        return None


def resolve_default(cfg, key, default):
    """Get config value with default fallback."""
    try:
        value = OmegaConf.select(cfg, key)
        return value if value is not None else default
    except:
        return default


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""

    print("=" * 60)
    print("LSTM Baseline Training")
    print("=" * 60)
    print(f"\nDataset: {cfg.dataset.name}")
    print(f"Feature type: {resolve_default(cfg, 'train.feature_type', 'raw')}")

    # Set up paths
    cfg.paths = convert_dict_to_path(cfg.paths)

    # Setup GPU/device selection
    print("\n" + "=" * 60)
    print("GPU Setup")
    print("=" * 60)
    selected_device = setup_device(cfg)

    # Use context manager to run all computations on selected device
    device_context = (
        jax.default_device(selected_device) if selected_device else contextlib.nullcontext()
    )

    with device_context:
        # Save configuration
        temp_cfg = cfg.copy()
        temp_cfg.paths = convert_dict_to_string(temp_cfg.paths)
        print(f"Configuration:\n{OmegaConf.to_yaml(temp_cfg)}")
        OmegaConf.save(temp_cfg, cfg.paths.log_dir / "run_config.yaml")

        # Load data
        print("\n" + "=" * 60)
        print("Loading Data")
        print("=" * 60)
        data_dict = load_data(cfg)

        # Convert to JAX arrays
        train_inputs = jnp.array(data_dict["inputs_train"], dtype=jnp.float32)
        val_inputs = jnp.array(data_dict["inputs_val"], dtype=jnp.float32)
        test_inputs = jnp.array(data_dict["inputs_test"], dtype=jnp.float32)

        print(f"Train data shape: {train_inputs.shape}")
        print(f"Val data shape: {val_inputs.shape}")
        print(f"Test data shape: {test_inputs.shape}")

        # Stack data into sequences if enabled
        if cfg.train.stack_inputs:
            print(f"\nStacking data with sequence_length={cfg.train.sequence_length}")
            overlap = cfg.train.sequence_length // resolve_default(cfg, "train.overlap_factor", 2)
            train_inputs = stack_data(train_inputs, cfg.train.sequence_length, overlap=overlap)
            val_inputs = stack_data(val_inputs, cfg.train.sequence_length, overlap=overlap)
            test_inputs = stack_data(test_inputs, cfg.train.sequence_length, overlap=overlap)

            print(f"After stacking:")
            print(f"  Train: {train_inputs.shape}")
            print(f"  Val: {val_inputs.shape}")
            print(f"  Test: {test_inputs.shape}")

        input_dim = train_inputs.shape[-1]
        print(f"\nInput dimension: {input_dim}")

        # Initialize model
        print("\n" + "=" * 60)
        print("Initializing Model")
        print("=" * 60)

        hidden_dim = resolve_default(cfg, "model.hidden_dim", 64)
        num_layers = resolve_default(cfg, "model.num_layers", 1)
        dropout_rate = resolve_default(cfg, "model.dropout_rate", 0.0)

        print(f"Hidden dim: {hidden_dim}")
        print(f"Num layers: {num_layers}")
        print(f"Dropout rate: {dropout_rate}")

        rngs = nnx.Rngs(0)
        model = LSTMBaseline(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=input_dim,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        rngs=rngs,
        )

        # Count parameters
        num_params = sum(param.size for param in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
        print(f"\nTotal parameters: {num_params:,}")

        # Create optimizer
        learning_rate = resolve_default(cfg, "train.learning_rate", 1e-3)
        weight_decay = resolve_default(cfg, "train.weight_decay", 1e-4)
        use_schedule = resolve_default(cfg, "train.use_schedule", False)

        print(f"\nOptimizer settings:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Use schedule: {use_schedule}")

        optimizer_tx = create_optimizer(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        use_schedule=use_schedule,
        schedule_transition_steps=resolve_default(cfg, "train.schedule_transition_steps", 200),
        schedule_decay_rate=resolve_default(cfg, "train.schedule_decay", 0.96),
        )
        optimizer = nnx.Optimizer(model, optimizer_tx)

        # Check for existing checkpoints and load if found
        start_epoch = 0
        restore_checkpoint = resolve_default(cfg, "restore_checkpoint", "")

        if restore_checkpoint and Path(restore_checkpoint).exists():
            print(f"\n" + "=" * 60)
            print("Loading Checkpoint")
            print("=" * 60)
            try:
                model, optimizer, start_epoch = load_lstm_checkpoint(
                    model, optimizer, restore_checkpoint
                )
                print(f"Resuming from epoch {start_epoch}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                print("Starting fresh training...")
                start_epoch = 0
        elif cfg.paths.ckpt_dir.exists():
            # Auto-discover latest checkpoint
            latest_ckpt = discover_lstm_checkpoint(str(cfg.paths.ckpt_dir))
            if latest_ckpt is not None:
                print(f"\n" + "=" * 60)
                print("Auto-discovered Checkpoint")
                print("=" * 60)
                print(f"Found checkpoint: {latest_ckpt}")
                try:
                    model, optimizer, start_epoch = load_lstm_checkpoint(
                        model, optimizer, str(latest_ckpt)
                    )
                    print(f"Resuming from epoch {start_epoch}")
                except Exception as e:
                    print(f"Failed to load checkpoint: {e}")
                    print("Starting fresh training...")
                    start_epoch = 0

        # Loss weights
        reconstruction_weight = resolve_default(cfg, "model.reconstruction_weight", 1.0)
        prediction_weight = resolve_default(cfg, "model.prediction_weight", 1.0)

        print(f"\nLoss weights:")
        print(f"  Reconstruction: {reconstruction_weight}")
        print(f"  Prediction: {prediction_weight}")

        # Initialize WandB
        if resolve_default(cfg, "train.use_wandb", False):
            print("\n" + "=" * 60)
            print("Initializing WandB")
            print("=" * 60)

            run_id = f"lstm_{cfg.dataset.name}_{int(time.time())}"
            wandb.init(
                dir=cfg.paths.log_dir,
                project=cfg.train.wandb_project,
                config=OmegaConf.to_container(cfg, resolve=True),
                notes=resolve_default(cfg, "note", ""),
                id=run_id,
                resume="allow",
                name=f"LSTM_{cfg.dataset.name}_{hidden_dim}h_{num_layers}L",
            )
            print(f"WandB run: {run_id}")
        else:
            wandb = None
    
        # Training loop
        print("\n" + "=" * 60)
        print("Training")
        print("=" * 60)

        num_epochs = cfg.train.num_epochs
        batch_size = resolve_default(cfg, "train.batch_size", None)
        show_progress = resolve_default(cfg, "train.show_progress", True)
        checkpoint_max_to_keep = resolve_default(cfg, "model.checkpoint_max_to_keep", 5)

        best_train_loss = float("inf")
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_since_validation = 0

        # Validation cooldown period (minimum epochs between validation runs)
        validation_cooldown = resolve_default(cfg, "train.validation_cooldown", 5)

        # Create Orbax checkpointer (persistent for async saves)
        checkpointer = ocp.StandardCheckpointer()

        epoch_iterator = (tqdm(range(start_epoch, num_epochs), desc="Training") if show_progress else range(start_epoch, num_epochs))

        for epoch in epoch_iterator:
            # Train
            train_metrics = train_epoch(
                model,
                optimizer,
                train_inputs,
                batch_size=batch_size,
                reconstruction_weight=reconstruction_weight,
                prediction_weight=prediction_weight,
            )

            # Track current training loss
            current_train_loss = float(train_metrics["loss"])

            # Increment validation cooldown counter
            epochs_since_validation += 1

            # Validation - run when training loss improves AND cooldown period has passed
            should_validate = False
            new_train_minimum = current_train_loss < best_train_loss

            if new_train_minimum:
                if epochs_since_validation >= validation_cooldown:
                    should_validate = True
                    if show_progress:
                        tqdm.write(
                            f"\nNew training loss minimum: {current_train_loss:.6f} (prev: {best_train_loss:.6f}). Running validation..."
                        )
                else:
                    if show_progress:
                        tqdm.write(
                            f"\nNew training loss minimum: {current_train_loss:.6f}, but cooldown active ({epochs_since_validation}/{validation_cooldown} epochs)"
                        )

                # Update best training loss
                best_train_loss = current_train_loss

            if should_validate:
                # Reset cooldown counter
                epochs_since_validation = 0

                val_metrics = evaluate(
                    model,
                    val_inputs,
                    batch_size=batch_size,
                    reconstruction_weight=reconstruction_weight,
                    prediction_weight=prediction_weight,
                )

                # Save checkpoint if validation loss improved
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    best_epoch = epoch

                    # Save checkpoint
                    try:
                        checkpoint_lstm_model(
                            model,
                            optimizer,
                            epoch + 1,
                            str(cfg.paths.ckpt_dir),
                            checkpointer,
                            max_to_keep=checkpoint_max_to_keep,
                        )
                        if show_progress:
                            tqdm.write(
                                f"Validation loss improved to {val_metrics['loss']:.6f}. Checkpoint saved at epoch {epoch + 1}"
                            )
                    except Exception as e:
                        print(f"Warning: Failed to save checkpoint: {e}")
                else:
                    if show_progress:
                        tqdm.write(
                            f"Validation loss: {val_metrics['loss']:.6f} (best: {best_val_loss:.6f})"
                        )
            else:
                # For non-validation epochs, use placeholder metrics for logging
                val_metrics = {
                    "loss": best_val_loss,
                    "reconstruction_loss": 0.0,
                    "prediction_loss": 0.0,
                }

            # Log to WandB
            if wandb is not None:
                log_dict = {
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/reconstruction_loss": train_metrics["reconstruction_loss"],
                    "train/prediction_loss": train_metrics["prediction_loss"],
                    "train/grad_norm": train_metrics["grad_norm"],
                    "val/loss": val_metrics["loss"],
                    "val/reconstruction_loss": val_metrics["reconstruction_loss"],
                    "val/prediction_loss": val_metrics["prediction_loss"],
                    "best_train_loss": best_train_loss,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                }
                wandb.log(log_dict, step=epoch)

            # Print progress
            if show_progress and epoch % 10 == 0:
                tqdm.write(
                    f"Epoch {epoch:4d} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Best: {best_val_loss:.4f} @ {best_epoch}"
                )

            # Save final checkpoint
            print("\n" + "=" * 60)
            print("Saving Final Checkpoint")
            print("=" * 60)
            try:
                checkpoint_lstm_model(
                    model, optimizer, num_epochs, str(cfg.paths.ckpt_dir), checkpointer, final=True
                )
                print(f"Final checkpoint saved to {cfg.paths.ckpt_dir}/final/")
            except Exception as e:
                print(f"Warning: Failed to save final checkpoint: {e}")

            # Wait for all async checkpoint saves to complete
            checkpointer.wait_until_finished()
            print("All checkpoints saved successfully")

        # Save predictions and hidden states if requested
        save_results = resolve_default(cfg, "model.save_results", False)

        if save_results:
            print("\n" + "=" * 60)
            print("Saving Predictions and Hidden States")
            print("=" * 60)

            results_file = cfg.paths.log_dir / "lstm_results.h5"
            results_dict = {}

            # Determine which splits to save
            save_train = resolve_default(cfg, "model.save_train_results", False)
            save_val = resolve_default(cfg, "model.save_val_results", True)
            save_test = resolve_default(cfg, "model.save_test_results", True)

            # Save train results
            if save_train:
                print("Recording train set...")
                train_loss, train_results = evaluate_lstm_record(
                    model,
                    train_inputs,
                    reconstruction_weight=reconstruction_weight,
                    prediction_weight=prediction_weight,
                )
                results_dict["train"] = train_results
                print(f"  Train loss: {train_loss:.4f}")

            # Save validation results
            if save_val:
                print("Recording validation set...")
                val_loss, val_results = evaluate_lstm_record(
                    model,
                    val_inputs,
                    reconstruction_weight=reconstruction_weight,
                    prediction_weight=prediction_weight,
                )
                results_dict["val"] = val_results
                print(f"  Val loss: {val_loss:.4f}")

            # Save test results
            if save_test:
                print("Recording test set...")
                test_loss, test_results = evaluate_lstm_record(
                    model,
                    test_inputs,
                    reconstruction_weight=reconstruction_weight,
                    prediction_weight=prediction_weight,
                )
                results_dict["test"] = test_results
                print(f"  Test loss: {test_loss:.4f}")

            # Add metadata
            results_dict["metadata"] = {
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "input_dim": input_dim,
                "sequence_length": cfg.train.sequence_length,
                "reconstruction_weight": reconstruction_weight,
                "prediction_weight": prediction_weight,
            }

            # Save to HDF5
            print(f"\nSaving results to {results_file}...")
            ioh5.save(results_file, results_dict)
            print(f"✓ Results saved successfully")

            if wandb is not None:
                wandb.save(str(results_file))
                print("✓ Results uploaded to WandB")

        # Final evaluation on test set
        print("\n" + "=" * 60)
        print("Final Evaluation")
        print("=" * 60)

        test_metrics = evaluate(
        model,
        test_inputs,
        batch_size=batch_size,
        reconstruction_weight=reconstruction_weight,
        prediction_weight=prediction_weight,
    )

    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Reconstruction Loss: {test_metrics['reconstruction_loss']:.4f}")
    print(f"  Prediction Loss: {test_metrics['prediction_loss']:.4f}")

    if wandb is not None:
        wandb.log(
            {
                "test/loss": test_metrics["loss"],
                "test/reconstruction_loss": test_metrics["reconstruction_loss"],
                "test/prediction_loss": test_metrics["prediction_loss"],
            }
        )
        wandb.finish()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nBest validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")
    print(f"Checkpoints saved to: {cfg.paths.ckpt_dir}")


if __name__ == "__main__":
    main()
