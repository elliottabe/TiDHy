"""
Training utilities for TiDHy NNX model.

This module provides training loops, evaluation functions, and utilities
for working with the NNX-based TiDHy implementation.
"""

from typing import Tuple, Dict, Optional
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from tqdm.auto import tqdm

from TiDHy.models.TiDHy_nnx import TiDHy


def create_optimizer(
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    schedule: Optional[optax.Schedule] = None
) -> optax.GradientTransformation:
    """
    Create optimizer for model parameters.

    Args:
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        schedule: Optional learning rate schedule

    Returns:
        optax optimizer
    """
    if schedule is not None:
        return optax.adamw(schedule, weight_decay=weight_decay)
    else:
        return optax.adamw(learning_rate, weight_decay=weight_decay)


def train_step(
    model: TiDHy,
    optimizer: nnx.Optimizer,
    X: jax.Array
) -> Dict[str, float]:
    """
    Single training step.

    Args:
        model: TiDHy model instance
        optimizer: NNX optimizer
        X: Input batch of shape (batch_size, T, input_dim)

    Returns:
        Dictionary of metrics
    """

    def loss_fn(model: TiDHy):
        """Compute loss for training"""
        spatial_loss_rhat, spatial_loss_rbar, temp_loss, r2_losses, _, _ = model(X, training=True)

        total_loss = spatial_loss_rhat + spatial_loss_rbar + temp_loss

        if model.use_r2_decoder:
            total_loss += r2_losses

        return total_loss, {
            'spatial_loss_rhat': spatial_loss_rhat,
            'spatial_loss_rbar': spatial_loss_rbar,
            'temp_loss': temp_loss,
            'r2_losses': r2_losses
        }

    # Compute gradients
    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

    # Update parameters
    optimizer.update(grads)

    # Normalize if needed
    if model.normalize_spatial or model.normalize_temporal:
        model.normalize()

    metrics['loss'] = loss

    return metrics


def train_epoch(
    model: TiDHy,
    optimizer: nnx.Optimizer,
    train_data: jax.Array,
    batch_size: Optional[int] = None
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: TiDHy model instance
        optimizer: NNX optimizer
        train_data: Training data of shape (n_samples, T, input_dim)
        batch_size: Batch size (if None, use all data)

    Returns:
        Dictionary of averaged metrics
    """
    n_samples = train_data.shape[0]

    if batch_size is None:
        # Single batch
        metrics = train_step(model, optimizer, train_data)
        return metrics

    # Multiple batches
    n_batches = n_samples // batch_size
    all_metrics = []

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch = train_data[start_idx:end_idx]

        metrics = train_step(model, optimizer, batch)
        all_metrics.append(metrics)

    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = jnp.mean(jnp.array([m[key] for m in all_metrics]))

    return avg_metrics


def train_model(
    model: TiDHy,
    train_data: jax.Array,
    n_epochs: int,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: Optional[int] = None,
    val_data: Optional[jax.Array] = None,
    verbose: bool = True
) -> Tuple[TiDHy, Dict[str, list]]:
    """
    Train model for multiple epochs.

    Args:
        model: TiDHy model instance
        train_data: Training data of shape (n_samples, T, input_dim)
        n_epochs: Number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        batch_size: Batch size (if None, use all data)
        val_data: Optional validation data
        verbose: Whether to print progress

    Returns:
        Tuple of (trained model, history dictionary)
    """
    # Create optimizer
    optimizer_tx = create_optimizer(learning_rate, weight_decay)
    optimizer = nnx.Optimizer(model=model, tx=optimizer_tx, wrt=nnx.Param)

    # History
    history = {
        'loss': [],
        'spatial_loss_rhat': [],
        'spatial_loss_rbar': [],
        'temp_loss': [],
        'r2_losses': []
    }

    if val_data is not None:
        history['val_loss'] = []

    # Training loop
    epoch_iter = tqdm(range(n_epochs), desc="Training") if verbose else range(n_epochs)

    for epoch in epoch_iter:
        # Train
        metrics = train_epoch(model, optimizer, train_data, batch_size)

        # Record history
        for key in ['loss', 'spatial_loss_rhat', 'spatial_loss_rbar', 'temp_loss', 'r2_losses']:
            history[key].append(float(metrics[key]))

        # Validation
        if val_data is not None:
            val_metrics = evaluate_batch(model, val_data)
            history['val_loss'].append(float(val_metrics['loss']))

        # Print progress
        if verbose:
            msg = f"Epoch {epoch + 1}/{n_epochs} - Loss: {metrics['loss']:.4f}"
            if val_data is not None:
                msg += f" - Val Loss: {val_metrics['loss']:.4f}"
            if isinstance(epoch_iter, tqdm):
                epoch_iter.set_postfix_str(msg)
            else:
                print(msg)

    return model, history


@nnx.jit
def evaluate_batch(model: TiDHy, X: jax.Array) -> Dict[str, float]:
    """
    Evaluate model on a batch (JIT compiled).

    Args:
        model: TiDHy model instance
        X: Input batch of shape (batch_size, T, input_dim)

    Returns:
        Dictionary of metrics
    """
    spatial_loss_rhat, spatial_loss_rbar, temp_loss, r2_losses, _, _ = model(X, training=False)

    total_loss = spatial_loss_rhat + spatial_loss_rbar + temp_loss

    if model.use_r2_decoder:
        total_loss += r2_losses

    return {
        'loss': total_loss,
        'spatial_loss_rhat': spatial_loss_rhat,
        'spatial_loss_rbar': spatial_loss_rbar,
        'temp_loss': temp_loss,
        'r2_losses': r2_losses
    }


def evaluate_record(
    model: TiDHy,
    data_batch: jax.Array,
    verbose: bool = False
) -> Tuple[float, float, float, Dict[str, jax.Array]]:
    """
    Forward pass for evaluation with full recording.

    Args:
        model: TiDHy model instance
        data_batch: Input data of shape (batch_size, T, input_dim)
        verbose: Whether to show progress

    Returns:
        Tuple of (spatial_loss_rhat_avg, spatial_loss_rbar_avg, temp_loss_avg, result_dict)
    """
    batch_size, T, input_dim = data_batch.shape

    # Initialize storage arrays
    I_bar = jnp.zeros((batch_size, T, input_dim))
    I_hat = jnp.zeros((batch_size, T, input_dim))
    I = data_batch.copy()
    R_bar = jnp.zeros((batch_size, T, model.r_dim))
    R_hat = jnp.zeros((batch_size, T, model.r_dim))
    R2_hat = jnp.zeros((batch_size, T, model.r2_dim))
    W = jnp.zeros((batch_size, T, model.mix_dim))
    temp_loss = jnp.zeros((batch_size, T, model.r_dim))
    spatial_loss_rhat = jnp.zeros((batch_size, T, input_dim))
    spatial_loss_rbar = jnp.zeros((batch_size, T, input_dim))
    Ut = jnp.zeros((batch_size, T, model.r_dim, model.r_dim))

    if model.dyn_bias:
        b = jnp.zeros((batch_size, T, model.r_dim))

    # Initialize codes
    r, r2 = model.init_code(batch_size)

    # Store initial values
    R_bar = R_bar.at[:, 0].set(r)
    R2_hat = R2_hat.at[:, 0].set(r2)
    I_bar = I_bar.at[:, 0].set(model.spatial_decoder(r))

    spat_loss_fn = model.get_spatial_loss_fn()
    spatial_loss_rbar = spatial_loss_rbar.at[:, 0].set(
        spat_loss_fn(model.spatial_decoder(r), data_batch[:, 0])
    )

    # First step inference
    r = model.inf_first_step(data_batch[:, 0])

    R_hat = R_hat.at[:, 0].set(r)
    I_hat = I_hat.at[:, 0].set(model.spatial_decoder(r))
    spatial_loss_rhat = spatial_loss_rhat.at[:, 0].set(
        spat_loss_fn(model.spatial_decoder(r), data_batch[:, 0])
    )

    # Main loop through time
    t_range = tqdm(range(1, T), leave=False, desc="Evaluating") if verbose else range(1, T)

    for t in t_range:
        r_p = r.copy()
        r2_p = r2.copy()

        # Temporal prediction
        r_bar, V_t, w = model.temporal_prediction(r_p, r2)
        R_bar = R_bar.at[:, t].set(r_bar)
        x_bar = model.spatial_decoder(r_bar)
        I_bar = I_bar.at[:, t].set(x_bar)

        # Inference
        r, r2, _ = model.inf(data_batch[:, t], r_p, r2)

        R_hat = R_hat.at[:, t].set(r)
        R2_hat = R2_hat.at[:, t].set(r2)
        x_hat = model.spatial_decoder(r)
        I_hat = I_hat.at[:, t].set(x_hat)

        # Store mixture weights and dynamics
        wb = model.hypernet(r2)
        W = W.at[:, t].set(wb[:, :model.mix_dim])
        if model.dyn_bias:
            b = b.at[:, t].set(wb[:, model.mix_dim:])
        Ut = Ut.at[:, t].set(V_t)

        # Compute losses
        spatial_loss_rhat = spatial_loss_rhat.at[:, t].set(
            spat_loss_fn(x_hat, data_batch[:, t])
        )
        spatial_loss_rbar = spatial_loss_rbar.at[:, t].set(
            spat_loss_fn(x_bar, data_batch[:, t])
        )
        temp_loss = temp_loss.at[:, t].set((r - r_bar) ** 2)

    # Compile results
    result_dict = {
        'I_bar': I_bar,
        'I_hat': I_hat,
        'I': I,
        'R_bar': R_bar,
        'R_hat': R_hat,
        'R2_hat': R2_hat,
        'W': W,
        'Ut': Ut,
        'temp_loss': temp_loss,
        'spatial_loss_rhat': spatial_loss_rhat,
        'spatial_loss_rbar': spatial_loss_rbar,
    }

    if model.dyn_bias:
        result_dict['b'] = b

    # Compute average losses
    spatial_loss_rhat_avg = spatial_loss_rhat.reshape(batch_size, -1).sum(1).mean(0)
    spatial_loss_rbar_avg = spatial_loss_rbar.reshape(batch_size, -1).sum(1).mean(0)
    temp_loss_avg = model.temp_weight * temp_loss.reshape(batch_size, -1).sum(1).mean(0)

    return spatial_loss_rhat_avg, spatial_loss_rbar_avg, temp_loss_avg, result_dict


def save_model(model: TiDHy, filepath: str):
    """
    Save model to disk.

    Args:
        model: TiDHy model instance
        filepath: Path to save model
    """
    # Extract state
    _, state = nnx.split(model)

    # Save state
    with open(filepath, 'wb') as f:
        import pickle
        pickle.dump(state, f)

    print(f"Model saved to {filepath}")


def load_model(model: TiDHy, filepath: str) -> TiDHy:
    """
    Load model from disk.

    Args:
        model: TiDHy model instance (for structure)
        filepath: Path to saved model

    Returns:
        Loaded model
    """
    # Load state
    with open(filepath, 'rb') as f:
        import pickle
        state = pickle.load(f)

    # Merge state into model
    graphdef, _ = nnx.split(model)
    loaded_model = nnx.merge(graphdef, state)

    print(f"Model loaded from {filepath}")
    return loaded_model


def checkpoint_model(
    model: TiDHy,
    optimizer: nnx.Optimizer,
    epoch: int,
    filepath: str
):
    """
    Save checkpoint including optimizer state.

    Args:
        model: TiDHy model instance
        optimizer: NNX optimizer
        epoch: Current epoch
        filepath: Path to save checkpoint
    """
    _, model_state = nnx.split(model)
    _, opt_state = nnx.split(optimizer)

    checkpoint = {
        'model_state': model_state,
        'optimizer_state': opt_state,
        'epoch': epoch
    }

    with open(filepath, 'wb') as f:
        import pickle
        pickle.dump(checkpoint, f)

    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    model: TiDHy,
    optimizer: nnx.Optimizer,
    filepath: str
) -> Tuple[TiDHy, nnx.Optimizer, int]:
    """
    Load checkpoint including optimizer state.

    Args:
        model: TiDHy model instance (for structure)
        optimizer: NNX optimizer (for structure)
        filepath: Path to saved checkpoint

    Returns:
        Tuple of (loaded model, loaded optimizer, epoch)
    """
    with open(filepath, 'rb') as f:
        import pickle
        checkpoint = pickle.load(f)

    # Restore model
    model_graphdef, _ = nnx.split(model)
    loaded_model = nnx.merge(model_graphdef, checkpoint['model_state'])

    # Restore optimizer
    opt_graphdef, _ = nnx.split(optimizer)
    loaded_optimizer = nnx.merge(opt_graphdef, checkpoint['optimizer_state'])

    epoch = checkpoint['epoch']

    print(f"Checkpoint loaded from {filepath}, epoch {epoch}")
    return loaded_model, loaded_optimizer, epoch
