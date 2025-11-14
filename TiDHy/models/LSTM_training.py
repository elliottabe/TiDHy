"""
Training utilities for LSTM baseline model.
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
import shutil
from pathlib import Path
from natsort import natsorted
from typing import Tuple, Dict, Any, Optional
from TiDHy.models.LSTM_baseline import LSTMBaseline, compute_lstm_losses


def create_optimizer(learning_rate: float = 1e-3,
                    weight_decay: float = 1e-4,
                    use_schedule: bool = False,
                    schedule_transition_steps: int = 200,
                    schedule_decay_rate: float = 0.96,
                    clip_norm: float = 1.0) -> optax.GradientTransformation:
    """
    Create AdamW optimizer with optional learning rate schedule and gradient clipping.

    Args:
        learning_rate: Base learning rate
        weight_decay: Weight decay coefficient
        use_schedule: Whether to use exponential decay schedule
        schedule_transition_steps: Steps before decay starts
        schedule_decay_rate: Decay rate
        clip_norm: Gradient clipping norm

    Returns:
        Optax gradient transformation
    """
    if use_schedule:
        schedule = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=schedule_transition_steps,
            decay_rate=schedule_decay_rate,
            staircase=False
        )
    else:
        schedule = learning_rate

    # Create optimizer chain
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
    )

    return optimizer


@nnx.jit
def train_step(model: LSTMBaseline,
              optimizer: nnx.Optimizer,
              x: jnp.ndarray,
              reconstruction_weight: float = 1.0,
              prediction_weight: float = 1.0) -> Dict[str, Any]:
    """
    Single training step for LSTM model.

    Args:
        model: LSTM model
        optimizer: NNX optimizer
        x: Input sequence (T, input_dim) or batch (B, T, input_dim)
        reconstruction_weight: Weight for reconstruction loss
        prediction_weight: Weight for prediction loss

    Returns:
        Dictionary with loss and metrics
    """
    # Define loss function
    def loss_fn(model):
        # Check if batched
        if x.ndim == 3:
            # Batched: vmap over batch dimension
            losses_and_metrics = jax.vmap(
                lambda xi: compute_lstm_losses(
                    model, xi,
                    reconstruction_weight=reconstruction_weight,
                    prediction_weight=prediction_weight,
                    training=True
                )
            )(x)
            # Average over batch
            total_loss = jnp.mean(losses_and_metrics[0])
            metrics = jax.tree_util.tree_map(lambda m: jnp.mean(m), losses_and_metrics[1])
        else:
            # Single sequence
            total_loss, metrics = compute_lstm_losses(
                model, x,
                reconstruction_weight=reconstruction_weight,
                prediction_weight=prediction_weight,
                training=True
            )

        return total_loss, metrics

    # Compute gradients
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(model)

    # Update parameters
    optimizer.update(model, grads)

    # Add gradient norm to metrics
    grad_norm = optax.global_norm(grads)
    metrics['grad_norm'] = grad_norm

    return metrics


@nnx.jit
def eval_step(model: LSTMBaseline,
             x: jnp.ndarray,
             reconstruction_weight: float = 1.0,
             prediction_weight: float = 1.0) -> Dict[str, Any]:
    """
    Single evaluation step for LSTM model.

    Args:
        model: LSTM model
        x: Input sequence (T, input_dim) or batch (B, T, input_dim)
        reconstruction_weight: Weight for reconstruction loss
        prediction_weight: Weight for prediction loss

    Returns:
        Dictionary with loss and metrics
    """
    # Check if batched
    if x.ndim == 3:
        # Batched: vmap over batch dimension
        losses_and_metrics = jax.vmap(
            lambda xi: compute_lstm_losses(
                model, xi,
                reconstruction_weight=reconstruction_weight,
                prediction_weight=prediction_weight,
                training=False
            )
        )(x)
        # Average over batch
        metrics = jax.tree_util.tree_map(lambda m: jnp.mean(m), losses_and_metrics[1])
    else:
        # Single sequence
        _, metrics = compute_lstm_losses(
            model, x,
            reconstruction_weight=reconstruction_weight,
            prediction_weight=prediction_weight,
            training=False
        )

    return metrics


def train_epoch(model: LSTMBaseline,
               optimizer: nnx.Optimizer,
               train_data: jnp.ndarray,
               batch_size: int = None,
               reconstruction_weight: float = 1.0,
               prediction_weight: float = 1.0) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: LSTM model
        optimizer: NNX optimizer
        train_data: Training data (num_sequences, T, input_dim)
        batch_size: Batch size (None = use all data)
        reconstruction_weight: Weight for reconstruction loss
        prediction_weight: Weight for prediction loss

    Returns:
        Dictionary of averaged metrics over epoch
    """
    num_sequences = train_data.shape[0]

    # Initialize metrics accumulator
    epoch_metrics = {}

    if batch_size is None:
        # Use all data
        metrics = train_step(
            model, optimizer, train_data,
            reconstruction_weight=reconstruction_weight,
            prediction_weight=prediction_weight
        )
        epoch_metrics = {k: float(v) for k, v in metrics.items()}
    else:
        # Mini-batch training
        num_batches = (num_sequences + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_sequences)
            batch = train_data[start_idx:end_idx]

            metrics = train_step(
                model, optimizer, batch,
                reconstruction_weight=reconstruction_weight,
                prediction_weight=prediction_weight
            )

            # Accumulate metrics
            for k, v in metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = 0.0
                epoch_metrics[k] += float(v)

        # Average metrics over batches
        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches

    return epoch_metrics


def evaluate(model: LSTMBaseline,
            eval_data: jnp.ndarray,
            batch_size: int = None,
            reconstruction_weight: float = 1.0,
            prediction_weight: float = 1.0) -> Dict[str, float]:
    """
    Evaluate model on validation/test data.

    Args:
        model: LSTM model
        eval_data: Evaluation data (num_sequences, T, input_dim)
        batch_size: Batch size (None = use all data)
        reconstruction_weight: Weight for reconstruction loss
        prediction_weight: Weight for prediction loss

    Returns:
        Dictionary of averaged metrics
    """
    num_sequences = eval_data.shape[0]

    # Initialize metrics accumulator
    eval_metrics = {}

    if batch_size is None:
        # Use all data
        metrics = eval_step(
            model, eval_data,
            reconstruction_weight=reconstruction_weight,
            prediction_weight=prediction_weight
        )
        eval_metrics = {k: float(v) for k, v in metrics.items()}
    else:
        # Mini-batch evaluation
        num_batches = (num_sequences + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_sequences)
            batch = eval_data[start_idx:end_idx]

            metrics = eval_step(
                model, batch,
                reconstruction_weight=reconstruction_weight,
                prediction_weight=prediction_weight
            )

            # Accumulate metrics
            for k, v in metrics.items():
                if k not in eval_metrics:
                    eval_metrics[k] = 0.0
                eval_metrics[k] += float(v)

        # Average metrics over batches
        for k in eval_metrics:
            eval_metrics[k] /= num_batches

    return eval_metrics


# ============================================================================
# Checkpointing Functions (matching TiDHy's approach)
# ============================================================================

def checkpoint_lstm_model(
    model: LSTMBaseline,
    optimizer: nnx.Optimizer,
    epoch: int,
    checkpoint_dir: str,
    checkpointer: ocp.StandardCheckpointer,
    max_to_keep: int = 5,
    final: bool = False
) -> None:
    """
    Save LSTM model and optimizer checkpoint using Orbax.

    Args:
        model: LSTM model to save
        optimizer: NNX optimizer to save
        epoch: Current epoch number
        checkpoint_dir: Directory to save checkpoints
        checkpointer: Orbax StandardCheckpointer instance
        max_to_keep: Maximum number of checkpoints to keep (0 = keep all)
        final: Whether this is the final checkpoint
    """
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # Extract states from model and optimizer
    _, model_state = nnx.split(model)
    _, opt_state = nnx.split(optimizer)

    # Create checkpoint directory name
    if final:
        epoch_dir = ckpt_path / 'final'
    else:
        epoch_dir = ckpt_path / f'epoch_{epoch:04d}'

    epoch_dir.mkdir(exist_ok=True, parents=True)

    # Prepare checkpoint data
    # Note: Wrap epoch in dict to avoid Orbax scalar restrictions
    checkpoint_data = {
        'model_state': model_state,
        'optimizer_state': opt_state,
        'epoch': {'value': int(epoch)}
    }

    # Save checkpoint
    checkpointer.save(epoch_dir / 'checkpoint', checkpoint_data)

    # Cleanup old checkpoints (only for non-final checkpoints)
    if max_to_keep > 0 and not final:
        # Get all epoch checkpoints (exclude 'final')
        checkpoint_dirs = natsorted([
            d for d in ckpt_path.iterdir()
            if d.is_dir() and d.name.startswith('epoch_')
        ])

        # Remove oldest checkpoints if exceeding max_to_keep
        while len(checkpoint_dirs) > max_to_keep:
            oldest_ckpt = checkpoint_dirs.pop(0)
            shutil.rmtree(oldest_ckpt)


def load_lstm_checkpoint(
    model: LSTMBaseline,
    optimizer: nnx.Optimizer,
    checkpoint_dir: str,
    epoch: Optional[int] = None
) -> Tuple[LSTMBaseline, nnx.Optimizer, int]:
    """
    Load LSTM model and optimizer from checkpoint.

    Args:
        model: LSTM model (used as template for structure)
        optimizer: NNX optimizer (used as template for structure)
        checkpoint_dir: Directory containing checkpoints
        epoch: Specific epoch to load (None = load latest)

    Returns:
        Tuple of (loaded_model, loaded_optimizer, loaded_epoch)

    Raises:
        FileNotFoundError: If checkpoint directory doesn't exist or no checkpoints found
        ValueError: If specified epoch not found
    """
    ckpt_path = Path(checkpoint_dir)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Find available checkpoints
    checkpoint_dirs = natsorted([
        d for d in ckpt_path.iterdir()
        if d.is_dir() and d.name.startswith('epoch_')
    ])

    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    # Determine which checkpoint to load
    if epoch is None:
        # Load latest checkpoint
        epoch_dir = checkpoint_dirs[-1]
        print(f"Loading latest checkpoint: {epoch_dir.name}")
    else:
        # Load specific epoch
        epoch_dir = ckpt_path / f'epoch_{epoch:04d}'
        if not epoch_dir.exists():
            raise ValueError(f"Checkpoint for epoch {epoch} not found")
        print(f"Loading checkpoint: epoch_{epoch:04d}")

    # Create checkpointer
    checkpointer = ocp.StandardCheckpointer()

    # Get current states as templates
    _, current_model_state = nnx.split(model)
    _, current_opt_state = nnx.split(optimizer)

    # Create template for restoration
    checkpoint_template = {
        'model_state': current_model_state,
        'optimizer_state': current_opt_state,
        'epoch': {'value': 0}
    }

    # Restore checkpoint
    try:
        restored = checkpointer.restore(
            epoch_dir / 'checkpoint',
            target=checkpoint_template
        )
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        # Try loading model only (optimizer may have structure mismatch)
        try:
            print("Attempting to load model state only...")
            model_template = {'model_state': current_model_state, 'epoch': {'value': 0}}
            restored_partial = checkpointer.restore(
                epoch_dir / 'checkpoint',
                target=model_template
            )
            # Merge model, keep fresh optimizer
            model_graphdef, _ = nnx.split(model)
            loaded_model = nnx.merge(model_graphdef, restored_partial['model_state'])
            print("Successfully loaded model state. Using fresh optimizer.")
            return loaded_model, optimizer, int(restored_partial['epoch']['value'])
        except Exception as e2:
            raise RuntimeError(f"Failed to load checkpoint: {e2}") from e

    # Merge restored states with graph definitions
    model_graphdef, _ = nnx.split(model)
    loaded_model = nnx.merge(model_graphdef, restored['model_state'])

    opt_graphdef, _ = nnx.split(optimizer)
    loaded_optimizer = nnx.merge(opt_graphdef, restored['optimizer_state'])

    loaded_epoch = int(restored['epoch']['value'])

    print(f"Successfully loaded checkpoint from epoch {loaded_epoch}")

    return loaded_model, loaded_optimizer, loaded_epoch


def discover_lstm_checkpoint(checkpoint_dir: str) -> Optional[Path]:
    """
    Discover the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory to search for checkpoints

    Returns:
        Path to latest checkpoint directory, or None if no checkpoints found
    """
    ckpt_path = Path(checkpoint_dir)

    if not ckpt_path.exists():
        return None

    # Find all epoch checkpoints
    checkpoint_dirs = natsorted([
        d for d in ckpt_path.iterdir()
        if d.is_dir() and d.name.startswith('epoch_')
    ])

    if not checkpoint_dirs:
        return None

    # Return latest checkpoint
    return checkpoint_dirs[-1]


def list_lstm_checkpoints(checkpoint_dir: str) -> list:
    """
    List all available checkpoint epochs.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        List of epoch numbers for available checkpoints
    """
    ckpt_path = Path(checkpoint_dir)

    if not ckpt_path.exists():
        return []

    checkpoint_dirs = natsorted([
        d for d in ckpt_path.iterdir()
        if d.is_dir() and d.name.startswith('epoch_')
    ])

    # Extract epoch numbers
    epochs = []
    for d in checkpoint_dirs:
        try:
            epoch_num = int(d.name.split('_')[1])
            epochs.append(epoch_num)
        except (ValueError, IndexError):
            continue

    return epochs


# ============================================================================
# Inference and Recording Functions (for analysis)
# ============================================================================

def evaluate_lstm_record(
    model: LSTMBaseline,
    X: jnp.ndarray,
    reconstruction_weight: float = 1.0,
    prediction_weight: float = 1.0
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    """
    Run LSTM inference and record all hidden states, predictions, and losses.

    This function is analogous to TiDHy's evaluate_record() and provides
    comprehensive recording of LSTM internal states for analysis.

    Args:
        model: LSTM model
        X: Input data of shape (batch, T, input_dim) or (T, input_dim)
        reconstruction_weight: Weight for reconstruction loss
        prediction_weight: Weight for prediction loss

    Returns:
        avg_loss: Average total loss across all timesteps
        result_dict: Dictionary containing:
            - 'I': Input sequences (batch, T, input_dim)
            - 'H': Hidden states (batch, T, hidden_dim)
            - 'C': Cell states (batch, T, hidden_dim)
            - 'I_hat': Reconstructions (batch, T, input_dim)
            - 'predictions': Next-step predictions (batch, T-1, input_dim)
            - 'reconstruction_loss': Per-timestep reconstruction losses (batch, T)
            - 'prediction_loss': Per-timestep prediction losses (batch, T-1)
            - 'total_loss': Per-timestep total losses (batch, T)
    """
    # Handle both batched and unbatched inputs
    if X.ndim == 2:
        X = X[None, ...]  # Add batch dimension: (T, input_dim) -> (1, T, input_dim)
        unbatched = True
    else:
        unbatched = False

    batch_size, T, input_dim = X.shape
    hidden_dim = model.hidden_dim

    # Process each sequence in the batch
    def process_sequence(x_seq):
        """Process a single sequence and record all states."""
        # Initialize lists to collect states
        hidden_states = []
        cell_states = []

        # Run encoder to get hidden states
        # Note: encoder returns (T, hidden_dim)
        hiddens = model.encoder(x_seq, training=False)

        # For LSTM, we need to re-run through cells to get cell states
        # Initialize carry
        carry = model.encoder.init_carry()

        # Process each timestep to extract both h and c
        def step_fn(carry, x_t):
            # Process through all layers
            layer_carry = carry
            layer_input = x_t

            for layer_idx, lstm_cell in enumerate(model.encoder.lstm_cells):
                h, c = layer_carry[layer_idx]
                new_carry, new_h = lstm_cell((h, c), layer_input)
                layer_carry = layer_carry[:layer_idx] + (new_carry,) + layer_carry[layer_idx+1:]
                layer_input = new_h

            # Return final layer's h and c
            final_h, final_c = layer_carry[-1]
            return layer_carry, (final_h, final_c)

        # Initialize multi-layer carry
        num_layers = model.encoder.num_layers
        init_carry = tuple([model.encoder.init_carry() for _ in range(num_layers)])

        # Scan over sequence
        final_carry, (h_sequence, c_sequence) = jax.lax.scan(step_fn, init_carry, x_seq)

        # Decode to get reconstructions
        reconstructions = jax.vmap(model.decoder)(hiddens)  # (T, input_dim)

        # Get next-step predictions (use h[:-1] to predict x[1:])
        predictions = jax.vmap(model.decoder)(hiddens[:-1])  # (T-1, input_dim)

        # Compute per-timestep losses
        recon_losses = jnp.sum((x_seq - reconstructions) ** 2, axis=-1)  # (T,)
        pred_losses = jnp.sum((x_seq[1:] - predictions) ** 2, axis=-1)   # (T-1,)

        # Pad prediction loss to match length
        pred_losses_padded = jnp.concatenate([pred_losses, jnp.array([0.0])])

        # Total loss per timestep
        total_losses = reconstruction_weight * recon_losses + prediction_weight * pred_losses_padded

        return {
            'H': h_sequence,
            'C': c_sequence,
            'I_hat': reconstructions,
            'predictions': predictions,
            'reconstruction_loss': recon_losses,
            'prediction_loss': pred_losses,
            'total_loss': total_losses
        }

    # Process all sequences in batch
    batch_results = jax.vmap(process_sequence)(X)

    # Compute average loss
    avg_loss = jnp.mean(batch_results['total_loss'])

    # Construct result dictionary
    result_dict = {
        'I': X,  # Inputs
        'H': batch_results['H'],  # Hidden states
        'C': batch_results['C'],  # Cell states
        'I_hat': batch_results['I_hat'],  # Reconstructions
        'predictions': batch_results['predictions'],  # Next-step predictions
        'reconstruction_loss': batch_results['reconstruction_loss'],
        'prediction_loss': batch_results['prediction_loss'],
        'total_loss': batch_results['total_loss']
    }

    # Remove batch dimension if input was unbatched
    if unbatched:
        result_dict = {k: v[0] for k, v in result_dict.items()}

    return float(avg_loss), result_dict
