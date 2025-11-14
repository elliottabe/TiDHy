"""
Training utilities for TiDHy NNX vmap model.

This module demonstrates how to use vmap to train the single-example TiDHy model
on batches of data.

Key concept:
- The model works on single sequences: (T, input_dim)
- Use jax.vmap to process batches: (batch_size, T, input_dim)
"""

from typing import Tuple, Dict, Optional
from pathlib import Path
from tqdm.auto import tqdm
import functools
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
from orbax.checkpoint import PyTreeCheckpointHandler
import wandb

from TiDHy.models.TiDHy_nnx_vmap import TiDHy, cos_sim_mat
from TiDHy.utils.logging import log_training_step, log_sparsity_analysis, log_optimization_metrics


@nnx.jit
def compute_task_gradient_norms(
    model: TiDHy, X_batch: jax.Array, weights: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Compute gradient norms for each task loss using vmap.

    Args:
        model: TiDHy model
        X_batch: Input batch of shape (batch_size, T, input_dim)
        weights: Current task weights

    Returns:
        Tuple of (grad_norms, losses, grads, cos_reg)
    """

    def weighted_loss_fn(mdl: TiDHy):
        """Compute weighted loss for gradient computation with sparsity regularization"""
        # Generate random keys for each sequence in the batch
        batch_size = X_batch.shape[0]
        base_key = mdl.rngs()  # Get key outside vmap
        keys = jax.random.split(base_key, batch_size)

        # Create vmapped version of model
        def forward_single(seq, key):
            return mdl(seq, return_internals=True, rng_key=key)

        vmapped_forward = jax.vmap(forward_single, in_axes=(0, 0))
        model_outputs = vmapped_forward(X_batch, keys)
        losses_per_seq, internals_per_seq, inf_stats_seq = model_outputs

        spatial_loss_rhat, spatial_loss_rbar, temp_loss = losses_per_seq

        # Average over batch
        spatial_loss_rhat_mean = jnp.mean(spatial_loss_rhat)
        spatial_loss_rbar_mean = jnp.mean(spatial_loss_rbar)
        temp_loss_mean = jnp.mean(temp_loss)

        # Compute cosine regularization
        cos_reg = (1.0 / mdl.mix_dim) * cos_sim_mat(
            mdl.temporal.value.reshape(mdl.mix_dim, mdl.r_dim, mdl.r_dim)
        )

        # Extract internals for sparsity regularization
        r_values, r2_values, w_values = internals_per_seq

        # Compute sparsity regularization
        sparsity_reg, _ = mdl.compute_sparsity_regularization(
            r_values.reshape(-1, mdl.r_dim),
            r2_values.reshape(-1, mdl.r2_dim),
            w_values.reshape(-1, mdl.mix_dim),
        )

        temp_with_cos = temp_loss_mean + mdl.cos_eta * cos_reg

        # Include sparsity in temporal loss term for GradNorm balancing
        temp_with_cos_and_sparsity = temp_with_cos + sparsity_reg

        losses = jnp.array(
            [spatial_loss_rhat_mean, spatial_loss_rbar_mean, temp_with_cos_and_sparsity]
        )

        # Weighted sum
        weighted = jnp.dot(weights, losses)

        return weighted, (
            spatial_loss_rhat_mean,
            spatial_loss_rbar_mean,
            temp_with_cos_and_sparsity,
            cos_reg,
            sparsity_reg,
        )

    # Compute gradients for weighted loss
    (_, (spatial_rhat, spatial_rbar, temp_with_cos, cos_reg, sparsity_reg)), grads = (
        nnx.value_and_grad(weighted_loss_fn, has_aux=True)(model)
    )

    # Now compute gradient norm for each individual task
    grad_norms = []
    losses = jnp.array([spatial_rhat, spatial_rbar, temp_with_cos])

    for i in range(3):
        # Compute gradient for this individual weighted task
        def single_task_loss(mdl: TiDHy, task_idx: int):
            # Generate random keys for each sequence in the batch
            batch_size = X_batch.shape[0]
            base_key = mdl.rngs()  # Get key outside vmap
            keys = jax.random.split(base_key, batch_size)

            def forward_single(seq, key):
                return mdl(seq, return_internals=True, rng_key=key)

            vmapped_forward = jax.vmap(forward_single, in_axes=(0, 0))
            model_outputs = vmapped_forward(X_batch, keys)
            losses_per_seq, internals_per_seq, inf_stats_seq = model_outputs

            spatial_loss_rhat, spatial_loss_rbar, temp_loss = losses_per_seq

            spatial_loss_rhat_mean = jnp.mean(spatial_loss_rhat)
            spatial_loss_rbar_mean = jnp.mean(spatial_loss_rbar)
            temp_loss_mean = jnp.mean(temp_loss)

            cos_reg = (1.0 / mdl.mix_dim) * cos_sim_mat(
                mdl.temporal.value.reshape(mdl.mix_dim, mdl.r_dim, mdl.r_dim)
            )

            # Extract internals for sparsity regularization
            r_values, r2_values, w_values = internals_per_seq

            # Compute sparsity regularization
            sparsity_reg, _ = mdl.compute_sparsity_regularization(
                r_values.reshape(-1, mdl.r_dim),
                r2_values.reshape(-1, mdl.r2_dim),
                w_values.reshape(-1, mdl.mix_dim),
            )

            temp_with_cos = temp_loss_mean + mdl.cos_eta * cos_reg
            temp_with_cos_and_sparsity = temp_with_cos + sparsity_reg

            task_losses = jnp.array(
                [spatial_loss_rhat_mean, spatial_loss_rbar_mean, temp_with_cos_and_sparsity]
            )
            return weights[task_idx] * task_losses[task_idx]

        task_loss_fn = functools.partial(single_task_loss, task_idx=i)
        task_grads = nnx.grad(task_loss_fn)(model)

        # Compute L2 norm of gradients
        grad_norm = jnp.sqrt(
            sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(nnx.state(task_grads, nnx.Param)))
        )
        grad_norms.append(grad_norm)

    return jnp.array(grad_norms), losses, grads, cos_reg


@nnx.jit
def apply_gradnorm(
    model: TiDHy, X_batch: jax.Array, grad_alpha: float
) -> Tuple[jax.Array, jax.Array, jax.Array, Dict[str, float]]:
    """
    Apply GradNorm algorithm to balance task losses.

    Args:
        model: TiDHy model
        X_batch: Input batch of shape (batch_size, T, input_dim)
        grad_alpha: GradNorm hyperparameter (typically 1.5)

    Returns:
        Tuple of (loss_weights, weighted_loss, grads, metrics)
    """
    # Initialize on first iteration
    if model.loss_weights is None:
        # Get initial losses using vmap
        # Generate random keys for each sequence in the batch
        batch_size = X_batch.shape[0]
        base_key = model.rngs()  # Get key outside vmap
        keys = jax.random.split(base_key, batch_size)

        def forward_single(seq, key):
            return model(seq, return_internals=True, rng_key=key)

        vmapped_forward = jax.vmap(forward_single, in_axes=(0, 0))
        model_outputs = vmapped_forward(X_batch, keys)
        losses_per_seq, internals_per_seq, inf_stats_seq = model_outputs

        spatial_loss_rhat, spatial_loss_rbar, temp_loss = losses_per_seq

        spatial_loss_rhat_mean = jnp.mean(spatial_loss_rhat)
        spatial_loss_rbar_mean = jnp.mean(spatial_loss_rbar)
        temp_loss_mean = jnp.mean(temp_loss)

        cos_reg = (1.0 / model.mix_dim) * cos_sim_mat(
            model.temporal.value.reshape(model.mix_dim, model.r_dim, model.r_dim)
        )

        # Compute sparsity regularization for initialization
        r_values, r2_values, w_values = internals_per_seq
        sparsity_reg_init, _ = model.compute_sparsity_regularization(
            r_values.reshape(-1, model.r_dim),
            r2_values.reshape(-1, model.r2_dim),
            w_values.reshape(-1, model.mix_dim),
        )

        temp_with_cos = temp_loss_mean + model.cos_eta * cos_reg + sparsity_reg_init

        model.l0 = jnp.array([spatial_loss_rhat_mean, spatial_loss_rbar_mean, temp_with_cos])
        model.loss_weights = jnp.ones(3)
        model.iters = 0

    weights = model.loss_weights
    T = jnp.sum(weights)  # Sum of weights for normalization

    # Compute gradient norms and losses
    grad_norms, losses, grads, cos_reg = compute_task_gradient_norms(model, X_batch, weights)

    # Compute loss ratios
    loss_ratio = losses / (model.l0 + 1e-10)
    rt = loss_ratio / (jnp.mean(loss_ratio) + 1e-10)

    # Compute GradNorm target
    gw_avg = jnp.mean(grad_norms)
    constant = gw_avg * jnp.power(rt, grad_alpha)

    # Update weights using simple gradient descent
    lr_weights = model.lr_weights if hasattr(model, "lr_weights") else 0.025
    gradnorm_diff = grad_norms - constant

    # Simple update: move weights to reduce discrepancy
    weight_updates = -lr_weights * jnp.sign(gradnorm_diff) * jnp.abs(loss_ratio)
    new_weights = weights + weight_updates

    # Clip before exponential to prevent explosion
    new_weights = jnp.clip(new_weights, -5.0, 5.0)

    # Renormalize: use exponential and normalize to sum to T
    new_weights = jnp.exp(new_weights)

    # Add safety check for NaN
    new_weights = jnp.where(jnp.isnan(new_weights), jnp.ones(3), new_weights)

    # Renormalize
    new_weights = (new_weights / (jnp.sum(new_weights) + 1e-8)) * T

    # Update model state
    model.loss_weights = new_weights
    model.iters += 1

    # Compute weighted loss
    weighted_loss = jnp.dot(new_weights, losses)

    metrics = {
        "spatial_loss_rhat": losses[0],
        "spatial_loss_rbar": losses[1],
        "temp_loss": losses[2],
        "cos_reg": cos_reg,
        "loss_weights": new_weights,
        "grad_norms": grad_norms,
    }

    return new_weights, weighted_loss, grads, metrics


def create_optimizer(
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    schedule: Optional[optax.Schedule] = None,
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


def create_multi_lr_optimizer(
    model: TiDHy,
    learning_rate_s: float = 1e-3,  # spatial decoder
    learning_rate_t: float = 1e-3,  # temporal parameters
    learning_rate_h: float = 1e-3,  # hypernetwork
    weight_decay: float = 1e-4,
    schedule_s: Optional[optax.Schedule] = None,
    schedule_t: Optional[optax.Schedule] = None,
    schedule_h: Optional[optax.Schedule] = None,
    use_schedule: bool = False,
    schedule_transition_steps: int = 200,
    schedule_decay_rate: float = 0.96,
) -> nnx.Optimizer:
    """
    Create optimizer with different learning rates for different model components.

    Args:
        model: TiDHy model instance
        learning_rate_s: Learning rate for spatial decoder
        learning_rate_t: Learning rate for temporal parameters
        learning_rate_h: Learning rate for hypernetwork
        weight_decay: Weight decay coefficient
        schedule_s: Optional learning rate schedule for spatial decoder
        schedule_t: Optional learning rate schedule for temporal parameters
        schedule_h: Optional learning rate schedule for hypernetwork
        use_schedule: Whether to create exponential decay schedules
        schedule_transition_steps: Steps between learning rate decay
        schedule_decay_rate: Decay rate for exponential schedule

    Returns:
        nnx.Optimizer with separate optimizers for each component
    """
    # Create schedules if requested
    if use_schedule:
        if schedule_s is None:
            schedule_s = optax.exponential_decay(
                init_value=learning_rate_s,
                transition_steps=schedule_transition_steps,
                decay_rate=schedule_decay_rate,
            )
        if schedule_t is None:
            schedule_t = optax.exponential_decay(
                init_value=learning_rate_t,
                transition_steps=schedule_transition_steps,
                decay_rate=schedule_decay_rate,
            )
        if schedule_h is None:
            schedule_h = optax.exponential_decay(
                init_value=learning_rate_h,
                transition_steps=schedule_transition_steps,
                decay_rate=schedule_decay_rate,
            )

    # Create separate optimizers for each component
    lr_s = schedule_s if schedule_s is not None else learning_rate_s
    lr_t = schedule_t if schedule_t is not None else learning_rate_t
    lr_h = schedule_h if schedule_h is not None else learning_rate_h

    # Create mask functions for each parameter group
    def spatial_mask(path):
        """Mask for spatial decoder parameters"""
        path_str = "/".join(str(p) for p in path)
        return "spatial_decoder" in path_str

    def temporal_mask(path):
        """Mask for temporal parameters"""
        path_str = "/".join(str(p) for p in path)
        return "temporal" in path_str

    def hyper_mask(path):
        """Mask for hypernetwork parameters"""
        path_str = "/".join(str(p) for p in path)
        return "hypernet" in path_str

    # Create masked optimizers for each component
    spatial_tx = optax.masked(optax.adamw(lr_s, weight_decay=weight_decay), spatial_mask)
    temporal_tx = optax.masked(optax.adamw(lr_t, weight_decay=weight_decay), temporal_mask)
    hyper_tx = optax.masked(optax.adamw(lr_h, weight_decay=weight_decay), hyper_mask)

    # Add gradient clipping to prevent NaN from exploding gradients
    # clip_by_global_norm(1.0) is already quite conservative - it clips the global norm
    # of all gradients to 1.0. This helps prevent instability but may need adjustment
    # if convergence is too slow (increase value) or still seeing NaNs (decrease value).
    clip_tx = optax.clip_by_global_norm(1.0)

    # Chain gradient clipping with the masked optimizers
    multi_tx = optax.chain(clip_tx, spatial_tx, temporal_tx, hyper_tx)

    # Create NNX optimizer
    return nnx.Optimizer(model=model, tx=multi_tx, wrt=nnx.Param)


@nnx.jit(static_argnames=["use_gradnorm"])
def train_step_single(
    model: TiDHy,
    optimizer: nnx.Optimizer,
    X_batch: jax.Array,
    use_gradnorm: bool = False,
    grad_alpha: float = 1.5,
) -> Dict[str, float]:
    """
    JIT-compiled training step using vmap to process batch.

    The model's __call__ works on single sequences (T, input_dim).
    We vmap it to process batches (batch_size, T, input_dim).

    Args:
        model: TiDHy model instance
        optimizer: NNX optimizer
        X_batch: Input batch of shape (batch_size, T, input_dim)
        use_gradnorm: Whether to use GradNorm (static, known at compile time)
        grad_alpha: GradNorm hyperparameter

    Returns:
        Dictionary of metrics
    """
    if use_gradnorm:
        # Use GradNorm for balanced training
        _, weighted_loss, grads, metrics = apply_gradnorm(model, X_batch, grad_alpha)

        # Update parameters using computed gradients
        optimizer.update(model, grads)

        metrics["loss"] = weighted_loss
    else:
        # Standard training without GradNorm
        def loss_fn(model: TiDHy):
            """
            Compute loss for training with sparsity regularization.

            We vmap the model's __call__ to process all sequences in the batch.
            """
            # Create vmapped version of model that processes batches
            # Generate random keys for each sequence in the batch
            batch_size = X_batch.shape[0]
            base_key = model.rngs()  # Get key outside vmap
            keys = jax.random.split(base_key, batch_size)

            # Define function that processes single sequence with internals
            def forward_single(seq, key):
                return model(seq, return_internals=True, rng_key=key)

            # Vmap over batch dimension and keys
            vmapped_forward = jax.vmap(forward_single, in_axes=(0, 0))

            # Get per-sequence losses and internals
            model_outputs = vmapped_forward(X_batch, keys)
            losses_per_seq, internals_per_seq, inf_stats_seq = model_outputs

            # losses_per_seq is a tuple of 3 arrays, each of shape (batch_size,)
            spatial_loss_rhat, spatial_loss_rbar, temp_loss = losses_per_seq

            # Average over batch
            spatial_loss_rhat_mean = jnp.mean(spatial_loss_rhat)
            spatial_loss_rbar_mean = jnp.mean(spatial_loss_rbar)
            temp_loss_mean = jnp.mean(temp_loss)

            # Compute cosine regularization on temporal matrices
            cos_reg = (1.0 / model.mix_dim) * cos_sim_mat(
                model.temporal.value.reshape(model.mix_dim, model.r_dim, model.r_dim)
            )

            # Extract internals for sparsity regularization
            r_values, r2_values, w_values = internals_per_seq
            # r_values: (batch_size, T, r_dim), r2_values: (batch_size, T, r2_dim), w_values: (batch_size, T, mix_dim)

            # Compute sparsity regularization
            sparsity_reg, sparsity_breakdown = model.compute_sparsity_regularization(
                r_values.reshape(-1, model.r_dim),  # Flatten to (batch_size*T, r_dim)
                r2_values.reshape(-1, model.r2_dim),  # Flatten to (batch_size*T, r2_dim)
                w_values.reshape(-1, model.mix_dim),  # Flatten to (batch_size*T, mix_dim)
            )

            # Compute r2 continuity loss for overlapping windows
            r2_continuity_loss = 0.0
            if model.r2_continuity_weight > 0:
                # Calculate overlap length from config
                # overlap = sequence_length - stride
                # stride = sequence_length // overlap_factor
                overlap_length = X_batch.shape[1] - (
                    X_batch.shape[1] // 10
                )  # Default overlap_factor=10
                r2_continuity_loss = model.compute_r2_continuity_loss(
                    r2_values, overlap_length=overlap_length  # (batch_size, T, r2_dim)
                )

            # Compute r2 temporal smoothness loss
            r2_temporal_smooth_loss = 0.0
            if model.r2_temporal_smoothness_train > 0:
                # r2_values: (batch_size, T, r2_dim)
                # Compute differences between consecutive timesteps
                r2_diffs = r2_values[:, 1:, :] - r2_values[:, :-1, :]  # (batch_size, T-1, r2_dim)
                r2_temporal_smooth_loss = jnp.mean(r2_diffs**2)

            # Total loss including sparsity regularization, r2 continuity, and r2 temporal smoothness
            total_loss = (
                spatial_loss_rhat_mean
                + spatial_loss_rbar_mean
                + temp_loss_mean
                + model.cos_eta * cos_reg
                + sparsity_reg
                + model.r2_continuity_weight * r2_continuity_loss
                + model.r2_temporal_smoothness_train * r2_temporal_smooth_loss
            )

            # Prepare metrics
            metrics = {
                "spatial_loss_rhat": spatial_loss_rhat_mean,
                "spatial_loss_rbar": spatial_loss_rbar_mean,
                "temp_loss": temp_loss_mean,
                "cos_reg": cos_reg,
                "sparsity_reg": sparsity_reg,
                "r2_continuity_loss": r2_continuity_loss,
                "r2_temporal_smooth_loss": r2_temporal_smooth_loss,
                **inf_stats_seq,
            }

            # Add sparsity breakdown to metrics
            for key, value in sparsity_breakdown.items():
                metrics[f"sparsity_{key}"] = value

            # Add sparsity metrics
            sparsity_metrics = model.compute_sparsity_metrics(
                r_values.reshape(-1, model.r_dim),
                r2_values.reshape(-1, model.r2_dim),
                w_values.reshape(-1, model.mix_dim),
            )
            metrics.update(sparsity_metrics)

            return total_loss, metrics

        # Compute gradients
        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

        # Compute gradient norm for monitoring
        grad_norm = jnp.sqrt(
            sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(nnx.state(grads, nnx.Param)))
        )
        metrics["grad_norm"] = grad_norm

        # Update parameters
        optimizer.update(model, grads)

        metrics["loss"] = loss

    # Normalize if needed
    if model.normalize_spatial or model.normalize_temporal:
        model.normalize()

    return metrics


def train_epoch(
    model: TiDHy,
    optimizer: nnx.Optimizer,
    train_data: jax.Array,
    batch_size: Optional[int] = None,
    use_gradnorm: bool = False,
    grad_alpha: float = 1.5,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: TiDHy model instance
        optimizer: NNX optimizer
        train_data: Training data of shape (n_samples, T, input_dim)
        batch_size: Batch size (if None, use all data)
        use_gradnorm: Whether to use GradNorm (static, known at compile time)
        grad_alpha: GradNorm hyperparameter

    Returns:
        Dictionary of averaged metrics
    """
    n_samples = train_data.shape[0]

    if batch_size is None:
        # Single batch - use JIT-compiled function
        metrics = train_step_single(model, optimizer, train_data, use_gradnorm, grad_alpha)

        # Check for NaN/Inf after training
        if jnp.isnan(metrics["loss"]) or jnp.isinf(metrics["loss"]):
            print(f"Warning: NaN/Inf detected in loss! Loss value: {float(metrics['loss'])}")
            raise ValueError("Training became unstable (NaN/Inf loss)")

        return metrics

    # Multiple batches - accumulate metrics efficiently
    n_batches = n_samples // batch_size
    metrics_list = []

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch = train_data[start_idx:end_idx]

        # Train on batch
        metrics = train_step_single(model, optimizer, batch, use_gradnorm, grad_alpha)
        metrics_list.append(metrics)

    # Efficiently average metrics: stack all metrics first, then compute mean
    # This reduces blocking host-device transfers from N to 1
    stacked_metrics = jax.tree.map(
        lambda *args: jnp.stack(args), *metrics_list
    )  # pylint: disable=no-value-for-parameter
    averaged_metrics = jax.tree.map(
        lambda x: jnp.mean(x), stacked_metrics
    )  # pylint: disable=unnecessary-lambda

    # Convert to Python floats once at the end
    final_metrics = {k: float(v) for k, v in averaged_metrics.items()}

    # Check for NaN/Inf after training
    if jnp.isnan(averaged_metrics["loss"]) or jnp.isinf(averaged_metrics["loss"]):
        print(f"Warning: NaN/Inf detected in loss! Loss value: {final_metrics['loss']}")
        raise ValueError("Training became unstable (NaN/Inf loss)")

    return final_metrics


def train_model(
    model: TiDHy,
    train_data: jax.Array,
    n_epochs: int,
    learning_rate: float = 1e-3,
    learning_rate_s: Optional[float] = None,
    learning_rate_t: Optional[float] = None,
    learning_rate_h: Optional[float] = None,
    use_schedule: bool = True,
    schedule_transition_steps: int = 200,
    schedule_decay_rate: float = 0.96,
    weight_decay: float = 1e-4,
    batch_size: Optional[int] = None,
    val_data: Optional[jax.Array] = None,
    use_gradnorm: bool = False,
    grad_alpha: float = 1.5,
    verbose: bool = True,
    val_every_n_epochs: int = 1,
    # Wandb logging parameters
    use_wandb: bool = False,
    log_params_every: int = 10,
    log_sparsity_every: int = 5,
) -> Tuple[TiDHy, Dict[str, list]]:
    """
    Train model for multiple epochs (simple wrapper without checkpointing).

    This is a convenience wrapper around train_model_with_checkpointing that
    disables checkpointing. For more control, use train_model_with_checkpointing directly.

    Args:
        model: TiDHy model instance
        train_data: Training data of shape (n_samples, T, input_dim)
        n_epochs: Number of epochs
        learning_rate: Default learning rate
        learning_rate_s: Learning rate for spatial decoder
        learning_rate_t: Learning rate for temporal parameters
        learning_rate_h: Learning rate for hypernetwork
        use_schedule: Whether to use learning rate schedules
        schedule_transition_steps: Steps between LR decay
        schedule_decay_rate: LR decay rate
        weight_decay: Weight decay coefficient
        batch_size: Batch size (if None, use all data)
        val_data: Optional validation data
        use_gradnorm: Whether to use GradNorm for loss balancing
        grad_alpha: GradNorm hyperparameter (default: 1.5)
        verbose: Whether to print progress
        val_every_n_epochs: Run validation every N epochs
        use_wandb: Whether to log to Weights & Biases
        log_params_every: Log model parameters every N epochs
        log_sparsity_every: Log sparsity analysis every N epochs

    Returns:
        Tuple of (trained model, history dictionary)
    """
    # Create a simple config object to pass to train_model_with_checkpointing
    from types import SimpleNamespace

    config = SimpleNamespace(
        train=SimpleNamespace(
            num_epochs=n_epochs,
            batch_size=batch_size,
            batch_size_input=(batch_size is None),
            learning_rate=learning_rate,
            learning_rate_s=learning_rate_s,
            learning_rate_t=learning_rate_t,
            learning_rate_h=learning_rate_h,
            weight_decay=weight_decay,
            grad_norm=use_gradnorm,
            grad_alpha=grad_alpha,
            save_summary_steps=val_every_n_epochs,
            use_schedule=use_schedule,
            schedule_transition_steps=schedule_transition_steps,
            schedule_decay=schedule_decay_rate,
        )
    )

    # Call the full training function without checkpointing
    return _train_model_internal(
        model=model,
        train_data=train_data,
        config=config,
        checkpoint_dir=None,  # No checkpointing
        val_data=val_data,
        start_epoch=0,
        optimizer=None,
        checkpoint_every=10,
        max_checkpoints_to_keep=5,
        verbose=verbose,
        use_wandb=use_wandb,
        log_params_every=log_params_every,
        log_sparsity_every=log_sparsity_every,
    )


@nnx.jit
def _evaluate_batch_basic(model: TiDHy, X: jax.Array) -> Dict[str, float]:
    """JIT-compiled basic evaluation without sparsity metrics."""
    # Generate random keys for each sequence in the batch
    batch_size = X.shape[0]
    base_key = model.rngs()  # Get key outside vmap
    keys = jax.random.split(base_key, batch_size)

    # Vmap the model over batch dimension
    def forward_single(seq, key):
        return model(seq, return_internals=False, rng_key=key)

    vmapped_forward = jax.vmap(forward_single, in_axes=(0, 0))
    losses_per_seq = vmapped_forward(X, keys)
    spatial_loss_rhat, spatial_loss_rbar, temp_loss = losses_per_seq

    # Average over batch
    spatial_loss_rhat_mean = jnp.mean(spatial_loss_rhat)
    spatial_loss_rbar_mean = jnp.mean(spatial_loss_rbar)
    temp_loss_mean = jnp.mean(temp_loss)

    # Compute cosine regularization
    cos_reg = (1.0 / model.mix_dim) * cos_sim_mat(
        model.temporal.value.reshape(model.mix_dim, model.r_dim, model.r_dim)
    )

    total_loss = (
        spatial_loss_rhat_mean + spatial_loss_rbar_mean + temp_loss_mean + model.cos_eta * cos_reg
    )

    return {
        "loss": total_loss,
        "spatial_loss_rhat": spatial_loss_rhat_mean,
        "spatial_loss_rbar": spatial_loss_rbar_mean,
        "temp_loss": temp_loss_mean,
        "cos_reg": cos_reg,
    }


def _evaluate_batch_with_sparsity(model: TiDHy, X: jax.Array) -> Dict[str, float]:
    """Evaluation with sparsity metrics (not JIT-compiled due to dynamic structure)."""
    # Generate random keys for each sequence in the batch
    batch_size = X.shape[0]
    base_key = model.rngs()  # Get key outside vmap
    keys = jax.random.split(base_key, batch_size)

    # Vmap the model over batch dimension
    def forward_single(seq, key):
        return model(seq, return_internals=True, rng_key=key)

    vmapped_forward = jax.vmap(forward_single, in_axes=(0, 0))
    model_outputs = vmapped_forward(X, keys)
    losses_per_seq, internals_per_seq, inf_stats_seq = model_outputs
    spatial_loss_rhat, spatial_loss_rbar, temp_loss = losses_per_seq

    # Extract internals for sparsity computation
    r_values, r2_values, w_values = internals_per_seq

    # Average over batch
    spatial_loss_rhat_mean = jnp.mean(spatial_loss_rhat)
    spatial_loss_rbar_mean = jnp.mean(spatial_loss_rbar)
    temp_loss_mean = jnp.mean(temp_loss)

    # Compute cosine regularization
    cos_reg = (1.0 / model.mix_dim) * cos_sim_mat(
        model.temporal.value.reshape(model.mix_dim, model.r_dim, model.r_dim)
    )

    total_loss = (
        spatial_loss_rhat_mean + spatial_loss_rbar_mean + temp_loss_mean + model.cos_eta * cos_reg
    )

    metrics = {
        "loss": total_loss,
        "spatial_loss_rhat": spatial_loss_rhat_mean,
        "spatial_loss_rbar": spatial_loss_rbar_mean,
        "temp_loss": temp_loss_mean,
        "cos_reg": cos_reg,
    }

    # Compute sparsity regularization (but don't add to loss for evaluation)
    sparsity_reg, sparsity_breakdown = model.compute_sparsity_regularization(
        r_values.reshape(-1, model.r_dim),
        r2_values.reshape(-1, model.r2_dim),
        w_values.reshape(-1, model.mix_dim),
    )

    metrics["sparsity_reg"] = sparsity_reg

    # Add sparsity breakdown
    for key, value in sparsity_breakdown.items():
        metrics[f"sparsity_{key}"] = value

    # Add sparsity metrics
    sparsity_metrics = model.compute_sparsity_metrics(
        r_values.reshape(-1, model.r_dim),
        r2_values.reshape(-1, model.r2_dim),
        w_values.reshape(-1, model.mix_dim),
    )
    metrics.update(sparsity_metrics)

    return metrics


def evaluate_batch(model: TiDHy, X: jax.Array, include_sparsity: bool = False) -> Dict[str, float]:
    """
    Evaluate model on a batch using vmap.

    Args:
        model: TiDHy model instance
        X: Input batch of shape (batch_size, T, input_dim)
        include_sparsity: Whether to include sparsity regularization in evaluation

    Returns:
        Dictionary of metrics
    """
    if include_sparsity:
        return _evaluate_batch_with_sparsity(model, X)
    else:
        return _evaluate_batch_basic(model, X)


def evaluate_record(
    model: TiDHy,
    X: jax.Array,
    rng,
) -> Tuple[float, float, float, Dict[str, jax.Array]]:
    """
    Forward pass for evaluation with full recording using vmap.

    This function processes each sequence in the batch and records all intermediate values.

    Args:
        model: TiDHy model instance
        X: Input data of shape (batch_size, T, input_dim)
        rng: JAX random key for initialization

    Returns:
        Tuple of (spatial_loss_rhat_avg, spatial_loss_rbar_avg, temp_loss_avg, result_dict)
    """
    batch_size, T, input_dim = X.shape

    def record_single_sequence(x_seq, rng_key):
        """
        Process a single sequence and record all values.

        Args:
            x_seq: Single sequence of shape (T, input_dim)
            rng_key: JAX random key for this sequence

        Returns:
            Dictionary of recorded values
        """
        init_key, first_step_key = jax.random.split(rng_key)
        # Initialize
        _, r2_init = model.init_code(init_key)

        spat_loss_fn = model.get_spatial_loss_fn()

        # First step (return_stats=False for performance in training)
        r0, r2_0 = model.inf_first_step(x_seq[0], first_step_key, return_stats=False)
        x_hat_0 = model.spatial_decoder(r0)
        x_bar_0 = model.spatial_decoder(jnp.zeros(model.r_dim))

        # Record t=0
        r_hats = [r0]
        r2_hats = [r2_0]
        x_hats = [x_hat_0]
        x_bars = [x_bar_0]
        ws = [jnp.zeros(model.mix_dim)]
        V_ts = [jnp.zeros((model.r_dim, model.r_dim))]

        spatial_losses_rhat = [jnp.sum(spat_loss_fn(x_hat_0, x_seq[0]))]
        spatial_losses_rbar = [jnp.sum(spat_loss_fn(x_bar_0, x_seq[0]))]
        temp_losses = [jnp.zeros(())]

        # Process remaining timesteps
        def scan_fn(carry, x_t):
            r_prev, r2_prev = carry

            # Temporal prediction
            r_bar, V_t, w = model.temporal_prediction(r_prev, r2_prev)
            x_bar = model.spatial_decoder(r_bar)

            # Inference (return_stats=False for performance in training)
            r, r2 = model.inf(x_t, r_prev, r2_prev, return_stats=False)
            x_hat = model.spatial_decoder(r)

            # Compute losses
            loss_rhat = jnp.sum(spat_loss_fn(x_hat, x_t))
            loss_rbar = jnp.sum(spat_loss_fn(x_bar, x_t))
            tloss = jnp.sum((r - r_bar) ** 2)

            outputs = {
                "r_hat": r,
                "r_bar": r_bar,
                "r2_hat": r2,
                "x_hat": x_hat,
                "x_bar": x_bar,
                "w": w,
                "V_t": V_t,
                "spatial_loss_rhat": loss_rhat,
                "spatial_loss_rbar": loss_rbar,
                "temp_loss": tloss,
            }

            return (r, r2), outputs

        # Run scan
        if T > 1:
            init_carry = (r0, r2_0)
            _, scan_outputs = jax.lax.scan(scan_fn, init_carry, x_seq[1:])

            # Concatenate with t=0
            r_hats = jnp.concatenate([r0[None, :], scan_outputs["r_hat"]], axis=0)
            r_bars = jnp.concatenate([r0[None, :], scan_outputs["r_bar"]], axis=0)
            r2_hats = jnp.concatenate([r2_0[None, :], scan_outputs["r2_hat"]], axis=0)
            x_hats = jnp.concatenate([x_hat_0[None, :], scan_outputs["x_hat"]], axis=0)
            x_bars = jnp.concatenate([x_bar_0[None, :], scan_outputs["x_bar"]], axis=0)
            ws = jnp.concatenate([jnp.zeros(model.mix_dim)[None, :], scan_outputs["w"]], axis=0)
            V_ts = jnp.concatenate(
                [jnp.zeros((model.r_dim, model.r_dim))[None, :, :], scan_outputs["V_t"]], axis=0
            )
            spatial_losses_rhat = jnp.concatenate(
                [spatial_losses_rhat[0][None], scan_outputs["spatial_loss_rhat"]], axis=0
            )
            spatial_losses_rbar = jnp.concatenate(
                [spatial_losses_rbar[0][None], scan_outputs["spatial_loss_rbar"]], axis=0
            )
            temp_losses = jnp.concatenate([temp_losses[0][None], scan_outputs["temp_loss"]], axis=0)
        else:
            # T == 1 case
            r_hats = r0[None, :]
            r_bars = r0[None, :]
            r2_hats = r2_0[None, :]
            x_hats = x_hat_0[None, :]
            x_bars = x_bar_0[None, :]
            ws = jnp.zeros(model.mix_dim)[None, :]
            V_ts = jnp.zeros((model.r_dim, model.r_dim))[None, :, :]
            spatial_losses_rhat = jnp.array(spatial_losses_rhat)
            spatial_losses_rbar = jnp.array(spatial_losses_rbar)
            temp_losses = jnp.array(temp_losses)

        return {
            "R_hat": r_hats,
            "R_bar": r_bars,
            "R2_hat": r2_hats,
            "I_hat": x_hats,
            "I_bar": x_bars,
            "W": ws,
            "Ut": V_ts,
            "spatial_loss_rhat": spatial_losses_rhat,
            "spatial_loss_rbar": spatial_losses_rbar,
            "temp_loss": temp_losses,
        }

    # Generate random keys for each sequence in the batch
    keys = jax.random.split(rng, batch_size)

    # Vmap over batch
    vmapped_record = jax.vmap(record_single_sequence, in_axes=(0, 0))
    result_dict = vmapped_record(X, keys)

    # Add input data
    result_dict["I"] = X

    # Compute average losses
    spatial_loss_rhat_avg = jnp.mean(result_dict["spatial_loss_rhat"])
    spatial_loss_rbar_avg = jnp.mean(result_dict["spatial_loss_rbar"])
    temp_loss_avg = model.temp_weight * jnp.mean(result_dict["temp_loss"])

    return spatial_loss_rhat_avg, spatial_loss_rbar_avg, temp_loss_avg, result_dict


def unwrap_values(d):
    """
    Unwrap nested dictionaries with 'value' keys.

    Helper function for loading checkpoints.

    Args:
        d: Dictionary to unwrap

    Returns:
        Unwrapped dictionary
    """
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            if "value" in v and len(v) == 1:
                # It's a leaf node with {'value': Array}, unwrap it
                result[k] = v["value"]
            else:
                # It's a nested structure, recurse
                result[k] = unwrap_values(v)
        else:
            result[k] = v
    return result


def load_model(model: TiDHy, filepath: str) -> TiDHy:
    """
    Load only model weights from a checkpoint (ignoring optimizer state).

    This is useful when you want to load a trained model without needing
    to create a matching optimizer structure.

    Args:
        model: TiDHy model instance (for structure)
        filepath: Path to saved checkpoint directory

    Returns:
        Loaded model with weights from checkpoint
    """
    # Convert to Path object
    load_path = Path(filepath)

    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint directory {load_path} does not exist")

    checkpoint_path = load_path / "checkpoint"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Use PyTreeCheckpointHandler to load the full checkpoint
    # We load the full checkpoint but only use the model_state
    handler = PyTreeCheckpointHandler()

    # Load the full checkpoint structure without providing a target
    # This gives us access to all components
    full_checkpoint = handler.restore(checkpoint_path)

    # Extract just the model_state
    if "model_state" not in full_checkpoint:
        raise ValueError(f"Checkpoint at {checkpoint_path} does not contain 'model_state'")

    model_state = full_checkpoint["model_state"]

    # Merge state into model
    graphdef, old_state = nnx.split(model)
    unwrapped_model_state = unwrap_values(model_state)
    nnx.replace_by_pure_dict(old_state, unwrapped_model_state)
    loaded_model = nnx.merge(graphdef, model_state)

    print(f"Model weights loaded from {load_path}")
    return loaded_model


def checkpoint_model(
    model: TiDHy,
    optimizer: nnx.Optimizer,
    epoch: int,
    checkpoint_dir: str,
    checkpointer: ocp.StandardCheckpointer,
    max_to_keep: int = 3,
    final: bool = False,
):
    """
    Save checkpoint including optimizer state using Orbax.

    Args:
        model: TiDHy model instance
        optimizer: NNX optimizer
        epoch: Current epoch
        checkpoint_dir: Directory to save checkpoints
        checkpointer: Persistent StandardCheckpointer instance for async saving
        max_to_keep: Maximum number of checkpoints to keep (default: 3)
        final: Whether this is the final checkpoint (default: False)
    """
    # Convert epoch to int if it's a JAX array
    if hasattr(epoch, "item"):
        epoch = int(epoch.item())
    else:
        epoch = int(epoch)

    # Convert to Path object
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # Extract states for saving
    _, model_state = nnx.split(model)
    _, opt_state = nnx.split(optimizer)

    # Create epoch-specific checkpoint directory
    if final:
        epoch_dir = ckpt_path / "final"
    else:
        epoch_dir = ckpt_path / f"epoch_{epoch:04d}"
    epoch_dir.mkdir(exist_ok=True)

    # Save checkpoint components - bundle everything in a dict to avoid scalar issues
    checkpoint_data = {
        "model_state": model_state,
        "optimizer_state": opt_state,
        "epoch": {
            "value": epoch
        },  # Wrap scalar in dict to avoid StandardCheckpointer scalar restriction
    }

    # Save with StandardCheckpointer (it automatically handles sharding metadata)
    checkpointer.save(epoch_dir / "checkpoint", checkpoint_data)

    # Clean up old checkpoints if needed
    if max_to_keep > 0:
        # Get all checkpoint directories
        checkpoint_dirs = [
            d for d in ckpt_path.iterdir() if d.is_dir() and d.name.startswith("epoch_")
        ]
        checkpoint_dirs.sort(key=lambda x: int(x.name.split("_")[1]))

        # Remove old checkpoints
        while len(checkpoint_dirs) > max_to_keep:
            old_dir = checkpoint_dirs.pop(0)
            import shutil

            shutil.rmtree(old_dir)

    print(f"Checkpoint saved to {epoch_dir}")


def load_checkpoint(
    model: TiDHy, optimizer: nnx.Optimizer, checkpoint_dir: str, epoch: Optional[int] = None
) -> Tuple[TiDHy, nnx.Optimizer, int]:
    """
    Load checkpoint including optimizer state using Orbax.

    Args:
        model: TiDHy model instance (for structure)
        optimizer: NNX optimizer (for structure)
        checkpoint_dir: Directory containing checkpoints
        epoch: Specific epoch to load (if None, loads latest)

    Returns:
        Tuple of (loaded model, loaded optimizer, epoch)
    """
    # Convert to Path object
    ckpt_path = Path(checkpoint_dir)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_path} does not exist")

    # Find available checkpoints
    checkpoint_dirs = [d for d in ckpt_path.iterdir() if d.is_dir() and d.name.startswith("epoch_")]
    if not checkpoint_dirs:
        raise ValueError(f"No checkpoints found in {ckpt_path}")

    checkpoint_dirs.sort(key=lambda x: int(x.name.split("_")[1]))

    # Determine which checkpoint to load
    if epoch is None:
        # Load latest
        epoch_dir = checkpoint_dirs[-1]
        loaded_epoch = {"value": int(epoch_dir.name.split("_")[1])}
    else:
        # Load specific epoch
        epoch_dir = ckpt_path / f"epoch_{epoch:04d}"
        if not epoch_dir.exists():
            available = [int(d.name.split("_")[1]) for d in checkpoint_dirs]
            raise ValueError(f"Checkpoint for epoch {epoch} not found. Available: {available}")
        loaded_epoch = {"value": epoch}

    # Load using Orbax
    orbax_path = epoch_dir / "checkpoint"
    if not orbax_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {orbax_path}")

    # Extract current states for structure
    _, current_model_state = nnx.split(model)
    _, current_opt_state = nnx.split(optimizer)

    # Create template for loading
    checkpoint_template = {
        "model_state": current_model_state,
        "optimizer_state": current_opt_state,
        "epoch": {"value": 0},  # Match the save format
    }

    # Use Orbax StandardCheckpointer
    checkpointer = ocp.StandardCheckpointer()

    # Try to load checkpoint with optimizer state
    try:
        # StandardCheckpointer automatically reads sharding info from checkpoint
        restored = checkpointer.restore(orbax_path, target=checkpoint_template)

        model_state = restored["model_state"]
        opt_state = restored["optimizer_state"]
        loaded_epoch = restored["epoch"]["value"]

        # Restore model and optimizer
        model_graphdef, _ = nnx.split(model)
        loaded_model = nnx.merge(model_graphdef, model_state)

        opt_graphdef, _ = nnx.split(optimizer)
        loaded_optimizer = nnx.merge(opt_graphdef, opt_state)

        print(f"Checkpoint loaded from {epoch_dir}")
        return loaded_model, loaded_optimizer, int(loaded_epoch)

    except ValueError as e:
        # If optimizer state structure mismatch, load only model weights
        if "tree structures do not match" in str(e):
            print(
                "Warning: Optimizer structure mismatch. Loading model weights only and creating fresh optimizer."
            )

            # Use PyTreeCheckpointHandler to load the full checkpoint
            handler = PyTreeCheckpointHandler()

            # Load the full checkpoint structure
            full_checkpoint = handler.restore(orbax_path)

            # Extract just the model_state
            model_state = full_checkpoint["model_state"]

            # Restore model with fresh optimizer (passed in)
            model_graphdef, _ = nnx.split(model)
            loaded_model = nnx.merge(model_graphdef, model_state)

            # Get epoch from checkpoint if available
            try:
                loaded_epoch = int(full_checkpoint["epoch"])
            except (KeyError, TypeError):
                loaded_epoch = int(epoch_dir.name.split("_")[1])

            print(f"Model weights loaded from {epoch_dir} (optimizer state reset)")
            return loaded_model, optimizer, loaded_epoch
        else:
            # Re-raise if it's a different error
            raise


def list_checkpoints(checkpoint_dir: str) -> list[int]:
    """
    List all available checkpoint epochs in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        List of available epoch numbers
    """
    ckpt_path = Path(checkpoint_dir)

    if not ckpt_path.exists():
        return []

    # Find all checkpoint directories
    checkpoint_dirs = [d for d in ckpt_path.iterdir() if d.is_dir() and d.name.startswith("epoch_")]
    epochs = [int(d.name.split("_")[1]) for d in checkpoint_dirs]

    return sorted(epochs)


def get_latest_checkpoint_epoch(checkpoint_dir: str) -> Optional[int]:
    """
    Get the epoch number of the latest checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Latest epoch number or None if no checkpoints exist
    """
    epochs = list_checkpoints(checkpoint_dir)
    return epochs[-1] if epochs else None


def _train_model_internal(
    model: TiDHy,
    train_data: jax.Array,
    config,
    checkpoint_dir: Optional[str],
    val_data: Optional[jax.Array] = None,
    start_epoch: int = 0,
    optimizer: Optional[nnx.Optimizer] = None,
    checkpoint_every: int = 10,  # Deprecated: kept for backward compatibility
    max_checkpoints_to_keep: int = 5,
    verbose: bool = True,
    # Wandb logging parameters
    use_wandb: bool = True,  # Default to True since this is typically called from Run script
    log_params_every: int = 10,
    log_sparsity_every: int = 5,
) -> Tuple[TiDHy, Dict[str, list]]:
    """
    Internal training function with optional checkpointing.

    Checkpoints are saved only when the validation loss decreases (if checkpoint_dir provided),
    ensuring that only the best models are saved. A final checkpoint is also saved at
    the end of training.

    Args:
        model: TiDHy model instance
        train_data: Training data of shape (n_samples, T, input_dim)
        config: Configuration object with config.model and config.train sections
        checkpoint_dir: Directory to save checkpoints (None to disable checkpointing)
        val_data: Optional validation data (required for best-model checkpointing)
        start_epoch: Starting epoch (for resuming training)
        optimizer: Pre-created optimizer (for resuming training)
        checkpoint_every: (Deprecated) No longer used - checkpoints saved on val loss improvement
        max_checkpoints_to_keep: Maximum number of checkpoints to keep (default: 5)
        max_checkpoints_to_keep: Maximum number of checkpoints to keep
        verbose: Whether to print progress
        use_wandb: Whether to log to Weights & Biases (expects existing session)
        log_params_every: Log model parameters every N epochs
        log_sparsity_every: Log sparsity analysis every N epochs

    Returns:
        Tuple of (trained model, history dictionary)
    """
    # Extract training parameters from config
    n_epochs = config.train.num_epochs
    batch_size = None if config.train.batch_size_input else config.train.batch_size

    # Extract learning rates - check if multi-LR is configured
    learning_rate_s = config.train.learning_rate_s
    learning_rate_t = config.train.learning_rate_t
    learning_rate_h = config.train.learning_rate_h

    # Fallback to default if not specified
    default_lr = config.train.learning_rate
    weight_decay = config.train.weight_decay

    # GradNorm settings
    use_gradnorm = config.train.grad_norm
    grad_alpha = config.train.grad_alpha

    # Validation frequency
    val_every_n_epochs = config.train.save_summary_steps

    # Learning rate schedule settings
    use_schedule = config.train.use_schedule
    schedule_transition_steps = config.train.schedule_transition_steps
    schedule_decay_rate = config.train.schedule_decay

    # Create optimizer if not provided (fresh training)
    if optimizer is None:
        if (
            learning_rate_s is not None
            or learning_rate_t is not None
            or learning_rate_h is not None
        ):
            # Use component-specific learning rates
            lr_s = learning_rate_s if learning_rate_s is not None else default_lr
            lr_t = learning_rate_t if learning_rate_t is not None else default_lr
            lr_h = learning_rate_h if learning_rate_h is not None else default_lr

            optimizer = create_multi_lr_optimizer(
                model,
                lr_s,
                lr_t,
                lr_h,
                weight_decay,
                use_schedule=use_schedule,
                schedule_transition_steps=schedule_transition_steps,
                schedule_decay_rate=schedule_decay_rate,
            )
        else:
            # Use single learning rate for all parameters
            optimizer_tx = create_optimizer(default_lr, weight_decay)
            optimizer = nnx.Optimizer(model=model, tx=optimizer_tx, wrt=nnx.Param)

    # Create persistent checkpointer for saving model checkpoints (if needed)
    checkpointer = ocp.StandardCheckpointer() if checkpoint_dir is not None else None

    # Warn if no validation data provided when checkpointing is enabled
    if checkpoint_dir is not None and val_data is None and verbose:
        print("Warning: No validation data provided. Only final checkpoint will be saved.")

    # History
    history = {
        "loss": [],
        "spatial_loss_rhat": [],
        "spatial_loss_rbar": [],
        "temp_loss": [],
        "cos_reg": [],
    }

    if val_data is not None:
        history["val_loss"] = []

    # Track best validation loss for checkpointing and display
    best_val_loss = float("inf")
    best_epoch = -1
    current_val_loss = None  # Track current validation loss for display

    # Check if wandb is available for logging
    wandb_available = use_wandb and wandb.run is not None

    # Training loop with checkpointing
    epoch_iter = (
        tqdm(range(start_epoch, n_epochs), desc="Training")
        if verbose
        else range(start_epoch, n_epochs)
    )

    for epoch in epoch_iter:
        # Train using JIT-compiled function
        metrics = train_epoch(model, optimizer, train_data, batch_size, use_gradnorm, grad_alpha)

        # Record history
        for key in ["loss", "spatial_loss_rhat", "spatial_loss_rbar", "temp_loss", "cos_reg"]:
            if key in metrics:
                history[key].append(float(metrics[key]))

        # Wandb logging for training metrics
        if wandb_available:
            log_training_step(
                metrics, step=epoch, prefix="train", log_sparsity=True, log_gradnorm=use_gradnorm
            )

        # Validation - only run every N epochs
        if val_data is not None and (epoch + 1) % val_every_n_epochs == 0:
            val_metrics = evaluate_batch(model, val_data, include_sparsity=True)
            current_val_loss = float(val_metrics["loss"])
            history["val_loss"].append(current_val_loss)

            # Log validation metrics to wandb
            if wandb_available:
                log_training_step(val_metrics, step=epoch, prefix="val", log_sparsity=True)

            # Save checkpoint if validation loss improved (and checkpointing enabled)
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_epoch = epoch + 1

                if checkpoint_dir is not None:
                    if verbose:
                        print(
                            f"\nValidation loss improved to {current_val_loss:.6f}. Saving checkpoint..."
                        )

                    checkpoint_model(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch + 1,
                        checkpoint_dir=checkpoint_dir,
                        checkpointer=checkpointer,
                        max_to_keep=max_checkpoints_to_keep,
                    )

                # Also log to wandb if available
                if wandb_available:
                    wandb.log(
                        {"best_val_loss": best_val_loss, "best_epoch": best_epoch}, step=epoch
                    )

        # Log model parameters periodically
        if wandb_available and (epoch + 1) % log_params_every == 0:
            log_optimization_metrics(optimizer, step=epoch)

        # Log detailed sparsity analysis periodically
        if wandb_available and (epoch + 1) % log_sparsity_every == 0:
            # Get a sample batch for sparsity analysis
            sample_size = min(
                batch_size if batch_size else len(train_data), 8
            )  # Limit to 8 samples
            sample_data = train_data[:sample_size]

            # Forward pass to get internals
            # Generate random keys for each sequence
            sample_size_actual = sample_data.shape[0]
            base_key = model.rngs()  # Get key outside vmap
            keys = jax.random.split(base_key, sample_size_actual)

            def forward_single(seq, key):
                return model(seq, return_internals=True, rng_key=key)

            vmapped_forward = jax.vmap(forward_single, in_axes=(0, 0))
            _, (r_values, r2_values, w_values), _ = vmapped_forward(sample_data, keys)

            log_sparsity_analysis(model, r_values, r2_values, w_values, step=epoch)

        # Print progress
        if verbose:
            msg = f"Epoch {epoch + 1}/{n_epochs} - Loss: {metrics['loss']:.4f}"
            if current_val_loss is not None:
                msg += f" - Val Loss: {current_val_loss:.4f}"
                if best_epoch == epoch + 1:
                    msg += " (✓ best)"
            if isinstance(epoch_iter, tqdm):
                epoch_iter.set_postfix_str(msg)
            else:
                print(msg)

    # Save final checkpoint (if checkpointing enabled)
    if checkpoint_dir is not None:
        if verbose:
            if val_data is not None:
                print(
                    f"\nTraining complete. Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}"
                )
            else:
                print("\nTraining complete.")

        checkpoint_model(
            model=model,
            optimizer=optimizer,
            epoch=n_epochs,
            checkpoint_dir=checkpoint_dir,
            checkpointer=checkpointer,
            max_to_keep=max_checkpoints_to_keep,
            final=True,
        )

        # Wait for all async checkpoint operations to complete before returning
        checkpointer.wait_until_finished()
    elif verbose and val_data is not None:
        print(
            f"\nTraining complete. Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}"
        )

    return model, history


def train_model_with_checkpointing(
    model: TiDHy,
    train_data: jax.Array,
    config,
    checkpoint_dir: str,
    val_data: Optional[jax.Array] = None,
    start_epoch: int = 0,
    optimizer: Optional[nnx.Optimizer] = None,
    checkpoint_every: int = 10,
    max_checkpoints_to_keep: int = 5,
    verbose: bool = True,
    use_wandb: bool = True,
    log_params_every: int = 10,
    log_sparsity_every: int = 5,
) -> Tuple[TiDHy, Dict[str, list]]:
    """
    Train model with automatic checkpointing when validation loss improves.

    This is a convenience wrapper around _train_model_internal with checkpointing enabled.
    Checkpoints are saved only when the validation loss decreases.

    Args:
        model: TiDHy model instance
        train_data: Training data of shape (n_samples, T, input_dim)
        config: Configuration object with config.model and config.train sections
        checkpoint_dir: Directory to save checkpoints
        val_data: Optional validation data (required for best-model checkpointing)
        start_epoch: Starting epoch (for resuming training)
        optimizer: Pre-created optimizer (for resuming training)
        checkpoint_every: (Deprecated) No longer used
        max_checkpoints_to_keep: Maximum number of checkpoints to keep
        verbose: Whether to print progress
        use_wandb: Whether to log to Weights & Biases
        log_params_every: Log model parameters every N epochs
        log_sparsity_every: Log sparsity analysis every N epochs

    Returns:
        Tuple of (trained model, history dictionary)
    """
    return _train_model_internal(
        model=model,
        train_data=train_data,
        config=config,
        checkpoint_dir=checkpoint_dir,
        val_data=val_data,
        start_epoch=start_epoch,
        optimizer=optimizer,
        checkpoint_every=checkpoint_every,
        max_checkpoints_to_keep=max_checkpoints_to_keep,
        verbose=verbose,
        use_wandb=use_wandb,
        log_params_every=log_params_every,
        log_sparsity_every=log_sparsity_every,
    )
