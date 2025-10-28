"""
JAX/Flax implementation of TiDHy model with full inference optimization loops.
This module provides optimization-based inference using optax.
"""

from typing import Tuple, Dict, Any
import jax
import jax.numpy as jnp
import optax
from flax.core import freeze, unfreeze


def soft_thresholding(r: jax.Array, lmda: float) -> jax.Array:
    """Non-negative proximal gradient for L1 regularization."""
    return jax.nn.relu(jnp.abs(r) - lmda) * jnp.sign(r)


def inf_first_step(
    model_apply_fn,
    params,
    x: jax.Array,
    lr_r: float = 0.1,
    max_iter: int = 100,
    tol: float = 1e-4,
    batch_converge: bool = False
) -> jax.Array:
    """
    First step inference: p(r_1 | x_1) (no temporal prior)

    Args:
        model_apply_fn: Model's apply function (spatial_decoder)
        params: Model parameters
        x: Observation at time t=1
        lr_r: Learning rate for r
        max_iter: Maximum iterations
        tol: Convergence tolerance
        batch_converge: Whether to check convergence across batch

    Returns:
        Inferred r at time 1
    """
    batch_size = x.shape[0]
    r_dim = params['params']['spatial_decoder']['Dense_0']['kernel'].shape[0]

    # Initialize r
    r = jnp.zeros((batch_size, r_dim))

    # Create optimizer
    optimizer = optax.adamw(lr_r)
    opt_state = optimizer.init(r)

    def loss_fn(r_val):
        x_bar = model_apply_fn(params, r_val, method='spatial_decoder')
        # MSE loss (can be modified for other loss types)
        loss = jnp.sum((x_bar - x) ** 2, axis=-1).mean()
        return loss

    def step(carry, _):
        r_val, opt_state_val = carry
        old_r = r_val

        # Compute gradients and update
        loss, grads = jax.value_and_grad(loss_fn)(r_val)
        updates, opt_state_new = optimizer.update(grads, opt_state_val)
        r_new = optax.apply_updates(r_val, updates)

        # Check convergence
        if batch_converge:
            r_diff = jnp.linalg.norm(r_new - old_r) / (jnp.linalg.norm(old_r) + 1e-16)
        else:
            r_diff = jnp.linalg.norm(r_new - old_r, axis=-1) / (
                jnp.linalg.norm(old_r, axis=-1) + 1e-16
            )

        converged = jnp.all(r_diff < tol)

        return (r_new, opt_state_new), (loss, converged)

    # Run optimization loop
    (r_final, _), (losses, converged_flags) = jax.lax.scan(
        step, (r, opt_state), None, length=max_iter
    )

    return r_final


def inf_step(
    model,
    params,
    x: jax.Array,
    r_p: jax.Array,
    r2: jax.Array,
    lr_r: float = 0.1,
    lr_r2: float = 0.1,
    lmda_r: float = 0.0,
    lmda_r2: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-4,
    temp_weight: float = 1.0,
    batch_converge: bool = False,
    use_r2_decoder: bool = False,
    L1_inf_w: float = 0.0,
    L1_inf_r2: float = 0.0,
    L1_inf_r: float = 0.0
) -> Tuple[jax.Array, jax.Array, float]:
    """
    Inference step: p(r_t, r2_t | x_t, r_t-1, r2_t-1)

    Args:
        model: TiDHy model instance
        params: Model parameters
        x: Current observation
        r_p: Previous r
        r2: Initial r2
        lr_r: Learning rate for r
        lr_r2: Learning rate for r2
        lmda_r: L1 penalty for r
        lmda_r2: L1 penalty for r2
        max_iter: Maximum iterations
        tol: Convergence tolerance
        temp_weight: Weight for temporal loss
        batch_converge: Whether to check convergence across batch
        use_r2_decoder: Whether to use r2 decoder
        L1_inf_w: L1 penalty for mixture weights
        L1_inf_r2: L1 penalty for r2 in inference
        L1_inf_r: L1 penalty for r in inference

    Returns:
        Tuple of (r, r2, r2_loss)
    """
    batch_size = x.shape[0]
    r_dim = r_p.shape[-1]

    # Initialize r
    r = jnp.zeros((batch_size, r_dim))

    # Create optimizers
    optimizer_r = optax.sgd(lr_r, momentum=0.9, nesterov=True)
    optimizer_r2 = optax.sgd(lr_r2, momentum=0.9, nesterov=True)

    opt_state_r = optimizer_r.init(r)
    opt_state_r2 = optimizer_r2.init(r2)

    # Create learning rate schedules
    milestones_r = list(range(max_iter // 10, max_iter, max_iter // 10))
    milestones_r2 = list(range(max_iter // 5, max_iter, max_iter // 10))

    def loss_fn(r_val, r2_val):
        # Spatial reconstruction
        x_bar = model.apply(
            params, r_val, method='spatial_decoder'
        )
        spatial_loss = jnp.sum((x_bar - x) ** 2, axis=-1).mean()

        # Temporal prediction
        r_bar, V_t, w = model.apply(
            params, r_p, r2_val, method='temporal_prediction'
        )
        temporal_loss = jnp.sum((r_val - r_bar) ** 2, axis=-1).mean()

        # R2 decoder loss (if applicable)
        r2_loss_val = 0.0
        if use_r2_decoder:
            r2_bar = model.apply(
                params,
                jnp.concatenate([r_val, r2_val], axis=-1),
                method='r2_decoder'
            )
            r2_loss_val = jnp.sum((r2_val - r2_bar) ** 2, axis=-1).mean()

        # Sparsity penalties
        inf_w_sparsity = L1_inf_w * jnp.linalg.norm(w.ravel(), ord=1)
        inf_r2_sparsity = L1_inf_r2 * jnp.linalg.norm(r2_val.ravel(), ord=1)
        inf_r_sparsity = L1_inf_r * jnp.linalg.norm(r_val.ravel(), ord=1)

        total_loss = (spatial_loss +
                     temp_weight * temporal_loss +
                     inf_w_sparsity +
                     inf_r2_sparsity +
                     inf_r_sparsity)

        if use_r2_decoder:
            total_loss += r2_loss_val

        return total_loss, (spatial_loss, temporal_loss, r2_loss_val)

    def step(carry, _):
        r_val, r2_val, opt_state_r_val, opt_state_r2_val, iteration = carry
        old_r = r_val
        old_r2 = r2_val

        # Compute gradients
        (loss, (spatial_loss, temporal_loss, r2_loss_val)), grads = jax.value_and_grad(
            lambda r, r2: loss_fn(r, r2)[0], argnums=(0, 1), has_aux=False
        )(r_val, r2_val)

        grad_r, grad_r2 = grads

        # Update r
        updates_r, opt_state_r_new = optimizer_r.update(grad_r, opt_state_r_val)
        r_new = optax.apply_updates(r_val, updates_r)

        # Update r2
        updates_r2, opt_state_r2_new = optimizer_r2.update(grad_r2, opt_state_r2_val)
        r2_new = optax.apply_updates(r2_val, updates_r2)

        # Apply soft thresholding
        r_new = soft_thresholding(r_new, lmda_r)
        r2_new = soft_thresholding(r2_new, lmda_r2)

        # Check convergence
        if batch_converge:
            r2_converge = jnp.linalg.norm(r2_new - old_r2) / (
                jnp.linalg.norm(old_r2) + 1e-16
            )
            r_converge = jnp.linalg.norm(r_new - old_r) / (
                jnp.linalg.norm(old_r) + 1e-16
            )
        else:
            r2_converge = jnp.linalg.norm(r2_new - old_r2, axis=-1) / (
                jnp.linalg.norm(old_r2, axis=-1) + 1e-16
            )
            r_converge = jnp.linalg.norm(r_new - old_r, axis=-1) / (
                jnp.linalg.norm(old_r, axis=-1) + 1e-16
            )

        converged = jnp.all(r_converge < tol) & jnp.all(r2_converge < tol)

        return (
            r_new,
            r2_new,
            opt_state_r_new,
            opt_state_r2_new,
            iteration + 1
        ), (loss, converged, spatial_loss, temporal_loss)

    # Run optimization loop
    (r_final, r2_final, _, _, _), (losses, converged_flags, _, _) = jax.lax.scan(
        step,
        (r, r2, opt_state_r, opt_state_r2, 0),
        None,
        length=max_iter
    )

    # Compute final r2 loss for reporting
    if use_r2_decoder:
        r2_bar = model.apply(
            params,
            jnp.concatenate([r_final, r2_final], axis=-1),
            method='r2_decoder'
        )
        final_r2_loss = jnp.sum((r2_final - r2_bar) ** 2, axis=-1).mean()
    else:
        final_r2_loss = 0.0

    return r_final, r2_final, final_r2_loss


def evaluate_record(
    model,
    params,
    data_batch: jax.Array,
    max_iter: int = 100,
    tol: float = 1e-4,
    **kwargs
) -> Tuple[float, float, float, Dict[str, jax.Array]]:
    """
    Forward pass for evaluation with full recording

    Args:
        model: TiDHy model instance
        params: Model parameters
        data_batch: Input data of shape (batch_size, T, input_dim)
        max_iter: Maximum iterations for inference
        tol: Convergence tolerance
        **kwargs: Additional keyword arguments for inference

    Returns:
        Tuple of (spatial_loss_rhat_avg, spatial_loss_rbar_avg, temp_loss_avg, result_dict)
    """
    batch_size, T, input_dim = data_batch.shape
    r_dim = model.r_dim
    r2_dim = model.r2_dim
    mix_dim = model.mix_dim

    # Initialize storage arrays
    I_bar = jnp.zeros((batch_size, T, input_dim))
    I_hat = jnp.zeros((batch_size, T, input_dim))
    I = data_batch.copy()
    R_bar = jnp.zeros((batch_size, T, r_dim))
    R_hat = jnp.zeros((batch_size, T, r_dim))
    R2_hat = jnp.zeros((batch_size, T, r2_dim))
    W = jnp.zeros((batch_size, T, mix_dim))
    temp_loss = jnp.zeros((batch_size, T, r_dim))
    spatial_loss_rhat = jnp.zeros((batch_size, T, input_dim))
    spatial_loss_rbar = jnp.zeros((batch_size, T, input_dim))
    Ut = jnp.zeros((batch_size, T, r_dim, r_dim))

    # Initialize codes
    r = jnp.zeros((batch_size, r_dim))
    r2 = jnp.zeros((batch_size, r2_dim))

    # Store initial values
    R_bar = R_bar.at[:, 0].set(r)
    R2_hat = R2_hat.at[:, 0].set(r2)
    I_bar = I_bar.at[:, 0].set(model.apply(params, r, method='spatial_decoder'))

    # First step inference
    r = inf_first_step(
        model.apply, params, data_batch[:, 0],
        lr_r=kwargs.get('lr_r', 0.1),
        max_iter=max_iter, tol=tol,
        batch_converge=kwargs.get('batch_converge', False)
    )

    R_hat = R_hat.at[:, 0].set(r)
    I_hat = I_hat.at[:, 0].set(model.apply(params, r, method='spatial_decoder'))

    # Compute losses for first step
    spat_loss_fn = model.get_spatial_loss_fn()
    spatial_loss_rhat = spatial_loss_rhat.at[:, 0].set(
        spat_loss_fn(I_hat[:, 0], data_batch[:, 0])
    )
    spatial_loss_rbar = spatial_loss_rbar.at[:, 0].set(
        spat_loss_fn(I_bar[:, 0], data_batch[:, 0])
    )

    # Main loop through time
    for t in range(1, T):
        r_p = r.copy()
        r2_p = r2.copy()

        # Temporal prediction
        r_bar, V_t, w = model.apply(params, r_p, r2, method='temporal_prediction')
        R_bar = R_bar.at[:, t].set(r_bar)
        x_bar = model.apply(params, r_bar, method='spatial_decoder')
        I_bar = I_bar.at[:, t].set(x_bar)

        # Inference
        r, r2, _ = inf_step(
            model, params, data_batch[:, t], r_p, r2,
            max_iter=max_iter, tol=tol, **kwargs
        )

        R_hat = R_hat.at[:, t].set(r)
        R2_hat = R2_hat.at[:, t].set(r2)
        x_hat = model.apply(params, r, method='spatial_decoder')
        I_hat = I_hat.at[:, t].set(x_hat)

        # Store mixture weights and dynamics
        wb = model.apply(params, r2, method='hypernet')
        W = W.at[:, t].set(wb[:, :mix_dim])
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

    # Compute average losses
    spatial_loss_rhat_avg = spatial_loss_rhat.reshape(batch_size, -1).sum(1).mean(0)
    spatial_loss_rbar_avg = spatial_loss_rbar.reshape(batch_size, -1).sum(1).mean(0)
    temp_loss_avg = model.temp_weight * temp_loss.reshape(batch_size, -1).sum(1).mean(0)

    return spatial_loss_rhat_avg, spatial_loss_rbar_avg, temp_loss_avg, result_dict
