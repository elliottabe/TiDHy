import logging
from typing import Any, Callable, Optional, Tuple
import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.core import freeze, unfreeze
import optax
from functools import partial
from tqdm.auto import tqdm

# Type aliases
PRNGKey = jax.Array
Array = jax.Array
Params = Any


def soft_thresholding(r: Array, lmda: float) -> Array:
    """Non-negative proximal gradient for L1 regularization."""
    return jax.nn.relu(jnp.abs(r) - lmda) * jnp.sign(r)


def soft_max(r: Array, axis: int = -1) -> Array:
    """Softmax activation."""
    return jax.nn.softmax(r, axis=axis)


def heavyside(r: Array, value: Array) -> Array:
    """Heaviside step function."""
    return jnp.heaviside(r, value)


def poissonLoss(predicted: Array, observed: Array) -> Array:
    """Custom loss function for Poisson model."""
    return predicted - observed * jnp.log(predicted + 1e-10)


class SpatialDecoder(nn.Module):
    """Spatial decoder: p(I | r)"""
    r_dim: int
    input_dim: int
    hyper_hid_dim: int
    loss_type: str
    nonlin_decoder: bool

    @nn.compact
    def __call__(self, r: Array) -> Array:
        if self.loss_type == 'BCE':
            x = nn.Dense(self.input_dim,
                        kernel_init=nn.initializers.xavier_normal())(r)
            return nn.sigmoid(x)

        if self.nonlin_decoder:
            x = nn.Dense(self.hyper_hid_dim,
                        kernel_init=nn.initializers.xavier_normal())(r)
            x = nn.LayerNorm()(x)
            x = nn.elu(x)
            x = nn.Dense(self.hyper_hid_dim)(x)
            x = nn.Dense(self.input_dim)(x)
            return x
        else:
            return nn.Dense(self.input_dim,
                          kernel_init=nn.initializers.xavier_normal())(r)


class HyperNetwork(nn.Module):
    """Hypernetwork for generating mixture weights"""
    r2_dim: int
    hyper_hid_dim: int
    mix_dim: int
    r_dim: int
    dyn_bias: bool

    @nn.compact
    def __call__(self, r2: Array) -> Array:
        x = nn.Dense(self.hyper_hid_dim,
                    kernel_init=nn.initializers.xavier_normal())(r2)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        x = nn.Dense(self.hyper_hid_dim,
                    kernel_init=nn.initializers.xavier_normal())(x)

        if self.dyn_bias:
            x = nn.Dense(self.mix_dim + self.r_dim, use_bias=False,
                        kernel_init=nn.initializers.xavier_normal())(x)
        else:
            x = nn.Dense(self.mix_dim, use_bias=True,
                        kernel_init=nn.initializers.xavier_normal())(x)

        return nn.relu(x)


class R2Decoder(nn.Module):
    """R2 decoder network"""
    r2_dim: int
    r_dim: int
    r2_decoder_hid_dim: int

    @nn.compact
    def __call__(self, r_r2_concat: Array) -> Array:
        x = nn.Dense(self.r2_decoder_hid_dim,
                    kernel_init=nn.initializers.xavier_normal())(r_r2_concat)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        x = nn.Dense(self.r2_decoder_hid_dim,
                    kernel_init=nn.initializers.xavier_normal())(x)
        x = nn.Dense(self.r2_dim,
                    kernel_init=nn.initializers.xavier_normal())(x)
        return x


class TiDHy(nn.Module):
    """TiDHy model in JAX/Flax"""
    # Model dimensions
    r_dim: int
    r2_dim: int
    mix_dim: int
    input_dim: int
    hyper_hid_dim: int

    # Architecture options
    loss_type: str = 'MSE'
    nonlin_decoder: bool = False
    low_rank_temp: bool = False
    dyn_bias: bool = False
    use_r2_decoder: bool = False
    r2_decoder_hid_dim: int = 64

    # Learning rates (for inference)
    lr_r: float = 0.1
    lr_r2: float = 0.1
    lr_weights: float = 0.025
    lr_weights_inf: float = 0.025

    # Regularization
    temp_weight: float = 1.0
    lmda_r: float = 0.0
    lmda_r2: float = 0.0
    weight_decay: float = 0.0
    L1_alpha: float = 0.0
    L1_inf_w: float = 0.0
    L1_inf_r2: float = 0.0
    L1_inf_r: float = 0.0
    L1_alpha_inf: float = 0.0
    L1_alpha_r2: float = 0.0
    grad_alpha: float = 1.5
    grad_alpha_inf: float = 1.5
    clip_grad: float = 1.0
    grad_norm_inf: bool = False

    # Training params
    max_iter: int = 100
    tol: float = 1e-4
    normalize_spatial: bool = False
    normalize_temporal: bool = False
    stateful: bool = False
    batch_converge: bool = False
    spat_weight: float = 1.0
    learning_rate_gamma: float = 0.5

    # Display
    show_progress: bool = True
    show_inf_progress: bool = False

    def setup(self):
        """Initialize model components"""
        # Spatial decoder
        self.spatial_decoder = SpatialDecoder(
            r_dim=self.r_dim,
            input_dim=self.input_dim,
            hyper_hid_dim=self.hyper_hid_dim,
            loss_type=self.loss_type,
            nonlin_decoder=self.nonlin_decoder
        )

        # Temporal parameters
        if self.low_rank_temp:
            self.temporal = self.param(
                'temporal',
                nn.initializers.orthogonal(),
                (self.mix_dim, self.r_dim, 2)
            )
        else:
            self.temporal = self.param(
                'temporal',
                nn.initializers.orthogonal(),
                (self.mix_dim, self.r_dim * self.r_dim)
            )

        if self.dyn_bias:
            self.temporal_bias = self.param(
                'temporal_bias',
                nn.initializers.xavier_normal(),
                (1, self.r_dim)
            )

        # Hypernetwork
        self.hypernet = HyperNetwork(
            r2_dim=self.r2_dim,
            hyper_hid_dim=self.hyper_hid_dim,
            mix_dim=self.mix_dim,
            r_dim=self.r_dim,
            dyn_bias=self.dyn_bias
        )

        # R2 decoder (optional)
        if self.use_r2_decoder:
            self.r2_decoder = R2Decoder(
                r2_dim=self.r2_dim,
                r_dim=self.r_dim,
                r2_decoder_hid_dim=self.r2_decoder_hid_dim
            )

    def temporal_prediction(self, r: Array, r2: Array) -> Tuple[Array, Array, Array]:
        """Temporal prediction: p(r_t | r_(t-1), r2)"""
        batch_size = r.shape[0]

        # Get mixture weights and bias from hypernetwork
        wb = self.hypernet(r2)
        w = wb[:, :self.mix_dim]
        b = wb[:, self.mix_dim:] if self.dyn_bias else None

        # Compute temporal dynamics matrix
        if self.low_rank_temp:
            Vk = jnp.einsum('mij,mkj->mik', self.temporal, self.temporal)
            Vk = Vk.reshape(self.mix_dim, -1)
            V_t = jnp.einsum('bm,mij->bij', w, Vk.reshape(self.mix_dim, self.r_dim, self.r_dim))
        else:
            V_t = jnp.einsum('bm,mij->bij', w,
                           self.temporal.reshape(self.mix_dim, self.r_dim, self.r_dim))

        # Apply temporal dynamics
        r_hat = jnp.einsum('bij,bj->bi', V_t, r)

        if self.dyn_bias:
            r_hat = r_hat + b * self.temporal_bias

        return r_hat, V_t, w

    def get_spatial_loss_fn(self):
        """Get spatial loss function based on loss_type"""
        if self.loss_type == 'Poisson':
            return poissonLoss
        elif self.loss_type == 'BCE':
            return lambda pred, obs: -obs * jnp.log(pred + 1e-10) - (1 - obs) * jnp.log(1 - pred + 1e-10)
        else:  # MSE
            return lambda pred, obs: (pred - obs) ** 2

    def __call__(self, X: Array, rng: PRNGKey) -> Tuple[Array, Array, Array, Array, Array, Array]:
        """
        Forward pass through the model

        Args:
            X: Input data of shape (batch_size, T, input_dim)
            rng: PRNG key for random number generation

        Returns:
            Tuple of (spatial_loss_rhat, spatial_loss_rbar, temp_loss, r2_losses, r_first, r2_final)
        """
        batch_size, T, _ = X.shape

        # Initialize codes
        r = jnp.zeros((batch_size, self.r_dim))
        r2 = jnp.zeros((batch_size, self.r2_dim))

        r_first = r.copy()

        spat_loss_fn = self.get_spatial_loss_fn()

        # First timestep
        spatial_loss_rhat = spat_loss_fn(
            self.spatial_decoder(r), X[:, 0]
        ).reshape(batch_size, -1).sum(1).mean(0)

        spatial_loss_rbar = jnp.zeros_like(spatial_loss_rhat)
        temp_loss = 0.0
        r2_losses = 0.0

        # Iterate through time
        for t in range(1, T):
            r_p = r.copy()
            r2_p = r2.copy()

            # Inference step (simplified - actual implementation would need optimization loop)
            # This is a placeholder for the inference logic
            r, r2, r2_loss = self.inf_step(X[:, t], r_p, r2, rng)

            # Predictions
            x_hat = self.spatial_decoder(r)
            r_bar, V_t, w = self.temporal_prediction(r_p, r2)
            x_bar = self.spatial_decoder(r_bar)

            # Compute losses
            if self.use_r2_decoder:
                r2_hat = self.r2_decoder(jnp.concatenate([r_p, r2_p], axis=-1))
                r2_losses += jnp.sum((r2 - r2_hat) ** 2, axis=-1).mean()

            spatial_loss_rhat += spat_loss_fn(x_hat, X[:, t]).reshape(batch_size, -1).sum(1).mean(0)
            spatial_loss_rbar += spat_loss_fn(x_bar, X[:, t]).reshape(batch_size, -1).sum(1).mean(0)
            temp_loss += jnp.sum((r - r_bar) ** 2, axis=-1).mean()

        return (spatial_loss_rhat, spatial_loss_rbar,
                self.temp_weight * temp_loss, r2_losses, r_first, r2)

    def inf_step(self, x: Array, r_p: Array, r2: Array, rng: PRNGKey) -> Tuple[Array, Array, float]:
        """
        Single inference step - placeholder for optimization-based inference

        In the full implementation, this would contain:
        - Optimization loop with SGD/Adam
        - Convergence checking
        - Soft thresholding

        Args:
            x: Current observation
            r_p: Previous r
            r2: Previous r2
            rng: PRNG key

        Returns:
            Tuple of (r, r2, r2_loss)
        """
        # This is a simplified version - full implementation would need
        # JAX optimization with optax and convergence checking
        batch_size = x.shape[0]
        r = jnp.zeros((batch_size, self.r_dim))
        r2_loss = 0.0

        return r, r2, r2_loss


def create_train_state(model: TiDHy, rng: PRNGKey, learning_rate: float,
                       batch_size: int, input_dim: int, T: int) -> Any:
    """
    Create training state with optimizer

    Args:
        model: TiDHy model instance
        rng: PRNG key
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for initialization
        input_dim: Input dimension
        T: Sequence length

    Returns:
        Training state with parameters and optimizer
    """
    # Initialize model with dummy input
    dummy_input = jnp.ones((batch_size, T, input_dim))
    variables = model.init(rng, dummy_input, rng)

    # Create optimizer
    tx = optax.adamw(learning_rate)

    return variables, tx


def train_step(state: Any, X: Array, model: TiDHy, rng: PRNGKey) -> Tuple[Any, dict]:
    """
    Single training step

    Args:
        state: Current training state
        X: Input batch
        model: TiDHy model
        rng: PRNG key

    Returns:
        Updated state and metrics dictionary
    """
    def loss_fn(params):
        outputs = model.apply(params, X, rng)
        spatial_loss_rhat, spatial_loss_rbar, temp_loss, r2_losses, _, _ = outputs

        total_loss = spatial_loss_rhat + spatial_loss_rbar + temp_loss
        if model.use_r2_decoder:
            total_loss += r2_losses

        return total_loss, {
            'spatial_loss_rhat': spatial_loss_rhat,
            'spatial_loss_rbar': spatial_loss_rbar,
            'temp_loss': temp_loss,
            'r2_losses': r2_losses
        }

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state['params'])

    # Update parameters
    updates, opt_state = state['tx'].update(grads, state['opt_state'])
    params = optax.apply_updates(state['params'], updates)

    state['params'] = params
    state['opt_state'] = opt_state

    metrics['loss'] = loss

    return state, metrics


def normalize_params(params: dict, normalize_spatial: bool = False,
                    normalize_temporal: bool = False) -> dict:
    """
    Normalize model parameters

    Args:
        params: Model parameters
        normalize_spatial: Whether to normalize spatial decoder weights
        normalize_temporal: Whether to normalize temporal parameters

    Returns:
        Normalized parameters
    """
    params = unfreeze(params)

    if normalize_spatial and 'spatial_decoder' in params['params']:
        decoder_params = params['params']['spatial_decoder']['Dense_0']
        if 'kernel' in decoder_params:
            kernel = decoder_params['kernel']
            # Normalize along dim 0
            kernel_norm = kernel / (jnp.linalg.norm(kernel, axis=0, keepdims=True) + 1e-10)
            decoder_params['kernel'] = kernel_norm

    if normalize_temporal and 'temporal' in params['params']:
        temporal = params['params']['temporal']
        # Normalize along last dimension
        temporal_norm = temporal / (jnp.linalg.norm(temporal, axis=-1, keepdims=True) + 1e-10)
        params['params']['temporal'] = temporal_norm

    return freeze(params)
