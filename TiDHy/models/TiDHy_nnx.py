"""
TiDHy model implementation using Flax NNX.

Flax NNX is the new stateful API that provides a more PyTorch-like experience
while maintaining JAX's functional benefits.
"""

import logging
from typing import Tuple
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from tqdm.auto import tqdm


def soft_thresholding(r: jax.Array, lmda: float) -> jax.Array:
    """Non-negative proximal gradient for L1 regularization."""
    return jax.nn.relu(jnp.abs(r) - lmda) * jnp.sign(r)


def soft_max(r: jax.Array, axis: int = -1) -> jax.Array:
    """Softmax activation."""
    return jax.nn.softmax(r, axis=axis)


def heavyside(r: jax.Array, value: jax.Array) -> jax.Array:
    """Heaviside step function."""
    return jnp.heaviside(r, value)


def poisson_loss(predicted: jax.Array, observed: jax.Array) -> jax.Array:
    """Custom loss function for Poisson model."""
    return predicted - observed * jnp.log(predicted + 1e-10)


class SpatialDecoder(nnx.Module):
    """Spatial decoder: p(I | r)"""

    def __init__(
        self,
        r_dim: int,
        input_dim: int,
        hyper_hid_dim: int,
        loss_type: str = 'MSE',
        nonlin_decoder: bool = False,
        *,
        rngs: nnx.Rngs
    ):
        self.loss_type = loss_type
        self.nonlin_decoder = nonlin_decoder

        if loss_type == 'BCE':
            self.dense = nnx.Linear(
                r_dim, input_dim, use_bias=True,
                kernel_init=nnx.initializers.xavier_normal(),
                rngs=rngs
            )
        elif nonlin_decoder:
            self.dense1 = nnx.Linear(
                r_dim, hyper_hid_dim, use_bias=True,
                kernel_init=nnx.initializers.xavier_normal(),
                rngs=rngs
            )
            self.norm1 = nnx.LayerNorm(hyper_hid_dim, rngs=rngs)
            self.dense2 = nnx.Linear(hyper_hid_dim, hyper_hid_dim, rngs=rngs)
            self.dense3 = nnx.Linear(hyper_hid_dim, input_dim, rngs=rngs)
        else:
            self.dense = nnx.Linear(
                r_dim, input_dim, use_bias=True,
                kernel_init=nnx.initializers.xavier_normal(),
                rngs=rngs
            )

    def __call__(self, r: jax.Array) -> jax.Array:
        if self.loss_type == 'BCE':
            x = self.dense(r)
            return nnx.sigmoid(x)

        if self.nonlin_decoder:
            x = self.dense1(r)
            x = self.norm1(x)
            x = nnx.elu(x)
            x = self.dense2(x)
            x = self.dense3(x)
            return x
        else:
            return self.dense(r)


class HyperNetwork(nnx.Module):
    """Hypernetwork for generating mixture weights"""

    def __init__(
        self,
        r2_dim: int,
        hyper_hid_dim: int,
        mix_dim: int,
        r_dim: int,
        dyn_bias: bool,
        *,
        rngs: nnx.Rngs
    ):
        self.dyn_bias = dyn_bias
        self.mix_dim = mix_dim

        self.dense1 = nnx.Linear(
            r2_dim, hyper_hid_dim,
            kernel_init=nnx.initializers.xavier_normal(),
            rngs=rngs
        )
        self.norm1 = nnx.LayerNorm(hyper_hid_dim, rngs=rngs)
        self.dense2 = nnx.Linear(
            hyper_hid_dim, hyper_hid_dim,
            kernel_init=nnx.initializers.xavier_normal(),
            rngs=rngs
        )

        out_dim = mix_dim + r_dim if dyn_bias else mix_dim
        self.dense3 = nnx.Linear(
            hyper_hid_dim, out_dim,
            use_bias=not dyn_bias,
            kernel_init=nnx.initializers.xavier_normal(),
            rngs=rngs
        )

    def __call__(self, r2: jax.Array) -> jax.Array:
        x = self.dense1(r2)
        x = self.norm1(x)
        x = nnx.elu(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return nnx.relu(x)


class R2Decoder(nnx.Module):
    """R2 decoder network"""

    def __init__(
        self,
        r2_dim: int,
        r_dim: int,
        r2_decoder_hid_dim: int,
        *,
        rngs: nnx.Rngs
    ):
        self.dense1 = nnx.Linear(
            r2_dim + r_dim, r2_decoder_hid_dim,
            kernel_init=nnx.initializers.xavier_normal(),
            rngs=rngs
        )
        self.norm1 = nnx.LayerNorm(r2_decoder_hid_dim, rngs=rngs)
        self.dense2 = nnx.Linear(
            r2_decoder_hid_dim, r2_decoder_hid_dim,
            kernel_init=nnx.initializers.xavier_normal(),
            rngs=rngs
        )
        self.dense3 = nnx.Linear(
            r2_decoder_hid_dim, r2_dim,
            kernel_init=nnx.initializers.xavier_normal(),
            rngs=rngs
        )

    def __call__(self, r_r2_concat: jax.Array) -> jax.Array:
        x = self.dense1(r_r2_concat)
        x = self.norm1(x)
        x = nnx.elu(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


class TiDHy(nnx.Module):
    """
    TiDHy model using Flax NNX.

    This implementation is more similar to PyTorch, with stateful parameters
    and a familiar API.
    """

    def __init__(
        self,
        # Model dimensions
        r_dim: int,
        r2_dim: int,
        mix_dim: int,
        input_dim: int,
        hyper_hid_dim: int,
        # Architecture options
        loss_type: str = 'MSE',
        nonlin_decoder: bool = False,
        low_rank_temp: bool = False,
        dyn_bias: bool = False,
        use_r2_decoder: bool = False,
        r2_decoder_hid_dim: int = 64,
        # Learning rates (for inference)
        lr_r: float = 0.1,
        lr_r2: float = 0.1,
        lr_weights: float = 0.025,
        lr_weights_inf: float = 0.025,
        # Regularization
        temp_weight: float = 1.0,
        lmda_r: float = 0.0,
        lmda_r2: float = 0.0,
        weight_decay: float = 0.0,
        L1_alpha: float = 0.0,
        L1_inf_w: float = 0.0,
        L1_inf_r2: float = 0.0,
        L1_inf_r: float = 0.0,
        L1_alpha_inf: float = 0.0,
        L1_alpha_r2: float = 0.0,
        grad_alpha: float = 1.5,
        grad_alpha_inf: float = 1.5,
        clip_grad: float = 1.0,
        grad_norm_inf: bool = False,
        # Training params
        max_iter: int = 100,
        tol: float = 1e-4,
        normalize_spatial: bool = False,
        normalize_temporal: bool = False,
        stateful: bool = False,
        batch_converge: bool = False,
        spat_weight: float = 1.0,
        learning_rate_gamma: float = 0.5,
        cos_eta: float = 0.001,
        # Display
        show_progress: bool = True,
        show_inf_progress: bool = False,
        *,
        rngs: nnx.Rngs
    ):
        """Initialize TiDHy model with NNX."""

        # Store hyperparameters
        self.r_dim = r_dim
        self.r2_dim = r2_dim
        self.mix_dim = mix_dim
        self.input_dim = input_dim
        self.hyper_hid_dim = hyper_hid_dim
        self.loss_type = loss_type
        self.nonlin_decoder = nonlin_decoder
        self.low_rank_temp = low_rank_temp
        self.dyn_bias = dyn_bias
        self.use_r2_decoder = use_r2_decoder
        self.r2_decoder_hid_dim = r2_decoder_hid_dim

        # Learning rates
        self.lr_r = lr_r
        self.lr_r2 = lr_r2
        self.lr_weights = lr_weights
        self.lr_weights_inf = lr_weights_inf

        # Regularization
        self.temp_weight = temp_weight
        self.lmda_r = lmda_r
        self.lmda_r2 = lmda_r2
        self.weight_decay = weight_decay
        self.L1_alpha = L1_alpha
        self.L1_inf_w = L1_inf_w
        self.L1_inf_r2 = L1_inf_r2
        self.L1_inf_r = L1_inf_r
        self.L1_alpha_inf = L1_alpha_inf
        self.L1_alpha_r2 = L1_alpha_r2
        self.grad_alpha = grad_alpha
        self.grad_alpha_inf = grad_alpha_inf
        self.clip_grad = clip_grad
        self.grad_norm_inf = grad_norm_inf

        # Training params
        self.max_iter = max_iter
        self.tol = tol
        self.normalize_spatial = normalize_spatial
        self.normalize_temporal = normalize_temporal
        self.stateful = stateful
        self.batch_converge = batch_converge
        self.spat_weight = spat_weight
        self.learning_rate_gamma = learning_rate_gamma
        self.cos_eta = cos_eta

        # Display
        self.show_progress = show_progress
        self.show_inf_progress = show_inf_progress

        # Initialize spatial decoder
        self.spatial_decoder = SpatialDecoder(
            r_dim=r_dim,
            input_dim=input_dim,
            hyper_hid_dim=hyper_hid_dim,
            loss_type=loss_type,
            nonlin_decoder=nonlin_decoder,
            rngs=rngs
        )

        # Initialize temporal parameters
        if low_rank_temp:
            self.temporal = nnx.Param(
                nnx.initializers.orthogonal()(
                    rngs.params(), (mix_dim, r_dim, 2)
                )
            )
        else:
            self.temporal = nnx.Param(
                nnx.initializers.orthogonal()(
                    rngs.params(), (mix_dim, r_dim * r_dim)
                )
            )

        if dyn_bias:
            self.temporal_bias = nnx.Param(
                nnx.initializers.xavier_normal()(
                    rngs.params(), (1, r_dim)
                )
            )
        else:
            self.temporal_bias = None

        # Initialize hypernetwork
        self.hypernet = HyperNetwork(
            r2_dim=r2_dim,
            hyper_hid_dim=hyper_hid_dim,
            mix_dim=mix_dim,
            r_dim=r_dim,
            dyn_bias=dyn_bias,
            rngs=rngs
        )

        # Initialize R2 decoder (optional)
        if use_r2_decoder:
            self.r2_decoder = R2Decoder(
                r2_dim=r2_dim,
                r_dim=r_dim,
                r2_decoder_hid_dim=r2_decoder_hid_dim,
                rngs=rngs
            )
        else:
            self.r2_decoder = None

        # State variables (if stateful)
        if stateful:
            self.r_state = nnx.Variable(jnp.zeros((1, r_dim)))
            self.r2_state = nnx.Variable(jnp.zeros((1, r2_dim)))
        else:
            self.r_state = None
            self.r2_state = None

        # Loss weights for grad norm (if used)
        self.loss_weights = None
        self.loss_weights_inf = None

    def temporal_prediction(
        self,
        r: jax.Array,
        r2: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Temporal prediction: p(r_t | r_(t-1), r2)

        Args:
            r: Previous latent state (batch_size, r_dim)
            r2: Higher-order latent (batch_size, r2_dim)

        Returns:
            Tuple of (r_hat, V_t, w)
        """
        batch_size = r.shape[0]

        # Get mixture weights and bias from hypernetwork
        wb = self.hypernet(r2)
        w = wb[:, :self.mix_dim]
        b = wb[:, self.mix_dim:] if self.dyn_bias else None

        # Compute temporal dynamics matrix
        if self.low_rank_temp:
            # Low-rank temporal dynamics
            Vk = jnp.einsum('mij,mkj->mik', self.temporal.value, self.temporal.value)
            Vk = Vk.reshape(self.mix_dim, -1)
            V_t = jnp.einsum('bm,mij->bij', w,
                           Vk.reshape(self.mix_dim, self.r_dim, self.r_dim))
        else:
            # Full-rank temporal dynamics
            V_t = jnp.einsum('bm,mij->bij', w,
                           self.temporal.value.reshape(self.mix_dim, self.r_dim, self.r_dim))

        # Apply temporal dynamics
        r_hat = jnp.einsum('bij,bj->bi', V_t, r)

        # Add bias if applicable
        if self.dyn_bias and self.temporal_bias is not None:
            r_hat = r_hat + b * self.temporal_bias.value

        return r_hat, V_t, w

    def get_spatial_loss_fn(self) -> callable:
        """Get spatial loss function based on loss_type"""
        if self.loss_type == 'Poisson':
            return poisson_loss
        elif self.loss_type == 'BCE':
            return lambda pred, obs: -obs * jnp.log(pred + 1e-10) - (1 - obs) * jnp.log(1 - pred + 1e-10)
        else:  # MSE
            return lambda pred, obs: (pred - obs) ** 2

    def init_code(self, batch_size: int) -> Tuple[jax.Array, jax.Array]:
        """Initialize latent codes"""
        r = jnp.zeros((batch_size, self.r_dim))
        r2 = jnp.zeros((batch_size, self.r2_dim))
        return r, r2

    def __call__(
        self,
        X: jax.Array,
        training: bool = True
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Forward pass through the model.

        Args:
            X: Input data of shape (batch_size, T, input_dim)
            training: Whether in training mode

        Returns:
            Tuple of (spatial_loss_rhat, spatial_loss_rbar, temp_loss,
                     r2_losses, r_first, r2_final)
        """
        batch_size, T, _ = X.shape

        # Initialize codes
        r, r2 = self.init_code(batch_size)
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
        t_range = tqdm(range(1, T), leave=False, dynamic_ncols=True) if self.show_progress else range(1, T)

        for t in t_range:
            r_p = r.copy()
            r2_p = r2.copy()

            # Inference step
            r, r2, r2_loss = self.inf(X[:, t], r_p, r2)

            # Predictions
            x_hat = self.spatial_decoder(r)
            r_bar, V_t, w = self.temporal_prediction(r_p, r2)
            x_bar = self.spatial_decoder(r_bar)

            # Compute losses
            if self.use_r2_decoder and self.r2_decoder is not None:
                r2_hat = self.r2_decoder(jnp.concatenate([r_p, r2_p], axis=-1))
                r2_losses += jnp.sum((r2 - r2_hat) ** 2, axis=-1).mean()

            spatial_loss_rhat += spat_loss_fn(x_hat, X[:, t]).reshape(batch_size, -1).sum(1).mean(0)
            spatial_loss_rbar += spat_loss_fn(x_bar, X[:, t]).reshape(batch_size, -1).sum(1).mean(0)
            temp_loss += jnp.sum((r - r_bar) ** 2, axis=-1).mean()

        # Update state if stateful and training
        if training and self.stateful:
            self.r_state.value = r.copy()
            self.r2_state.value = r2.copy()

        return (spatial_loss_rhat, spatial_loss_rbar,
                self.temp_weight * temp_loss, r2_losses, r_first, r2)

    def inf(
        self,
        x: jax.Array,
        r_p: jax.Array,
        r2: jax.Array
    ) -> Tuple[jax.Array, jax.Array, float]:
        """
        Inference step: p(r_t, r2_t | x_t, r_t-1, r2_t-1)

        Args:
            x: Current observation
            r_p: Previous r
            r2: Previous r2

        Returns:
            Tuple of (r, r2, r2_loss)
        """
        batch_size = x.shape[0]

        # Initialize r
        r = jnp.zeros((batch_size, self.r_dim))

        # Create optimizers
        optimizer_r = optax.sgd(self.lr_r, momentum=0.9, nesterov=True)
        optimizer_r2 = optax.sgd(self.lr_r2, momentum=0.9, nesterov=True)

        opt_state_r = optimizer_r.init(r)
        opt_state_r2 = optimizer_r2.init(r2)

        # Get graphdef for applying model components
        spat_loss_fn = self.get_spatial_loss_fn()

        def loss_fn(r_val, r2_val):
            # Spatial reconstruction
            x_bar = self.spatial_decoder(r_val)
            spatial_loss = spat_loss_fn(x_bar, x).reshape(batch_size, -1).sum(1).mean(0)

            # Temporal prediction
            r_bar, V_t, w = self.temporal_prediction(r_p, r2_val)
            temporal_loss = jnp.sum((r_val - r_bar) ** 2, axis=-1).mean()

            # R2 decoder loss (if applicable)
            r2_loss_val = 0.0
            if self.use_r2_decoder and self.r2_decoder is not None:
                r2_bar = self.r2_decoder(jnp.concatenate([r_val, r2_val], axis=-1))
                r2_loss_val = jnp.sum((r2_val - r2_bar) ** 2, axis=-1).mean()

            # Sparsity penalties
            inf_w_sparsity = self.L1_inf_w * jnp.linalg.norm(w.ravel(), ord=1)
            inf_r2_sparsity = self.L1_inf_r2 * jnp.linalg.norm(r2_val.ravel(), ord=1)
            inf_r_sparsity = self.L1_inf_r * jnp.linalg.norm(r_val.ravel(), ord=1)

            total_loss = (spatial_loss +
                         self.temp_weight * temporal_loss +
                         inf_w_sparsity +
                         inf_r2_sparsity +
                         inf_r_sparsity)

            if self.use_r2_decoder:
                total_loss += r2_loss_val

            return total_loss, (spatial_loss, temporal_loss, r2_loss_val)

        # Optimization loop
        converged = False
        i = 0
        r2_loss_out = 0.0

        while not converged and i < self.max_iter:
            old_r = r.copy()
            old_r2 = r2.copy()

            # Compute gradients
            (loss_val, (spatial_loss, temporal_loss, r2_loss_val)), (grad_r, grad_r2) = jax.value_and_grad(
                lambda r_v, r2_v: loss_fn(r_v, r2_v),
                argnums=(0, 1),
                has_aux=True
            )(r, r2)

            # Update r
            updates_r, opt_state_r = optimizer_r.update(grad_r, opt_state_r)
            r = optax.apply_updates(r, updates_r)

            # Update r2
            updates_r2, opt_state_r2 = optimizer_r2.update(grad_r2, opt_state_r2)
            r2 = optax.apply_updates(r2, updates_r2)

            # Apply soft thresholding
            r = soft_thresholding(r, self.lmda_r)
            r2 = soft_thresholding(r2, self.lmda_r2)

            # Check convergence
            if self.batch_converge:
                r2_converge = jnp.linalg.norm(r2 - old_r2) / (jnp.linalg.norm(old_r2) + 1e-16)
                r_converge = jnp.linalg.norm(r - old_r) / (jnp.linalg.norm(old_r) + 1e-16)
            else:
                r2_converge = jnp.linalg.norm(r2 - old_r2, axis=-1) / (
                    jnp.linalg.norm(old_r2, axis=-1) + 1e-16
                )
                r_converge = jnp.linalg.norm(r - old_r, axis=-1) / (
                    jnp.linalg.norm(old_r, axis=-1) + 1e-16
                )

            converged = jnp.all(r_converge < self.tol) and jnp.all(r2_converge < self.tol)

            if self.show_inf_progress:
                logging.info(
                    f'inf_it:{i}, r2_conv:{jnp.sum(r2_converge > self.tol):.03f}, '
                    f'r_conv:{jnp.sum(r_converge > self.tol):.03f}, '
                    f'spat_loss:{spatial_loss:.02f}, temp_loss:{temporal_loss:.02f}'
                )

            r2_loss_out = r2_loss_val
            i += 1

        # Convergence warning
        if i >= self.max_iter:
            logging.info(
                f'r/r2 did not converge: r2_conv:{jnp.sum(r2_converge > self.tol)}, '
                f'r_conv:{jnp.sum(r_converge > self.tol)}'
            )

        return r, r2, r2_loss_out

    def inf_first_step(self, x: jax.Array) -> jax.Array:
        """
        First step inference: p(r_1 | x_1) (no temporal prior)

        Args:
            x: First observation

        Returns:
            Inferred r at time 1
        """
        batch_size = x.shape[0]
        r = jnp.zeros((batch_size, self.r_dim))

        optimizer = optax.adamw(self.lr_r)
        opt_state = optimizer.init(r)

        spat_loss_fn = self.get_spatial_loss_fn()

        def loss_fn(r_val):
            x_bar = self.spatial_decoder(r_val)
            return spat_loss_fn(x_bar, x).reshape(batch_size, -1).sum(1).mean(0)

        converged = False
        i = 0

        while not converged and i < self.max_iter:
            old_r = r.copy()

            # Compute gradient and update
            loss_val, grad_r = jax.value_and_grad(loss_fn)(r)
            updates, opt_state = optimizer.update(grad_r, opt_state)
            r = optax.apply_updates(r, updates)

            # Check convergence
            r_converge = jnp.linalg.norm(r - old_r, axis=-1) / (
                jnp.linalg.norm(old_r, axis=-1) + 1e-16
            )
            converged = jnp.all(r_converge < self.tol)

            i += 1

        if i >= self.max_iter:
            logging.info("first step r did not converge")

        return r

    def normalize(self):
        """Normalize model parameters"""
        if self.normalize_spatial:
            # Normalize spatial decoder weights
            if hasattr(self.spatial_decoder, 'dense'):
                kernel = self.spatial_decoder.dense.kernel.value
                kernel_norm = kernel / (jnp.linalg.norm(kernel, axis=0, keepdims=True) + 1e-10)
                self.spatial_decoder.dense.kernel.value = kernel_norm

        if self.normalize_temporal:
            # Normalize temporal parameters
            temporal = self.temporal.value
            temporal_norm = temporal / (jnp.linalg.norm(temporal, axis=-1, keepdims=True) + 1e-10)
            self.temporal.value = temporal_norm
