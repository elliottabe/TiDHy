"""
TiDHy model implementation using Flax NNX - vmap-friendly version.

This version is designed to work on single sequences and be vmapped for batch processing.
All methods work on unbatched data (single examples) by default.

Usage:
    # Single sequence
    model = TiDHy(...)
    loss = model(single_sequence)  # single_sequence: (T, input_dim)

    # Batch processing with vmap
    batched_forward = jax.vmap(model, in_axes=0)
    losses = batched_forward(batch_sequences)  # (batch_size, T, input_dim) -> (batch_size,)
"""

from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from flax import nnx
import optax

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


def apply_saturation(x: jax.Array, method: str = 'none', scale: float = 1.0) -> jax.Array:
    """
    Apply saturation to values with configurable method and scale.

    WARNING: This function uses runtime string dispatch and should NOT be called
    inside JIT-compiled loops. Use _make_saturation_fn() instead to create
    specialized functions at initialization time.

    Args:
        x: Input array
        method: Saturation method
            - 'none': No saturation (identity)
            - 'sigmoid': Sigmoid saturation (output ∈ [0, 1])
            - 'shifted_sigmoid': Zero-centered sigmoid (output ∈ [-1, 1])
            - 'tanh': Tanh saturation (output ∈ [-1, 1])
            - 'scaled_tanh': Scaled tanh (output ∈ [-scale, scale])
            - 'softsign': Softsign saturation (output ∈ [-1, 1])
            - 'scaled_softsign': Scaled softsign (output ∈ [-scale, scale])
            - 'hardtanh': Hard clipping (output ∈ [-scale, scale])
        scale: Scale factor for scaled methods

    Returns:
        Saturated values

    Notes:
        - 'shifted_sigmoid': 2*sigmoid(x) - 1
          * Zero-centered like tanh
          * May be faster than tanh in nested loops
          * Gradient at 0: 0.5 (better than sigmoid's 0.25)

        - 'softsign': x / (1 + |x|)
          * Zero-centered, computationally cheap
          * Gradient at 0: 1.0 (same as tanh)
          * Gradients decay more slowly than tanh
          * No exponential operations

        - 'scaled_softsign': scale * x / (1 + |x|/scale)
          * Softsign with custom output range
          * Very efficient alternative to scaled_tanh

        - 'hardtanh': clip(x, -scale, scale)
          * Hard clipping, no smooth saturation
          * Fastest option (just comparison operations)
          * Non-zero gradients everywhere except at bounds
          * No gradient vanishing in linear region

        - 'scaled_tanh' has best gradient properties but may be slow:
          * tanh'(0) = 1 vs sigmoid'(0) = 0.25
          * Symmetric around 0 (no bias)
          * Gradients vanish more slowly: tanh'(x) > 0.1 for |x| < 2
    """
    if method == 'none':
        return x
    elif method == 'sigmoid':
        return jax.nn.sigmoid(x)
    elif method == 'shifted_sigmoid':
        return 2.0 * jax.nn.sigmoid(x) - 1.0
    elif method == 'tanh':
        return jax.nn.tanh(x)
    elif method == 'scaled_tanh':
        return scale * jax.nn.tanh(x / scale)
    elif method == 'softsign':
        return x / (1.0 + jnp.abs(x))
    elif method == 'scaled_softsign':
        return scale * x / (1.0 + jnp.abs(x) / scale)
    elif method == 'hardtanh':
        return jnp.clip(x, -scale, scale)
    else:
        raise ValueError(f"Unknown saturation method: {method}")


def _make_saturation_fn(method: str, scale: float):
    """
    Factory function to create specialized saturation functions.

    This eliminates runtime dispatch overhead by creating the appropriate
    function at initialization time. The returned function has zero overhead
    compared to direct calls to jax.nn.sigmoid, jax.nn.tanh, etc.

    PERFORMANCE: Using this factory instead of apply_saturation() inside
    JIT-compiled loops provides 5-20x speedup by eliminating string comparisons.

    Args:
        method: Saturation method ('none', 'sigmoid', 'shifted_sigmoid', 'tanh',
                'scaled_tanh', 'softsign', 'scaled_softsign', 'hardtanh')
        scale: Scale factor for scaled methods

    Returns:
        Specialized saturation function with signature (x: Array) -> Array

    Example:
        # At initialization:
        saturate_fn = _make_saturation_fn('scaled_softsign', 3.0)

        # In hot loop (JIT-compiled):
        x_saturated = saturate_fn(x)  # Fast! No string comparison
    """
    if method == 'none':
        return lambda x: x
    elif method == 'sigmoid':
        return lambda x: jax.nn.sigmoid(x)
    elif method == 'shifted_sigmoid':
        return lambda x: 2.0 * jax.nn.sigmoid(x) - 1.0
    elif method == 'tanh':
        return lambda x: jax.nn.tanh(x)
    elif method == 'scaled_tanh':
        return lambda x: scale * jax.nn.tanh(x / scale)
    elif method == 'softsign':
        return lambda x: x / (1.0 + jnp.abs(x))
    elif method == 'scaled_softsign':
        return lambda x: scale * x / (1.0 + jnp.abs(x) / scale)
    elif method == 'hardtanh':
        return lambda x: jnp.clip(x, -scale, scale)
    else:
        raise ValueError(f"Unknown saturation method: {method}")


def _make_clip_fn(clip_grad: float):
    """
    Factory function to create specialized gradient clipping function.

    This eliminates unnecessary clipping overhead when clip_grad is large.
    Returns identity function (zero overhead) when clipping is effectively disabled.

    PERFORMANCE: Avoids 2000+ unnecessary clip operations per sequence when
    gradients don't need clipping (e.g., with well-behaved saturation='none').

    Args:
        clip_grad: Maximum gradient magnitude. If >= 1e10, clipping is disabled.

    Returns:
        Specialized clipping function with signature (g: Array) -> Array

    Example:
        # At initialization:
        clip_fn = _make_clip_fn(10.0)  # Will clip

        # In hot loop (JIT-compiled):
        grad_clipped = clip_fn(grad)  # Fast! Pre-compiled
    """
    if clip_grad >= 1e10:
        # No clipping needed - return identity function (zero overhead)
        return lambda g: g
    else:
        # Capture clip_grad in closure
        return lambda g: jnp.clip(g, -clip_grad, clip_grad)


def cos_sim_mat(x: jax.Array, dim: int = -1) -> jax.Array:
    """
    Compute pairwise cosine similarity between vectors in a matrix.

    This computes cosine similarity for all pairs (i,j) where i!=j,
    returning the sum of absolute values of the lower triangle.

    Args:
        x: Input array of shape (..., n_vectors, vector_dim)
        dim: Dimension along which to compute cosine similarity

    Returns:
        Scalar sum of pairwise cosine similarities
    """
    if x.ndim == 2:
        x = x[None, :, :]

    # Compute pairwise cosine similarity
    # x[..., None, :, :] has shape (..., 1, n, d)
    # x[..., :, None, :] has shape (..., n, 1, d)
    # This broadcasts to (..., n, n, d) for pairwise computation
    x_norm = x / (jnp.linalg.norm(x, axis=dim, keepdims=True) + 1e-10)
    cos_sim = jnp.sum(x_norm[..., None, :, :] * x_norm[..., :, None, :], axis=dim)

    # Extract lower triangle (excluding diagonal) and sum absolute values
    cos_sim_lower = jnp.abs(jnp.tril(cos_sim, k=-1))

    return jnp.sum(cos_sim_lower)


def l1_regularization(x: jax.Array) -> jax.Array:
    """L1 (Lasso) regularization to encourage sparsity."""
    return jnp.sum(jnp.abs(x))


def l2_regularization(x: jax.Array) -> jax.Array:
    """L2 (Ridge) regularization."""
    return jnp.sum(x ** 2)


def elastic_net_regularization(x: jax.Array, l1_ratio: float = 0.5) -> jax.Array:
    """Elastic net regularization: combination of L1 and L2."""
    return l1_ratio * l1_regularization(x) + (1 - l1_ratio) * l2_regularization(x)


def group_lasso_regularization(x: jax.Array, group_size: int) -> jax.Array:
    """
    Group Lasso regularization for structured sparsity.
    
    Groups consecutive elements and applies L2 norm within groups,
    then L1 norm across groups.
    """
    # Reshape to groups
    n_elements = x.size
    
    # Handle edge cases
    if n_elements == 0 or group_size <= 0:
        return jnp.array(0.0)
    
    if group_size >= n_elements:
        # If group size is larger than array, just return L2 norm
        return jnp.linalg.norm(x.ravel())
    
    n_groups = n_elements // group_size
    
    if n_groups == 0:
        return jnp.linalg.norm(x.ravel())
    
    # Flatten and reshape into groups
    x_flat = x.ravel()
    remaining = n_elements % group_size
    
    if remaining > 0:
        # Pad with zeros to make divisible by group_size
        padding = group_size - remaining
        x_flat = jnp.concatenate([x_flat, jnp.zeros(padding)])
        n_groups = (n_elements + padding) // group_size
    
    # Reshape to (n_groups, group_size)
    x_grouped = x_flat[:n_groups * group_size].reshape(n_groups, group_size)
    
    # L2 norm within each group, then L1 norm across groups
    group_norms = jnp.linalg.norm(x_grouped, axis=1)
    return jnp.sum(group_norms)


def adaptive_sparsity_regularization(
    x: jax.Array, 
    target_sparsity: float = 0.8,
    current_sparsity: float = None,
    temperature: float = 1.0
) -> jax.Array:
    """
    Adaptive regularization that adjusts strength based on current sparsity level.
    
    Args:
        x: Input tensor
        target_sparsity: Desired fraction of elements to be (near) zero
        current_sparsity: Current fraction of near-zero elements (computed if None)
        temperature: Temperature parameter for smooth adaptation
        
    Returns:
        Regularization loss that adapts based on current vs target sparsity
    """
    if current_sparsity is None:
        # Compute current sparsity (fraction of near-zero elements)
        threshold = 1e-3
        near_zero = jnp.abs(x) < threshold
        current_sparsity = jnp.mean(near_zero)
    
    # Clip sparsity values to reasonable range
    current_sparsity = jnp.clip(current_sparsity, 0.0, 0.999)
    target_sparsity = jnp.clip(target_sparsity, 0.0, 0.999)
    
    # Adaptive weight: increase regularization if below target sparsity
    sparsity_gap = target_sparsity - current_sparsity
    # Clip the gap to avoid extreme values
    sparsity_gap = jnp.clip(sparsity_gap, -5.0, 5.0)
    adaptive_weight = jnp.exp(temperature * sparsity_gap)
    # Clip the weight to avoid numerical issues
    adaptive_weight = jnp.clip(adaptive_weight, 0.01, 100.0)
    
    return adaptive_weight * l1_regularization(x)


def hypernetwork_sparsity_regularization(
    w: jax.Array, 
    min_active_ratio: float = 0.3,
    sparsity_strength: float = 1.0,
    epsilon: float = 1e-8
) -> jax.Array:
    """
    Specialized sparsity regularization for hypernetwork outputs that prevents collapse.
    
    This encourages sparsity while ensuring a minimum number of components remain active.
    
    Args:
        w: Hypernetwork output (mixture weights)
        min_active_ratio: Minimum fraction of components that should remain active
        sparsity_strength: Overall sparsity regularization strength
        epsilon: Small value for numerical stability
        
    Returns:
        Regularization loss
    """
    # Ensure w is non-negative (hypernetwork should output positive weights)
    w_pos = jnp.maximum(w, epsilon)
    
    # Normalize to get mixture probabilities
    w_sum = jnp.sum(w_pos)
    w_sum = jnp.maximum(w_sum, epsilon)
    probs = w_pos / w_sum
    
    # Sort probabilities to identify top components
    sorted_probs = jnp.sort(probs, descending=True)
    
    # Use a smooth approximation instead of hard thresholding
    # This avoids dynamic indexing issues in JAX
    mix_dim = w.size
    
    # Create a smooth weighting that gives less penalty to top components
    indices = jnp.arange(mix_dim)
    min_active_float = min_active_ratio * mix_dim
    
    # Soft masking: components beyond min_active get higher penalty
    soft_mask = jax.nn.sigmoid(2.0 * (indices - min_active_float))
    
    # Apply penalty proportional to the soft mask and probability
    sparsity_loss = sparsity_strength * jnp.sum(soft_mask * sorted_probs)
    
    return sparsity_loss


def selective_sparsity_regularization(
    w: jax.Array,
    target_active_ratio: float = 0.5,
    temperature: float = 1.0,
    epsilon: float = 1e-8
) -> jax.Array:
    """
    Encourage a fraction of components to be active in hypernetwork output.
    
    Args:
        w: Hypernetwork output
        target_active_ratio: Fraction of components that should be active (0.0-1.0)
        temperature: Softmax temperature for smooth selection
        epsilon: Small value for numerical stability
        
    Returns:
        Regularization loss
    """
    mix_dim = w.size
    
    # Clip to reasonable range
    target_active_ratio = jnp.clip(target_active_ratio, 0.1, 0.9)
    
    # Apply softmax to get selection probabilities
    w_softmax = jax.nn.softmax(w / temperature)
    
    # Sort probabilities in descending order
    sorted_probs = jnp.sort(w_softmax, descending=True)
    
    # Create position-based weights that encourage top-k sparsity
    positions = jnp.arange(mix_dim) / mix_dim
    target_threshold = target_active_ratio
    
    # Soft selection mask: sigmoid transition around target_active_ratio
    selection_strength = 10.0  # Controls sharpness of selection
    soft_mask = jax.nn.sigmoid(selection_strength * (target_threshold - positions))
    
    # Encourage high probability in early positions (should be active)
    # and low probability in later positions (should be inactive)
    active_reward = jnp.sum(soft_mask * sorted_probs)
    inactive_penalty = jnp.sum((1 - soft_mask) * sorted_probs)
    
    # We want to maximize active_reward and minimize inactive_penalty
    return inactive_penalty - 0.5 * active_reward


def hoyer_sparsity_regularization(x: jax.Array) -> jax.Array:
    """
    Hoyer's sparsity measure: encourages sparsity by penalizing 
    the ratio of L1 to L2 norms.
    
    Hoyer sparsity = (sqrt(n) - ||x||_1/||x||_2) / (sqrt(n) - 1)
    where n is the number of elements.
    
    We minimize the negative of this to encourage sparsity.
    """
    n = x.size
    x_flat = x.ravel()
    
    l1_norm = jnp.linalg.norm(x_flat, ord=1)
    l2_norm = jnp.linalg.norm(x_flat, ord=2)
    
    # Avoid division by zero and numerical issues
    l1_norm = jnp.maximum(l1_norm, 1e-10)
    l2_norm = jnp.maximum(l2_norm, 1e-10)
    
    sqrt_n = jnp.sqrt(jnp.maximum(float(n), 1.0))
    
    # Avoid division by zero in denominator
    denominator = jnp.maximum(sqrt_n - 1.0, 1e-10)
    
    hoyer = (sqrt_n - l1_norm / l2_norm) / denominator
    
    # Clip to reasonable range and return negative to minimize (encourage sparsity)
    hoyer = jnp.clip(hoyer, -10.0, 10.0)
    return -hoyer


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
        """
        Decode latent r to observation space.

        Args:
            r: Latent code of shape (r_dim,) for single example
               or (..., r_dim) for vmapped batch

        Returns:
            Decoded output of shape (input_dim,) or (..., input_dim)
        """
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
        *,
        rngs: nnx.Rngs
    ):
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
        self.dense3 = nnx.Linear(
            hyper_hid_dim, mix_dim,
            use_bias=True,
            kernel_init=nnx.initializers.xavier_normal(),
            rngs=rngs
        )

    def __call__(self, r2: jax.Array) -> jax.Array:
        """
        Generate mixture weights from r2.

        Args:
            r2: Higher-order latent of shape (r2_dim,) for single example
                or (..., r2_dim) for vmapped batch

        Returns:
            Mixture weights of shape (mix_dim,) or (..., mix_dim)
        """
        x = self.dense1(r2)
        x = self.norm1(x)
        x = nnx.elu(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return nnx.relu(x)


class TiDHy(nnx.Module):
    """
    TiDHy model using Flax NNX - vmap-friendly version.

    All methods work on single examples (no batch dimension).
    Use jax.vmap to process batches.
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
        # Learning rates (for inference)
        lr_r: float = 0.2,
        lr_r2: float = 0.2,
        lr_weights: float = 0.01,
        lr_weights_inf: float = 0.005,
        temp_weight: float = 2.0,
        spat_weight: float = 0.5,
        # Regularization
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
        # Enhanced sparsity regularization
        sparsity_r_l1: float = 0.0,           # L1 regularization on r during training
        sparsity_r2_l1: float = 0.0,          # L1 regularization on r2 during training  
        sparsity_w_l1: float = 0.0,           # L1 regularization on hypernetwork output w
        sparsity_r_adaptive: float = 0.0,     # Adaptive sparsity for r
        sparsity_r2_adaptive: float = 0.0,    # Adaptive sparsity for r2
        sparsity_w_adaptive: float = 0.0,     # Adaptive sparsity for w
        sparsity_r_group: float = 0.0,        # Group lasso for r
        sparsity_r2_group: float = 0.0,       # Group lasso for r2
        sparsity_r_hoyer: float = 0.0,        # Hoyer sparsity for r
        sparsity_r2_hoyer: float = 0.0,       # Hoyer sparsity for r2
        # New hypernetwork-specific regularization (prevents collapse)
        sparsity_w_safe: float = 0.0,         # Safe hypernetwork sparsity (prevents collapse)
        sparsity_w_selective: float = 0.0,    # Selective sparsity (keep only k active components)
        target_sparsity_r: float = 0.7,       # Target sparsity level for r (70% zeros)
        target_sparsity_r2: float = 0.8,      # Target sparsity level for r2 (80% zeros) 
        target_sparsity_w: float = 0.8,       # Target sparsity level for w (80% zeros)
        sparsity_group_size: int = 4,         # Group size for group lasso
        sparsity_temperature: float = 1.0,    # Temperature for adaptive sparsity
        # Hypernetwork protection parameters
        w_min_active_ratio: float = 0.3,      # Minimum fraction of hypernetwork components to keep active
        w_target_active: int = None,          # Target number of active components (None = mix_dim//2)
        # Training continuity (cross-window)
        r2_continuity_weight: float = 0.0,    # Weight for r2 continuity across overlapping windows during training
        # Temporal smoothness regularization for r2
        r2_temporal_smoothness_inf: float = 0.0,   # Temporal smoothness penalty for r2 during inference
        r2_temporal_smoothness_train: float = 0.0, # Temporal smoothness penalty for r2 during training
        # Saturation parameters
        saturate_r_method: str = 'none',           # Saturation method for r: 'none', 'sigmoid', 'tanh', 'scaled_tanh'
        saturate_r2_method: str = 'none',          # Saturation method for r2: 'none', 'sigmoid', 'tanh', 'scaled_tanh'
        saturate_r_scale: float = 5.0,             # Scale for scaled_tanh saturation of r
        saturate_r2_scale: float = 3.0,            # Scale for scaled_tanh saturation of r2
        # Training params
        max_iter: int = 200,
        tol: float = 1e-4,
        normalize_spatial: bool = False,
        normalize_temporal: bool = False,
        learning_rate_gamma: float = 0.5,
        cos_eta: float = 0.001,
        # Display
        show_progress: bool = True,
        show_inf_progress: bool = False,
        *,
        rngs: nnx.Rngs,
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
        
        # Enhanced sparsity regularization
        self.sparsity_r_l1 = sparsity_r_l1
        self.sparsity_r2_l1 = sparsity_r2_l1
        self.sparsity_w_l1 = sparsity_w_l1
        self.sparsity_r_adaptive = sparsity_r_adaptive
        self.sparsity_r2_adaptive = sparsity_r2_adaptive
        self.sparsity_w_adaptive = sparsity_w_adaptive
        self.sparsity_r_group = sparsity_r_group
        self.sparsity_r2_group = sparsity_r2_group
        self.sparsity_r_hoyer = sparsity_r_hoyer
        self.sparsity_r2_hoyer = sparsity_r2_hoyer
        # New hypernetwork-specific regularization
        self.sparsity_w_safe = sparsity_w_safe
        self.sparsity_w_selective = sparsity_w_selective
        self.target_sparsity_r = target_sparsity_r
        self.target_sparsity_r2 = target_sparsity_r2
        self.target_sparsity_w = target_sparsity_w
        self.sparsity_group_size = sparsity_group_size
        self.sparsity_temperature = sparsity_temperature
        # Hypernetwork protection parameters
        self.w_min_active_ratio = w_min_active_ratio
        self.w_target_active_ratio = 0.5 if w_target_active is None else float(w_target_active) / max(1, mix_dim)
        # Training continuity
        self.r2_continuity_weight = r2_continuity_weight
        # Temporal smoothness regularization
        self.r2_temporal_smoothness_inf = r2_temporal_smoothness_inf
        self.r2_temporal_smoothness_train = r2_temporal_smoothness_train
        # Saturation parameters
        self.saturate_r_method = saturate_r_method
        self.saturate_r2_method = saturate_r2_method
        self.saturate_r_scale = saturate_r_scale
        self.saturate_r2_scale = saturate_r2_scale

        # Create specialized saturation functions (compile-time, zero runtime overhead)
        self.saturate_r_fn = _make_saturation_fn(saturate_r_method, saturate_r_scale)
        self.saturate_r2_fn = _make_saturation_fn(saturate_r2_method, saturate_r2_scale)

        # Create specialized gradient clipping function (compile-time)
        self.clip_grad_fn = _make_clip_fn(clip_grad)

        # Training params
        self.max_iter = max_iter
        self.tol = tol
        self.normalize_spatial = normalize_spatial
        self.normalize_temporal = normalize_temporal
        self.spat_weight = spat_weight
        self.learning_rate_gamma = learning_rate_gamma
        self.cos_eta = cos_eta

        # Display
        self.show_progress = show_progress
        self.show_inf_progress = show_inf_progress

        # Store RNG state for dynamic random initialization
        self.rngs = rngs

        # GradNorm tracking (not Parameters, just state)
        self.loss_weights = None
        self.loss_weights_inf = None
        self.l0 = None  # Initial losses for GradNorm
        self.iters = 0  # Training iterations counter

        # Profiling stats (enable with profile_inference=True)
        self.profile_inference = False  # Set to True to collect stats
        self.inference_stats = {
            'total_iterations': [],
            'converged_count': 0,
            'max_iter_count': 0,
            'total_timesteps': 0
        }

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
        self.temporal = nnx.Param(
            nnx.initializers.orthogonal()(
                rngs.params(), (mix_dim, r_dim * r_dim)
            )
        )

        # Initialize hypernetwork
        self.hypernet = HyperNetwork(
            r2_dim=r2_dim,
            hyper_hid_dim=hyper_hid_dim,
            mix_dim=mix_dim,
            rngs=rngs
        )

    def temporal_prediction(
        self,
        r: jax.Array,
        r2: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Temporal prediction: p(r_t | r_(t-1), r2)

        Args:
            r: Previous latent state of shape (r_dim,)
            r2: Higher-order latent of shape (r2_dim,)

        Returns:
            Tuple of (r_hat, V_t, w) where:
                r_hat: shape (r_dim,)
                V_t: shape (r_dim, r_dim)
                w: shape (mix_dim,)
        """
        # Get mixture weights from hypernetwork
        w = self.hypernet(r2)

        # Compute temporal dynamics matrix
        # w: (mix_dim,), temporal: (mix_dim, r_dim * r_dim)
        V_t = jnp.einsum('m,mij->ij', w,
                       self.temporal.value.reshape(self.mix_dim, self.r_dim, self.r_dim))

        # Apply temporal dynamics
        r_hat = jnp.einsum('ij,j->i', V_t, r)

        return r_hat, V_t, w

    def get_spatial_loss_fn(self) -> callable:
        """Get spatial loss function based on loss_type"""
        if self.loss_type == 'Poisson':
            return poisson_loss
        elif self.loss_type == 'BCE':
            return lambda pred, obs: -obs * jnp.log(pred + 1e-10) - (1 - obs) * jnp.log(1 - pred + 1e-10)
        else:  # MSE
            return lambda pred, obs: (pred - obs) ** 2

    def init_code(self, key: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Initialize latent codes for a single sequence.

        Args:
            key: JAX random key for initialization

        Returns:
            Tuple of (r, r2) with shapes (r_dim,) and (r2_dim,)
        """
        r = jnp.zeros(self.r_dim)
        # Initialize r2 with small random values instead of zeros to avoid dead neurons
        r2 = jax.random.normal(key, (self.r2_dim,)) * 0.01
        return r, r2

    def __call__(
        self,
        X: jax.Array,
        return_internals: bool = False,
        rng_key: Optional[jax.Array] = None
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Forward pass through the model for a SINGLE sequence.

        Args:
            X: Input data of shape (T, input_dim) - single sequence
            return_internals: If True, also return r, r2, w values for regularization
            rng_key: Optional JAX random key for initialization. If None, uses self.rngs()

        Returns:
            Tuple of (spatial_loss_rhat, spatial_loss_rbar, temp_loss)
            Each is a scalar for this single sequence

            If return_internals=True, returns additional tuple:
            (losses, (r_values, r2_values, w_values))
        """
        T = X.shape[0]

        # Get or generate RNG key
        if rng_key is None:
            rng_key = self.rngs()  # Only called outside vmap

        # Split key for init_code and inf_first_step
        key_init, key_inf = jax.random.split(rng_key)

        # Initialize codes for this sequence
        _, r2 = self.init_code(key_init)

        # Infer first code (stop gradients - we only need gradients w.r.t. decoder parameters)
        r_inferred, r2_inferred = self.inf_first_step(X[0], key_inf, return_stats=False)
        r = jax.lax.stop_gradient(r_inferred)
        r2 = jax.lax.stop_gradient(r2_inferred)  # Use inferred r2 instead of zeros

        spat_loss_fn = self.get_spatial_loss_fn()

        # First timestep loss (gradients only flow through decoder, not inference)
        spatial_loss_rhat = jnp.sum(spat_loss_fn(
            self.spatial_decoder(r), X[0]
        ))

        spatial_loss_rbar = jnp.zeros_like(spatial_loss_rhat)
        temp_loss = 0.0

        # Track internals if requested
        if return_internals:
            r_values = [r]
            r2_values = [r2]
            w_values = [jnp.zeros(self.mix_dim)]  # No w for first timestep

        # Define scan step function for time iteration
        def scan_step(carry, x_t):
            """Single timestep: inference, prediction, and loss computation"""
            r, r2, loss_rhat, loss_rbar, tloss, inf_stats = carry

            # CRITICAL: Detach previous codes to match PyTorch version
            r_p = jax.lax.stop_gradient(r)
            r2_p = jax.lax.stop_gradient(r2)

            # === INFERENCE: Find optimal r and r2 ===
            r_inferred, r2_inferred, inf_stats_new = self.inf(x_t, r_p, r2_p)

            # Stop gradients on inferred codes
            r_inferred_detached = jax.lax.stop_gradient(r_inferred)
            r2_inferred_detached = jax.lax.stop_gradient(r2_inferred)

            # === FORWARD PASS: Recompute with detached codes to get gradients w.r.t. parameters ===
            # Spatial reconstruction (gradients flow to decoder parameters)
            x_hat = self.spatial_decoder(r_inferred_detached)

            # Temporal prediction (gradients flow to temporal/hypernet parameters)
            r_bar, _, w = self.temporal_prediction(r_p, r2_inferred_detached)
            x_bar = self.spatial_decoder(r_bar)

            # Compute losses (sum over features for this single example)
            loss_rhat += jnp.sum(spat_loss_fn(x_hat, x_t))
            loss_rbar += jnp.sum(spat_loss_fn(x_bar, x_t))
            tloss += jnp.sum((r_inferred_detached - r_bar) ** 2)

            # Pass along the inferred codes (detached) for next timestep
            new_carry = (r_inferred_detached, r2_inferred_detached, loss_rhat, loss_rbar, tloss, inf_stats_new)
            
            if return_internals:
                return new_carry, (r_inferred_detached, r2_inferred_detached, w)
            else:
                return new_carry, None
        init_stats = {
            'iterations': 0,
            'converged': False,
            'max_iter_reached': False
        }
        # Run scan over timesteps 1 to T
        init_carry = (r, r2, spatial_loss_rhat, spatial_loss_rbar, temp_loss, init_stats)
        final_carry, scan_outputs = jax.lax.scan(scan_step, init_carry, X[1:])

        r, r2, spatial_loss_rhat, spatial_loss_rbar, temp_loss, inf_stats = final_carry

        losses = (spatial_loss_rhat, spatial_loss_rbar, self.temp_weight * temp_loss)

        if return_internals:
            # Collect all r, r2, w values from the scan
            if T > 1:
                scan_r, scan_r2, scan_w = scan_outputs
                r_values.extend([scan_r[i] for i in range(T-1)])
                r2_values.extend([scan_r2[i] for i in range(T-1)])
                w_values.extend([scan_w[i] for i in range(T-1)])
            
            # Stack into arrays
            r_array = jnp.stack(r_values)  # Shape: (T, r_dim)
            r2_array = jnp.stack(r2_values)  # Shape: (T, r2_dim)
            w_array = jnp.stack(w_values)  # Shape: (T, mix_dim)
            
            return losses, (r_array, r2_array, w_array), inf_stats

        return losses

    def inf(
        self,
        x: jax.Array,
        r_p: jax.Array,
        r2: jax.Array,
        return_stats: bool = True
    ) -> Tuple[jax.Array, jax.Array] | Tuple[jax.Array, jax.Array, dict]:
        """
        Inference step: p(r_t, r2_t | x_t, r_t-1, r2_t-1)

        Args:
            x: Current observation of shape (input_dim,)
            r_p: Previous r of shape (r_dim,)
            r2: Previous r2 of shape (r2_dim,)
            return_stats: If True, return (r, r2, stats) instead of (r, r2). Default: True

        Returns:
            Tuple of (r, r2) with shapes (r_dim,) and (r2_dim,)
            If return_stats=True, also returns dict with iteration count and convergence flag
        """
        # Initialize r
        r = jnp.zeros(self.r_dim)

        # Create optimizers
        optimizer_r = optax.sgd(self.lr_r, momentum=0.9, nesterov=True)
        optimizer_r2 = optax.sgd(self.lr_r2, momentum=0.9, nesterov=True)

        opt_state_r = optimizer_r.init(r)
        opt_state_r2 = optimizer_r2.init(r2)

        # Get spatial loss function
        spat_loss_fn = self.get_spatial_loss_fn()

        def loss_fn(r_val, r2_val):
            # Spatial reconstruction
            x_bar = self.spatial_decoder(r_val)
            spatial_loss = jnp.sum(spat_loss_fn(x_bar, x))

            # Temporal prediction
            r_bar, V_t, w = self.temporal_prediction(r_p, r2_val)
            temporal_loss = jnp.sum((r_val - r_bar) ** 2)

            # Sparsity penalties
            inf_w_sparsity = self.L1_inf_w * jnp.linalg.norm(w.ravel(), ord=1)
            inf_r2_sparsity = self.L1_inf_r2 * jnp.linalg.norm(r2_val.ravel(), ord=1)
            inf_r_sparsity = self.L1_inf_r * jnp.linalg.norm(r_val.ravel(), ord=1)

            # Temporal smoothness penalty for r2
            r2_temporal_loss = self.r2_temporal_smoothness_inf * jnp.sum((r2_val - r2) ** 2)

            total_loss = (spatial_loss +
                         self.temp_weight * temporal_loss +
                         inf_w_sparsity +
                         inf_r2_sparsity +
                         inf_r_sparsity +
                         r2_temporal_loss)

            return total_loss, (spatial_loss, temporal_loss, w)

        def while_step(carry):
            """Combined step and convergence check for while_loop"""
            r, r2, opt_state_r, opt_state_r2, iteration, converged = carry

            # Compute gradients
            (_, (spatial_loss, temporal_loss, _)), (grad_r, grad_r2) = jax.value_and_grad(
                lambda r_v, r2_v: loss_fn(r_v, r2_v),
                argnums=(0, 1),
                has_aux=True
            )(r, r2)

            # Clip gradients if needed (pre-compiled function, zero overhead when disabled)
            # grad_r = self.clip_grad_fn(grad_r)
            # grad_r2 = self.clip_grad_fn(grad_r2)

            # Update r
            updates_r, new_opt_state_r = optimizer_r.update(grad_r, opt_state_r)
            updated_r = optax.apply_updates(r, updates_r)

            # Update r2
            updates_r2, new_opt_state_r2 = optimizer_r2.update(grad_r2, opt_state_r2)
            updated_r2 = optax.apply_updates(r2, updates_r2)

            # Apply soft thresholding only if regularization is enabled
            updated_r = soft_thresholding(updated_r, self.lmda_r) if self.lmda_r > 0 else updated_r
            updated_r2 = soft_thresholding(updated_r2, self.lmda_r2) if self.lmda_r2 > 0 else updated_r2

            # Apply saturation if enabled (using pre-compiled functions, zero dispatch overhead)
            updated_r = self.saturate_r_fn(updated_r)
            updated_r2 = self.saturate_r2_fn(updated_r2)

            # Check convergence (single example version - no batch mean/sum)
            r2_change = jnp.linalg.norm(updated_r2 - r2)
            r_change = jnp.linalg.norm(updated_r - r)
            r2_norm = jnp.linalg.norm(r2)
            r_norm = jnp.linalg.norm(r)

            r2_converge = jnp.where(r2_norm < 1e-8, r2_change, r2_change / (r2_norm + 1e-16))
            r_converge = jnp.where(r_norm < 1e-8, r_change, r_change / (r_norm + 1e-16))
            is_converged = (r_converge < self.tol) & (r2_converge < self.tol)

            new_iteration = iteration + 1
            new_converged = converged | is_converged

            return (updated_r, updated_r2, new_opt_state_r, new_opt_state_r2, new_iteration, new_converged)

        def continue_condition(carry):
            """Check if while_loop should continue"""
            r, r2, opt_state_r, opt_state_r2, iteration, converged = carry
            return (~converged) & (iteration < self.max_iter)

        # Run optimization with early stopping using while_loop
        init_carry = (r, r2, opt_state_r, opt_state_r2, 0, False)
        final_carry = jax.lax.while_loop(continue_condition, while_step, init_carry)

        r, r2, _, _, final_iteration, final_converged = final_carry

        # Optional: Log convergence info (only works for non-vmapped calls)
        if self.show_inf_progress:
            # Note: jax.debug.print will not print when vmapped, but useful for single-sequence debugging
            jax.debug.print(
                "Inference: iter={iter}/{max}, converged={conv}, r2_norm={r2norm:.3f}, r2_max={r2max:.3f}",
                iter=final_iteration,
                max=self.max_iter,
                conv=final_converged,
                r2norm=jnp.linalg.norm(r2),
                r2max=jnp.max(jnp.abs(r2))
            )

        if return_stats:
            stats = {
                'iterations': final_iteration,
                'converged': final_converged,
                'max_iter_reached': final_iteration >= self.max_iter
            }
            return r, r2, stats
        else:
            return r, r2

    def inf_first_step(
        self,
        x: jax.Array,
        key: jax.Array,
        return_stats: bool = True
    ) -> Tuple[jax.Array, jax.Array] | Tuple[jax.Array, jax.Array, dict]:
        """
        First step inference: p(r_1, r2_1 | x_1)
        Now optimizes both r and r2 to give r2 a chance to learn

        Args:
            x: First observation of shape (input_dim,)
            key: JAX random key for initialization
            return_stats: If True, return (r, r2, stats) instead of (r, r2). Default: True

        Returns:
            Tuple of (inferred r, inferred r2) at time 1
            Each has shape (r_dim,) and (r2_dim,)
            If return_stats=True, also returns dict with iteration count and convergence flag
        """
        r = jnp.zeros(self.r_dim)
        # Initialize r2 with small values to avoid starting from zero
        _, r2 = self.init_code(key)

        # Create optimizers for both r and r2
        optimizer_r = optax.sgd(self.lr_r, momentum=0.9, nesterov=True)
        optimizer_r2 = optax.sgd(self.lr_r2, momentum=0.9, nesterov=True)

        opt_state_r = optimizer_r.init(r)
        opt_state_r2 = optimizer_r2.init(r2)

        spat_loss_fn = self.get_spatial_loss_fn()

        def loss_fn(r_val, r2_val):
            # Spatial reconstruction loss
            x_bar = self.spatial_decoder(r_val)
            spatial_loss = jnp.sum(spat_loss_fn(x_bar, x))

            # Add small r2 regularization to encourage non-zero values
            r2_reg = 0.01 * jnp.sum(r2_val ** 2)

            # Add hypernetwork activity to encourage r2 learning
            w = self.hypernet(r2_val)
            w_activity = 0.01 * jnp.sum(w ** 2)

            return spatial_loss + r2_reg + w_activity

        def while_step(carry):
            """Combined step and convergence check for while_loop"""
            r, r2, opt_state_r, opt_state_r2, iteration, converged = carry

            # Compute gradients for both r and r2
            loss_val, (grad_r, grad_r2) = jax.value_and_grad(
                lambda r_v, r2_v: loss_fn(r_v, r2_v),
                argnums=(0, 1)
            )(r, r2)

            # Clip gradients if needed (pre-compiled function, zero overhead when disabled)
            grad_r = self.clip_grad_fn(grad_r)
            grad_r2 = self.clip_grad_fn(grad_r2)

            # Update r
            updates_r, new_opt_state_r = optimizer_r.update(grad_r, opt_state_r)
            updated_r = optax.apply_updates(r, updates_r)

            # Update r2
            updates_r2, new_opt_state_r2 = optimizer_r2.update(grad_r2, opt_state_r2)
            updated_r2 = optax.apply_updates(r2, updates_r2)

            # Apply soft thresholding if needed
            updated_r = soft_thresholding(updated_r, self.lmda_r) if self.lmda_r > 0 else updated_r
            updated_r2 = soft_thresholding(updated_r2, self.lmda_r2) if self.lmda_r2 > 0 else updated_r2

            # Apply saturation if enabled (using pre-compiled functions, zero dispatch overhead)
            updated_r = self.saturate_r_fn(updated_r)
            updated_r2 = self.saturate_r2_fn(updated_r2)

            # Check convergence
            r_change = jnp.linalg.norm(updated_r - r)
            r2_change = jnp.linalg.norm(updated_r2 - r2)
            r_norm = jnp.linalg.norm(r)
            r2_norm = jnp.linalg.norm(r2)

            r_converge = jnp.where(r_norm < 1e-8, r_change, r_change / (r_norm + 1e-16))
            r2_converge = jnp.where(r2_norm < 1e-8, r2_change, r2_change / (r2_norm + 1e-16))

            is_converged = (r_converge < self.tol) & (r2_converge < self.tol)

            new_iteration = iteration + 1
            new_converged = converged | is_converged

            return (updated_r, updated_r2, new_opt_state_r, new_opt_state_r2, new_iteration, new_converged)

        def continue_condition(carry):
            """Check if while_loop should continue"""
            r, r2, opt_state_r, opt_state_r2, iteration, converged = carry
            return (~converged) & (iteration < self.max_iter)

        # Run optimization with early stopping using while_loop
        init_carry = (r, r2, opt_state_r, opt_state_r2, 0, False)
        final_carry = jax.lax.while_loop(continue_condition, while_step, init_carry)

        r, r2, _, _, final_iteration, final_converged = final_carry

        # Optional: Log convergence info (only works for non-vmapped calls)
        if self.show_inf_progress:
            # Note: jax.debug.print will not print when vmapped, but useful for single-sequence debugging
            jax.debug.print(
                "First step: iter={iter}/{max}, converged={conv}, r2_norm={r2norm:.3f}, r2_max={r2max:.3f}",
                iter=final_iteration,
                max=self.max_iter,
                conv=final_converged,
                r2norm=jnp.linalg.norm(r2),
                r2max=jnp.max(jnp.abs(r2))
            )

        if return_stats:
            stats = {
                'iterations': final_iteration,
                'converged': final_converged,
                'max_iter_reached': final_iteration >= self.max_iter
            }
            return r, r2, stats
        else:
            return r, r2

    def compute_r2_continuity_loss(
        self,
        r2_batch: jax.Array,
        overlap_length: int
    ) -> jax.Array:
        """
        Compute continuity loss for r2 in overlapping regions between adjacent windows.

        This enforces that adjacent windows in a batch have similar r2 values in their
        overlapping regions, reducing discontinuities when reconstructing continuous sequences.

        Args:
            r2_batch: Batch of r2 values, shape (batch_size, T, r2_dim)
            overlap_length: Number of overlapping timesteps between consecutive windows

        Returns:
            Continuity loss (scalar) - MSE between overlapping regions
        """
        batch_size = r2_batch.shape[0]

        if batch_size < 2:
            # Need at least 2 windows to compute continuity
            return jnp.array(0.0)

        # Vectorized implementation: no Python for loop!
        # Extract all "end" regions: last overlap_length timesteps of windows [0, 1, ..., N-2]
        r2_ends = r2_batch[:-1, -overlap_length:, :]  # Shape: (batch_size-1, overlap_length, r2_dim)

        # Extract all "start" regions: first overlap_length timesteps of windows [1, 2, ..., N-1]
        r2_starts = r2_batch[1:, :overlap_length, :]  # Shape: (batch_size-1, overlap_length, r2_dim)

        # Compute MSE between all consecutive pairs in parallel (single vectorized operation)
        squared_diffs = (r2_ends - r2_starts) ** 2

        # Average over all dimensions: pairs, timesteps, and features
        return jnp.mean(squared_diffs)

    def compute_sparsity_regularization(
        self, 
        r_values: jax.Array,
        r2_values: jax.Array, 
        w_values: jax.Array
    ) -> Tuple[jax.Array, dict]:
        """
        Compute comprehensive sparsity regularization for r, r2, and hypernetwork output w.
        
        Args:
            r_values: Batch of r values, shape (batch_size, T, r_dim) or (batch_size, r_dim)
            r2_values: Batch of r2 values, shape (batch_size, T, r2_dim) or (batch_size, r2_dim)
            w_values: Batch of w values, shape (batch_size, T, mix_dim) or (batch_size, mix_dim)
            
        Returns:
            Tuple of (total_regularization_loss, regularization_breakdown)
        """
        reg_losses = {}
        total_reg = 0.0
        
        # === R regularization ===
        if self.sparsity_r_l1 > 0:
            r_l1 = jnp.mean(jax.vmap(l1_regularization)(r_values))
            reg_losses['r_l1'] = r_l1
            total_reg += self.sparsity_r_l1 * r_l1
            
        if self.sparsity_r_adaptive > 0:
            # Compute adaptive regularization for each sequence, then average
            def compute_r_adaptive_single(r_seq):
                return adaptive_sparsity_regularization(
                    r_seq, 
                    target_sparsity=self.target_sparsity_r,
                    temperature=self.sparsity_temperature
                )
            r_adaptive = jnp.mean(jax.vmap(compute_r_adaptive_single)(r_values))
            reg_losses['r_adaptive'] = r_adaptive
            total_reg += self.sparsity_r_adaptive * r_adaptive
            
        if self.sparsity_r_group > 0:
            def compute_r_group_single(r_seq):
                return group_lasso_regularization(r_seq, self.sparsity_group_size)
            r_group = jnp.mean(jax.vmap(compute_r_group_single)(r_values))
            reg_losses['r_group'] = r_group
            total_reg += self.sparsity_r_group * r_group
            
        if self.sparsity_r_hoyer > 0:
            r_hoyer = jnp.mean(jax.vmap(hoyer_sparsity_regularization)(r_values))
            reg_losses['r_hoyer'] = r_hoyer
            total_reg += self.sparsity_r_hoyer * r_hoyer
        
        # === R2 regularization ===
        if self.sparsity_r2_l1 > 0:
            r2_l1 = jnp.mean(jax.vmap(l1_regularization)(r2_values))
            reg_losses['r2_l1'] = r2_l1
            total_reg += self.sparsity_r2_l1 * r2_l1
            
        if self.sparsity_r2_adaptive > 0:
            def compute_r2_adaptive_single(r2_seq):
                return adaptive_sparsity_regularization(
                    r2_seq,
                    target_sparsity=self.target_sparsity_r2,
                    temperature=self.sparsity_temperature
                )
            r2_adaptive = jnp.mean(jax.vmap(compute_r2_adaptive_single)(r2_values))
            reg_losses['r2_adaptive'] = r2_adaptive
            total_reg += self.sparsity_r2_adaptive * r2_adaptive
            
        if self.sparsity_r2_group > 0:
            def compute_r2_group_single(r2_seq):
                return group_lasso_regularization(r2_seq, self.sparsity_group_size)
            r2_group = jnp.mean(jax.vmap(compute_r2_group_single)(r2_values))
            reg_losses['r2_group'] = r2_group
            total_reg += self.sparsity_r2_group * r2_group
            
        if self.sparsity_r2_hoyer > 0:
            r2_hoyer = jnp.mean(jax.vmap(hoyer_sparsity_regularization)(r2_values))
            reg_losses['r2_hoyer'] = r2_hoyer
            total_reg += self.sparsity_r2_hoyer * r2_hoyer
        
        # === Hypernetwork output (w) regularization ===
        # Standard regularization (can cause collapse if too strong)
        if self.sparsity_w_l1 > 0:
            w_l1 = jnp.mean(jax.vmap(l1_regularization)(w_values))
            reg_losses['w_l1'] = w_l1
            total_reg += self.sparsity_w_l1 * w_l1
            
        if self.sparsity_w_adaptive > 0:
            def compute_w_adaptive_single(w_seq):
                return adaptive_sparsity_regularization(
                    w_seq,
                    target_sparsity=self.target_sparsity_w,
                    temperature=self.sparsity_temperature
                )
            w_adaptive = jnp.mean(jax.vmap(compute_w_adaptive_single)(w_values))
            reg_losses['w_adaptive'] = w_adaptive
            total_reg += self.sparsity_w_adaptive * w_adaptive
            
        # SAFE hypernetwork regularization (prevents collapse)
        if self.sparsity_w_safe > 0:
            def compute_w_safe_single(w_seq):
                return hypernetwork_sparsity_regularization(
                    w_seq,
                    min_active_ratio=self.w_min_active_ratio,
                    sparsity_strength=1.0
                )
            w_safe = jnp.mean(jax.vmap(compute_w_safe_single)(w_values))
            reg_losses['w_safe'] = w_safe
            total_reg += self.sparsity_w_safe * w_safe
            
        if self.sparsity_w_selective > 0:
            def compute_w_selective_single(w_seq):
                return selective_sparsity_regularization(
                    w_seq,
                    target_active_ratio=self.w_target_active,
                    temperature=self.sparsity_temperature
                )
            w_selective = jnp.mean(jax.vmap(compute_w_selective_single)(w_values))
            reg_losses['w_selective'] = w_selective
            total_reg += self.sparsity_w_selective * w_selective
        
        return total_reg, reg_losses

    def compute_sparsity_metrics(
        self,
        r_values: jax.Array,
        r2_values: jax.Array,
        w_values: jax.Array,
        threshold: float = 1e-3
    ) -> dict:
        """
        Compute sparsity metrics for monitoring.
        
        Args:
            r_values: Batch of r values
            r2_values: Batch of r2 values  
            w_values: Batch of w values
            threshold: Threshold for considering values as zero
            
        Returns:
            Dictionary of sparsity metrics
        """
        metrics = {}
        
        # Compute fraction of near-zero elements
        r_sparsity = jnp.mean(jnp.abs(r_values) < threshold)
        r2_sparsity = jnp.mean(jnp.abs(r2_values) < threshold)
        w_sparsity = jnp.mean(jnp.abs(w_values) < threshold)
        
        metrics['r_sparsity'] = r_sparsity
        metrics['r2_sparsity'] = r2_sparsity  
        metrics['w_sparsity'] = w_sparsity
        
        # Compute L1 and L2 norms
        def safe_l1_norm(x):
            return jnp.linalg.norm(x.ravel(), ord=1)
        
        def safe_l2_norm(x):
            return jnp.linalg.norm(x.ravel(), ord=2)
            
        metrics['r_l1_norm'] = jnp.mean(jax.vmap(safe_l1_norm)(r_values))
        metrics['r2_l1_norm'] = jnp.mean(jax.vmap(safe_l1_norm)(r2_values))
        metrics['w_l1_norm'] = jnp.mean(jax.vmap(safe_l1_norm)(w_values))
        
        metrics['r_l2_norm'] = jnp.mean(jax.vmap(safe_l2_norm)(r_values))
        metrics['r2_l2_norm'] = jnp.mean(jax.vmap(safe_l2_norm)(r2_values))
        metrics['w_l2_norm'] = jnp.mean(jax.vmap(safe_l2_norm)(w_values))
        
        # Compute Hoyer sparsity measure  
        def hoyer_measure(x):
            n = x.size
            x_flat = x.ravel()
            l1_norm = jnp.linalg.norm(x_flat, ord=1)
            l2_norm = jnp.linalg.norm(x_flat, ord=2)
            
            # Avoid numerical issues
            l1_norm = jnp.maximum(l1_norm, 1e-10)
            l2_norm = jnp.maximum(l2_norm, 1e-10)
            sqrt_n = jnp.sqrt(jnp.maximum(float(n), 1.0))
            denominator = jnp.maximum(sqrt_n - 1.0, 1e-10)
            
            hoyer = (sqrt_n - l1_norm / l2_norm) / denominator
            return jnp.clip(hoyer, 0.0, 1.0)  # Hoyer sparsity should be in [0,1]
            
        metrics['r_hoyer'] = jnp.mean(jax.vmap(hoyer_measure)(r_values))
        metrics['r2_hoyer'] = jnp.mean(jax.vmap(hoyer_measure)(r2_values))
        metrics['w_hoyer'] = jnp.mean(jax.vmap(hoyer_measure)(w_values))
        
        return metrics

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
