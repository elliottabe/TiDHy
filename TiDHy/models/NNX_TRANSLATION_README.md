# TiDHy Flax NNX Implementation

This directory contains a complete translation of the TiDHy PyTorch model to Flax NNX, the new stateful API for Flax.

## Files

- **TiDHy_nnx.py**: Core model architecture using Flax NNX
- **TiDHy_nnx_training.py**: Training utilities, evaluation, and checkpointing
- **TiDHy_nnx_example.py**: Comprehensive usage examples
- **TiDHy.py**: Original PyTorch implementation

## What is Flax NNX?

Flax NNX is the new stateful API for Flax that provides a **more PyTorch-like experience** while maintaining JAX's functional benefits. Key differences from the original Flax (`flax.linen`):

### NNX vs Linen

| Feature | Flax Linen | Flax NNX |
|---------|------------|----------|
| **Style** | Functional | Stateful (PyTorch-like) |
| **Parameters** | Separated from model | Stored in model |
| **State Management** | External | Internal (via Variables) |
| **API Complexity** | More verbose | More intuitive |
| **Model Modification** | Requires `apply()` | Direct access |
| **Learning Curve** | Steeper | Gentler (if familiar with PyTorch) |

### Why NNX for TiDHy?

The TiDHy model benefits from NNX because:

1. **Stateful Inference**: The model performs optimization-based inference that naturally benefits from mutable state
2. **PyTorch Similarity**: The original implementation is in PyTorch, making NNX a more natural translation
3. **Iterative Updates**: Inference loops with gradient descent are more intuitive with stateful parameters
4. **Direct Parameter Access**: No need to pass parameters explicitly to every function call

## Installation

```bash
# Install JAX
pip install -U jax jaxlib

# For GPU support (CUDA 12)
pip install -U "jax[cuda12]"

# Install Flax with NNX support (Flax >= 0.7.0)
pip install -U flax

# Install optax for optimizers
pip install -U optax
```

## Quick Start

```python
from flax import nnx
from TiDHy_nnx import TiDHy
from TiDHy_nnx_training import train_model

# Create model
rngs = nnx.Rngs(0)
model = TiDHy(
    r_dim=20,
    r2_dim=10,
    mix_dim=5,
    input_dim=100,
    hyper_hid_dim=64,
    rngs=rngs
)

# Create data
import jax.numpy as jnp
X_train = jnp.ones((32, 50, 100))  # (batch, time, features)

# Train
trained_model, history = train_model(
    model,
    X_train,
    n_epochs=10,
    learning_rate=1e-3,
    batch_size=8
)
```

## Key Differences: PyTorch vs NNX

### 1. Model Definition

**PyTorch**:
```python
class TiDHy(nn.Module):
    def __init__(self, params, device):
        super().__init__()
        self.spatial_decoder = nn.Sequential(...)
        self.temporal = nn.Parameter(torch.randn(...))
```

**NNX**:
```python
class TiDHy(nnx.Module):
    def __init__(self, r_dim, r2_dim, ..., *, rngs: nnx.Rngs):
        self.spatial_decoder = SpatialDecoder(..., rngs=rngs)
        self.temporal = nnx.Param(
            nnx.initializers.orthogonal()(rngs.params(), shape)
        )
```

### 2. Forward Pass

**PyTorch**:
```python
model = TiDHy(params, device)
outputs = model(X)  # Direct call
```

**NNX**:
```python
model = TiDHy(..., rngs=rngs)
outputs = model(X)  # Also direct call! (like PyTorch)
```

Unlike Flax Linen, NNX doesn't require `apply()` - you call the model directly!

### 3. Training Loop

**PyTorch**:
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(n_epochs):
    loss = compute_loss(model, X)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**NNX**:
```python
optimizer_tx = optax.adamw(1e-3)
optimizer = nnx.Optimizer(model, optimizer_tx)

for epoch in range(n_epochs):
    def loss_fn(model):
        return compute_loss(model, X)

    grads = nnx.grad(loss_fn)(model)
    optimizer.update(grads)
```

### 4. Parameter Access

**PyTorch**:
```python
# Direct access
model.temporal.data = new_value

# Iteration
for param in model.parameters():
    print(param.shape)
```

**NNX**:
```python
# Direct access
model.temporal.value = new_value

# Iteration
for param in nnx.iter_graph(model, nnx.Param):
    print(param.shape)
```

### 5. State Management

**PyTorch**:
```python
# Stateful by default
model.r_state = torch.zeros(...)
model.r_state = new_value  # Mutable
```

**NNX**:
```python
# Use nnx.Variable for mutable state
self.r_state = nnx.Variable(jnp.zeros(...))
self.r_state.value = new_value  # Mutable within JAX
```

## Architecture Overview

### Spatial Decoder

Decodes latent representations to observations: `x = g(r)`

```python
class SpatialDecoder(nnx.Module):
    def __init__(self, r_dim, input_dim, ..., *, rngs: nnx.Rngs):
        self.dense = nnx.Linear(r_dim, input_dim, rngs=rngs)

    def __call__(self, r):
        return self.dense(r)
```

### Temporal Dynamics

Hypernetwork-based temporal prediction: `r_t = f(r_{t-1}, r2_t)`

```python
def temporal_prediction(self, r, r2):
    # Get mixture weights from hypernetwork
    wb = self.hypernet(r2)
    w = wb[:, :self.mix_dim]

    # Compute dynamics matrix
    V_t = jnp.einsum('bm,mij->bij', w,
                    self.temporal.value.reshape(...))

    # Apply dynamics
    r_hat = jnp.einsum('bij,bj->bi', V_t, r)
    return r_hat, V_t, w
```

### Inference

Optimization-based inference using gradient descent:

```python
def inf(self, x, r_p, r2):
    # Initialize latent
    r = jnp.zeros((batch_size, self.r_dim))

    # Create optimizers
    optimizer_r = optax.sgd(self.lr_r, momentum=0.9)
    opt_state_r = optimizer_r.init(r)

    # Optimization loop
    for i in range(self.max_iter):
        # Compute loss
        loss = spatial_loss + temporal_loss + regularization

        # Gradient step
        grad = jax.grad(loss_fn)(r)
        updates, opt_state_r = optimizer_r.update(grad, opt_state_r)
        r = optax.apply_updates(r, updates)

        # Proximal operator
        r = soft_thresholding(r, self.lmda_r)

        # Check convergence
        if converged:
            break

    return r, r2
```

## Training

### Basic Training

```python
from TiDHy_nnx_training import train_model

model, history = train_model(
    model,
    train_data,
    n_epochs=20,
    learning_rate=1e-3,
    batch_size=16,
    val_data=val_data
)
```

### Manual Training Loop

```python
from TiDHy_nnx_training import train_step

optimizer_tx = optax.adamw(1e-3)
optimizer = nnx.Optimizer(model, optimizer_tx)

for epoch in range(n_epochs):
    metrics = train_step(model, optimizer, X_batch)
    print(f"Loss: {metrics['loss']:.4f}")
```

### Custom Training with Full Control

```python
def custom_loss_fn(model, X):
    spatial_loss_rhat, spatial_loss_rbar, temp_loss, r2_losses, _, _ = model(X)

    # Custom weighting
    total_loss = (
        1.0 * spatial_loss_rhat +
        0.5 * spatial_loss_rbar +
        2.0 * temp_loss
    )

    return total_loss

# Compute gradients
grads = nnx.grad(lambda m: custom_loss_fn(m, X))(model)

# Update
optimizer.update(grads)
```

## Evaluation

### Batch Evaluation

```python
from TiDHy_nnx_training import evaluate_batch

metrics = evaluate_batch(model, X_test)
print(f"Test loss: {metrics['loss']:.4f}")
```

### Full Recording

```python
from TiDHy_nnx_training import evaluate_record

spatial_loss_rhat, spatial_loss_rbar, temp_loss, result_dict = evaluate_record(
    model, X_test, verbose=True
)

# Access recorded variables
R_hat = result_dict['R_hat']  # Inferred latents
R_bar = result_dict['R_bar']  # Predicted latents
W = result_dict['W']          # Mixture weights
Ut = result_dict['Ut']        # Dynamics matrices
```

## Saving and Loading

### Save/Load Model

```python
from TiDHy_nnx_training import save_model, load_model

# Save
save_model(model, "model.pkl")

# Load (need model structure)
new_model = TiDHy(..., rngs=rngs)  # Same architecture
loaded_model = load_model(new_model, "model.pkl")
```

### Checkpointing

```python
from TiDHy_nnx_training import checkpoint_model, load_checkpoint

# Save checkpoint (includes optimizer state)
checkpoint_model(model, optimizer, epoch=10, filepath="checkpoint.pkl")

# Load checkpoint
loaded_model, loaded_optimizer, epoch = load_checkpoint(
    model, optimizer, "checkpoint.pkl"
)

# Continue training from epoch
```

## Advanced Features

### Stateful Mode

Enable state persistence across batches:

```python
model = TiDHy(
    ...,
    stateful=True,  # Enable stateful mode
    rngs=rngs
)

# Process sequential batches
for batch in sequential_data:
    outputs = model(batch, training=True)
    # model.r_state and model.r2_state are updated automatically
```

### Normalization

Normalize parameters during training:

```python
model = TiDHy(
    ...,
    normalize_spatial=True,   # Normalize decoder weights
    normalize_temporal=True,  # Normalize dynamics
    rngs=rngs
)

# Normalization happens automatically during training
# Or manually:
model.normalize()
```

### Different Loss Types

```python
# MSE loss (default)
model = TiDHy(..., loss_type='MSE', rngs=rngs)

# Binary cross-entropy
model = TiDHy(..., loss_type='BCE', rngs=rngs)

# Poisson loss
model = TiDHy(..., loss_type='Poisson', rngs=rngs)
```

### Architectural Variants

```python
# Nonlinear decoder
model = TiDHy(..., nonlin_decoder=True, rngs=rngs)

# Low-rank temporal dynamics
model = TiDHy(..., low_rank_temp=True, rngs=rngs)

# With R2 decoder
model = TiDHy(..., use_r2_decoder=True, rngs=rngs)

# With dynamic bias
model = TiDHy(..., dyn_bias=True, rngs=rngs)
```

## NNX-Specific Features

### Graph Surgery

Split and merge model components:

```python
# Split model into definition and state
graphdef, state = nnx.split(model)

# Save/load just the state
import pickle
with open('state.pkl', 'wb') as f:
    pickle.dump(state, f)

# Recreate model
with open('state.pkl', 'rb') as f:
    state = pickle.load(f)
loaded_model = nnx.merge(graphdef, state)
```

### Parameter Filtering

Access specific parameter types:

```python
# Get all parameters
params = nnx.state(model, nnx.Param)

# Get all variables (mutable state)
variables = nnx.state(model, nnx.Variable)

# Iterate over parameters
for path, param in nnx.iter_graph(model, nnx.Param):
    print(f"{path}: {param.value.shape}")
```

### JIT Compilation

```python
# JIT compile evaluation
@nnx.jit
def fast_eval(model, X):
    return model(X, training=False)

# Use compiled function
outputs = fast_eval(model, X_test)
```

### Lifted Transforms

Apply JAX transformations to model methods:

```python
# Vectorize over batch dimension
vmapped_forward = nnx.vmap(
    lambda m, x: m(x),
    in_axes=(None, 0)  # Don't vmap over model
)

# Gradient over time
grad_temporal = nnx.grad(
    lambda m, r, r2: m.temporal_prediction(r, r2)[0].sum()
)
```

## Debugging

### Check Model Structure

```python
# Print model
print(model)

# Check parameter shapes
for path, param in nnx.iter_graph(model, nnx.Param):
    print(f"{path}: {param.value.shape}")

# Count parameters
n_params = sum(p.value.size for _, p in nnx.iter_graph(model, nnx.Param))
print(f"Total parameters: {n_params:,}")
```

### Disable JIT for Debugging

```python
import jax
jax.config.update('jax_disable_jit', True)

# Now you can use Python debugger
import pdb; pdb.set_trace()
```

## Performance Tips

1. **Use JIT compilation** for evaluation and inference loops
2. **Batch processing** for multiple sequences
3. **Pre-allocate arrays** when possible
4. **Use `jnp.einsum`** for efficient tensor operations
5. **Profile with `jax.profiler`** to find bottlenecks

## Comparison: Linen vs NNX Implementation

Both implementations are provided in this directory:

- **TiDHy_jax.py**: Functional Flax Linen implementation
- **TiDHy_nnx.py**: Stateful Flax NNX implementation

Choose NNX if:
- You're familiar with PyTorch
- Your model has complex stateful logic
- You prefer direct parameter access
- You want simpler training loops

Choose Linen if:
- You need maximum functional purity
- You're building very large models
- You need advanced parallelization (pmap, etc.)
- You prefer explicit parameter passing

## Common Issues

### Issue: `rngs` not provided

```python
# ❌ Wrong
model = TiDHy(r_dim=20, r2_dim=10, ...)

# ✅ Correct
rngs = nnx.Rngs(0)
model = TiDHy(r_dim=20, r2_dim=10, ..., rngs=rngs)
```

### Issue: Modifying parameters directly

```python
# ❌ Wrong
model.temporal = new_value

# ✅ Correct
model.temporal.value = new_value
```

### Issue: Accessing state in stateless model

```python
# ❌ Wrong (if stateful=False)
model.r_state.value = new_value  # AttributeError

# ✅ Correct
model = TiDHy(..., stateful=True, rngs=rngs)
model.r_state.value = new_value
```

## Migration from PyTorch

### Step-by-Step Guide

1. **Replace imports**:
   ```python
   # PyTorch
   import torch
   import torch.nn as nn

   # NNX
   import jax.numpy as jnp
   from flax import nnx
   ```

2. **Change model definition**:
   - Inherit from `nnx.Module` instead of `nn.Module`
   - Add `*, rngs: nnx.Rngs` to `__init__`
   - Use `nnx.Linear`, `nnx.LayerNorm`, etc.
   - Wrap parameters with `nnx.Param()`

3. **Update forward pass**:
   - Replace `torch.Tensor` operations with `jnp` equivalents
   - Use `jnp.einsum` instead of `torch.einsum`
   - No device management needed

4. **Modify training loop**:
   - Use `nnx.Optimizer` instead of PyTorch optimizers
   - Replace `.backward()` with `nnx.grad()`
   - Use `optimizer.update()` instead of `.step()`

5. **Handle state**:
   - Use `nnx.Variable` for mutable state
   - Access with `.value` property

## References

- [Flax NNX Documentation](https://flax.readthedocs.io/en/latest/nnx/index.html)
- [Flax NNX Tutorial](https://flax.readthedocs.io/en/latest/nnx_basics.html)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Optax Documentation](https://optax.readthedocs.io/)
- [NNX vs Linen Guide](https://flax.readthedocs.io/en/latest/guides/flax_fundamentals/linen_vs_nnx.html)

## Examples

See `TiDHy_nnx_example.py` for comprehensive examples including:
1. Basic usage
2. Training
3. Manual training loops
4. Evaluation with recording
5. Saving and loading
6. Checkpointing
7. Stateful models
8. Different architectures

Run examples:
```bash
python TiDHy_nnx_example.py
```
