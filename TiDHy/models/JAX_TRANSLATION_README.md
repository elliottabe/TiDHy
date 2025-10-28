# TiDHy JAX/Flax Translation

This directory contains a complete translation of the TiDHy PyTorch model to JAX/Flax.

## Files

- **TiDHy_jax.py**: Core model architecture in Flax
- **TiDHy_jax_inference.py**: Inference optimization loops using optax
- **TiDHy_jax_example.py**: Example usage and training scripts
- **TiDHy.py**: Original PyTorch implementation

## Key Differences: PyTorch vs JAX/Flax

### 1. Functional Programming Paradigm

**PyTorch** (Object-Oriented):
```python
class TiDHy(nn.Module):
    def __init__(self, params, device):
        super().__init__()
        self.spatial_decoder = nn.Sequential(...)
        self.temporal = nn.Parameter(torch.randn(...))
```

**JAX/Flax** (Functional):
```python
class TiDHy(nn.Module):
    r_dim: int  # Dataclass-style attributes

    def setup(self):
        self.spatial_decoder = SpatialDecoder(...)
        self.temporal = self.param('temporal', init_fn, shape)
```

### 2. Parameter Management

**PyTorch**:
- Parameters stored as instance attributes
- Automatically tracked by `nn.Module`
- Mutable state

**JAX/Flax**:
- Parameters separated from model definition
- Immutable pytree structures
- Explicit parameter passing to `apply()`

### 3. Gradient Computation

**PyTorch**:
```python
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

**JAX/Flax**:
```python
grad_fn = jax.grad(loss_fn)
grads = grad_fn(params)
updates, opt_state = optimizer.update(grads, opt_state)
params = optax.apply_updates(params, updates)
```

### 4. Optimization Loops

**PyTorch** uses imperative loops:
```python
for i in range(max_iter):
    loss.backward()
    optim.step()
    if converged:
        break
```

**JAX** uses `jax.lax.scan` for efficiency:
```python
def step(carry, _):
    # compute gradient and update
    return new_carry, outputs

final_carry, all_outputs = jax.lax.scan(step, init_carry, None, length=max_iter)
```

### 5. Device Management

**PyTorch**:
```python
model.to(device)
data = data.to(device)
```

**JAX**:
- Automatically uses available accelerators
- No explicit device placement needed (usually)
- Can use `jax.device_put()` when needed

## Architecture Translation

### Spatial Decoder

The spatial decoder translates observations to latent representations.

**PyTorch**:
```python
self.spatial_decoder = nn.Sequential(
    nn.Linear(r_dim, input_dim, bias=True),
    nn.Sigmoid()  # if loss_type == 'BCE'
)
```

**Flax**:
```python
class SpatialDecoder(nn.Module):
    @nn.compact
    def __call__(self, r):
        x = nn.Dense(self.input_dim)(r)
        if self.loss_type == 'BCE':
            return nn.sigmoid(x)
        return x
```

### Temporal Dynamics

Implements hypernetwork-based temporal prediction: `r_t = f(r_{t-1}, r2_t)`

**PyTorch**:
```python
V_t = torch.matmul(w, self.temporal).reshape(batch_size, r_dim, r_dim)
r_hat = torch.bmm(V_t, r.unsqueeze(2)).squeeze(dim=-1)
```

**JAX**:
```python
V_t = jnp.einsum('bm,mij->bij', w,
                self.temporal.reshape(mix_dim, r_dim, r_dim))
r_hat = jnp.einsum('bij,bj->bi', V_t, r)
```

### Hypernetwork

Generates mixture weights from higher-order latents.

Both implementations use similar MLP architectures with LayerNorm and ELU activations.

## Initialization

### PyTorch
```python
model = TiDHy(params, device)
```

### JAX/Flax
```python
model = TiDHy(r_dim=20, r2_dim=10, ...)  # Define structure

# Initialize with dummy input
variables = model.init(rng, dummy_input, rng)
params = variables['params']

# Use in forward pass
outputs = model.apply(params, X, rng)
```

## Training

### PyTorch Training Loop
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    loss, spatial_loss, temp_loss, _, _, _ = model(X)
    total_loss = loss + spatial_loss + temp_loss

    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### JAX Training Loop
```python
tx = optax.adamw(learning_rate=1e-3)
opt_state = tx.init(params)

for epoch in range(num_epochs):
    def loss_fn(params):
        outputs = model.apply(params, X, rng)
        # compute total loss
        return total_loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
```

## Inference

The inference step optimizes latent codes given observations.

### PyTorch
```python
def inf(self, x, r_p, r2):
    r = torch.zeros((batch_size, r_dim), requires_grad=True)
    optim_r = torch.optim.SGD([r], lr=self.lr_r)

    for i in range(max_iter):
        loss = compute_loss(r, r2, x, r_p)
        loss.backward()
        optim_r.step()
        optim_r.zero_grad()

        # soft thresholding
        r.data = soft_thresholding(r, lmda)

    return r.detach()
```

### JAX
```python
def inf_step(model, params, x, r_p, r2, ...):
    r = jnp.zeros((batch_size, r_dim))
    optimizer = optax.sgd(lr_r, momentum=0.9, nesterov=True)
    opt_state = optimizer.init(r)

    def step(carry, _):
        r, opt_state = carry
        grads = jax.grad(loss_fn)(r)
        updates, opt_state = optimizer.update(grads, opt_state)
        r = optax.apply_updates(r, updates)
        r = soft_thresholding(r, lmda)
        return (r, opt_state), metrics

    (r_final, _), all_metrics = jax.lax.scan(
        step, (r, opt_state), None, length=max_iter
    )
    return r_final
```

## Key JAX Concepts

### 1. Pure Functions
All JAX transformations require pure functions (no side effects).

### 2. Immutability
Arrays and parameters are immutable. Use `.at[].set()` for updates:
```python
# PyTorch
R_hat[:, t] = r

# JAX
R_hat = R_hat.at[:, t].set(r)
```

### 3. PRNG State
Random number generation requires explicit key management:
```python
rng = jax.random.PRNGKey(0)
rng, subkey = jax.random.split(rng)
data = jax.random.normal(subkey, shape)
```

### 4. JIT Compilation
Functions can be compiled for performance:
```python
@jax.jit
def train_step(params, X):
    # computation
    return updated_params, loss
```

### 5. Automatic Vectorization
Use `vmap` for batch operations:
```python
batch_fn = jax.vmap(single_item_fn)
results = batch_fn(batch_data)
```

## Performance Considerations

### JAX Advantages
1. **XLA Compilation**: JIT compilation can significantly speed up computation
2. **Automatic Parallelization**: Easy to scale to multiple devices
3. **Functional Purity**: Enables powerful transformations (grad, vmap, pmap)
4. **Memory Efficiency**: Immutable arrays with copy-on-write

### JAX Challenges
1. **Learning Curve**: Functional paradigm differs from PyTorch
2. **Debugging**: Compiled code harder to debug (use `jax.disable_jit()`)
3. **Static Shapes**: Array shapes must be known at compile time
4. **Control Flow**: Need special handling for conditionals/loops

## Migration Checklist

If migrating existing PyTorch code to JAX:

- [ ] Replace `torch.nn.Module` with `flax.linen.Module`
- [ ] Convert `__init__` parameter registration to `setup()` with `self.param()`
- [ ] Replace `nn.Parameter` with `self.param()`
- [ ] Change in-place operations to functional updates (`.at[].set()`)
- [ ] Replace PyTorch optimizers with optax optimizers
- [ ] Convert `.backward()` + `.step()` to `jax.grad()` + `optax.apply_updates()`
- [ ] Handle random number generation with explicit PRNG keys
- [ ] Replace imperative loops with `jax.lax.scan` where possible
- [ ] Separate model definition from parameter initialization
- [ ] Use `model.apply()` instead of direct call in training

## Installation

```bash
# JAX (CPU)
pip install jax jaxlib

# JAX (GPU - CUDA 12)
pip install -U "jax[cuda12]"

# Flax and optax
pip install flax optax
```

## Usage Example

```python
from TiDHy_jax import TiDHy
from TiDHy_jax_inference import evaluate_record
import jax.numpy as jnp
from jax import random

# Create model
model = TiDHy(
    r_dim=20,
    r2_dim=10,
    mix_dim=5,
    input_dim=100,
    hyper_hid_dim=64,
)

# Initialize
rng = random.PRNGKey(0)
X = jnp.ones((8, 50, 100))  # (batch, time, features)
variables = model.init(rng, X, rng)

# Forward pass
outputs = model.apply(variables, X, rng)

# Evaluation
results = evaluate_record(model, variables, X, max_iter=100, tol=1e-4)
```

See `TiDHy_jax_example.py` for complete examples.

## Testing

To verify the translation is correct, compare outputs:

```python
# PyTorch
import torch
from TiDHy import TiDHy as TiDHyPT

# JAX
from TiDHy_jax import TiDHy as TiDHyJAX

# Create equivalent models and compare outputs on same input
# (after converting random seeds and ensuring same initialization)
```

## Known Limitations

1. **Gradient Normalization**: The `grad_norm_inf_step` method is simplified in JAX version
2. **Learning Rate Scheduling**: Milestone scheduling differs between implementations
3. **Stateful Training**: JAX version doesn't maintain state between calls (fully functional)
4. **Progress Bars**: JAX's scan doesn't support interactive progress bars during compilation

## Future Improvements

- [ ] Add JIT compilation decorators
- [ ] Implement gradient normalization fully
- [ ] Add checkpointing utilities
- [ ] Multi-device parallelization with `pmap`
- [ ] Mixed precision training support
- [ ] Advanced learning rate schedules with optax

## References

- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
- [Optax Documentation](https://optax.readthedocs.io/)
- [JAX vs PyTorch](https://pytorch.org/blog/JAX-vs-PyTorch/)

## Contact

For questions about the translation, please open an issue in the repository.
