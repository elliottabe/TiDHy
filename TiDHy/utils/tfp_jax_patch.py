"""
Compatibility patch for TensorFlow Probability with JAX 0.8+

TFP 0.25.0 uses jax.interpreters.xla.pytype_aval_mappings which was removed in JAX 0.8.
This patch restores the missing API by redirecting to the new location.
"""

import jax
from jax import core


def apply_tfp_jax_patch():
    """
    Apply monkeypatch to make TensorFlow Probability compatible with JAX 0.8+
    
    This function must be called before importing tensorflow_probability.
    """
    # Check if patch is needed (JAX 0.8+)
    jax_version = tuple(map(int, jax.__version__.split('.')[:2]))
    
    if jax_version >= (0, 5):
        # Restore the removed API
        if not hasattr(jax.interpreters, 'xla'):
            # Create xla module if it doesn't exist
            import types
            jax.interpreters.xla = types.ModuleType('xla')
        
        # Map the old API to the new location
        if hasattr(core, 'pytype_aval_mappings'):
            jax.interpreters.xla.pytype_aval_mappings = core.pytype_aval_mappings
        else:
            # Fallback: create the mapping directly
            import numpy as np
            jax.interpreters.xla.pytype_aval_mappings = {
                np.ndarray: lambda x: core.ShapedArray(x.shape, x.dtype),
            }
        
        print(f"✓ Applied TFP compatibility patch for JAX {jax.__version__}")
