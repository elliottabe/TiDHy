"""
Example usage of the JAX/Flax TiDHy model.

This script demonstrates how to:
1. Create a TiDHy model instance
2. Initialize parameters
3. Train the model
4. Evaluate the model
"""

import jax
import jax.numpy as jnp
from jax import random
import optax
from flax.training import train_state

# Import the model (adjust import path as needed)
from TiDHy.utils.TiDHy_jax import TiDHy, create_train_state, train_step, normalize_params
from TiDHy_jax_inference import inf_step, evaluate_record


class SimpleParams:
    """Simple parameter container to mimic PyTorch version"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def create_model_params():
    """Create model parameters matching PyTorch version"""
    params = SimpleParams(
        # Model dimensions
        r_dim=20,
        r2_dim=10,
        mix_dim=5,
        input_dim=100,
        hyper_hid_dim=64,

        # Architecture options
        loss_type='MSE',
        nonlin_decoder=False,
        low_rank_temp=False,
        dyn_bias=True,
        use_r2_decoder=True,
        r2_decoder_hid_dim=64,

        # Learning rates
        lr_r=0.1,
        lr_r2=0.1,
        lr_weights=0.025,
        lr_weights_inf=0.025,

        # Regularization
        temp_weight=1.0,
        lmda_r=0.01,
        lmda_r2=0.01,
        weight_decay=1e-4,
        L1_alpha=0.0,
        L1_inf_w=0.0,
        L1_inf_r2=0.0,
        L1_inf_r=0.0,
        L1_alpha_inf=0.0,
        L1_alpha_r2=0.0,
        grad_alpha=1.5,
        grad_alpha_inf=1.5,
        clip_grad=1.0,
        grad_norm_inf=False,

        # Training
        max_iter=100,
        tol=1e-4,
        normalize_spatial=False,
        normalize_temporal=False,
        stateful=False,
        batch_converge=False,
        spat_weight=1.0,
        learning_rate_gamma=0.5,
        cos_eta=0.001,

        # Display
        show_progress=True,
        show_inf_progress=False,
    )
    return params


def example_training():
    """Example training loop"""

    # Set random seed
    rng = random.PRNGKey(0)

    # Create model parameters
    model_params = create_model_params()

    # Create model
    model = TiDHy(
        r_dim=model_params.r_dim,
        r2_dim=model_params.r2_dim,
        mix_dim=model_params.mix_dim,
        input_dim=model_params.input_dim,
        hyper_hid_dim=model_params.hyper_hid_dim,
        loss_type=model_params.loss_type,
        nonlin_decoder=model_params.nonlin_decoder,
        low_rank_temp=model_params.low_rank_temp,
        dyn_bias=model_params.dyn_bias,
        use_r2_decoder=model_params.use_r2_decoder,
        r2_decoder_hid_dim=model_params.r2_decoder_hid_dim,
        lr_r=model_params.lr_r,
        lr_r2=model_params.lr_r2,
        temp_weight=model_params.temp_weight,
        lmda_r=model_params.lmda_r,
        lmda_r2=model_params.lmda_r2,
        max_iter=model_params.max_iter,
        tol=model_params.tol,
        normalize_spatial=model_params.normalize_spatial,
        normalize_temporal=model_params.normalize_temporal,
        show_progress=model_params.show_progress,
    )

    # Create dummy data
    batch_size = 8
    T = 50  # time steps
    input_dim = model_params.input_dim

    rng, data_rng = random.split(rng)
    X_train = random.normal(data_rng, (batch_size, T, input_dim))

    # Initialize model
    rng, init_rng = random.split(rng)
    variables, tx = create_train_state(
        model, init_rng, learning_rate=1e-3,
        batch_size=batch_size, input_dim=input_dim, T=T
    )

    # Create training state
    state = {
        'params': variables['params'],
        'tx': tx,
        'opt_state': tx.init(variables['params'])
    }

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        rng, train_rng = random.split(rng)
        state, metrics = train_step(state, X_train, model, train_rng)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Total loss: {metrics['loss']:.4f}")
        print(f"  Spatial loss (rhat): {metrics['spatial_loss_rhat']:.4f}")
        print(f"  Spatial loss (rbar): {metrics['spatial_loss_rbar']:.4f}")
        print(f"  Temporal loss: {metrics['temp_loss']:.4f}")

        # Optional: normalize parameters
        if model_params.normalize_spatial or model_params.normalize_temporal:
            state['params'] = normalize_params(
                {'params': state['params']},
                normalize_spatial=model_params.normalize_spatial,
                normalize_temporal=model_params.normalize_temporal
            )['params']

    return model, state, X_train


def example_evaluation():
    """Example evaluation"""

    # Train model first
    print("Training model...")
    model, state, X_train = example_training()

    print("\nEvaluating model...")

    # Create test data
    rng = random.PRNGKey(42)
    batch_size = 4
    T = 30
    X_test = random.normal(rng, (batch_size, T, model.input_dim))

    # Evaluate
    params = {'params': state['params']}

    spatial_loss_rhat, spatial_loss_rbar, temp_loss, result_dict = evaluate_record(
        model, params, X_test,
        max_iter=model.max_iter,
        tol=model.tol,
        lr_r=model.lr_r,
        lr_r2=model.lr_r2,
        lmda_r=model.lmda_r,
        lmda_r2=model.lmda_r2,
        temp_weight=model.temp_weight,
        batch_converge=model.batch_converge,
        use_r2_decoder=model.use_r2_decoder,
    )

    print(f"\nEvaluation Results:")
    print(f"  Spatial loss (rhat): {spatial_loss_rhat:.4f}")
    print(f"  Spatial loss (rbar): {spatial_loss_rbar:.4f}")
    print(f"  Temporal loss: {temp_loss:.4f}")

    print(f"\nResult dictionary shapes:")
    for key, value in result_dict.items():
        print(f"  {key}: {value.shape}")

    return result_dict


def example_inference():
    """Example of running inference on a single timestep"""

    # Create model
    model_params = create_model_params()
    model = TiDHy(**{k: v for k, v in model_params.__dict__.items()
                     if k in TiDHy.__annotations__})

    # Initialize model
    rng = random.PRNGKey(0)
    batch_size = 4
    T = 10

    dummy_input = jnp.ones((batch_size, T, model_params.input_dim))
    variables = model.init(rng, dummy_input, rng)

    # Create single observation
    x_t = random.normal(rng, (batch_size, model_params.input_dim))
    r_prev = random.normal(rng, (batch_size, model_params.r_dim))
    r2_prev = random.normal(rng, (batch_size, model_params.r2_dim))

    # Run inference
    r_new, r2_new, r2_loss = inf_step(
        model, variables, x_t, r_prev, r2_prev,
        lr_r=model_params.lr_r,
        lr_r2=model_params.lr_r2,
        lmda_r=model_params.lmda_r,
        lmda_r2=model_params.lmda_r2,
        max_iter=model_params.max_iter,
        tol=model_params.tol,
        temp_weight=model_params.temp_weight,
        batch_converge=model_params.batch_converge,
        use_r2_decoder=model_params.use_r2_decoder,
    )

    print("Inference Results:")
    print(f"  r shape: {r_new.shape}")
    print(f"  r2 shape: {r2_new.shape}")
    print(f"  r2 loss: {r2_loss:.4f}")

    return r_new, r2_new


if __name__ == "__main__":
    print("=" * 60)
    print("TiDHy JAX/Flax Example")
    print("=" * 60)

    # Run training example
    print("\n1. Training Example")
    print("-" * 60)
    example_training()

    # Run evaluation example
    print("\n2. Evaluation Example")
    print("-" * 60)
    example_evaluation()

    # Run inference example
    print("\n3. Inference Example")
    print("-" * 60)
    example_inference()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
