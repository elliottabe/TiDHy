"""
Example usage of the Flax NNX TiDHy model.

This script demonstrates how to:
1. Create a TiDHy model with NNX
2. Train the model
3. Evaluate the model
4. Save and load models
"""

import jax
import jax.numpy as jnp
from flax import nnx

from TiDHy.models.TiDHy_nnx import TiDHy
from TiDHy.models.TiDHy_nnx_training import (
    train_model,
    train_step,
    evaluate_batch,
    evaluate_record,
    save_model,
    load_model,
    checkpoint_model,
    load_checkpoint,
    create_optimizer
)


def example_basic_usage():
    """Basic example of creating and using the model"""

    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # Create RNG
    rngs = nnx.Rngs(0)

    # Create model
    model = TiDHy(
        r_dim=20,
        r2_dim=10,
        mix_dim=5,
        input_dim=100,
        hyper_hid_dim=64,
        loss_type='MSE',
        nonlin_decoder=False,
        low_rank_temp=False,
        dyn_bias=True,
        use_r2_decoder=True,
        r2_decoder_hid_dim=64,
        lr_r=0.1,
        lr_r2=0.1,
        temp_weight=1.0,
        lmda_r=0.01,
        lmda_r2=0.01,
        max_iter=100,
        tol=1e-4,
        show_progress=False,
        show_inf_progress=False,
        rngs=rngs
    )

    print(f"Model created with:")
    print(f"  r_dim: {model.r_dim}")
    print(f"  r2_dim: {model.r2_dim}")
    print(f"  input_dim: {model.input_dim}")

    # Create dummy data
    batch_size = 8
    T = 20
    X = jax.random.normal(jax.random.PRNGKey(42), (batch_size, T, model.input_dim))

    print(f"\nInput shape: {X.shape}")

    # Forward pass
    spatial_loss_rhat, spatial_loss_rbar, temp_loss, r2_losses, r_first, r2_final = model(X)

    print(f"\nForward pass results:")
    print(f"  Spatial loss (rhat): {spatial_loss_rhat:.4f}")
    print(f"  Spatial loss (rbar): {spatial_loss_rbar:.4f}")
    print(f"  Temporal loss: {temp_loss:.4f}")
    print(f"  R2 losses: {r2_losses:.4f}")
    print(f"  r_first shape: {r_first.shape}")
    print(f"  r2_final shape: {r2_final.shape}")

    return model, X


def example_training():
    """Example of training the model"""

    print("\n" + "=" * 60)
    print("Example 2: Training")
    print("=" * 60)

    # Create RNG
    rngs = nnx.Rngs(0)

    # Create model
    model = TiDHy(
        r_dim=15,
        r2_dim=8,
        mix_dim=4,
        input_dim=50,
        hyper_hid_dim=32,
        loss_type='MSE',
        use_r2_decoder=True,
        lr_r=0.1,
        lr_r2=0.1,
        max_iter=50,
        tol=1e-4,
        show_progress=False,
        rngs=rngs
    )

    # Create training data
    n_samples = 32
    T = 30
    train_data = jax.random.normal(
        jax.random.PRNGKey(1), (n_samples, T, model.input_dim)
    )

    # Create validation data
    val_data = jax.random.normal(
        jax.random.PRNGKey(2), (8, T, model.input_dim)
    )

    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")

    # Train model
    trained_model, history = train_model(
        model,
        train_data,
        n_epochs=5,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=8,
        val_data=val_data,
        verbose=True
    )

    print(f"\nTraining complete!")
    print(f"Final training loss: {history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")

    return trained_model, history


def example_manual_training():
    """Example of manual training loop with more control"""

    print("\n" + "=" * 60)
    print("Example 3: Manual Training Loop")
    print("=" * 60)

    # Create RNG
    rngs = nnx.Rngs(0)

    # Create model
    model = TiDHy(
        r_dim=10,
        r2_dim=5,
        mix_dim=3,
        input_dim=30,
        hyper_hid_dim=32,
        show_progress=False,
        rngs=rngs
    )

    # Create optimizer
    optimizer_tx = create_optimizer(learning_rate=1e-3)
    optimizer = nnx.Optimizer(model, optimizer_tx, wrt=nnx.Param)

    # Create data
    batch_size = 4
    T = 25
    X = jax.random.normal(jax.random.PRNGKey(10), (batch_size, T, model.input_dim))

    # Manual training loop
    n_epochs = 3
    for epoch in range(n_epochs):
        metrics = train_step(model, optimizer, X)

        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Spatial loss (rhat): {metrics['spatial_loss_rhat']:.4f}")
        print(f"  Temporal loss: {metrics['temp_loss']:.4f}")

    return model, optimizer


def example_evaluation():
    """Example of evaluating the model with recording"""

    print("\n" + "=" * 60)
    print("Example 4: Evaluation with Recording")
    print("=" * 60)

    # Create and train a model first
    rngs = nnx.Rngs(0)
    model = TiDHy(
        r_dim=10,
        r2_dim=5,
        mix_dim=3,
        input_dim=30,
        hyper_hid_dim=32,
        show_progress=False,
        show_inf_progress=False,
        rngs=rngs
    )

    # Create test data
    batch_size = 4
    T = 20
    test_data = jax.random.normal(
        jax.random.PRNGKey(100), (batch_size, T, model.input_dim)
    )

    print(f"Test data shape: {test_data.shape}")

    # Evaluate with recording
    spatial_loss_rhat, spatial_loss_rbar, temp_loss, result_dict = evaluate_record(
        model, test_data, verbose=False
    )

    print(f"\nEvaluation Results:")
    print(f"  Spatial loss (rhat): {spatial_loss_rhat:.4f}")
    print(f"  Spatial loss (rbar): {spatial_loss_rbar:.4f}")
    print(f"  Temporal loss: {temp_loss:.4f}")

    print(f"\nRecorded outputs:")
    for key, value in result_dict.items():
        print(f"  {key}: {value.shape}")

    return result_dict


def example_save_load():
    """Example of saving and loading models"""

    print("\n" + "=" * 60)
    print("Example 5: Save and Load Model")
    print("=" * 60)

    # Create model
    rngs = nnx.Rngs(0)
    model = TiDHy(
        r_dim=10,
        r2_dim=5,
        mix_dim=3,
        input_dim=30,
        hyper_hid_dim=32,
        show_progress=False,
        rngs=rngs
    )

    # Save model
    save_path = "/tmp/tidhy_model.pkl"
    save_model(model, save_path)

    # Create new model with same structure
    rngs_new = nnx.Rngs(42)  # Different seed
    new_model = TiDHy(
        r_dim=10,
        r2_dim=5,
        mix_dim=3,
        input_dim=30,
        hyper_hid_dim=32,
        show_progress=False,
        rngs=rngs_new
    )

    # Load saved weights
    loaded_model = load_model(new_model, save_path)

    # Test that loaded model works
    X = jax.random.normal(jax.random.PRNGKey(50), (4, 15, 30))
    outputs = loaded_model(X)

    print(f"\nLoaded model works! Loss: {outputs[0]:.4f}")

    return loaded_model


def example_checkpoint():
    """Example of saving and loading checkpoints"""

    print("\n" + "=" * 60)
    print("Example 6: Checkpointing")
    print("=" * 60)

    # Create model and optimizer
    rngs = nnx.Rngs(0)
    model = TiDHy(
        r_dim=10,
        r2_dim=5,
        mix_dim=3,
        input_dim=30,
        hyper_hid_dim=32,
        show_progress=False,
        rngs=rngs
    )

    optimizer_tx = create_optimizer(learning_rate=1e-3)
    optimizer = nnx.Optimizer(model, optimizer_tx, wrt=nnx.Param)

    # Train for a few epochs
    X = jax.random.normal(jax.random.PRNGKey(10), (4, 15, 30))

    for epoch in range(3):
        metrics = train_step(model, optimizer, X)
        print(f"Epoch {epoch + 1}: Loss = {metrics['loss']:.4f}")

    # Save checkpoint
    checkpoint_path = "/tmp/tidhy_checkpoint.pkl"
    checkpoint_model(model, optimizer, epoch=3, filepath=checkpoint_path)

    # Create new model and optimizer
    rngs_new = nnx.Rngs(42)
    new_model = TiDHy(
        r_dim=10,
        r2_dim=5,
        mix_dim=3,
        input_dim=30,
        hyper_hid_dim=32,
        show_progress=False,
        rngs=rngs_new
    )

    new_optimizer_tx = create_optimizer(learning_rate=1e-3)
    new_optimizer = nnx.Optimizer(new_model, new_optimizer_tx, wrt=nnx.Param)

    # Load checkpoint
    loaded_model, loaded_optimizer, loaded_epoch = load_checkpoint(
        new_model, new_optimizer, checkpoint_path
    )

    # Continue training
    print(f"\nContinuing training from epoch {loaded_epoch + 1}")
    for epoch in range(loaded_epoch, loaded_epoch + 2):
        metrics = train_step(loaded_model, loaded_optimizer, X)
        print(f"Epoch {epoch + 1}: Loss = {metrics['loss']:.4f}")

    return loaded_model, loaded_optimizer


def example_stateful_model():
    """Example of using stateful mode"""

    print("\n" + "=" * 60)
    print("Example 7: Stateful Model")
    print("=" * 60)

    # Create model with stateful=True
    rngs = nnx.Rngs(0)
    model = TiDHy(
        r_dim=10,
        r2_dim=5,
        mix_dim=3,
        input_dim=30,
        hyper_hid_dim=32,
        stateful=True,  # Enable stateful mode
        show_progress=False,
        rngs=rngs
    )

    print(f"Model created with stateful=True")
    print(f"Initial r_state: {model.r_state.value.shape}")
    print(f"Initial r2_state: {model.r2_state.value.shape}")

    # Process sequential batches
    for i in range(3):
        X = jax.random.normal(
            jax.random.PRNGKey(i), (4, 10, model.input_dim)
        )

        outputs = model(X, training=True)

        print(f"\nBatch {i + 1}:")
        print(f"  Loss: {outputs[0]:.4f}")
        print(f"  r_state mean: {jnp.mean(model.r_state.value):.4f}")
        print(f"  r2_state mean: {jnp.mean(model.r2_state.value):.4f}")

    return model


def example_different_architectures():
    """Example showing different architectural configurations"""

    print("\n" + "=" * 60)
    print("Example 8: Different Architectures")
    print("=" * 60)

    X = jax.random.normal(jax.random.PRNGKey(0), (4, 15, 50))

    configs = [
        {
            "name": "Basic (MSE loss)",
            "params": {
                "loss_type": "MSE",
                "nonlin_decoder": False,
                "low_rank_temp": False,
            }
        },
        {
            "name": "BCE loss with sigmoid decoder",
            "params": {
                "loss_type": "BCE",
                "nonlin_decoder": False,
                "low_rank_temp": False,
            }
        },
        {
            "name": "Nonlinear decoder",
            "params": {
                "loss_type": "MSE",
                "nonlin_decoder": True,
                "low_rank_temp": False,
            }
        },
        {
            "name": "Low-rank temporal dynamics",
            "params": {
                "loss_type": "MSE",
                "nonlin_decoder": False,
                "low_rank_temp": True,
            }
        },
    ]

    for config in configs:
        rngs = nnx.Rngs(0)
        model = TiDHy(
            r_dim=10,
            r2_dim=5,
            mix_dim=3,
            input_dim=50,
            hyper_hid_dim=32,
            show_progress=False,
            rngs=rngs,
            **config["params"]
        )

        outputs = model(X, training=False)
        print(f"\n{config['name']}:")
        print(f"  Loss: {outputs[0]:.4f}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TiDHy Flax NNX Examples")
    print("=" * 60)

    # Run all examples
    example_basic_usage()
    example_training()
    example_manual_training()
    example_evaluation()
    example_save_load()
    example_checkpoint()
    example_stateful_model()
    example_different_architectures()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
