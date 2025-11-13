"""
LSTM Baseline Model for comparison with TiDHy.

This module implements a simple LSTM autoencoder baseline using Flax NNX.
It supports both reconstruction (autoencoder) and next-step prediction tasks.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple, Optional


class LSTMCell(nnx.Module):
    """
    Simple LSTM cell implementation using Flax NNX.
    """
    def __init__(self, input_dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        """
        Initialize LSTM cell.

        Args:
            input_dim: Dimension of input
            hidden_dim: Dimension of hidden state
            rngs: Random number generators
        """
        self.hidden_dim = hidden_dim

        # LSTM gates: input, forget, cell, output
        # Combined weight matrix for efficiency: [input + hidden] -> [4 * hidden]
        self.W_ih = nnx.Linear(input_dim, 4 * hidden_dim, use_bias=False, rngs=rngs)
        self.W_hh = nnx.Linear(hidden_dim, 4 * hidden_dim, use_bias=True, rngs=rngs)

    def __call__(self, carry: Tuple[jnp.ndarray, jnp.ndarray], x: jnp.ndarray) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """
        Forward pass through LSTM cell.

        Args:
            carry: Tuple of (h, c) where h is hidden state and c is cell state
            x: Input at current timestep (input_dim,)

        Returns:
            new_carry: Updated (h, c)
            h: New hidden state
        """
        h, c = carry

        # Compute all gates at once
        gates = self.W_ih(x) + self.W_hh(h)

        # Split into individual gates
        i, f, g, o = jnp.split(gates, 4, axis=-1)

        # Apply activations
        i = jax.nn.sigmoid(i)  # Input gate
        f = jax.nn.sigmoid(f)  # Forget gate
        g = jnp.tanh(g)        # Cell gate
        o = jax.nn.sigmoid(o)  # Output gate

        # Update cell state and hidden state
        new_c = f * c + i * g
        new_h = o * jnp.tanh(new_c)

        return (new_h, new_c), new_h


class LSTMEncoder(nnx.Module):
    """
    LSTM encoder that processes a sequence and returns hidden states at each timestep.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1,
                 dropout_rate: float = 0.0, *, rngs: nnx.Rngs):
        """
        Initialize LSTM encoder.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden state
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate (0 = no dropout)
            rngs: Random number generators
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # Create LSTM layers
        # self.lstm_cells = nnx.List([])
        self.lstm_cells = []
        for i in range(num_layers):
            layer_input_dim = input_dim if i == 0 else hidden_dim
            self.lstm_cells.append(LSTMCell(layer_input_dim, hidden_dim, rngs=rngs))

        # Dropout (if enabled)
        if dropout_rate > 0.0:
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    def init_carry(self, batch_shape: Tuple = ()) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Initialize hidden and cell states to zeros.

        Args:
            batch_shape: Shape for batch dimensions

        Returns:
            Initial (h, c) tuple
        """
        shape = batch_shape + (self.hidden_dim,)
        return (jnp.zeros(shape), jnp.zeros(shape))

    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Process sequence through LSTM encoder.

        Args:
            x: Input sequence of shape (T, input_dim)
            training: Whether in training mode (for dropout)

        Returns:
            Hidden states of shape (T, hidden_dim)
        """
        T = x.shape[0]

        # Process through each layer
        layer_input = x
        for layer_idx, lstm_cell in enumerate(self.lstm_cells):
            # Initialize carry for this layer
            carry = self.init_carry()

            # Process sequence with scan
            def scan_fn(carry, x_t):
                new_carry, h_t = lstm_cell(carry, x_t)
                return new_carry, h_t

            _, hiddens = jax.lax.scan(scan_fn, carry, layer_input)

            # Apply dropout between layers (except last layer)
            if self.dropout is not None and layer_idx < self.num_layers - 1 and training:
                hiddens = self.dropout(hiddens)

            layer_input = hiddens

        return hiddens


class LSTMBaseline(nnx.Module):
    """
    LSTM baseline model for comparison with TiDHy.
    Supports both autoencoder reconstruction and next-step prediction.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: Optional[int] = None,
                 num_layers: int = 1, dropout_rate: float = 0.0, *, rngs: nnx.Rngs):
        """
        Initialize LSTM baseline model.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of LSTM hidden state
            output_dim: Dimension of output (defaults to input_dim)
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate
            rngs: Random number generators
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.num_layers = num_layers

        # Encoder
        self.encoder = LSTMEncoder(input_dim, hidden_dim, num_layers, dropout_rate, rngs=rngs)

        # Decoder (linear projection from hidden state to output)
        self.decoder = nnx.Linear(hidden_dim, self.output_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray, training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass: reconstruction + prediction.

        Args:
            x: Input sequence of shape (T, input_dim)
            training: Whether in training mode

        Returns:
            reconstruction: Reconstructed sequence (T, output_dim)
            prediction: Next-step predictions (T-1, output_dim)
        """
        # Encode sequence
        hiddens = self.encoder(x, training=training)  # (T, hidden_dim)

        # Decode for reconstruction
        reconstruction = jax.vmap(self.decoder)(hiddens)  # (T, output_dim)

        # Predict next step: use hidden at t to predict x at t+1
        # prediction[t] = decoder(hidden[t]) should match x[t+1]
        predictions = jax.vmap(self.decoder)(hiddens[:-1])  # (T-1, output_dim)

        return reconstruction, predictions

    def encode(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Encode input sequence to hidden states.

        Args:
            x: Input sequence of shape (T, input_dim)
            training: Whether in training mode

        Returns:
            Hidden states of shape (T, hidden_dim)
        """
        return self.encoder(x, training=training)

    def reconstruct(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Reconstruct input sequence (autoencoder).

        Args:
            x: Input sequence of shape (T, input_dim)
            training: Whether in training mode

        Returns:
            Reconstructed sequence of shape (T, output_dim)
        """
        hiddens = self.encoder(x, training=training)
        return jax.vmap(self.decoder)(hiddens)

    def predict_next(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Predict next timestep from current input.

        Args:
            x: Input sequence of shape (T, input_dim)
            training: Whether in training mode

        Returns:
            Next-step predictions of shape (T-1, output_dim)
        """
        hiddens = self.encoder(x, training=training)
        # Use hidden[:-1] to predict x[1:]
        return jax.vmap(self.decoder)(hiddens[:-1])


def compute_lstm_losses(model: LSTMBaseline, x: jnp.ndarray,
                       reconstruction_weight: float = 1.0,
                       prediction_weight: float = 1.0,
                       training: bool = True) -> Tuple[jnp.ndarray, dict]:
    """
    Compute LSTM losses: reconstruction + next-step prediction.

    Args:
        model: LSTM model
        x: Input sequence (T, input_dim)
        reconstruction_weight: Weight for reconstruction loss
        prediction_weight: Weight for prediction loss
        training: Whether in training mode

    Returns:
        total_loss: Weighted sum of losses
        metrics: Dictionary of individual loss components
    """
    # Forward pass
    reconstruction, predictions = model(x, training=training)

    # Reconstruction loss: MSE between input and reconstruction
    recon_loss = jnp.mean((x - reconstruction) ** 2)

    # Prediction loss: MSE between x[1:] and predictions
    pred_loss = jnp.mean((x[1:] - predictions) ** 2)

    # Total loss
    total_loss = reconstruction_weight * recon_loss + prediction_weight * pred_loss

    # Metrics dictionary
    metrics = {
        'loss': total_loss,
        'reconstruction_loss': recon_loss,
        'prediction_loss': pred_loss,
    }

    return total_loss, metrics
