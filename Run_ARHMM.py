# Apply TFP compatibility patch BEFORE importing modules that use dynamax/TFP
from TiDHy.utils.tfp_jax_patch import apply_tfp_jax_patch
apply_tfp_jax_patch()

import logging
import hydra
import numpy as np
import itertools
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import jax.random as jr
from dynamax.hidden_markov_model import LinearAutoregressiveHMM

import TiDHy.utils.io_dict_to_hdf5 as ioh5
from TiDHy.datasets.load_data import load_data
from TiDHy.utils.path_utils import convert_dict_to_path, convert_dict_to_string


def fit_ARHMM(cfg, data_dict, num_lags=1):
    """
    Fit ARHMM model to partial_superposition data using exact EM.

    ARHMM (Autoregressive Hidden Markov Model):
    - Discrete states z_t that switch between different AR dynamics
    - Emissions: y_t = A_{z_t} y_{t-1:t-L} + b_{z_t} + noise
    - Unlike SLDS: No continuous latent states, models AR directly in observation space
    - Advantage: EXACT EM (tractable inference) vs SLDS's approximate Laplace-EM

    Args:
        cfg: Hydra configuration
        data_dict: Dictionary with 'inputs_train', 'inputs_test', 'states_z_test'
        num_lags: AR order (1 = AR(1), 2 = AR(2), etc.)

    Returns:
        arhmm: Fitted model
        fitted_params: Learned parameters
        results_dict: Evaluation results
    """
    print("=" * 70)
    print("Fitting ARHMM with Exact EM")
    print("=" * 70)

    # Extract data
    emissions_train = np.asarray(data_dict['inputs_train'])  # (time_bins_train, obs_dim)
    emissions_test = np.asarray(data_dict['inputs_test'])     # (time_bins_test, obs_dim)
    states_z_test = np.asarray(data_dict['states_z_test'])    # (Nlds, time_bins_test)

    ssm_params = cfg.dataset.ssm_params

    # Setup ground truth discrete states (combinatorial mapping)
    # Map 3 binary states to 8 combined states: (0,0,0)→0, (0,0,1)→1, ..., (1,1,1)→7
    lst = list(itertools.product([0, 1], repeat=ssm_params['Nlds']))
    full_state_z = np.zeros(ssm_params['time_bins_test'], dtype=int)
    for n in range(len(lst)):
        full_state_z[np.apply_along_axis(
            lambda x: np.all(x == lst[n]), 0, states_z_test
        )] = n

    # Model hyperparameters
    num_states = 2 ** ssm_params['Nlds']  # 8 states for 3 binary SLDS systems
    emission_dim = emissions_train.shape[-1]  # 5 for partial_sup config

    print(f"\nARHMM Configuration:")
    print(f"  Number of discrete states (K): {num_states}")
    print(f"  Emission dimension: {emission_dim}")
    print(f"  AR order (num_lags): {num_lags}")
    print(f"  Training time bins: {emissions_train.shape[0]:,}")
    print(f"  Test time bins: {emissions_test.shape[0]:,}")
    print(f"  Ground truth unique states in test: {len(np.unique(full_state_z))}")

    # Create ARHMM model
    arhmm = LinearAutoregressiveHMM(num_states, emission_dim, num_lags=num_lags)

    # CRITICAL STEP: Compute lagged inputs for AR model
    # This creates y_{t-1}, y_{t-2}, ..., y_{t-L} features
    print("\nComputing lagged inputs for AR dynamics...")
    inputs_train = arhmm.compute_inputs(emissions_train)
    inputs_test = arhmm.compute_inputs(emissions_test)
    print(f"  Lagged inputs shape: {inputs_train.shape}")

    # Initialize parameters with K-means (recommended for real data)
    print("\nInitializing parameters with K-means clustering...")
    key = jr.PRNGKey(ssm_params['seed'])
    params, props = arhmm.initialize(
        key=key,
        method="kmeans",
        emissions=emissions_train
    )

    # Fit with exact EM
    print("\nFitting ARHMM ...")
    fitted_params, log_probs = arhmm.fit_em(
        params,
        props,
        emissions_train,
        inputs=inputs_train,
        num_iters=100,
        verbose=True
    )

    # Convert JAX arrays to numpy
    log_probs = np.array(log_probs)

    print(f"\nTraining complete!")
    print(f"  Initial log probability: {log_probs[0]:.2f}")
    print(f"  Final log probability: {log_probs[-1]:.2f}")
    print(f"  Improvement: {log_probs[-1] - log_probs[0]:.2f}")

    # Inference on test set
    print("\nRunning inference on test set...")

    # Compute posterior (smoothing) - exact forward-backward algorithm
    posterior = arhmm.smoother(fitted_params, emissions_test, inputs=inputs_test)
    smoothed_probs = np.array(posterior.smoothed_probs)  # (T, num_states) - convert JAX to numpy

    # Most likely states (Viterbi decoding)
    most_likely_z = np.array(arhmm.most_likely_states(fitted_params, emissions_test, inputs=inputs_test))

    # Marginal log probability on test set
    test_log_prob = float(arhmm.marginal_log_prob(fitted_params, emissions_test, inputs=inputs_test))

    # Compute evaluation metrics
    print("\nComputing evaluation metrics...")

    # 1. Discrete state accuracy
    state_accuracy = np.nanmean(most_likely_z == full_state_z)

    # 2. Emission prediction MSE
    # Predict: y_t = A_{z_t} @ inputs_test[t] + b_{z_t}
    # Note: inputs_test already contains the lagged features from compute_inputs()
    predicted_emissions = np.zeros_like(emissions_test)
    for t in range(len(emissions_test)):
        z_t = int(most_likely_z[t])
        # Get AR parameters for this state
        A = np.array(fitted_params.emissions.weights[z_t])  # (emission_dim, emission_dim * num_lags)
        b = np.array(fitted_params.emissions.biases[z_t])   # (emission_dim,)

        # Predict: y_t = A @ inputs[t] + b
        # inputs[t] contains [y_{t-1}, y_{t-2}, ..., y_{t-L}] already
        predicted_emissions[t] = A @ np.array(inputs_test[t]) + b

    # Skip first num_lags timesteps (no previous observations for prediction)
    emission_mse = np.nanmean((predicted_emissions[num_lags:] - np.array(emissions_test)[num_lags:])**2)

    # 3. State prediction confidence (entropy of smoothed distribution)
    from scipy.stats import entropy
    avg_entropy = np.nanmean([entropy(smoothed_probs[t]) for t in range(len(smoothed_probs))])
    avg_entropy = float(avg_entropy) if not np.isnan(avg_entropy) else 0.0

    print(f"\nARHMM Evaluation Metrics:")
    print(f"  Discrete state accuracy: {state_accuracy:.4f}")
    print(f"  Emission prediction MSE: {emission_mse:.6f}")
    print(f"  Test log probability: {test_log_prob:.2f}")
    print(f"  Average state entropy: {avg_entropy:.4f} (lower = more confident)")
    print(f"  Final training log prob: {log_probs[-1]:.2f}")

    # Save results
    results_path = cfg.paths.log_dir / f'arhmm_K{num_states}_L{num_lags}_seed{ssm_params["seed"]}.h5'

    print(f"\nSaving results to: {results_path}")
    ioh5.save(results_path, {
        'ARHMM_states': most_likely_z,
        'ARHMM_smoothed_probs': smoothed_probs,
        'ARHMM_predictions': predicted_emissions,
        'ARHMM_log_probs': log_probs,
        'ground_truth_states_z': full_state_z,
        'ground_truth_emissions': emissions_test,
        'metrics': {
            'state_accuracy': state_accuracy,
            'emission_mse': float(emission_mse),
            'test_log_prob': float(test_log_prob),
            'final_train_log_prob': float(log_probs[-1]),
            'train_log_prob_improvement': float(log_probs[-1] - log_probs[0]),
            'avg_state_entropy': float(avg_entropy)
        },
        'config': {
            'num_states': num_states,
            'emission_dim': emission_dim,
            'num_lags': num_lags,
            'seed': int(ssm_params['seed'])
        }
    })

    print("\nARHMM fitting complete!")
    print("=" * 70)

    return arhmm, fitted_params, {
        'states': most_likely_z,
        'smoothed_probs': smoothed_probs,
        'predictions': predicted_emissions,
        'log_probs': log_probs,
        'metrics': {
            'state_accuracy': state_accuracy,
            'emission_mse': emission_mse,
            'test_log_prob': float(test_log_prob),
            'avg_entropy': avg_entropy
        }
    }


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for ARHMM baseline fitting.

    Usage:
        python Run_ARHMM.py dataset=SSM run_id=arhmm_baseline
        python Run_ARHMM.py dataset=SLDS run_id=arhmm_ar2 num_lags=2
    """
    # Convert path strings to Path objects (handles interpolations properly)
    cfg.paths = convert_dict_to_path(cfg.paths)

    print(f"\nDataset: {cfg.dataset.name}")
    print(f"Run ID: {cfg.run_id}")
    print(f"Log directory: {cfg.paths.log_dir}")

    # Load data
    print("\nLoading dataset...")
    data_dict = load_data(cfg)
    
    # Save configuration
    temp_cfg = cfg.copy()
    temp_cfg.paths = convert_dict_to_string(temp_cfg.paths)
    logging.info(f"Configuration:\n{OmegaConf.to_yaml(temp_cfg)}")
    OmegaConf.save(temp_cfg, cfg.paths.log_dir / 'run_config.yaml')
    
    # Fit ARHMM with default AR(1) - can override with num_lags argument
    num_lags = cfg.get('num_lags', 1)
    arhmm, fitted_params, results = fit_ARHMM(cfg, data_dict, num_lags=num_lags)

    print(f"\nResults summary:")
    print(f"  State accuracy: {results['metrics']['state_accuracy']:.4f}")
    print(f"  Emission MSE: {results['metrics']['emission_mse']:.6f}")
    print(f"  Test log prob: {results['metrics']['test_log_prob']:.2f}")

    print("\nDone! Compare with SLDS/TiDHy results for benchmarking.")


if __name__ == "__main__":
    main()
