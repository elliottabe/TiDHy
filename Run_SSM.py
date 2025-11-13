import hydra
import logging
import numpy as np
import itertools
import ssm
from ssm.util import random_rotation, find_permutation
from pathlib import Path
import pickle
from tqdm.auto import tqdm
from omegaconf import DictConfig, OmegaConf

import TiDHy.utils.io_dict_to_hdf5 as ioh5

# Local imports
from TiDHy.datasets.load_data import load_data, stack_data
from TiDHy.utils.path_utils import convert_dict_to_path, convert_dict_to_string
import ssm, itertools 
from ssm.util import find_permutation

def fit_SLDS(cfg,data_dict):

    print("Fitting SLDS with Laplace-EM")
    
    # Handle multiple sequences by concatenating if needed
    inputs_train_SLDS = np.asarray(data_dict['inputs_train'])
    inputs_test_SLDS = np.asarray(data_dict['inputs_test'])
    
    if inputs_train_SLDS.ndim == 3:  # (n_sequences, time, features)
        print(f"Concatenating {inputs_train_SLDS.shape[0]} training sequences")
        inputs_train_SLDS = inputs_train_SLDS.reshape(-1, inputs_train_SLDS.shape[-1])
    if inputs_test_SLDS.ndim == 3:
        print(f"Concatenating {inputs_test_SLDS.shape[0]} test sequences")
        inputs_test_SLDS = inputs_test_SLDS.reshape(-1, inputs_test_SLDS.shape[-1])
    
    # Get model hyperparameters with defaults
    K = getattr(cfg.model, 'num_states', getattr(cfg.dataset, 'K', 6))
    D = getattr(cfg.model, 'latent_dim', 6)
    N = inputs_train_SLDS.shape[-1]
    
    # Check if ground truth states are available
    has_ground_truth = 'states_x_test' in data_dict and 'states_z_test' in data_dict
    
    if has_ground_truth:
        states_x_test = np.asarray(data_dict['states_x_test'])
        states_z_test = np.asarray(data_dict['states_z_test'])
        
        # Handle multi-sequence ground truth
        if states_x_test.ndim == 3 and states_x_test.shape[0] != states_x_test.shape[1]:
            # Assume shape is (Nlds, time, latent_dim) for SLDS/SSM datasets
            states_x_test = states_x_test.transpose((1,0,2)).reshape(states_x_test.shape[1], -1)
        elif states_x_test.ndim == 3:
            # Shape is (n_sequences, time, latent_dim) for other datasets
            states_x_test = states_x_test.reshape(-1, states_x_test.shape[-1])
        
        # For SLDS/SSM: Create combinatorial discrete states
        ssm_params = getattr(cfg.dataset, 'ssm_params', None)
        if ssm_params is not None and 'Nlds' in ssm_params:
            Nlds = ssm_params['Nlds']
            n_disc_states = ssm_params.get('n_disc_states', 2)
            lst = list(itertools.product(range(n_disc_states), repeat=Nlds))
            time_bins = states_z_test.shape[-1] if states_z_test.ndim > 1 else len(states_z_test)
            full_state_z = np.zeros(time_bins, dtype=int)
            for n in range(len(lst)):
                if states_z_test.ndim > 1:
                    full_state_z[np.apply_along_axis(lambda x: np.all(x == lst[n]), 0, states_z_test)] = n
                else:
                    full_state_z = states_z_test  # Already in correct format
            # Override K if using theoretical max from SLDS structure
            K = len(lst)
        elif states_z_test.ndim == 1:
            full_state_z = states_z_test
        else:
            # Flatten multi-dimensional discrete states
            full_state_z = states_z_test.flatten()
    else:
        states_x_test = None
        full_state_z = None
        print("No ground truth latent states available - will skip latent evaluation metrics")
    
    # Log warning if using defaults
    if not hasattr(cfg.model, 'num_states'):
        print(f"Warning: model.num_states not specified, using default K={K}")
    if not hasattr(cfg.model, 'latent_dim'):
        print(f"Warning: model.latent_dim not specified, using default D={D}")
    
    print(f"N:{N} K: {K}, D: {D}")
    # Create the model and initialize its parameters
    slds = ssm.SLDS(N=N, #inputs_train_SLDS.shape[-1], # Input dimension
                    K=K, #len(np.unique(full_state_z)), # number of sets of dynamics dimensions
                    D=D, #data_dict['states_x_test'].shape[-1], # latent dim
                    emissions="gaussian")

    # Fit the model using Laplace-EM with a structured variational posterior
    q_lem_elbos, q_lem = slds.fit(inputs_train_SLDS, method="laplace_em",
                                variational_posterior="structured_meanfield",
                                initialize=False,
                                num_iters=100, alpha=0.0,)


    posterior = slds._make_variational_posterior(variational_posterior="structured_meanfield",datas=inputs_test_SLDS,inputs=None, masks=None, tags=None,method="laplace_em")
    q_lem_x = posterior.mean_continuous_states[0]
    q_lem_z = slds.most_likely_states(q_lem_x, inputs_test_SLDS)

    # Find the permutation that matches the true and inferred states (if ground truth available)
    if has_ground_truth and full_state_z is not None:
        slds.permute(find_permutation(full_state_z, q_lem_z))
        q_lem_z = slds.most_likely_states(q_lem_x, inputs_test_SLDS)

    # Smooth the data under the variational posterior
    q_lem_y = slds.smooth(q_lem_x, inputs_test_SLDS)

    # Compute evaluation metrics for benchmarking
    emission_mse = np.mean((q_lem_y - inputs_test_SLDS)**2)
    
    print(f"SLDS Evaluation Metrics:")
    if has_ground_truth:
        state_accuracy = np.mean(q_lem_z == full_state_z)
        latent_mse = np.mean((q_lem_x - states_x_test)**2)
        print(f"  Discrete state accuracy: {state_accuracy:.4f}")
        print(f"  Latent MSE: {latent_mse:.6f}")
    else:
        state_accuracy = None
        latent_mse = None
        print(f"  Discrete state accuracy: N/A (no ground truth)")
        print(f"  Latent MSE: N/A (no ground truth)")
    print(f"  Emission MSE: {emission_mse:.6f}")
    print(f"  Final ELBO: {q_lem_elbos[-1]:.2f}")

    # Get seed for filename (with fallbacks)
    seed = getattr(cfg.dataset, 'seed', None)
    if seed is None and hasattr(cfg.dataset, 'ssm_params'):
        seed = cfg.dataset.ssm_params.get('seed', 'unknown')
    if seed is None:
        seed = 'unknown'
    
    # Prepare save dictionary
    save_dict = {
        'SLDS_latents': q_lem_x,
        'SLDS_states': q_lem_z,
        'SLDS_emission': q_lem_y,
        'SLDS_elbos': q_lem_elbos,
        'ground_truth_inputs': inputs_test_SLDS,
        'metrics': {
            'emission_mse': emission_mse,
            'final_elbo': q_lem_elbos[-1]
        }
    }
    
    # Add ground truth data if available
    if has_ground_truth:
        save_dict['ground_truth_states_z'] = full_state_z
        save_dict['ground_truth_states_x'] = states_x_test
        save_dict['metrics']['state_accuracy'] = state_accuracy
        save_dict['metrics']['latent_mse'] = latent_mse
    
    ioh5.save(cfg.paths.log_dir/f'ssm_slds_test_full_{D}D_{K}K_{seed}seed.h5', save_dict)
    # with open(cfg.paths.log_dir/f'ssm_slds_test_full_{D}D_{K}K_{seed}seed.pickle', 'wb') as handle:
    #     pickle.dump(slds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(posterior, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(q_lem_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(q_lem_z, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(q_lem_y, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ###### rSLDS
    print('Fitting rSLDS with Laplace-EM')
    rslds = ssm.SLDS(N=N, #inputs_train_SLDS.shape[-1], # Input dimension
                     K=K, #len(np.unique(full_state_z)), # number of sets of dynamics dimensions
                     D=D, #data_dict['states_x_test'].shape[-1], # latent dim
                     transitions="recurrent",
                     emissions="gaussian")
    
    # rslds.initialize(inputs_train_SLDS)
    q_elbos_lem, q_lem = rslds.fit(inputs_train_SLDS, method="laplace_em",
                                variational_posterior="structured_meanfield",
                                initialize=False, num_iters=100, alpha=0.0)
    # xhat_lem = q_lem.mean_continuous_states[0]
    # rslds.permute(find_permutation(full_state_z, rslds.most_likely_states(xhat_lem, inputs_test_SLDS)))
    # zhat_lem = rslds.most_likely_states(xhat_lem, inputs_test_SLDS)

    posterior = rslds._make_variational_posterior( variational_posterior="structured_meanfield",datas=inputs_test_SLDS,inputs=None, masks=None, tags=None,method="laplace_em")
    q_lem_x = posterior.mean_continuous_states[0]
    q_lem_z = rslds.most_likely_states(q_lem_x, inputs_test_SLDS)

    # Find the permutation that matches the true and inferred states (if ground truth available)
    if has_ground_truth and full_state_z is not None:
        rslds.permute(find_permutation(full_state_z, q_lem_z))
        q_lem_z = rslds.most_likely_states(q_lem_x, inputs_test_SLDS)

    # Smooth the data under the variational posterior
    q_lem_y = rslds.smooth(q_lem_x, inputs_test_SLDS)

    # Compute evaluation metrics for benchmarking
    emission_mse_r = np.mean((q_lem_y - inputs_test_SLDS)**2)
    
    print(f"\nrSLDS Evaluation Metrics:")
    if has_ground_truth:
        state_accuracy_r = np.mean(q_lem_z == full_state_z)
        latent_mse_r = np.mean((q_lem_x - states_x_test)**2)
        print(f"  Discrete state accuracy: {state_accuracy_r:.4f}")
        print(f"  Latent MSE: {latent_mse_r:.6f}")
    else:
        state_accuracy_r = None
        latent_mse_r = None
        print(f"  Discrete state accuracy: N/A (no ground truth)")
        print(f"  Latent MSE: N/A (no ground truth)")
    print(f"  Emission MSE: {emission_mse_r:.6f}")
    print(f"  Final ELBO: {q_elbos_lem[-1]:.2f}")

    # Prepare save dictionary
    save_dict_r = {
        'rSLDS_latents': q_lem_x,
        'rSLDS_states': q_lem_z,
        'rSLDS_emission': q_lem_y,
        'rSLDS_elbos': q_elbos_lem,
        'ground_truth_inputs': inputs_test_SLDS,
        'metrics': {
            'emission_mse': emission_mse_r,
            'final_elbo': q_elbos_lem[-1]
        }
    }
    
    # Add ground truth data if available
    if has_ground_truth:
        save_dict_r['ground_truth_states_z'] = full_state_z
        save_dict_r['ground_truth_states_x'] = states_x_test
        save_dict_r['metrics']['state_accuracy'] = state_accuracy_r
        save_dict_r['metrics']['latent_mse'] = latent_mse_r
    
    ioh5.save(cfg.paths.log_dir/f'ssm_rslds_test_full_{D}D_{K}K_{seed}seed.h5', save_dict_r)
    
    # with open(cfg.paths.log_dir/f'ssm_rslds_test_full_{D}D_{K}K_{seed}seed.pickle', 'wb') as handle:
    #     pickle.dump(rslds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(posterior, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(q_lem_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(q_lem_z, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(q_lem_y, handle, protocol=pickle.HIGHEST_PROTOCOL)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def parse_hydra_config(cfg : DictConfig):
    
    ##### Convert paths to Path objects #####
    cfg.paths = convert_dict_to_path(cfg.paths)

    ##### Set Random Seed #####
    # set_seed(42)
    
    # ##### Load Dataset #####
    data_dict = load_data(cfg)
    # Save configuration
    temp_cfg = cfg.copy()
    temp_cfg.paths = convert_dict_to_string(temp_cfg.paths)
    logging.info(f"Configuration:\n{OmegaConf.to_yaml(temp_cfg)}")
    OmegaConf.save(temp_cfg, cfg.paths.log_dir / 'run_config.yaml')
    fit_SLDS(cfg, data_dict)
    

if __name__ == "__main__":
    parse_hydra_config()
    