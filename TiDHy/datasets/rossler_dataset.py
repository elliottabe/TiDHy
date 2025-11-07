"""
Hierarchical Multi-Timescale Rossler Attractor Dataset Generation

This module implements a hierarchical Rossler attractor system where a slow Rossler
system modulates the parameters of a fast Rossler system, creating dynamics with
multiple timescales in a single trajectory.
"""

import numpy as np
import jax.numpy as jnp
from scipy.integrate import solve_ivp
from TiDHy.utils import io_dict_to_hdf5 as ioh5

def hierarchical_rossler_ode(t, state, a_slow, b_slow, c_slow, a_fast, b_base, c_base,
                               coupling_b, coupling_c):
    """
    Coupled hierarchical Rossler system ODEs.

    The slow system modulates the b and c parameters of the fast system:
    - b_fast(t) = clip(b_base + coupling_b * x_slow(t), 0.01, 1.0)
    - c_fast(t) = clip(c_base + coupling_c * z_slow(t), 2.0, 10.0)

    The clipping prevents numerical instability by ensuring the fast system's
    parameters stay within stable Rossler attractor ranges.

    Args:
        t: time (not used, autonomous system)
        state: [xs, ys, zs, xf, yf, zf] - concatenated slow and fast states
        a_slow, b_slow, c_slow: slow system parameters
        a_fast, b_base, c_base: fast system base parameters
        coupling_b, coupling_c: modulation coupling strengths

    Returns:
        derivatives: [dxs, dys, dzs, dxf, dyf, dzf]
    """
    xs, ys, zs, xf, yf, zf = state

    # Slow Rossler system
    dxs = -ys - zs
    dys = xs + a_slow * ys
    dzs = b_slow + zs * (xs - c_slow)

    # Fast Rossler system with modulated parameters
    # Clip parameters to stable ranges to prevent numerical instability
    b_fast = np.clip(b_base + coupling_b * xs, 0.01, 1.0)
    c_fast = np.clip(c_base + coupling_c * zs, 2.0, 10.0)

    dxf = -yf - zf
    dyf = xf + a_fast * yf
    dzf = b_fast + zf * (xf - c_fast)

    return [dxs, dys, dzs, dxf, dyf, dzf]


def integrate_hierarchical_rossler(time_bins, dt, initial_state, params, transient_steps=1000):
    """
    Integrate the hierarchical Rossler system.

    Args:
        time_bins: number of time points to generate
        dt: integration timestep
        initial_state: [xs0, ys0, zs0, xf0, yf0, zf0] initial conditions
        params: dictionary with system parameters
        transient_steps: number of initial steps to discard (default: 1000)

    Returns:
        states: array of shape (time_bins, 6) containing the trajectory
    """
    # Total time including transient
    total_bins = time_bins + transient_steps
    t_span = (0, total_bins * dt)
    t_eval = np.linspace(0, total_bins * dt, total_bins)

    # Integrate using LSODA (good for stiff systems)
    sol = solve_ivp(
        hierarchical_rossler_ode,
        t_span,
        initial_state,
        args=(
            params['a_slow'], params['b_slow'], params['c_slow'],
            params['a_fast'], params['b_base'], params['c_base'],
            params['coupling_b'], params['coupling_c']
        ),
        t_eval=t_eval,
        method='LSODA',
        rtol=1e-6,   # Relaxed from 1e-9
        atol=1e-9,   # Relaxed from 1e-12
        max_step=dt * 10  # Prevent extremely small steps
    )

    # Check integration success
    if sol.status != 0:
        raise RuntimeError(
            f"Integration failed with status {sol.status}: {sol.message}"
        )

    if sol.y.shape[1] != total_bins:
        raise RuntimeError(
            f"Integration produced {sol.y.shape[1]} timesteps, expected {total_bins}. "
            f"Integration terminated early. Status: {sol.status}, Message: {sol.message}"
        )

    # Discard transient and return
    states = sol.y.T[transient_steps:, :]
    return states


def hierarchical_rossler_dataset(cfg, params):
    """
    Generate hierarchical multi-timescale Rossler attractor dataset.

    This function creates a dataset where a slow Rossler system modulates the parameters
    of a fast Rossler system, producing dynamics with multiple timescales.

    Args:
        cfg: OmegaConf configuration object
        params: dictionary with the following keys:
            - time_bins_train: int, length of training trajectory
            - time_bins_test: int, length of test/validation trajectories
            - dt: float, integration timestep (e.g., 0.01)
            - a_slow, b_slow, c_slow: float, slow system parameters
            - a_fast, b_base, c_base: float, fast system base parameters
            - coupling_b, coupling_c: float, modulation coupling strengths
            - observed_states: list of int, indices of states to observe (e.g., [0,1,5])
                                State ordering: [xs, ys, zs, xf, yf, zf] = [0,1,2,3,4,5]
            - noise_level: float, observation noise standard deviation
            - normalize: bool, whether to normalize the observations
            - seed: int, random seed for reproducibility
            - transient_steps: int, number of initial steps to discard (default: 1000)

    Returns:
        data_dict: dictionary with keys:
            - inputs_train: (time_bins_train, obs_dim) - training observations
            - inputs_val: (time_bins_test, obs_dim) - validation observations
            - inputs_test: (time_bins_test, obs_dim) - test observations
            - states_x_train: (time_bins_train, 6) - full training states
            - states_x_val: (time_bins_test, 6) - full validation states
            - states_x_test: (time_bins_test, 6) - full test states
    """
    # Set random seed
    np.random.seed(params['seed'])

    # Extract parameters
    time_bins_train = params['time_bins_train']
    time_bins_test = params['time_bins_test']
    dt = params['dt']
    observed_states = params.get('observed_states', [3, 4, 2])  # Default: xf, yf, zs
    noise_level = params.get('noise_level', 0.0)
    normalize = params.get('normalize', True)
    transient_steps = params.get('transient_steps', 1000)
    # Set the random seed
    filename = 'Rossler_N{}_seed{}.h5'.format(''.join(map(str, cfg.dataset.rossler_params['observed_states'])), cfg.dataset.rossler_params['seed'])
    
    if (cfg.paths.data_dir/filename).exists():
        print(f"Loading existing dataset from {cfg.paths.data_dir/filename}...")
        temp_dict = ioh5.load(cfg.paths.data_dir/filename)
        obs_train = temp_dict['obs_train']
        obs_val = temp_dict['obs_val']
        obs_test = temp_dict['obs_test']
        states_train = temp_dict['states_x_train']
        states_val = temp_dict['states_x_val']
        states_test = temp_dict['states_x_test']
    else:
        # System parameters dictionary
        sys_params = {
            'a_slow': params['a_slow'],
            'b_slow': params['b_slow'],
            'c_slow': params['c_slow'],
            'a_fast': params['a_fast'],
            'b_base': params['b_base'],
            'c_base': params['c_base'],
            'coupling_b': params['coupling_b'],
            'coupling_c': params['coupling_c']
        }

        # Generate initial conditions for each dataset split
        # Random initial conditions in a reasonable range around the attractor
        def random_initial_state():
            return np.random.randn(6) * 0.5 + np.array([0, 0, 0, 1, 1, 1])

        print(f"Generating hierarchical Rossler training data ({time_bins_train} timesteps)...")
        initial_train = random_initial_state()
        states_train = integrate_hierarchical_rossler(
            time_bins_train, dt, initial_train, sys_params, transient_steps
        )

        print(f"Generating hierarchical Rossler validation data ({time_bins_test} timesteps)...")
        initial_val = random_initial_state()
        states_val = integrate_hierarchical_rossler(
            time_bins_test, dt, initial_val, sys_params, transient_steps
        )

        print(f"Generating hierarchical Rossler test data ({time_bins_test} timesteps)...")
        initial_test = random_initial_state()
        states_test = integrate_hierarchical_rossler(
            time_bins_test, dt, initial_test, sys_params, transient_steps
        )

        # Select observed states
        print(f"Selecting observed states: {observed_states}")
        obs_train = states_train[:, observed_states]
        obs_val = states_val[:, observed_states]
        obs_test = states_test[:, observed_states]

        # Add observation noise if specified
        if noise_level > 0:
            print(f"Adding observation noise (sigma={noise_level})...")
            obs_train += np.random.randn(*obs_train.shape) * noise_level
            obs_val += np.random.randn(*obs_val.shape) * noise_level
            obs_test += np.random.randn(*obs_test.shape) * noise_level
        # Save generated dataset
        ioh5.save(cfg.paths.data_dir/filename, {
            'obs_train': obs_train,
            'obs_val': obs_val,
            'obs_test': obs_test,
            'states_x_train': states_train,
            'states_x_val': states_val,
            'states_x_test': states_test,
        })
    if params.get('rand_proj', False):
        print(f"Applying random projection to dimension {params['rand_proj_dim']}...")
        obs_dim = obs_train.shape[1]
        rand_matrix = np.random.randn(obs_dim, params['rand_proj_dim'])
        obs_train = obs_train @ rand_matrix
        obs_val = obs_val @ rand_matrix
        obs_test = obs_test @ rand_matrix
        
    # Normalize if requested
    if normalize:
        print("Normalizing observations...")
        max_vals = np.max(np.abs(obs_train), axis=0, keepdims=True)
        # Avoid division by zero
        max_vals = np.where(max_vals < 1e-10, 1.0, max_vals)
        obs_train = obs_train / max_vals
        obs_val = obs_val / max_vals
        obs_test = obs_test / max_vals

    # Package into data dictionary
    data_dict = {
        'inputs_train': jnp.asarray(obs_train),
        'inputs_val': jnp.asarray(obs_val),
        'inputs_test': jnp.asarray(obs_test),
        'states_x_train': jnp.asarray(states_train),
        'states_x_val': jnp.asarray(states_val),
        'states_x_test': jnp.asarray(states_test),
    }

    print(f"Dataset generated successfully!")
    print(f"  Training observations: {obs_train.shape}")
    print(f"  Validation observations: {obs_val.shape}")
    print(f"  Test observations: {obs_test.shape}")
    print(f"  Observation dimension: {obs_train.shape[1]}")

    return data_dict
