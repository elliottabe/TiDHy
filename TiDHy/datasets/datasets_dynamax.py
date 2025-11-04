import numpy as np
import TiDHy.utils.io_dict_to_hdf5 as ioh5
import jax.numpy as jnp
import jax.random
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.utils.utils import random_rotation
from dynamax.slds.inference import ParamsSLDS, LGParamsSLDS, DiscreteParamsSLDS
from dynamax.slds.models import SLDS

def namedtuple_to_dict(obj):
    """Recursively convert nested NamedTuples to dicts."""
    if hasattr(obj, '_asdict'):
        # It's a NamedTuple, convert it and recurse on values
        return {k: namedtuple_to_dict(v) for k, v in obj._asdict().items()}
    elif isinstance(obj, dict):
        # Already a dict, recurse on values
        return {k: namedtuple_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # List or tuple, recurse on elements
        return [namedtuple_to_dict(item) for item in obj]
    else:
        # Base case: return as-is (numbers, strings, arrays, etc.)
        return obj
    
def create_dyn_mat(desired_eigenvalues, key):
    """
    Create dynamics matrix with desired eigenvalues using JAX.

    Args:
        desired_eigenvalues: Array of desired eigenvalues
        key: JAX random key

    Returns:
        A0: Dynamics matrix with desired eigenvalues
    """
    from scipy import signal
    # Create an empty A matrix with the same dimensions as desired_eigenvalues
    A = np.zeros((len(desired_eigenvalues), len(desired_eigenvalues)))
    # Create the system with the desired eigenvalues using pole placement
    key, subkey = jax.random.split(key)
    B = np.array(jax.random.normal(subkey, (len(desired_eigenvalues), len(desired_eigenvalues))))
    # Calculate the state-space representation
    poles = signal.place_poles(A, B, desired_eigenvalues)
    A0 = A - B @ poles.gain_matrix
    return A0

def delay_embedding(signal, delay, skipt=1):
    '''
    Create a delay embedding of the signal.
    signal: (time_bins, obs_dim)
    delay: int
    skipt: skip parameter
    '''
    delayed_sig = np.stack([
        np.hstack((signal[n:, m], np.zeros(n)))
        for m in range(signal.shape[1])
        for n in range(0, skipt*delay, skipt)
    ], axis=1)
    return delayed_sig


def partial_superposition_SLDS(cfg, ssm_params, timescales=None, saved_evals=True):
    """
    Create a dataset of Switching Linear Dynamical Systems (SLDS) with partially superimposed inputs.

    This version uses dynamax with JAX instead of the ssm library.
    SLDS has discrete states (z) that switch between different continuous dynamics.

    Args:
        cfg: Configuration object
        ssm_params: Dictionary of SSM parameters including:
            - Nlds: Number of SLDS systems
            - n_disc_states: Number of discrete states per system
            - latent_dim: Continuous latent state dimension
            - obs_dim: Observation dimension
            - time_bins_train/test: Sequence lengths
            - z_timescale: List of timescales for discrete state transitions
            - normalize: Whether to normalize states (default: False)
            - random_projection: Whether to apply random projection (default: False)
            - partial_sup: Whether to use partial superposition (default: False)
            - partial_obs: Whether to use partial observations (default: False)
            - full_sup: Whether to use full superposition (default: False)
            - seed: Random seed (default: 0)
            - overlap: Number of overlapping observation dimensions (required if partial_sup=True)
            - rand_dim: Dimension for random projection (required if random_projection=True)
        timescales: List of timescales for continuous dynamics (optional)
        saved_evals: Whether to use saved eigenvalues (default: True)

    Returns:
        lds_dict: Dictionary of SLDS models
        data_dict: Dictionary containing generated data including discrete states (z)
    """
    # Extract parameters from ssm_params with defaults
    normalize = ssm_params.get('normalize', False)
    random_projection = ssm_params.get('random_projection', False)
    partial_sup = ssm_params.get('partial_sup', False)
    partial_obs = ssm_params.get('partial_obs', False)
    full_sup = ssm_params.get('full_sup', False)
    seed = ssm_params.get('seed', 0)
    ##### Fast to slow #####
    timescales_default = [-.025, -.25, -.1, -.5, -.015, -.75]
    freq = jnp.zeros(len(timescales_default), dtype=complex)
    freq = freq.at[:].set(freq + 1j * jnp.array([.0075, .05, .025, .075, .01, .075]))
    set_eigvalues = jnp.array([
        [jnp.exp(jnp.array(timescales_default)) + 2*jnp.pi*freq],
        [jnp.exp(jnp.array(timescales_default)) - 2*jnp.pi*freq]
    ]).squeeze().T

    if timescales is None:
        timescales = timescales_default

    # Set the random seed
    filename = 'SLDS_N{}_zD{}_xD{}_yD{}_seed{}.h5'.format(
        ssm_params['Nlds'], ssm_params['n_disc_states'],
        ssm_params['latent_dim'], ssm_params['obs_dim'],
        ssm_params['seed']
    )

    if (cfg.paths.data_dir/filename).exists():
        data_dict = ioh5.load(cfg.paths.data_dir / filename)
        print('Loaded SLDS data from {}'.format(cfg.paths.data_dir/filename))
        cstates_train_all = data_dict['cstates_train_all']
        cstates_val_all = data_dict['cstates_val_all']
        cstates_test_all = data_dict['cstates_test_all']
        dstates_train_all = data_dict['dstates_train_all']
        dstates_val_all = data_dict['dstates_val_all']
        dstates_test_all = data_dict['dstates_test_all']
        emissions_train_all = data_dict['emissions_train_all']
        emissions_val_all = data_dict['emissions_val_all']
        emissions_test_all = data_dict['emissions_test_all']
        As = data_dict['As']
        bs = data_dict['bs']
        lds_dict = data_dict['lds_params']
    else:
        # Make SLDS with interesting dynamics parameters
        lds_dict = {}
        dstates_train_all, cstates_train_all, emissions_train_all = [], [], []
        dstates_val_all, cstates_val_all, emissions_val_all = [], [], []
        dstates_test_all, cstates_test_all, emissions_test_all = [], [], []
        As, bs = [], []
        timescales_default = [-.025, -.25, -.1, -.5, -.015, -.75]
        freq = jnp.zeros(len(timescales_default), dtype=complex)
        freq = freq.at[:].set(freq + 1j * jnp.array([.0075, .05, .025, .075, .01, .075]))
        set_eigvalues = jnp.array([
            [jnp.exp(jnp.array(timescales_default)) + 2*jnp.pi*freq],
            [jnp.exp(jnp.array(timescales_default)) - 2*jnp.pi*freq]
        ]).squeeze().T
        p=0
        timescales = timescales_default
        # Create JAX random key
        key = jax.random.PRNGKey(seed)
            # Setup for sampling discrete states
        num_states = ssm_params['n_disc_states']
        state_dim = ssm_params['latent_dim']
        emission_dim = ssm_params['obs_dim']
        slds = SLDS(num_states, state_dim, emission_dim)
        for i in range(ssm_params['Nlds']):
            key, subkey = jax.random.split(key)

            # Create dynamics matrices for each discrete state
            As_lds, bs_lds = [], []
            Cs_lds, ds_lds = [], []
            Qs_lds, Rs_lds = [], []

            for n in range(ssm_params['n_disc_states']):
                # Create dynamics matrix for this discrete state
                if saved_evals:
                    key, subkey2 = jax.random.split(key)
                    A0 = create_dyn_mat(set_eigvalues[p], subkey2)
                else:
                    neg = True if i % 2 == 0 else False
                    key, subkey2 = jax.random.split(key)
                    theta = jnp.pi/2 * jax.random.uniform(subkey2)/(timescales[p])
                    A0 = .95 * random_rotation(ssm_params['latent_dim'], theta=float(theta), neg=neg)

                key, subkey = jax.random.split(key)
                b = np.array(jax.random.uniform(subkey, (ssm_params['latent_dim'],), minval=-5.0, maxval=5.0))
                As_lds.append(A0)
                bs_lds.append(b)

                # Create emission matrix for this discrete state
                key, subkey = jax.random.split(key)
                C = np.array(jax.random.normal(subkey, (ssm_params['obs_dim'], ssm_params['latent_dim'])))
                d = jnp.zeros(ssm_params['obs_dim'])
                Cs_lds.append(C)
                ds_lds.append(np.array(d))

                # Noise covariances
                Q = jnp.eye(ssm_params['latent_dim']) * 0.1
                R = jnp.eye(ssm_params['obs_dim']) * 0.1
                Qs_lds.append(Q)
                Rs_lds.append(R)

                p += 1
            # Stack to make (n_disc_states, latent_dim, latent_dim) arrays
            As_lds = jnp.stack(As_lds)
            bs_lds = jnp.stack(bs_lds)
            Cs_lds = jnp.stack(Cs_lds)
            ds_lds = jnp.stack(ds_lds)
            Qs_lds = jnp.stack(Qs_lds)
            Rs_lds = jnp.stack(Rs_lds)
            
            # Create transition matrix for discrete states
            K = ssm_params['n_disc_states']
            z_timescale = ssm_params['z_timescale'][i]
            Ps = z_timescale * np.eye(K) + (1 - z_timescale)
            Ps /= Ps.sum(axis=1, keepdims=True)
            print(f"Transition matrix for system {i}:")
            print(Ps)

            # Initial discrete state distribution
            pi0 = np.ones(K) / K

            discr_params = DiscreteParamsSLDS(
                initial_distribution=jnp.ones(num_states)/num_states,
                transition_matrix=jnp.array(Ps),
                proposal_transition_matrix=jnp.array(Ps)
            )

            lg_params = LGParamsSLDS(
                initial_mean=jnp.ones(state_dim),
                initial_cov=jnp.eye(state_dim),
                dynamics_weights=As_lds,
                dynamics_cov=Qs_lds,
                dynamics_bias=bs_lds,
                dynamics_input_weights=None,
                emission_weights=Cs_lds,
                emission_cov=Rs_lds,
                emission_bias=None,
                emission_input_weights=None
            )

            pre_params = ParamsSLDS(
                discrete=discr_params,
                linear_gaussian=lg_params
            )

            params = pre_params.initialize(num_states, state_dim, emission_dim)

            ## Sample states and emissions Train Set
            key, sub_key = jax.random.split(key)

            dstates_train, cstates_train, emissions_train = slds.sample(params, sub_key, ssm_params['time_bins_train'])
            _, counts = np.unique(dstates_train, return_counts=True)
            p_states_x = counts / np.sum(counts)
            print(f'Train {0}, state distribution: {p_states_x}')
            
            ## Sample states and emissions Val Set
            key, sub_key = jax.random.split(key)
            dstates_val, cstates_val, emissions_val = slds.sample(params, sub_key, ssm_params['time_bins_test'])
            _, counts = np.unique(dstates_val, return_counts=True)
            p_states_x = counts / np.sum(counts)
            print(f'Val {0}, state distribution: {p_states_x}')
            
            ## Sample states and emissions Test Set
            key, sub_key = jax.random.split(key)
            dstates_test, cstates_test, emissions_test = slds.sample(params, sub_key, ssm_params['time_bins_test'])
            _, counts = np.unique(dstates_test, return_counts=True)
            p_states_x = counts / np.sum(counts)
            print(f'Test {0}, state distribution: {p_states_x}')

            if normalize:
                cstates_train = cstates_train / np.max(np.abs(cstates_train), axis=0, keepdims=True)
                cstates_val = cstates_val / np.max(np.abs(cstates_val), axis=0, keepdims=True)
                cstates_test = cstates_test / np.max(np.abs(cstates_test), axis=0, keepdims=True)

            cstates_train_all.append(cstates_train)
            cstates_val_all.append(cstates_val)
            cstates_test_all.append(cstates_test)
            dstates_train_all.append(dstates_train)
            dstates_val_all.append(dstates_val)
            dstates_test_all.append(dstates_test)
            emissions_train_all.append(emissions_train)
            emissions_val_all.append(emissions_val)
            emissions_test_all.append(emissions_test)
            As.append(As_lds)
            bs.append(bs_lds)

            lds_dict['{}'.format(i)] = {
                # 'model': hmm,
                'params': namedtuple_to_dict(params),
                'As': As_lds,
                'bs': bs_lds,
                'Cs': Cs_lds,
                'ds': ds_lds
            }

        As = jnp.stack(As)
        bs = jnp.stack(bs)
        dstates_train_all = jnp.stack(dstates_train_all)
        dstates_val_all = jnp.stack(dstates_val_all)
        dstates_test_all = jnp.stack(dstates_test_all)
        cstates_train_all = jnp.stack(cstates_train_all)
        cstates_val_all = jnp.stack(cstates_val_all)
        cstates_test_all = jnp.stack(cstates_test_all)
        emissions_train_all = jnp.stack(emissions_train_all)
        emissions_val_all = jnp.stack(emissions_val_all)
        emissions_test_all = jnp.stack(emissions_test_all)

        data_dict = {
            'lds_params': lds_dict,
            'timescales': timescales,
            'As': As,
            'bs': bs,
            'dstates_train_all': dstates_train_all,
            'dstates_val_all': dstates_val_all,
            'dstates_test_all': dstates_test_all,
            'cstates_train_all': cstates_train_all,
            'cstates_val_all': cstates_val_all,
            'cstates_test_all': cstates_test_all,
            'emissions_train_all': emissions_train_all,
            'emissions_val_all': emissions_val_all,
            'emissions_test_all': emissions_test_all,
        }

        ioh5.save(cfg.paths.data_dir/filename, data_dict)
        print('Saved SLDS data to {}'.format(cfg.paths.data_dir/filename))
        
        ##### Print Saved Data Info #####
        import itertools
        lst = list(itertools.product([0, 1], repeat=3))
        lst2 = list(itertools.product(['F', 'S'], repeat=3))
        full_state_z = np.zeros(ssm_params['time_bins_train'], dtype=int)

        for n in range(len(lst)):
            full_state_z[np.apply_along_axis(
                lambda x: np.all(x == lst[n]), 0, data_dict['dstates_train_all']
            )] = n
        print('Unique train_states:', np.unique(full_state_z, return_counts=True))

        full_state_z = np.zeros(ssm_params['time_bins_test'], dtype=int)
        for n in range(len(lst)):
            full_state_z[np.apply_along_axis(
                lambda x: np.all(x == lst[n]), 0, data_dict['dstates_test_all']
            )] = n
        print('Unique test_states:', np.unique(full_state_z, return_counts=True))
        _, zcounts = np.unique(full_state_z, return_counts=True)
        print("Test", zcounts/np.sum(zcounts))

    # Reset random key for reproducibility
    key = jax.random.PRNGKey(seed+1)

    ##### Partial Superposition #####
    if partial_sup:
        assert ssm_params['overlap'] <= ssm_params['obs_dim'], \
            'Overlap must be less than or equal to the number of observations'

        # Reshape emissions from (n_systems, Time, obs_dim) to (Time, n_systems, obs_dim)
        inputs_train_structured = emissions_train_all.transpose(1, 0, 2)
        inputs_val_structured = emissions_val_all.transpose(1, 0, 2)
        inputs_test_structured = emissions_test_all.transpose(1, 0, 2)

        # For overlapping dimensions: sum across systems using einsum
        # tso -> to means sum over systems (s) for each time (t) and obs dimension (o)
        overlapped_train = np.einsum('tso->to', inputs_train_structured[:, :, :ssm_params['overlap']])
        overlapped_val = np.einsum('tso->to', inputs_val_structured[:, :, :ssm_params['overlap']])
        overlapped_test = np.einsum('tso->to', inputs_test_structured[:, :, :ssm_params['overlap']])

        # For non-overlapping dimensions: keep separate (flatten across systems)
        non_overlapped_train = inputs_train_structured[:, :, ssm_params['overlap']:].reshape(ssm_params['time_bins_train'], -1)
        non_overlapped_val = inputs_val_structured[:, :, ssm_params['overlap']:].reshape(ssm_params['time_bins_test'], -1)
        non_overlapped_test = inputs_test_structured[:, :, ssm_params['overlap']:].reshape(ssm_params['time_bins_test'], -1)

        # Concatenate overlapped and non-overlapped dimensions
        inputs_train = np.concatenate([overlapped_train, non_overlapped_train], axis=1)
        inputs_val = np.concatenate([overlapped_val, non_overlapped_val], axis=1)
        inputs_test = np.concatenate([overlapped_test, non_overlapped_test], axis=1)
        print('Input train shape (partial sup):', inputs_train.shape)
    elif partial_obs:
        # Transpose and remove last observation dimension
        inputs_train = emissions_train_all.transpose(1, 0, 2)[:, :, :-1].reshape(ssm_params['time_bins_train'], -1)
        inputs_test = emissions_test_all.transpose(1, 0, 2)[:, :, :-1].reshape(ssm_params['time_bins_test'], -1)
        inputs_val = emissions_val_all.transpose(1, 0, 2)[:, :, :-1].reshape(ssm_params['time_bins_test'], -1)
        print('Input train shape (partial_obs):', inputs_train.shape)

    elif full_sup:
        # Sum across all systems and all observation dimensions
        inputs_train = np.sum(emissions_train_all, axis=(0, -1))[:, np.newaxis]
        inputs_test = np.sum(emissions_test_all, axis=(0, -1))[:, np.newaxis]
        inputs_val = np.sum(emissions_val_all, axis=(0, -1))[:, np.newaxis]
        print('Input train shape (full sup):', inputs_train.shape)

    else:
        # No mixing, just flatten (Time, n_systems, obs_dim) -> (Time, n_systems * obs_dim)
        inputs_train = emissions_train_all.transpose(1, 0, 2).reshape(ssm_params['time_bins_train'], -1)
        inputs_test = emissions_test_all.transpose(1, 0, 2).reshape(ssm_params['time_bins_test'], -1)
        inputs_val = emissions_val_all.transpose(1, 0, 2).reshape(ssm_params['time_bins_test'], -1)
        print('Input train shape (no sup):', inputs_train.shape)

    if random_projection:
        key, subkey = jax.random.split(key)
        RandMat = np.array(jax.random.normal(subkey, (inputs_train.shape[-1], ssm_params['rand_dim'])))
        inputs_train = inputs_train @ RandMat
        inputs_test = inputs_test @ RandMat
        inputs_val = inputs_val @ RandMat
        print('Input train shape (random projection):', inputs_train.shape)

    if normalize:
        inputs_train = inputs_train / np.max(np.abs(inputs_train), axis=0, keepdims=True)
        inputs_test = inputs_test / np.max(np.abs(inputs_test), axis=0, keepdims=True)
        inputs_val = inputs_val / np.max(np.abs(inputs_val), axis=0, keepdims=True)

    if cfg.delay.delay_embed:
        inputs_train = delay_embedding(inputs_train, cfg.delay.delay_tau, skipt=cfg.delay.skipt)
        inputs_test = delay_embedding(inputs_test, cfg.delay.delay_tau, skipt=cfg.delay.skipt)
        inputs_val = delay_embedding(inputs_val, cfg.delay.delay_tau, skipt=cfg.delay.skipt)

    # Assign states from the SLDS samples
    # Discrete states (z) and continuous states (x) come from the stored data_dict or from the generated samples
    states_z = dstates_train_all
    states_z_test = dstates_test_all
    states_z_val = dstates_val_all
    states_x = cstates_train_all
    states_x_test = cstates_test_all
    states_x_val = cstates_val_all

    data_dict = {
        'states_z': jnp.asarray(states_z),
        'states_z_test': jnp.asarray(states_z_test),
        'states_z_val': jnp.asarray(states_z_val),
        'states_x': jnp.asarray(states_x),
        'states_x_test': jnp.asarray(states_x_test),
        'states_x_val': jnp.asarray(states_x_val),
        'inputs_train': jnp.asarray(inputs_train),
        'inputs_test': jnp.asarray(inputs_test),
        'inputs_val': jnp.asarray(inputs_val),
        'As': jnp.asarray(As),
        'bs': jnp.asarray(bs),
        'timescales': jnp.asarray(timescales),
    }
    print('x Timescales: {}, z Timescales: {}'.format(
        data_dict['timescales'], ssm_params['z_timescale']
    ))
    return lds_dict, data_dict


def _simulate_slds_continuous_states(z_seq, As, bs, Qs, mu0, Sigma0, key):
    """
    Simulate continuous states given discrete state sequence.

    Args:
        z_seq: Discrete state sequence (T,)
        As: List of dynamics matrices for each discrete state
        bs: List of bias vectors for each discrete state
        Qs: List of process noise covariances
        mu0: Initial mean
        Sigma0: Initial covariance
        key: JAX random key

    Returns:
        x_seq: Continuous state sequence (T, state_dim)
    """
    T = len(z_seq)
    state_dim = As[0].shape[0]
    x_seq = np.zeros((T, state_dim))

    # Sample initial state
    key, subkey = jax.random.split(key)
    x_seq[0] = mu0 + jax.random.normal(subkey, (state_dim,)) @ jnp.linalg.cholesky(Sigma0).T
    x_seq[0] = np.array(x_seq[0])

    # Simulate forward
    for t in range(1, T):
        z_prev = int(z_seq[t-1])
        A = As[z_prev]
        b = bs[z_prev]
        Q = Qs[z_prev]

        key, subkey = jax.random.split(key)
        noise = np.array(jax.random.multivariate_normal(subkey, jnp.zeros(state_dim), jnp.array(Q)))
        x_seq[t] = A @ x_seq[t-1] + b + noise

    return x_seq


def _simulate_slds_observations(x_seq, z_seq, Cs, ds, Rs, key):
    """
    Simulate observations given continuous and discrete state sequences.

    Args:
        x_seq: Continuous state sequence (T, state_dim)
        z_seq: Discrete state sequence (T,)
        Cs: List of emission matrices for each discrete state
        ds: List of emission biases for each discrete state
        Rs: List of observation noise covariances
        key: JAX random key

    Returns:
        y_seq: Observation sequence (T, emission_dim)
    """
    T = len(z_seq)
    emission_dim = Cs[0].shape[0]
    y_seq = np.zeros((T, emission_dim))

    for t in range(T):
        z_t = int(z_seq[t])
        C = Cs[z_t]
        d = ds[z_t]
        R = Rs[z_t]

        key, subkey = jax.random.split(key)
        noise = np.array(jax.random.multivariate_normal(subkey, jnp.zeros(emission_dim), jnp.array(R)))
        y_seq[t] = C @ x_seq[t] + d + noise

    return y_seq
