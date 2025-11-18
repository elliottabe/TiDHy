"""
SLDS Timescale Extraction and Analysis

This module provides tools for extracting and analyzing timescales from
Switching Linear Dynamical Systems (SLDS) models.

Author: Elliott Abe
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import h5py

from .analysis import compute_timescale_spectrum, cluster_timescales
import TiDHy.utils.io_dict_to_hdf5 as ioh5

def load_slds_results(filepath: Path, model_type: str = 'SLDS') -> Dict[str, Any]:
    """
    Load SLDS or rSLDS results from HDF5 file.

    Args:
        filepath: Path to HDF5 file with SLDS results
        model_type: 'SLDS' or 'rSLDS'

    Returns:
        Dictionary with model outputs and parameters
    """
    prefix = model_type

    results = ioh5.load(filepath)
    
    return results


def extract_slds_timescales(
    dynamics_matrices: np.ndarray,
    dt: float = 1.0
) -> Dict[str, Any]:
    """
    Extract timescales from SLDS dynamics matrices using eigenvalue analysis.

    Each discrete state k has its own dynamics matrix A_k that governs the
    continuous latent evolution: x_{t+1} = A_k @ x_t + b_k

    Args:
        dynamics_matrices: Array of shape [K, D, D] where K is number of discrete
                          states and D is continuous latent dimension
        dt: Time step size (default: 1.0)

    Returns:
        Dictionary with timescale information per state:
        - state_spectra: List of spectrum dicts for each state (from compute_timescale_spectrum)
        - state_timescales: List of timescale arrays for each state (only stable modes)
        - all_timescales: Concatenated array of all timescales across states
        - num_states: Number of discrete states K
        - latent_dim: Dimension of continuous latents D
    """
    K, D, _ = dynamics_matrices.shape

    state_spectra = []
    state_timescales = []

    for k in range(K):
        A_k = jnp.array(dynamics_matrices[k])
        spectrum = compute_timescale_spectrum(A_k, dt=dt)
        state_spectra.append(spectrum)

        # Extract only finite timescales (stable modes)
        timescales_k = spectrum['timescales']
        finite_timescales = timescales_k[jnp.isfinite(timescales_k)]
        state_timescales.append(np.array(finite_timescales))

    # Concatenate all timescales
    all_timescales = np.concatenate([ts for ts in state_timescales if len(ts) > 0])

    return {
        'state_spectra': state_spectra,
        'state_timescales': state_timescales,
        'all_timescales': all_timescales,
        'num_states': K,
        'latent_dim': D,
        'dt': dt,
    }


def compute_state_duration_timescales(
    transition_matrix: np.ndarray,
    dt: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Compute expected state duration timescales from transition matrix.

    The expected duration in state k (before transitioning out) is:
    τ_k = dt / (1 - P_kk)

    where P_kk is the self-transition probability.

    Args:
        transition_matrix: Transition probability matrix [K, K] where
                          P[i,j] = P(state_t = j | state_{t-1} = i)
        dt: Time step size (default: 1.0)

    Returns:
        Dictionary with:
        - state_durations: Expected duration in each state [K]
        - self_transition_probs: Diagonal elements P_kk [K]
        - num_states: Number of discrete states K
    """
    K = transition_matrix.shape[0]

    # Get self-transition probabilities (diagonal)
    self_probs = np.diag(transition_matrix)

    # Compute expected durations: τ = dt / (1 - p_kk)
    # Clip probabilities to avoid division by zero
    safe_probs = np.clip(self_probs, 0.0, 0.9999)
    state_durations = dt / (1.0 - safe_probs)

    return {
        'state_durations': state_durations,
        'self_transition_probs': self_probs,
        'num_states': K,
    }


def analyze_slds_state_usage(
    discrete_states: np.ndarray,
    num_states: Optional[int] = None
) -> Dict[str, Any]:
    """
    Analyze discrete state usage patterns in SLDS trajectory.

    Args:
        discrete_states: Array of discrete state indices [T]
        num_states: Total number of possible states (if None, inferred from data)

    Returns:
        Dictionary with:
        - state_counts: Number of timesteps in each state [K]
        - state_fractions: Fraction of time in each state [K]
        - state_switches: Total number of state transitions
        - switch_rate: State transitions per timestep
        - entropy: Shannon entropy of state distribution
        - active_states: States that were visited at least once
    """
    if num_states is None:
        num_states = int(discrete_states.max()) + 1

    T = len(discrete_states)

    # Count state occurrences
    state_counts = np.bincount(discrete_states, minlength=num_states)
    state_fractions = state_counts / T

    # Count state switches
    switches = np.sum(discrete_states[1:] != discrete_states[:-1])
    switch_rate = switches / (T - 1)

    # Compute Shannon entropy
    nonzero_fracs = state_fractions[state_fractions > 0]
    entropy = -np.sum(nonzero_fracs * np.log(nonzero_fracs))

    # Find active states
    active_states = np.where(state_counts > 0)[0]

    return {
        'state_counts': state_counts,
        'state_fractions': state_fractions,
        'state_switches': int(switches),
        'switch_rate': float(switch_rate),
        'entropy': float(entropy),
        'active_states': active_states,
        'num_active_states': len(active_states),
    }


def compute_effective_slds_timescales(
    dynamics_matrices: np.ndarray,
    discrete_states: np.ndarray,
    dt: float = 1.0
) -> Dict[str, Any]:
    """
    Compute effective timescales weighted by state usage.

    This combines:
    1. Within-state continuous dynamics timescales (from A_k eigenvalues)
    2. State usage frequencies (how often each state is active)

    Args:
        dynamics_matrices: Dynamics matrices [K, D, D]
        discrete_states: Discrete state trajectory [T]
        dt: Time step size

    Returns:
        Dictionary with weighted timescale information
    """
    # Extract timescales per state
    timescale_info = extract_slds_timescales(dynamics_matrices, dt=dt)

    # Analyze state usage
    usage_info = analyze_slds_state_usage(discrete_states, num_states=timescale_info['num_states'])

    # Weight timescales by state usage
    weighted_timescales = []
    state_weights = []

    for k, ts_k in enumerate(timescale_info['state_timescales']):
        if len(ts_k) > 0:
            weight = usage_info['state_fractions'][k]
            weighted_timescales.extend(ts_k)
            state_weights.extend([weight] * len(ts_k))

    weighted_timescales = np.array(weighted_timescales)
    state_weights = np.array(state_weights)

    # Compute statistics
    if len(weighted_timescales) > 0:
        mean_timescale = np.average(weighted_timescales, weights=state_weights)
        median_timescale = np.median(weighted_timescales)
        min_timescale = np.min(weighted_timescales)
        max_timescale = np.max(weighted_timescales)
    else:
        mean_timescale = median_timescale = min_timescale = max_timescale = np.nan

    return {
        'weighted_timescales': weighted_timescales,
        'state_weights': state_weights,
        'state_usage': usage_info,
        'mean_timescale': float(mean_timescale),
        'median_timescale': float(median_timescale),
        'min_timescale': float(min_timescale),
        'max_timescale': float(max_timescale),
        **timescale_info,
    }


def analyze_slds_comprehensive(
    slds_results: Dict[str, Any],
    dt: float = 1.0,
    cluster_tolerance: float = 0.2
) -> Dict[str, Any]:
    """
    Comprehensive timescale analysis for SLDS model.

    This performs multiple types of timescale extraction:
    1. Per-state dynamics timescales (eigenvalue analysis)
    2. State duration timescales (from transition probabilities)
    3. Effective timescales (weighted by state usage)
    4. Clustered timescales (grouping similar timescales)

    Args:
        slds_results: Dictionary from load_slds_results()
        dt: Time step size
        cluster_tolerance: Relative tolerance for clustering timescales

    Returns:
        Comprehensive dictionary with all timescale analyses
    """
    params = slds_results['model_params']

    # 1. Dynamics timescales
    dynamics_analysis = extract_slds_timescales(params['dynamics_matrices'], dt=dt)

    # 2. State duration timescales
    duration_analysis = compute_state_duration_timescales(params['transition_matrix'], dt=dt)

    # 3. Effective timescales (weighted by usage)
    effective_analysis = compute_effective_slds_timescales(
        params['dynamics_matrices'],
        slds_results['SLDS_states'],
        dt=dt
    )

    # 4. Cluster all timescales
    all_ts = dynamics_analysis['all_timescales']
    if len(all_ts) > 0:
        clustered = cluster_timescales(all_ts)
    else:
        clustered = {
            'cluster_centers': np.array([]),
            'cluster_counts': np.array([]),
            'num_clusters': 0,
        }

    return {
        'dynamics_timescales': dynamics_analysis,
        'state_durations': duration_analysis,
        'effective_timescales': effective_analysis,
        'clustered_timescales': clustered,
        'dt': dt,
        'summary': {
            'num_dynamics_timescales': len(all_ts),
            'num_clustered_timescales': clustered['num_clusters'],
            'min_dynamics_timescale': float(np.min(all_ts)) if len(all_ts) > 0 else np.nan,
            'max_dynamics_timescale': float(np.max(all_ts)) if len(all_ts) > 0 else np.nan,
            'mean_state_duration': float(np.mean(duration_analysis['state_durations'])),
            'max_state_duration': float(np.max(duration_analysis['state_durations'])),
        }
    }


def compare_slds_to_ground_truth(
    slds_analysis: Dict[str, Any],
    ground_truth_timescales: np.ndarray,
    tolerance: float = 0.2
) -> Dict[str, Any]:
    """
    Compare SLDS-discovered timescales to ground truth.

    Uses the Hungarian algorithm to find optimal matching between discovered
    and ground truth timescales.

    Args:
        slds_analysis: Output from analyze_slds_comprehensive()
        ground_truth_timescales: Array of true timescales
        tolerance: Relative error tolerance for matching (default: 20%)

    Returns:
        Dictionary with comparison metrics (recovery rate, precision, recall, etc.)
    """
    from .analysis import compare_discovered_to_ground_truth

    # Use effective timescales for comparison (most representative)
    discovered = slds_analysis['effective_timescales']['weighted_timescales']

    if len(discovered) == 0:
        return {
            'recovery_rate': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'mean_relative_error': np.nan,
            'num_discovered': 0,
            'num_ground_truth': len(ground_truth_timescales),
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': len(ground_truth_timescales),
        }

    return compare_discovered_to_ground_truth(
        discovered,
        ground_truth_timescales,
        tolerance=tolerance
    )
