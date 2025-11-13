"""
Analysis utilities for TiDHy model selection and interpretation.

This module provides functions for:
- Effective dimensionality analysis of latent states
- Spectral analysis of temporal dynamics to discover timescales
- Hypernetwork mixture component usage analysis
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Any, Optional
from scipy import stats


# ============================================================================
# Utility Functions
# ============================================================================

def is_valid_array(arr: jnp.ndarray, allow_inf: bool = False) -> bool:
    """
    Check if array contains valid numeric values (no NaN, optionally no Inf).

    Args:
        arr: Array to check
        allow_inf: If False, treat Inf as invalid (default: False)

    Returns:
        True if array is valid, False otherwise
    """
    has_nan = jnp.any(jnp.isnan(arr))
    if allow_inf:
        return not has_nan
    else:
        has_inf = jnp.any(jnp.isinf(arr))
        return not (has_nan or has_inf)


# ============================================================================
# Effective Dimension Analysis
# ============================================================================

def compute_effective_dimension(
    eigenvalues: jnp.ndarray,
    threshold: float = 0.95
) -> Tuple[int, jnp.ndarray]:
    """
    Compute effective dimension using eigenvalue spectrum.

    The effective dimension is the number of eigenvalues needed to explain
    a threshold fraction of the total variance.

    Args:
        eigenvalues: Eigenvalues (sorted in descending order)
        threshold: Cumulative variance threshold (default: 0.95 for 95%)

    Returns:
        - effective_dim: Number of dimensions explaining threshold variance
        - variance_explained: Cumulative variance ratio for each dimension
    """
    # Check for invalid inputs
    if not is_valid_array(eigenvalues, allow_inf=False):
        # Return defaults for invalid data
        n_dims = len(eigenvalues)
        return 0, jnp.zeros(n_dims)

    # Ensure eigenvalues are positive and sorted
    eigenvalues = jnp.abs(eigenvalues)
    eigenvalues = jnp.sort(eigenvalues)[::-1]

    # Compute cumulative variance
    total_variance = jnp.sum(eigenvalues)

    # Handle degenerate case (all eigenvalues are zero or near-zero)
    if total_variance < 1e-10 or jnp.isnan(total_variance):
        n_dims = len(eigenvalues)
        return 0, jnp.zeros(n_dims)

    variance_explained = jnp.cumsum(eigenvalues) / total_variance

    # Check for NaN in variance_explained
    if not is_valid_array(variance_explained, allow_inf=False):
        n_dims = len(eigenvalues)
        return 0, jnp.zeros(n_dims)

    # Find effective dimension
    effective_dim = jnp.searchsorted(variance_explained, threshold) + 1

    # Ensure effective_dim is a valid integer
    effective_dim = int(jnp.clip(effective_dim, 0, len(eigenvalues)))

    return effective_dim, variance_explained


def participation_ratio(eigenvalues: jnp.ndarray) -> float:
    """
    Compute participation ratio (Gao & Ganguli, 2015).

    PR = (Σ λᵢ)² / Σ(λᵢ²)

    Higher PR indicates more distributed representation across dimensions.
    PR ranges from 1 (single dimension) to N (uniform across N dimensions).

    Args:
        eigenvalues: Eigenvalues from covariance or correlation matrix

    Returns:
        Participation ratio (scalar)
    """
    # Check for invalid inputs
    if not is_valid_array(eigenvalues, allow_inf=False):
        return 0.0

    eigenvalues = jnp.abs(eigenvalues)
    sum_eig = jnp.sum(eigenvalues)
    sum_eig_sq = jnp.sum(eigenvalues ** 2)

    # Handle degenerate cases
    if sum_eig < 1e-10 or jnp.isnan(sum_eig):
        return 0.0

    pr = (sum_eig ** 2) / (sum_eig_sq + 1e-10)

    # Check result validity
    if jnp.isnan(pr) or jnp.isinf(pr):
        return 0.0

    return float(pr)


def analyze_latent_dimension(
    R: jnp.ndarray,
    method: str = 'pca',
    variance_thresholds: Tuple[float, ...] = (0.90, 0.95, 0.99)
) -> Dict[str, Any]:
    """
    Analyze effective dimensionality of latent trajectories.

    Args:
        R: Latent states - shape (n_timesteps, r_dim) or (batch, T, r_dim)
        method: Analysis method:
            - 'pca': Principal component analysis (default)
            - 'covariance': Covariance matrix eigenvalues
            - 'correlation': Correlation matrix eigenvalues
        variance_thresholds: Variance thresholds to compute effective dims for

    Returns:
        Dictionary containing:
        - effective_dim_XX: Effective dimension for each threshold
        - participation_ratio: PR metric
        - eigenvalues: Full eigenvalue spectrum
        - variance_explained: Cumulative variance explained
        - method: Analysis method used
    """
    # Check for invalid input data
    if not is_valid_array(R, allow_inf=False):
        r_dim = R.shape[-1]
        # Return default results for invalid data
        results = {
            'participation_ratio': 0.0,
            'eigenvalues': jnp.zeros(r_dim),
            'eigenvectors': jnp.eye(r_dim),
            'method': method,
            'total_dims': r_dim,
            'variance_explained': jnp.zeros(r_dim)
        }
        for threshold in variance_thresholds:
            key = f'effective_dim_{int(threshold*100)}'
            results[key] = 0
        return results

    # Reshape if batched
    if R.ndim == 3:
        R = R.reshape(-1, R.shape[-1])  # (batch*T, r_dim)

    # Center the data
    R_centered = R - jnp.mean(R, axis=0, keepdims=True)

    # Check if data has any variance
    if not is_valid_array(R_centered, allow_inf=False):
        r_dim = R.shape[-1]
        results = {
            'participation_ratio': 0.0,
            'eigenvalues': jnp.zeros(r_dim),
            'eigenvectors': jnp.eye(r_dim),
            'method': method,
            'total_dims': r_dim,
            'variance_explained': jnp.zeros(r_dim)
        }
        for threshold in variance_thresholds:
            key = f'effective_dim_{int(threshold*100)}'
            results[key] = 0
        return results

    if method in ['pca', 'covariance']:
        # Covariance matrix
        cov_matrix = jnp.cov(R_centered.T)
        eigenvalues, eigenvectors = jnp.linalg.eigh(cov_matrix)

    elif method == 'correlation':
        # Correlation matrix
        corr_matrix = jnp.corrcoef(R_centered.T)
        eigenvalues, eigenvectors = jnp.linalg.eigh(corr_matrix)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Check if eigenvalues are valid
    if not is_valid_array(eigenvalues, allow_inf=False):
        r_dim = R.shape[-1]
        results = {
            'participation_ratio': 0.0,
            'eigenvalues': jnp.zeros(r_dim),
            'eigenvectors': jnp.eye(r_dim),
            'method': method,
            'total_dims': r_dim,
            'variance_explained': jnp.zeros(r_dim)
        }
        for threshold in variance_thresholds:
            key = f'effective_dim_{int(threshold*100)}'
            results[key] = 0
        return results

    # Sort eigenvalues in descending order
    idx = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Compute metrics
    pr = participation_ratio(eigenvalues)

    # Compute effective dimensions for each threshold
    results = {
        'participation_ratio': pr,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'method': method,
        'total_dims': R.shape[-1]
    }

    for threshold in variance_thresholds:
        eff_dim, var_exp = compute_effective_dimension(eigenvalues, threshold)
        key = f'effective_dim_{int(threshold*100)}'
        results[key] = eff_dim
        if threshold == variance_thresholds[0]:  # Store variance explained once
            results['variance_explained'] = var_exp

    return results


# ============================================================================
# Spectral Timescale Analysis
# ============================================================================

def compute_timescale_spectrum(
    V: jnp.ndarray,
    dt: float = 1.0
) -> Dict[str, jnp.ndarray]:
    """
    Compute eigenvalue spectrum of temporal dynamics matrix to discover timescales.

    For linear dynamics: r_{t+1} = V @ r_t
    Eigenvalues λ determine stability and timescales:
    - |λ| < 1: Stable (decaying mode)
    - |λ| > 1: Unstable (growing mode)
    - Timescale: τ = -dt / log|λ| (for |λ| < 1)
    - Frequency: f = angle(λ) / (2π dt) (for complex λ)

    Args:
        V: Temporal dynamics matrix (r_dim, r_dim)
        dt: Time step size (default: 1.0)

    Returns:
        Dictionary with:
        - eigenvalues: Complex eigenvalues
        - magnitudes: |λᵢ|
        - timescales: τᵢ = -dt / log|λᵢ| (only for stable modes |λ| < 1)
        - frequencies: fᵢ = angle(λᵢ) / (2π dt)
        - is_stable: Boolean array indicating |λᵢ| < 1
        - decay_rates: -log|λᵢ| / dt (inverse timescales)
    """
    # Check for invalid input matrix
    if not is_valid_array(V, allow_inf=False):
        r_dim = V.shape[0]
        return {
            'eigenvalues': jnp.zeros(r_dim, dtype=complex),
            'magnitudes': jnp.zeros(r_dim),
            'timescales': jnp.full(r_dim, jnp.inf),
            'frequencies': jnp.zeros(r_dim),
            'is_stable': jnp.zeros(r_dim, dtype=bool),
            'decay_rates': jnp.zeros(r_dim)
        }

    # Compute eigenvalues
    eigenvalues = jnp.linalg.eigvals(V)

    # Check if eigenvalues are valid (can contain NaN from degenerate matrices)
    if jnp.any(jnp.isnan(eigenvalues)):
        r_dim = V.shape[0]
        return {
            'eigenvalues': jnp.zeros(r_dim, dtype=complex),
            'magnitudes': jnp.zeros(r_dim),
            'timescales': jnp.full(r_dim, jnp.inf),
            'frequencies': jnp.zeros(r_dim),
            'is_stable': jnp.zeros(r_dim, dtype=bool),
            'decay_rates': jnp.zeros(r_dim)
        }

    # Magnitudes
    magnitudes = jnp.abs(eigenvalues)

    # Check for NaN in magnitudes
    if jnp.any(jnp.isnan(magnitudes)):
        r_dim = V.shape[0]
        return {
            'eigenvalues': eigenvalues,
            'magnitudes': jnp.zeros(r_dim),
            'timescales': jnp.full(r_dim, jnp.inf),
            'frequencies': jnp.zeros(r_dim),
            'is_stable': jnp.zeros(r_dim, dtype=bool),
            'decay_rates': jnp.zeros(r_dim)
        }

    # Stability check
    is_stable = magnitudes < 1.0

    # Timescales (only for stable modes)
    # τ = -dt / log|λ| for |λ| < 1
    # Use small epsilon to avoid log(0) or log(values close to 1)
    safe_magnitudes = jnp.clip(magnitudes, 1e-10, 0.9999)
    timescales = -dt / jnp.log(safe_magnitudes)
    # Set infinite timescales for unstable modes
    timescales = jnp.where(is_stable, timescales, jnp.inf)

    # Replace any NaN timescales with inf
    timescales = jnp.where(jnp.isnan(timescales), jnp.inf, timescales)

    # Decay rates (inverse of timescales)
    decay_rates = -jnp.log(safe_magnitudes) / dt

    # Replace any NaN decay rates with 0
    decay_rates = jnp.where(jnp.isnan(decay_rates), 0.0, decay_rates)

    # Frequencies (for oscillatory modes)
    angles = jnp.angle(eigenvalues)
    frequencies = angles / (2 * jnp.pi * dt)

    # Replace any NaN frequencies with 0
    frequencies = jnp.where(jnp.isnan(frequencies), 0.0, frequencies)

    return {
        'eigenvalues': eigenvalues,
        'magnitudes': magnitudes,
        'timescales': timescales,
        'frequencies': frequencies,
        'is_stable': is_stable,
        'decay_rates': decay_rates
    }


def analyze_mixture_timescales(
    model,
    dt: float = 1.0,
    stability_threshold: float = 1.0
) -> Dict[str, Any]:
    """
    Analyze timescales across all mixture components V_m in trained TiDHy model.

    Args:
        model: Trained TiDHy model with temporal.value parameter
        dt: Time step size (matches data sampling rate)
        stability_threshold: Magnitude threshold for stable eigenvalues (default: 1.0)

    Returns:
        Dictionary containing:
        - timescales_per_component: List of timescale arrays, one per mixture component
        - all_timescales: Concatenated array of all finite timescales
        - unique_timescales: Clustered unique timescales
        - timescale_range: (min, max) of discovered timescales
        - n_stable_modes: Number of stable eigenvalues per component
        - spectral_results: Full spectral analysis per component
    """
    # Extract temporal matrices
    temporal = model.temporal.value  # Shape: (mix_dim, r_dim * r_dim)
    mix_dim = model.mix_dim
    r_dim = model.r_dim

    V_matrices = temporal.reshape(mix_dim, r_dim, r_dim)

    # Analyze each component
    spectral_results = []
    timescales_per_component = []
    all_timescales = []

    for m in range(mix_dim):
        V_m = V_matrices[m]
        spectrum = compute_timescale_spectrum(V_m, dt)
        spectral_results.append(spectrum)

        # Extract finite stable timescales
        finite_mask = jnp.isfinite(spectrum['timescales']) & spectrum['is_stable'] & (~jnp.isnan(spectrum['timescales']))
        component_timescales = spectrum['timescales'][finite_mask]
        timescales_per_component.append(component_timescales)

        if len(component_timescales) > 0:
            all_timescales.append(component_timescales)

    # Combine all timescales
    if len(all_timescales) > 0:
        all_timescales = jnp.concatenate(all_timescales)
        timescale_range = (float(jnp.min(all_timescales)), float(jnp.max(all_timescales)))

        # Cluster to find unique timescales (simple approach: log-space binning)
        unique_timescales = cluster_timescales(all_timescales)
    else:
        all_timescales = jnp.array([])
        timescale_range = (None, None)
        unique_timescales = jnp.array([])

    # Count stable modes per component
    n_stable_modes = []
    for spec in spectral_results:
        n_stable = jnp.sum(spec['is_stable'])
        if jnp.isnan(n_stable) or jnp.isinf(n_stable):
            n_stable_modes.append(0)
        else:
            n_stable_modes.append(int(n_stable))

    return {
        'timescales_per_component': timescales_per_component,
        'all_timescales': all_timescales,
        'unique_timescales': unique_timescales,
        'timescale_range': timescale_range,
        'n_stable_modes': n_stable_modes,
        'spectral_results': spectral_results,
        'mix_dim': mix_dim,
        'r_dim': r_dim
    }


def cluster_timescales(
    timescales: jnp.ndarray,
    n_clusters: Optional[int] = None,
    log_space: bool = True
) -> jnp.ndarray:
    """
    Cluster timescales to identify unique temporal scales.

    Uses log-space binning to group similar timescales, since timescales
    often span multiple orders of magnitude.

    Args:
        timescales: Array of timescales
        n_clusters: Number of clusters (auto-determined if None)
        log_space: Cluster in log space (recommended for timescales)

    Returns:
        Array of representative timescales (cluster centers)
    """
    if len(timescales) == 0:
        return jnp.array([])

    # Filter out invalid timescales (NaN and Inf)
    valid_mask = jnp.isfinite(timescales)
    valid_timescales = timescales[valid_mask]

    # Return empty if no valid timescales
    if len(valid_timescales) == 0:
        return jnp.array([])

    if log_space:
        # Use log10 space for clustering
        log_timescales = jnp.log10(valid_timescales + 1e-10)

        # Check if log_timescales has any invalid values
        if not is_valid_array(log_timescales, allow_inf=False):
            return jnp.array([])

        # Auto-determine clusters using Freedman-Diaconis rule
        if n_clusters is None:
            q75 = jnp.percentile(log_timescales, 75)
            q25 = jnp.percentile(log_timescales, 25)
            iqr = q75 - q25

            # Check for degenerate case (all values identical)
            if iqr < 1e-10 or jnp.isnan(iqr):
                # Return single representative value
                return jnp.array([jnp.median(valid_timescales)])

            bin_width = 2 * iqr / (len(log_timescales) ** (1/3))

            # Check if bin_width is valid
            if bin_width < 1e-10 or jnp.isnan(bin_width):
                return jnp.array([jnp.median(valid_timescales)])

            # Compute range
            log_range = jnp.max(log_timescales) - jnp.min(log_timescales)

            # Check if range is valid
            if jnp.isnan(log_range) or log_range < 1e-10:
                return jnp.array([jnp.median(valid_timescales)])

            n_clusters_computed = (log_range / bin_width)
            if jnp.isnan(n_clusters_computed) or jnp.isinf(n_clusters_computed):
                n_clusters = 1
            else:
                n_clusters = max(int(n_clusters_computed), 1)
            n_clusters = min(n_clusters, len(valid_timescales) // 2)  # Cap at half the data
            n_clusters = max(n_clusters, 1)  # At least 1 cluster

        # Create histogram to find cluster centers
        hist, bin_edges = jnp.histogram(log_timescales, bins=n_clusters)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Return only non-empty bins
        non_empty = hist > 0
        unique_log_timescales = bin_centers[non_empty]
        unique_timescales = 10 ** unique_log_timescales
    else:
        # Linear space clustering
        if n_clusters is None:
            n_clusters_computed = len(valid_timescales) ** 0.5
            if jnp.isnan(n_clusters_computed) or jnp.isinf(n_clusters_computed):
                n_clusters = 1
            else:
                n_clusters = max(int(n_clusters_computed), 1)

        hist, bin_edges = jnp.histogram(valid_timescales, bins=n_clusters)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        non_empty = hist > 0
        unique_timescales = bin_centers[non_empty]

    return unique_timescales


def fit_power_law_timescales(
    timescales: jnp.ndarray,
    plot: bool = False
) -> Tuple[float, float]:
    """
    Fit power law to timescale distribution: P(τ) ~ τ^(-α)

    Many complex systems exhibit power-law distributed timescales,
    indicating scale-free temporal dynamics.

    Args:
        timescales: Array of timescales
        plot: Whether to create diagnostic plot (requires matplotlib)

    Returns:
        - alpha: Power law exponent
        - fit_quality: R² of log-log fit
    """
    if len(timescales) < 3:
        return float('nan'), float('nan')

    # Filter out invalid timescales (NaN and Inf)
    valid_mask = jnp.isfinite(timescales)
    valid_timescales = timescales[valid_mask]

    # Check if enough valid timescales remain
    if len(valid_timescales) < 3:
        return float('nan'), float('nan')

    # Convert to numpy for scipy
    timescales_np = np.array(valid_timescales)

    # Create histogram (PDF)
    hist, bin_edges = np.histogram(timescales_np, bins=20, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Remove zero bins for log-log fit
    nonzero = hist > 0
    if np.sum(nonzero) < 3:
        return float('nan'), float('nan')

    x = bin_centers[nonzero]
    y = hist[nonzero]

    # Fit in log-log space: log(P) = -α*log(τ) + C
    log_x = np.log10(x)
    log_y = np.log10(y)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
    alpha = -slope
    r_squared = r_value ** 2

    if plot:
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Linear histogram
            ax1.hist(timescales_np, bins=30, density=True, alpha=0.7)
            ax1.set_xlabel('Timescale τ')
            ax1.set_ylabel('Probability Density')
            ax1.set_title('Timescale Distribution')

            # Log-log plot with fit
            ax2.scatter(log_x, log_y, alpha=0.6, label='Data')
            fit_line = slope * log_x + intercept
            ax2.plot(log_x, fit_line, 'r--', label=f'Fit: α={alpha:.2f}, R²={r_squared:.3f}')
            ax2.set_xlabel('log₁₀(τ)')
            ax2.set_ylabel('log₁₀(P(τ))')
            ax2.set_title('Power Law Fit')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Matplotlib not available for plotting")

    return float(alpha), float(r_squared)


def analyze_effective_timescales(
    result_dict: Dict[str, jnp.ndarray],
    dt: float = 1.0,
    stability_threshold: float = 1.0
) -> Dict[str, Any]:
    """
    Analyze timescales of actual effective temporal matrices V_t used during inference.

    The model uses time-varying temporal matrices V_t = sum_m w_m(r2_t) * V_m,
    where w_m are hypernetwork weights. This function analyzes the eigenvalue
    spectrum of these actual matrices rather than the basis matrices V_m.

    Args:
        result_dict: Output from evaluate_record() containing:
            - 'Ut': Temporal matrices V_t at each timestep, shape (batch, T, r_dim, r_dim)
            - 'W': Hypernetwork weights, shape (batch, T, mix_dim)
        dt: Time step size (matches data sampling rate)
        stability_threshold: Magnitude threshold for stable eigenvalues (default: 1.0)

    Returns:
        Dictionary containing:
        - all_timescales: All finite stable timescales from all V_t matrices
        - timescale_stats: Statistics (mean, median, std, min, max) of timescales
        - unique_timescales: Clustered unique timescales
        - timescale_range: (min, max) of discovered timescales
        - n_stable_per_timestep: Number of stable eigenvalues at each timestep
        - spectral_results_sample: Spectral analysis for sample timesteps
    """
    V_t_all = result_dict['Ut']  # Shape: (batch, T, r_dim, r_dim)

    # Flatten batch and time dimensions
    if V_t_all.ndim == 4:
        batch_size, T, r_dim, _ = V_t_all.shape
        V_t_flat = V_t_all.reshape(batch_size * T, r_dim, r_dim)
    else:
        # Single sequence: (T, r_dim, r_dim)
        T, r_dim, _ = V_t_all.shape
        V_t_flat = V_t_all
        batch_size = 1

    n_matrices = V_t_flat.shape[0]

    # Analyze a subset if there are too many matrices (for efficiency)
    max_analyze = 1000
    if n_matrices > max_analyze:
        # Sample uniformly
        indices = jnp.linspace(0, n_matrices - 1, max_analyze, dtype=int)
        V_t_sample = V_t_flat[indices]
        sample_rate = n_matrices / max_analyze
    else:
        V_t_sample = V_t_flat
        indices = jnp.arange(n_matrices)
        sample_rate = 1.0

    # Analyze each effective matrix
    all_timescales = []
    n_stable_per_matrix = []
    spectral_results_sample = []

    for i, idx in enumerate(indices):
        V_t = V_t_sample[i]
        spectrum = compute_timescale_spectrum(V_t, dt)

        # Store sample results (first 10 for inspection)
        if i < 10:
            spectral_results_sample.append(spectrum)

        # Extract finite stable timescales
        finite_mask = jnp.isfinite(spectrum['timescales']) & spectrum['is_stable'] & (~jnp.isnan(spectrum['timescales']))
        matrix_timescales = spectrum['timescales'][finite_mask]

        # Count stable modes with NaN check
        n_stable = jnp.sum(spectrum['is_stable'])
        if jnp.isnan(n_stable) or jnp.isinf(n_stable):
            n_stable_per_matrix.append(0)
        else:
            n_stable_per_matrix.append(int(n_stable))

        if len(matrix_timescales) > 0:
            all_timescales.append(matrix_timescales)

    # Combine all timescales
    if len(all_timescales) > 0:
        all_timescales = jnp.concatenate(all_timescales)

        timescale_stats = {
            'mean': float(jnp.mean(all_timescales)),
            'median': float(jnp.median(all_timescales)),
            'std': float(jnp.std(all_timescales)),
            'min': float(jnp.min(all_timescales)),
            'max': float(jnp.max(all_timescales))
        }

        timescale_range = (timescale_stats['min'], timescale_stats['max'])
        unique_timescales = cluster_timescales(all_timescales)
    else:
        all_timescales = jnp.array([])
        timescale_stats = {k: None for k in ['mean', 'median', 'std', 'min', 'max']}
        timescale_range = (None, None)
        unique_timescales = jnp.array([])

    return {
        'all_timescales': all_timescales,
        'timescale_stats': timescale_stats,
        'unique_timescales': unique_timescales,
        'timescale_range': timescale_range,
        'n_stable_per_timestep': n_stable_per_matrix,
        'n_matrices_analyzed': len(indices),
        'sample_rate': float(sample_rate),
        'spectral_results_sample': spectral_results_sample
    }


def analyze_weighted_mixture_timescales(
    model,
    W: jnp.ndarray,
    dt: float = 1.0,
    stability_threshold: float = 1.0
) -> Dict[str, Any]:
    """
    Analyze timescales of mixture components weighted by their usage frequency.

    Instead of treating all V_m equally, this weights each component's timescales
    by how often it's actually used (based on hypernetwork weights w).

    Args:
        model: Trained TiDHy model with temporal.value parameter
        W: Mixture weights over time from inference - shape (T, mix_dim) or (batch, T, mix_dim)
        dt: Time step size (matches data sampling rate)
        stability_threshold: Magnitude threshold for stable eigenvalues (default: 1.0)

    Returns:
        Dictionary containing:
        - weighted_timescales: Timescales weighted by component usage
        - component_usage: Average weight for each mixture component
        - timescales_per_component: Timescale arrays for each V_m
        - weighted_unique_timescales: Usage-weighted clustered timescales
        - comparison: Comparison with unweighted analysis
    """
    # Extract temporal matrices
    temporal = model.temporal.value  # Shape: (mix_dim, r_dim * r_dim)
    mix_dim = model.mix_dim
    r_dim = model.r_dim
    V_matrices = temporal.reshape(mix_dim, r_dim, r_dim)

    # Compute average usage of each component
    if W.ndim == 3:
        # Batched: (batch, T, mix_dim)
        W_flat = W.reshape(-1, mix_dim)
    else:
        # Single sequence: (T, mix_dim)
        W_flat = W

    component_usage = jnp.mean(W_flat, axis=0)  # (mix_dim,)

    # Analyze each component
    timescales_per_component = []
    weighted_timescales_list = []
    spectral_results = []

    for m in range(mix_dim):
        V_m = V_matrices[m]
        spectrum = compute_timescale_spectrum(V_m, dt)
        spectral_results.append(spectrum)

        # Extract finite stable timescales
        finite_mask = jnp.isfinite(spectrum['timescales']) & spectrum['is_stable'] & (~jnp.isnan(spectrum['timescales']))
        component_timescales = spectrum['timescales'][finite_mask]
        timescales_per_component.append(component_timescales)

        # Weight by component usage
        if len(component_timescales) > 0 and component_usage[m] > 1e-6:
            # Replicate timescales according to usage weight
            # (approximate: use weight as probability mass)
            weighted_timescales_list.append((component_timescales, component_usage[m]))

    # Create weighted timescale distribution
    # Method: weight each component's timescales by usage
    if len(weighted_timescales_list) > 0:
        # Concatenate all timescales with weights
        all_timescales = []
        all_weights = []
        for timescales, weight in weighted_timescales_list:
            all_timescales.append(timescales)
            all_weights.append(jnp.full(len(timescales), weight))

        weighted_timescales = jnp.concatenate(all_timescales)
        weights = jnp.concatenate(all_weights)

        # Normalize weights
        weights = weights / jnp.sum(weights)

        # Compute weighted statistics
        weighted_mean = float(jnp.sum(weighted_timescales * weights))
        weighted_median = float(jnp.median(weighted_timescales))  # Approximation

        # Cluster weighted timescales
        weighted_unique = cluster_timescales(weighted_timescales)
    else:
        weighted_timescales = jnp.array([])
        weights = jnp.array([])
        weighted_mean = None
        weighted_median = None
        weighted_unique = jnp.array([])

    # Compare with unweighted analysis
    all_unweighted = []
    for timescales in timescales_per_component:
        if len(timescales) > 0:
            all_unweighted.append(timescales)

    if len(all_unweighted) > 0:
        all_unweighted = jnp.concatenate(all_unweighted)
        unweighted_mean = float(jnp.mean(all_unweighted))
    else:
        all_unweighted = jnp.array([])
        unweighted_mean = None

    return {
        'weighted_timescales': weighted_timescales,
        'weights': weights,
        'component_usage': component_usage,
        'timescales_per_component': timescales_per_component,
        'weighted_unique_timescales': weighted_unique,
        'weighted_mean': weighted_mean,
        'weighted_median': weighted_median,
        'spectral_results': spectral_results,
        'comparison': {
            'unweighted_timescales': all_unweighted,
            'unweighted_mean': unweighted_mean,
            'weighted_mean': weighted_mean,
            'usage_entropy': float(compute_component_entropy(component_usage))
        }
    }


def analyze_time_varying_timescales(
    result_dict: Dict[str, jnp.ndarray],
    dt: float = 1.0,
    n_dominant: int = 3,
    stride: int = 1
) -> Dict[str, Any]:    
    """
    Analyze how timescales change over time by tracking dominant eigenvalues of V_t.

    Computes the N largest (slowest) timescales at each timestep to show
    transitions between dynamical regimes.

    Args:
        result_dict: Output from evaluate_record() containing 'Ut' (V_t matrices)
        dt: Time step size
        n_dominant: Number of dominant timescales to track (default: 3)
        stride: Analyze every Nth timestep (for efficiency, default: 1)

    Returns:
        Dictionary containing:
        - dominant_timescales: Array of shape (n_timesteps, n_dominant) with top timescales
        - dominant_eigenvalues: Corresponding eigenvalues
        - time_indices: Timestep indices analyzed
        - transitions: Detected regime transitions (where timescales change significantly)
    """
    V_t_all = result_dict['Ut']  # Shape: (batch, T, r_dim, r_dim) or (T, r_dim, r_dim)

    # Handle batched data
    if V_t_all.ndim == 4:
        # Use first batch element for time-varying analysis
        V_t_sequence = V_t_all[0]  # (T, r_dim, r_dim)
    else:
        V_t_sequence = V_t_all  # (T, r_dim, r_dim)

    T = V_t_sequence.shape[0]
    time_indices = jnp.arange(0, T, stride)
    n_timesteps = len(time_indices)

    # Track dominant timescales over time
    dominant_timescales_list = []
    dominant_eigenvalues_list = []

    for t_idx in time_indices:
        V_t = V_t_sequence[t_idx]
        spectrum = compute_timescale_spectrum(V_t, dt)

        # Get stable timescales sorted by magnitude (largest first)
        finite_mask = jnp.isfinite(spectrum['timescales']) & spectrum['is_stable'] & (~jnp.isnan(spectrum['timescales']))
        timescales = spectrum['timescales'][finite_mask]
        eigenvalues = spectrum['eigenvalues'][finite_mask]

        if len(timescales) > 0:
            # Sort by timescale (descending - slowest first)
            sort_idx = jnp.argsort(timescales)[::-1]
            timescales_sorted = timescales[sort_idx]
            eigenvalues_sorted = eigenvalues[sort_idx]

            # Take top N
            n_available = min(n_dominant, len(timescales_sorted))
            top_timescales = timescales_sorted[:n_available]
            top_eigenvalues = eigenvalues_sorted[:n_available]

            # Pad if necessary
            if n_available < n_dominant:
                top_timescales = jnp.concatenate([
                    top_timescales,
                    jnp.full(n_dominant - n_available, jnp.nan)
                ])
                top_eigenvalues = jnp.concatenate([
                    top_eigenvalues,
                    jnp.full(n_dominant - n_available, jnp.nan)
                ])
        else:
            # No stable timescales
            top_timescales = jnp.full(n_dominant, jnp.nan)
            top_eigenvalues = jnp.full(n_dominant, jnp.nan)

        dominant_timescales_list.append(top_timescales)
        dominant_eigenvalues_list.append(top_eigenvalues)

    dominant_timescales = jnp.stack(dominant_timescales_list)  # (n_timesteps, n_dominant)
    dominant_eigenvalues = jnp.stack(dominant_eigenvalues_list)  # (n_timesteps, n_dominant)

    # Detect transitions: where dominant timescale changes significantly
    transitions = []
    if n_timesteps > 1:
        # Compute relative change in slowest timescale
        slowest = dominant_timescales[:, 0]  # First column = slowest timescale

        # Handle NaNs
        valid_mask = jnp.isfinite(slowest)
        if jnp.sum(valid_mask) > 1:
            # Compute differences
            diffs = jnp.diff(slowest)
            rel_change = jnp.abs(diffs / (slowest[:-1] + 1e-10))

            # Threshold for significant change (e.g., 50%)
            threshold = 0.5
            transition_mask = rel_change > threshold
            transition_indices = jnp.where(transition_mask)[0]

            # Convert indices to integers with NaN check
            transitions = []
            for i in transition_indices:
                idx = time_indices[i + 1]
                if not (jnp.isnan(idx) or jnp.isinf(idx)):
                    transitions.append(int(idx))

    return {
        'dominant_timescales': dominant_timescales,
        'dominant_eigenvalues': dominant_eigenvalues,
        'time_indices': time_indices,
        'transitions': transitions,
        'n_dominant': n_dominant,
        'stride': stride
    }


# ============================================================================
# Hypernetwork Usage Analysis
# ============================================================================

def analyze_hypernetwork_usage(
    W: jnp.ndarray,
    threshold: float = 0.01
) -> Dict[str, Any]:
    """
    Analyze which mixture components are actively used by the hypernetwork.

    Args:
        W: Mixture weights over time - shape (T, mix_dim) or (batch, T, mix_dim)
        threshold: Weight threshold for considering component "active"

    Returns:
        Dictionary containing:
        - active_components: Number of components used above threshold
        - usage_fraction: Fraction of timesteps each component is active
        - mean_weights: Time-averaged weights per component
        - entropy: Shannon entropy of average weights (uniformity measure)
        - sparsity: Fraction of weights below threshold
        - dominant_component: Index of most-used component
    """
    # Reshape if batched
    if W.ndim == 3:
        W = W.reshape(-1, W.shape[-1])  # (batch*T, mix_dim)

    T, mix_dim = W.shape

    # Mean weights per component
    mean_weights = jnp.mean(W, axis=0)

    # Active components (above threshold in mean)
    active_comp_count = jnp.sum(mean_weights > threshold)
    if jnp.isnan(active_comp_count) or jnp.isinf(active_comp_count):
        active_components = 0
    else:
        active_components = int(active_comp_count)

    # Usage fraction: for each component, fraction of timesteps it's active
    usage_fraction = jnp.mean(W > threshold, axis=0)

    # Shannon entropy of mean weights (measure of uniformity)
    # H = -Σ p_i log(p_i)
    # Normalized: H / log(mix_dim) ∈ [0, 1]
    p = mean_weights / (jnp.sum(mean_weights) + 1e-10)
    entropy_raw = -jnp.sum(p * jnp.log(p + 1e-10))
    entropy_normalized = entropy_raw / jnp.log(mix_dim)

    # Sparsity: fraction of all weights below threshold
    sparsity = jnp.mean(W < threshold)

    # Dominant component
    dominant_idx = jnp.argmax(mean_weights)
    if jnp.isnan(dominant_idx) or jnp.isinf(dominant_idx):
        dominant_component = 0
    else:
        dominant_component = int(dominant_idx)

    return {
        'active_components': active_components,
        'usage_fraction': usage_fraction,
        'mean_weights': mean_weights,
        'entropy': float(entropy_normalized),
        'entropy_raw': float(entropy_raw),
        'sparsity': float(sparsity),
        'dominant_component': dominant_component,
        'mix_dim': mix_dim
    }


def compute_component_entropy(weights: jnp.ndarray) -> float:
    """
    Compute Shannon entropy of mixture weights.

    Normalized entropy H/log(N) ranges from 0 (single component)
    to 1 (uniform distribution).

    Args:
        weights: Mixture weights (mix_dim,) or (T, mix_dim)

    Returns:
        Normalized entropy [0, 1]
    """
    if weights.ndim > 1:
        weights = jnp.mean(weights, axis=0)

    # Normalize to probability distribution
    p = weights / (jnp.sum(weights) + 1e-10)

    # Shannon entropy
    H = -jnp.sum(p * jnp.log(p + 1e-10))

    # Normalize by maximum entropy
    max_entropy = jnp.log(len(weights))

    return float(H / max_entropy)


# ============================================================================
# Ground Truth Timescale Comparison (for validation)
# ============================================================================

def compute_rossler_ground_truth_timescales(
    trajectory: jnp.ndarray,
    dt: float,
    params: Dict[str, Any],
    max_lag: int = 1000
) -> Dict[str, Any]:
    """
    Compute ground truth timescales for hierarchical Rossler attractor.

    Uses both analytical (Jacobian eigenvalues at fixed point) and empirical
    (autocorrelation analysis) methods to determine true timescales.

    Args:
        trajectory: State trajectory, shape (T, state_dim)
                   For hierarchical Rossler: (T, 6) = [xs, ys, zs, xf, yf, zf]
        dt: Integration timestep
        params: Dictionary with Rossler parameters:
                - a_slow, b_slow, c_slow
                - a_fast, b_base, c_base
                - coupling_b, coupling_c
        max_lag: Maximum lag for autocorrelation (default: 1000)

    Returns:
        Dictionary containing:
        - slow_period: Oscillation period of slow system (empirical)
        - fast_period: Oscillation period of fast system (empirical)
        - slow_timescale: Characteristic timescale of slow dynamics
        - fast_timescale: Characteristic timescale of fast dynamics
        - analytical_timescales: Timescales from Jacobian analysis
        - autocorr_slow: Autocorrelation function for slow system
        - autocorr_fast: Autocorrelation function for fast system
    """
    # Convert to numpy for scipy operations
    traj_np = np.array(trajectory)
    T = len(traj_np)

    # Separate slow and fast states
    # Hierarchical Rossler: [xs, ys, zs, xf, yf, zf]
    slow_states = traj_np[:, :3]  # xs, ys, zs
    fast_states = traj_np[:, 3:]  # xf, yf, zf

    # ========== Empirical Analysis: Autocorrelation ==========

    def compute_autocorr_period(signal, max_lag, dt):
        """Find period from autocorrelation peaks."""
        # Normalize signal
        signal = signal - np.mean(signal)
        signal = signal / (np.std(signal) + 1e-10)

        # Compute autocorrelation
        n = min(len(signal), max_lag)
        autocorr = np.correlate(signal[:n], signal[:n], mode='same')
        autocorr = autocorr[n//2:]  # Keep only positive lags
        autocorr = autocorr / autocorr[0]  # Normalize

        # Find peaks (excluding lag=0)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(autocorr[1:], height=0.1)

        if len(peaks) > 0:
            # First peak gives period
            period = (peaks[0] + 1) * dt  # +1 because we excluded lag=0
            return period, autocorr
        else:
            # No clear peak - estimate from zero crossings
            zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
            if len(zero_crossings) >= 2:
                # Half period between first two crossings
                period = 2 * zero_crossings[0] * dt
                return period, autocorr
            else:
                return None, autocorr

    # Analyze slow system (use x component)
    slow_period, autocorr_slow = compute_autocorr_period(slow_states[:, 0], max_lag, dt)

    # Analyze fast system (use x component)
    fast_period, autocorr_fast = compute_autocorr_period(fast_states[:, 0], max_lag, dt)

    # ========== Analytical Analysis: Jacobian at Fixed Point ==========

    def compute_rossler_jacobian_timescales(a, b, c):
        """
        Compute timescales from Jacobian eigenvalues at Rossler fixed point.

        For Rossler system: dx/dt = -y-z, dy/dt = x+ay, dz/dt = b+z(x-c)
        Fixed point: x* ≈ (c - sqrt(c^2 - 4ab))/2, y* ≈ -x*/a, z* ≈ x*/a
        """
        # Fixed point (approximate for small a)
        discriminant = c**2 - 4*a*b
        if discriminant < 0:
            # No real fixed point - use approximate
            x_star = c / 2
        else:
            x_star = (c - np.sqrt(discriminant)) / 2

        y_star = -x_star / a
        z_star = x_star / a

        # Jacobian at fixed point
        J = np.array([
            [0, -1, -1],
            [1, a, 0],
            [z_star, 0, x_star - c]
        ])

        # Eigenvalues
        eigenvalues = np.linalg.eigvals(J)

        # Convert to timescales: τ = -1/Re(λ) for negative real parts
        timescales = []
        for lam in eigenvalues:
            re_lam = np.real(lam)
            if re_lam < 0:
                timescales.append(-1 / re_lam)
            elif np.abs(np.imag(lam)) > 1e-6:
                # Oscillatory mode: period = 2π/Im(λ)
                period = 2 * np.pi / np.abs(np.imag(lam))
                timescales.append(period)

        return timescales, eigenvalues

    # Analytical timescales for both systems
    slow_analytical, slow_eigs = compute_rossler_jacobian_timescales(
        params['a_slow'], params['b_slow'], params['c_slow']
    )

    fast_analytical, fast_eigs = compute_rossler_jacobian_timescales(
        params['a_fast'], params['b_base'], params['c_base']
    )

    # ========== Summary ==========

    # Use empirical if available, fallback to analytical
    slow_timescale = slow_period if slow_period is not None else (
        np.max(slow_analytical) if len(slow_analytical) > 0 else None
    )

    fast_timescale = fast_period if fast_period is not None else (
        np.max(fast_analytical) if len(fast_analytical) > 0 else None
    )

    return {
        # Empirical (from autocorrelation)
        'slow_period': slow_period,
        'fast_period': fast_period,
        'slow_timescale': slow_timescale,
        'fast_timescale': fast_timescale,
        # Analytical (from Jacobian)
        'analytical_slow': slow_analytical,
        'analytical_fast': fast_analytical,
        'slow_eigenvalues': slow_eigs,
        'fast_eigenvalues': fast_eigs,
        # Autocorrelation functions
        'autocorr_slow': autocorr_slow,
        'autocorr_fast': autocorr_fast,
        # Ratio
        'timescale_ratio': slow_timescale / fast_timescale if (
            slow_timescale and fast_timescale
        ) else None
    }


def compare_discovered_to_ground_truth(
    discovered_timescales: jnp.ndarray,
    ground_truth_timescales: list,
    tolerance: float = 0.2
) -> Dict[str, Any]:
    """
    Compare discovered timescales to ground truth with matching and error metrics.

    Uses Hungarian algorithm to find optimal matching between discovered and
    ground truth timescales, then computes recovery metrics.

    Args:
        discovered_timescales: Array of timescales found by model analysis
        ground_truth_timescales: List of true timescales from system
        tolerance: Relative error tolerance for considering a match (default: 0.2 = 20%)

    Returns:
        Dictionary containing:
        - matched_pairs: List of (discovered, ground_truth) pairs
        - relative_errors: Relative error for each matched pair
        - recovery_rate: Fraction of ground truth timescales successfully recovered
        - mean_relative_error: Average relative error across matched pairs
        - false_positives: Discovered timescales with no ground truth match
        - false_negatives: Ground truth timescales not discovered
        - precision: TP / (TP + FP)
        - recall: TP / (TP + FN)
    """
    from scipy.optimize import linear_sum_assignment

    discovered = np.array(discovered_timescales)
    ground_truth = np.array(ground_truth_timescales)

    if len(discovered) == 0:
        return {
            'matched_pairs': [],
            'relative_errors': [],
            'recovery_rate': 0.0,
            'mean_relative_error': None,
            'false_positives': [],
            'false_negatives': list(ground_truth),
            'precision': 0.0,
            'recall': 0.0,
            'n_discovered': 0,
            'n_ground_truth': len(ground_truth)
        }

    # Compute cost matrix: relative error between all pairs
    n_disc = len(discovered)
    n_true = len(ground_truth)

    cost_matrix = np.zeros((n_disc, n_true))
    for i, d in enumerate(discovered):
        for j, t in enumerate(ground_truth):
            # Relative error
            rel_error = np.abs(d - t) / (t + 1e-10)
            cost_matrix[i, j] = rel_error

    # Find optimal matching using Hungarian algorithm
    disc_indices, true_indices = linear_sum_assignment(cost_matrix)

    # Extract matches and filter by tolerance
    matched_pairs = []
    relative_errors = []
    matched_disc_set = set()
    matched_true_set = set()

    for i, j in zip(disc_indices, true_indices):
        rel_error = cost_matrix[i, j]
        if rel_error <= tolerance:
            # Valid match
            matched_pairs.append((discovered[i], ground_truth[j]))
            relative_errors.append(rel_error)
            matched_disc_set.add(i)
            matched_true_set.add(j)

    # Identify false positives and false negatives
    false_positives = [discovered[i] for i in range(n_disc) if i not in matched_disc_set]
    false_negatives = [ground_truth[j] for j in range(n_true) if j not in matched_true_set]

    # Compute metrics
    n_true_positives = len(matched_pairs)
    n_false_positives = len(false_positives)
    n_false_negatives = len(false_negatives)

    recovery_rate = n_true_positives / n_true if n_true > 0 else 0.0
    precision = n_true_positives / (n_true_positives + n_false_positives) if (
        n_true_positives + n_false_positives > 0
    ) else 0.0
    recall = recovery_rate  # Same as recovery rate

    mean_relative_error = np.mean(relative_errors) if len(relative_errors) > 0 else None

    return {
        'matched_pairs': matched_pairs,
        'relative_errors': relative_errors,
        'recovery_rate': recovery_rate,
        'mean_relative_error': mean_relative_error,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'n_discovered': n_disc,
        'n_ground_truth': n_true,
        'n_matched': n_true_positives,
        'cost_matrix': cost_matrix,
        'tolerance': tolerance
    }


def analyze_hierarchical_rossler_recovery(
    model,
    test_data: jnp.ndarray,
    rossler_trajectory: jnp.ndarray,
    rossler_params: Dict[str, Any],
    dt: float,
    rng_key,
    tolerance: float = 0.2
) -> Dict[str, Any]:
    """
    All-in-one analysis: compare discovered timescales to ground truth for Rossler.

    Performs complete timescale recovery analysis including:
    1. Compute ground truth from Rossler trajectory
    2. Run inference to get model predictions
    3. Analyze using basis, effective, and weighted methods
    4. Compare each to ground truth

    Args:
        model: Trained TiDHy model
        test_data: Test observations for inference
        rossler_trajectory: Raw state trajectory from Rossler system (for ground truth)
        rossler_params: Rossler system parameters
        dt: Integration timestep
        rng_key: JAX random key for inference
        tolerance: Matching tolerance (default: 0.2 = 20%)

    Returns:
        Dictionary containing:
        - ground_truth: Ground truth timescales dict
        - basis_analysis: Basis V_m analysis results
        - effective_analysis: Effective V_t analysis results (if available)
        - weighted_analysis: Weighted analysis results (if available)
        - basis_comparison: Comparison of basis to ground truth
        - effective_comparison: Comparison of effective to ground truth
        - weighted_comparison: Comparison of weighted to ground truth
        - result_dict: Full inference results for further analysis
    """
    from TiDHy.models.TiDHy_nnx_vmap_training import evaluate_record

    # 1. Compute ground truth
    ground_truth = compute_rossler_ground_truth_timescales(
        rossler_trajectory, dt, rossler_params
    )

    # Extract ground truth timescales as list
    gt_timescales = []
    if ground_truth['slow_timescale'] is not None:
        gt_timescales.append(ground_truth['slow_timescale'])
    if ground_truth['fast_timescale'] is not None:
        gt_timescales.append(ground_truth['fast_timescale'])

    # 2. Run inference
    _, _, _, result_dict = evaluate_record(model, test_data, rng_key)

    # 3. Analyze using different methods
    # Basis analysis
    basis_results = analyze_mixture_timescales(model, dt=dt)

    # Effective analysis (if V_t available)
    effective_results = None
    if 'Ut' in result_dict:
        effective_results = analyze_effective_timescales(result_dict, dt=dt)

    # Weighted analysis (if W available)
    weighted_results = None
    if 'W' in result_dict:
        weighted_results = analyze_weighted_mixture_timescales(model, result_dict['W'], dt=dt)

    # 4. Compare each to ground truth
    basis_comparison = compare_discovered_to_ground_truth(
        basis_results['all_timescales'],
        gt_timescales,
        tolerance=tolerance
    )

    effective_comparison = None
    if effective_results is not None and len(effective_results['all_timescales']) > 0:
        effective_comparison = compare_discovered_to_ground_truth(
            effective_results['all_timescales'],
            gt_timescales,
            tolerance=tolerance
        )

    weighted_comparison = None
    if weighted_results is not None and len(weighted_results['weighted_timescales']) > 0:
        weighted_comparison = compare_discovered_to_ground_truth(
            weighted_results['weighted_timescales'],
            gt_timescales,
            tolerance=tolerance
        )

    return {
        'ground_truth': ground_truth,
        'basis_analysis': basis_results,
        'effective_analysis': effective_results,
        'weighted_analysis': weighted_results,
        'basis_comparison': basis_comparison,
        'effective_comparison': effective_comparison,
        'weighted_comparison': weighted_comparison,
        'result_dict': result_dict,
        'ground_truth_timescales': gt_timescales
    }
