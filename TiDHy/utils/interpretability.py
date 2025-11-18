"""
Interpretability Analysis for Latent Representations

This module provides tools for analyzing and measuring the interpretability of
learned latent representations, particularly the R latent space in TiDHy.

Key metrics:
- Correlation analysis (dimension independence)
- Disentanglement metrics (Total Correlation, MIG, SAP)
- Dimension usage and importance
- Decoder orthogonality analysis

Author: Elliott Abe
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_regression


def compute_correlation_matrix(
    r_values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute correlation and covariance matrices for R dimensions.

    Args:
        r_values: R latent values, shape (T, r_dim) or (batch, T, r_dim)

    Returns:
        Tuple of (correlation_matrix, covariance_matrix)
    """
    # Flatten if needed
    if r_values.ndim == 3:
        r_flat = r_values.reshape(-1, r_values.shape[-1])
    else:
        r_flat = r_values

    # Compute correlation matrix
    corr = np.corrcoef(r_flat.T)

    # Compute covariance matrix
    cov = np.cov(r_flat.T)

    return corr, cov


def compute_total_correlation(
    r_values: np.ndarray,
    n_bins: int = 20
) -> float:
    """
    Compute Total Correlation (TC) as a measure of statistical independence.

    TC measures the KL divergence between the joint distribution and the product
    of marginals. TC = 0 means perfect independence, higher TC means more dependence.

    TC = ∑ H(z_i) - H(z)  where H is entropy

    Args:
        r_values: R latent values, shape (T, r_dim) or (batch, T, r_dim)
        n_bins: Number of bins for histogram-based entropy estimation

    Returns:
        Total correlation score (lower is better, 0 = perfect independence)
    """
    # Flatten if needed
    if r_values.ndim == 3:
        r_flat = r_values.reshape(-1, r_values.shape[-1])
    else:
        r_flat = r_values

    n_samples, n_dims = r_flat.shape

    # Compute marginal entropies
    marginal_entropies = []
    for dim in range(n_dims):
        hist, _ = np.histogram(r_flat[:, dim], bins=n_bins, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        marginal_h = -np.sum(hist * np.log(hist + 1e-10))
        marginal_entropies.append(marginal_h)

    sum_marginal_h = np.sum(marginal_entropies)

    # Estimate joint entropy using correlation determinant approximation
    # H(z) ≈ 0.5 * log((2πe)^n * |Σ|) where Σ is covariance matrix
    cov = np.cov(r_flat.T)
    sign, logdet = np.linalg.slogdet(cov + 1e-6 * np.eye(n_dims))

    if sign <= 0:
        # Degenerate covariance, return large TC
        return sum_marginal_h

    joint_h = 0.5 * (n_dims * np.log(2 * np.pi * np.e) + logdet)

    # TC = sum of marginal entropies - joint entropy
    tc = sum_marginal_h - joint_h

    return float(max(0, tc))  # Clamp to non-negative


def compute_mutual_info_gap(
    r_values: np.ndarray,
    annotations: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute Mutual Information Gap (MIG) as a measure of disentanglement.

    MIG measures how much each latent dimension is specific to a single factor.
    Higher MIG means better disentanglement.

    Args:
        r_values: R latent values, shape (T, r_dim) or (batch, T, r_dim)
        annotations: Optional behavioral annotations, shape (T,)

    Returns:
        Dictionary with MIG score and related metrics
    """
    # Flatten if needed
    if r_values.ndim == 3:
        r_flat = r_values.reshape(-1, r_values.shape[-1])
    else:
        r_flat = r_values

    if annotations is None:
        # Without ground truth factors, use PCA components as pseudo-factors
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(10, r_flat.shape[1]))
        factors = pca.fit_transform(r_flat)
    else:
        # Use annotations as factors (one-hot encode if discrete)
        if annotations.ndim == 1:
            from sklearn.preprocessing import OneHotEncoder
            enc = OneHotEncoder(sparse=False)
            factors = enc.fit_transform(annotations.reshape(-1, 1))
        else:
            factors = annotations

    # Truncate to same length
    min_len = min(len(r_flat), len(factors))
    r_flat = r_flat[:min_len]
    factors = factors[:min_len]

    # Compute mutual information between each latent dim and each factor
    n_latents = r_flat.shape[1]
    n_factors = factors.shape[1]

    mi_matrix = np.zeros((n_latents, n_factors))

    for i in range(n_latents):
        for j in range(n_factors):
            mi = mutual_info_regression(
                r_flat[:, i].reshape(-1, 1),
                factors[:, j],
                random_state=42
            )[0]
            mi_matrix[i, j] = mi

    # MIG: For each factor, find gap between top 2 latents
    mig_scores = []
    for j in range(n_factors):
        sorted_mi = np.sort(mi_matrix[:, j])[::-1]
        if len(sorted_mi) >= 2:
            gap = sorted_mi[0] - sorted_mi[1]
            mig_scores.append(gap / (np.sum(mi_matrix[:, j]) + 1e-10))

    mig = np.mean(mig_scores) if mig_scores else 0.0

    return {
        'mig': float(mig),
        'mi_matrix': mi_matrix,
        'n_latents': n_latents,
        'n_factors': n_factors
    }


def compute_sap_score(
    r_values: np.ndarray,
    annotations: Optional[np.ndarray] = None
) -> float:
    """
    Compute Separated Attribute Predictability (SAP) score.

    SAP measures how well each factor can be predicted from a single latent dimension.

    Args:
        r_values: R latent values, shape (T, r_dim)
        annotations: Optional behavioral annotations, shape (T,)

    Returns:
        SAP score (higher is better, range [0, 1])
    """
    # Similar to MIG but uses regression/classification accuracy
    # Simplified implementation
    if annotations is None:
        return 0.0

    # Flatten if needed
    if r_values.ndim == 3:
        r_flat = r_values.reshape(-1, r_values.shape[-1])
    else:
        r_flat = r_values

    min_len = min(len(r_flat), len(annotations))
    r_flat = r_flat[:min_len]
    annotations = annotations[:min_len]

    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import cross_val_score

    # For each latent dimension, predict annotations
    scores = []
    for dim in range(r_flat.shape[1]):
        model = DecisionTreeRegressor(max_depth=3, random_state=42)
        cv_scores = cross_val_score(
            model,
            r_flat[:, dim].reshape(-1, 1),
            annotations,
            cv=3,
            scoring='r2'
        )
        scores.append(np.mean(cv_scores))

    # SAP: difference between top 2 scores
    sorted_scores = np.sort(scores)[::-1]
    if len(sorted_scores) >= 2:
        sap = sorted_scores[0] - sorted_scores[1]
    else:
        sap = sorted_scores[0] if len(sorted_scores) > 0 else 0.0

    return float(max(0, sap))


def analyze_dimension_usage(
    r_values: np.ndarray,
    threshold: float = 0.01
) -> Dict[str, Any]:
    """
    Analyze which R dimensions are actually being used (non-zero variance).

    Args:
        r_values: R latent values, shape (T, r_dim) or (batch, T, r_dim)
        threshold: Variance threshold for considering a dimension "dead"

    Returns:
        Dictionary with usage statistics
    """
    # Flatten if needed
    if r_values.ndim == 3:
        r_flat = r_values.reshape(-1, r_values.shape[-1])
    else:
        r_flat = r_values

    # Compute variance per dimension
    variances = np.var(r_flat, axis=0)

    # Compute mean absolute value per dimension
    mean_abs = np.mean(np.abs(r_flat), axis=0)

    # Identify active vs dead dimensions
    active_mask = variances > threshold
    n_active = np.sum(active_mask)
    n_dead = np.sum(~active_mask)

    # Compute sparsity per dimension (fraction of near-zero values)
    sparsity_per_dim = np.mean(np.abs(r_flat) < threshold, axis=0)

    return {
        'n_dimensions': r_flat.shape[1],
        'n_active': int(n_active),
        'n_dead': int(n_dead),
        'active_fraction': float(n_active / r_flat.shape[1]),
        'variances': variances,
        'mean_abs_values': mean_abs,
        'sparsity_per_dim': sparsity_per_dim,
        'active_indices': np.where(active_mask)[0],
        'dead_indices': np.where(~active_mask)[0]
    }


def compute_decoder_orthogonality(
    decoder_weights: np.ndarray
) -> Dict[str, float]:
    """
    Measure orthogonality of decoder weight matrix.

    Args:
        decoder_weights: Decoder weight matrix, shape (input_dim, r_dim)

    Returns:
        Dictionary with orthogonality metrics
    """
    W = decoder_weights

    # Normalize columns
    W_norm = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-10)

    # Compute Gram matrix
    gram = W_norm.T @ W_norm

    # Perfect orthogonality → Gram = Identity
    identity = np.eye(gram.shape[0])
    off_diagonal_error = np.sum((gram - identity) ** 2)

    # Compute average absolute off-diagonal correlation
    off_diag_mask = 1 - identity
    avg_off_diag = np.sum(np.abs(gram * off_diag_mask)) / (np.sum(off_diag_mask) + 1e-10)

    # Compute condition number (measure of ill-conditioning)
    try:
        cond_number = np.linalg.cond(W)
    except:
        cond_number = np.inf

    return {
        'orthogonality_error': float(off_diagonal_error),
        'avg_off_diagonal_corr': float(avg_off_diag),
        'condition_number': float(cond_number),
        'gram_matrix': gram
    }


def plot_r_correlation_matrix(
    r_values: np.ndarray,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot correlation matrix heatmap for R dimensions.

    Args:
        r_values: R latent values, shape (T, r_dim)
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    corr, _ = compute_correlation_matrix(r_values)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        corr,
        annot=False,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={'label': 'Correlation'}
    )

    ax.set_title('R Dimension Correlation Matrix', fontsize=14)
    ax.set_xlabel('R Dimension', fontsize=12)
    ax.set_ylabel('R Dimension', fontsize=12)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_dimension_trajectories(
    r_values: np.ndarray,
    annotations: Optional[np.ndarray] = None,
    n_dims_to_plot: int = 8,
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot time evolution of R dimensions.

    Args:
        r_values: R latent values, shape (T, r_dim)
        annotations: Optional behavioral annotations for coloring
        n_dims_to_plot: Number of dimensions to plot
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    # Flatten if needed
    if r_values.ndim == 3:
        r_flat = r_values[0]  # Take first batch
    else:
        r_flat = r_values

    n_dims = min(n_dims_to_plot, r_flat.shape[1])
    n_rows = int(np.ceil(n_dims / 2))

    fig, axes = plt.subplots(n_rows, 2, figsize=figsize, sharex=True)
    axes = axes.flatten() if n_rows > 1 else [axes]

    for i in range(n_dims):
        ax = axes[i]

        if annotations is not None:
            # Color by annotations
            scatter = ax.scatter(
                range(len(r_flat)),
                r_flat[:, i],
                c=annotations[:len(r_flat)],
                s=1,
                alpha=0.5,
                cmap='tab10'
            )
        else:
            ax.plot(r_flat[:, i], linewidth=0.5, alpha=0.7)

        ax.set_ylabel(f'R[{i}]', fontsize=10)
        ax.grid(alpha=0.3)

        # Add variance in title
        var = np.var(r_flat[:, i])
        ax.set_title(f'Dim {i} (var={var:.3f})', fontsize=10)

    axes[-2].set_xlabel('Time', fontsize=12)
    axes[-1].set_xlabel('Time', fontsize=12)

    # Hide unused subplots
    for i in range(n_dims, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('R Latent Dimension Trajectories', fontsize=14, y=0.995)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_dimension_importance(
    r_values: np.ndarray,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot dimension importance based on variance and usage.

    Args:
        r_values: R latent values, shape (T, r_dim)
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    usage = analyze_dimension_usage(r_values)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot variances
    dims = np.arange(usage['n_dimensions'])
    ax1.bar(dims, usage['variances'], alpha=0.7)
    ax1.set_xlabel('Dimension', fontsize=12)
    ax1.set_ylabel('Variance', fontsize=12)
    ax1.set_title('Dimension Variance (Activity)', fontsize=13)
    ax1.grid(axis='y', alpha=0.3)

    # Mark dead dimensions
    dead_dims = usage['dead_indices']
    if len(dead_dims) > 0:
        ax1.bar(dead_dims, usage['variances'][dead_dims], color='red', alpha=0.5, label='Dead')
        ax1.legend()

    # Plot sparsity
    ax2.bar(dims, usage['sparsity_per_dim'], alpha=0.7, color='orange')
    ax2.set_xlabel('Dimension', fontsize=12)
    ax2.set_ylabel('Sparsity (fraction zeros)', fontsize=12)
    ax2.set_title('Dimension Sparsity', fontsize=13)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def identify_interpretable_dimensions(
    r_values: np.ndarray,
    min_variance: float = 0.01,
    max_correlation: float = 0.5
) -> List[int]:
    """
    Identify most interpretable dimensions based on criteria.

    Interpretable dimensions should have:
    - Non-zero variance (actually used)
    - Low correlation with other dimensions (independent)

    Args:
        r_values: R latent values, shape (T, r_dim)
        min_variance: Minimum variance threshold
        max_correlation: Maximum absolute correlation with other dims

    Returns:
        List of dimension indices that are interpretable
    """
    usage = analyze_dimension_usage(r_values, threshold=min_variance)
    corr, _ = compute_correlation_matrix(r_values)

    # Start with active dimensions
    candidates = usage['active_indices']

    # Filter by correlation
    interpretable = []
    for dim in candidates:
        # Get correlations with other dimensions
        other_dims = [d for d in range(corr.shape[0]) if d != dim]
        max_corr_with_others = np.max(np.abs(corr[dim, other_dims]))

        if max_corr_with_others < max_correlation:
            interpretable.append(dim)

    return interpretable
