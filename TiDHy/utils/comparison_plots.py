"""
TiDHy vs SLDS Comparison Visualization

Functions for creating comparison plots between TiDHy and SLDS models.

Author: Elliott Abe
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.2)
    sns.set_palette("Set2")


def plot_reconstruction_comparison(
    comparison_results: Dict[str, Any],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 4)
):
    """
    Plot reconstruction quality comparison between models.

    Args:
        comparison_results: Output from compare_models()
        save_path: Optional path to save figure
        figsize: Figure size
    """
    set_publication_style()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Extract metrics
    models = []
    r2_scores = []
    mse_scores = []
    rmse_scores = []

    if 'tidhy_r2' in comparison_results:
        models.append('TiDHy')
        r2_scores.append(comparison_results['tidhy_r2'])
        mse_scores.append(comparison_results['tidhy_mse'])
        rmse_scores.append(comparison_results['tidhy_rmse'])

    if 'slds_r2' in comparison_results:
        models.append('SLDS')
        r2_scores.append(comparison_results['slds_r2'])
        mse_scores.append(comparison_results['slds_mse'])
        rmse_scores.append(comparison_results['slds_rmse'])

    # Plot R²
    axes[0].bar(models, r2_scores, color=['#1f77b4', '#ff7f0e'])
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('Reconstruction Quality (R²)')
    axes[0].set_ylim([0, 1.0])
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    for i, v in enumerate(r2_scores):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

    # Plot MSE
    axes[1].bar(models, mse_scores, color=['#1f77b4', '#ff7f0e'])
    axes[1].set_ylabel('MSE')
    axes[1].set_title('Mean Squared Error')
    axes[1].set_yscale('log')
    for i, v in enumerate(mse_scores):
        axes[1].text(i, v * 1.1, f'{v:.2e}', ha='center', va='bottom', fontsize=9)

    # Plot RMSE
    axes[2].bar(models, rmse_scores, color=['#1f77b4', '#ff7f0e'])
    axes[2].set_ylabel('RMSE')
    axes[2].set_title('Root Mean Squared Error')
    for i, v in enumerate(rmse_scores):
        axes[2].text(i, v + v * 0.05, f'{v:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved reconstruction comparison to {save_path}")

    return fig


def plot_dimensionality_comparison(
    comparison_results: Dict[str, Any],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 5)
):
    """
    Plot effective dimensionality comparison.

    Args:
        comparison_results: Output from compare_models()
        save_path: Optional path to save figure
        figsize: Figure size
    """
    set_publication_style()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Participation ratio
    models = []
    pr_values = []
    nom_dims = []

    if 'tidhy_r_participation_ratio' in comparison_results:
        models.append('TiDHy (r)')
        pr_values.append(comparison_results['tidhy_r_participation_ratio'])
        nom_dims.append(comparison_results['tidhy_r_nominal_dim'])

    if 'tidhy_r2_participation_ratio' in comparison_results:
        models.append('TiDHy (r2)')
        pr_values.append(comparison_results['tidhy_r2_participation_ratio'])
        nom_dims.append(comparison_results['tidhy_r2_nominal_dim'])

    if 'slds_participation_ratio' in comparison_results:
        models.append('SLDS')
        pr_values.append(comparison_results['slds_participation_ratio'])
        nom_dims.append(comparison_results['slds_nominal_dim'])

    # Plot participation ratio
    x_pos = np.arange(len(models))
    bars = axes[0].bar(x_pos, pr_values, color=sns.color_palette("Set2", len(models)))
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(models, rotation=15, ha='right')
    axes[0].set_ylabel('Participation Ratio')
    axes[0].set_title('Effective Dimensionality\n(Participation Ratio)')

    for i, (v, nd) in enumerate(zip(pr_values, nom_dims)):
        axes[0].text(i, v + 0.1, f'{v:.2f}\n({nd}D)', ha='center', va='bottom', fontsize=9)

    # Effective dimensions at different thresholds
    thresholds = ['90%', '95%', '99%']
    width = 0.25

    for i, model in enumerate(models):
        prefix = model.lower().replace(' ', '_').replace('(', '').replace(')', '')
        eff_dims = [
            comparison_results.get(f'{prefix}_eff_dim_90', 0),
            comparison_results.get(f'{prefix}_eff_dim_95', 0),
            comparison_results.get(f'{prefix}_eff_dim_99', 0),
        ]
        x_pos = np.arange(len(thresholds)) + i * width
        axes[1].bar(x_pos, eff_dims, width, label=model)

    axes[1].set_xlabel('Variance Explained')
    axes[1].set_ylabel('Number of Dimensions')
    axes[1].set_title('Effective Dimensions\n(at variance thresholds)')
    axes[1].set_xticks(np.arange(len(thresholds)) + width)
    axes[1].set_xticklabels(thresholds)
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved dimensionality comparison to {save_path}")

    return fig


def plot_timescale_comparison(
    comparison_results: Dict[str, Any],
    ground_truth_timescales: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 5)
):
    """
    Plot timescale discovery comparison.

    Args:
        comparison_results: Output from compare_models()
        ground_truth_timescales: Optional ground truth timescales to plot
        save_path: Optional path to save figure
        figsize: Figure size
    """
    set_publication_style()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Extract SLDS timescales
    slds_analysis = comparison_results.get('slds_timescales', {})
    slds_ts = slds_analysis.get('dynamics_timescales', {}).get('all_timescales', np.array([]))

    # Plot 1: Timescale distribution
    if len(slds_ts) > 0:
        axes[0].hist(slds_ts, bins=20, alpha=0.6, label='SLDS', color='#ff7f0e')

    if ground_truth_timescales is not None:
        for gt_ts in ground_truth_timescales:
            axes[0].axvline(gt_ts, color='red', linestyle='--', linewidth=2, alpha=0.7)
        axes[0].axvline(gt_ts, color='red', linestyle='--', linewidth=2, alpha=0.7,
                       label='Ground Truth')

    axes[0].set_xlabel('Timescale (timesteps)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Discovered Timescale Distribution')
    axes[0].legend()
    axes[0].set_xscale('log')

    # Plot 2: Per-state timescales (SLDS)
    state_timescales = slds_analysis.get('dynamics_timescales', {}).get('state_timescales', [])
    if state_timescales:
        state_indices = []
        state_ts_values = []
        for k, ts_k in enumerate(state_timescales):
            if len(ts_k) > 0:
                state_indices.extend([k] * len(ts_k))
                state_ts_values.extend(ts_k)

        if state_indices:
            axes[1].scatter(state_indices, state_ts_values, alpha=0.6, s=50, color='#ff7f0e')
            axes[1].set_xlabel('Discrete State Index')
            axes[1].set_ylabel('Timescale (timesteps)')
            axes[1].set_title('SLDS: Timescales per State')
            axes[1].set_yscale('log')

    # Plot 3: State duration timescales
    state_durations = slds_analysis.get('state_durations', {}).get('state_durations', np.array([]))
    if len(state_durations) > 0:
        K = len(state_durations)
        axes[2].bar(np.arange(K), state_durations, color='#ff7f0e', alpha=0.6)
        axes[2].set_xlabel('Discrete State Index')
        axes[2].set_ylabel('Expected Duration (timesteps)')
        axes[2].set_title('SLDS: State Duration Timescales')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved timescale comparison to {save_path}")

    return fig


def plot_state_transition_matrix(
    transition_matrix: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 7)
):
    """
    Plot SLDS state transition matrix heatmap.

    Args:
        transition_matrix: Transition probability matrix [K, K]
        save_path: Optional path to save figure
        figsize: Figure size
    """
    set_publication_style()

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        transition_matrix,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        square=True,
        cbar_kws={'label': 'Transition Probability'},
        vmin=0,
        vmax=1,
        ax=ax
    )

    ax.set_xlabel('Next State')
    ax.set_ylabel('Current State')
    ax.set_title('SLDS State Transition Matrix')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved transition matrix to {save_path}")

    return fig


def plot_latent_trajectories_2d(
    slds_latents: np.ndarray,
    slds_states: np.ndarray,
    tidhy_r: Optional[np.ndarray] = None,
    tidhy_r2: Optional[np.ndarray] = None,
    time_window: Tuple[int, int] = (0, 1000),
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 5)
):
    """
    Plot 2D latent trajectory comparison.

    Args:
        slds_latents: SLDS continuous latents [T, D]
        slds_states: SLDS discrete states [T]
        tidhy_r: Optional TiDHy fast latents [T, r_dim]
        tidhy_r2: Optional TiDHy slow latents [T, r2_dim]
        time_window: Time window to plot (start, end)
        save_path: Optional path to save figure
        figsize: Figure size
    """
    set_publication_style()

    t_start, t_end = time_window
    n_plots = 1 + (tidhy_r is not None) + (tidhy_r2 is not None)

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Plot SLDS latents (first 2 dimensions)
    slds_2d = slds_latents[t_start:t_end, :2]
    states_window = slds_states[t_start:t_end]

    scatter = axes[plot_idx].scatter(
        slds_2d[:, 0],
        slds_2d[:, 1],
        c=states_window,
        cmap='tab10',
        s=10,
        alpha=0.6
    )
    axes[plot_idx].set_xlabel('Dimension 1')
    axes[plot_idx].set_ylabel('Dimension 2')
    axes[plot_idx].set_title('SLDS Latent Trajectory\n(colored by discrete state)')
    plt.colorbar(scatter, ax=axes[plot_idx], label='Discrete State')
    plot_idx += 1

    # Plot TiDHy r (first 2 dimensions)
    if tidhy_r is not None:
        r_2d = tidhy_r[t_start:t_end, :2]
        axes[plot_idx].scatter(
            r_2d[:, 0],
            r_2d[:, 1],
            c=np.arange(len(r_2d)),
            cmap='viridis',
            s=10,
            alpha=0.6
        )
        axes[plot_idx].set_xlabel('r Dimension 1')
        axes[plot_idx].set_ylabel('r Dimension 2')
        axes[plot_idx].set_title('TiDHy Fast Latent (r)\n(colored by time)')
        plot_idx += 1

    # Plot TiDHy r2 (first 2 dimensions)
    if tidhy_r2 is not None:
        r2_2d = tidhy_r2[t_start:t_end, :2]
        axes[plot_idx].scatter(
            r2_2d[:, 0],
            r2_2d[:, 1],
            c=np.arange(len(r2_2d)),
            cmap='plasma',
            s=10,
            alpha=0.6
        )
        axes[plot_idx].set_xlabel('r2 Dimension 1')
        axes[plot_idx].set_ylabel('r2 Dimension 2')
        axes[plot_idx].set_title('TiDHy Slow Latent (r2)\n(colored by time)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved latent trajectories to {save_path}")

    return fig


def plot_ground_truth_comparison(
    comparison_results: Dict[str, Any],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 4)
):
    """
    Plot ground truth comparison metrics.

    Args:
        comparison_results: Output from compare_models()
        save_path: Optional path to save figure
        figsize: Figure size
    """
    set_publication_style()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # State accuracy
    models = []
    state_acc = []
    if 'slds_state_accuracy' in comparison_results:
        models.append('SLDS')
        state_acc.append(comparison_results['slds_state_accuracy'])

    if models:
        axes[0].bar(models, state_acc, color='#ff7f0e')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Discrete State Recovery')
        axes[0].set_ylim([0, 1.0])
        for i, v in enumerate(state_acc):
            axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

    # Latent MSE
    models = []
    latent_mse = []
    if 'tidhy_latent_mse' in comparison_results:
        models.append('TiDHy')
        latent_mse.append(comparison_results['tidhy_latent_mse'])
    if 'slds_latent_mse' in comparison_results:
        models.append('SLDS')
        latent_mse.append(comparison_results['slds_latent_mse'])

    if models:
        axes[1].bar(models, latent_mse, color=['#1f77b4', '#ff7f0e'][:len(models)])
        axes[1].set_ylabel('MSE')
        axes[1].set_title('Continuous Latent Recovery')
        axes[1].set_yscale('log')
        for i, v in enumerate(latent_mse):
            axes[1].text(i, v * 1.1, f'{v:.2e}', ha='center', va='bottom', fontsize=9)

    # Latent correlation
    models = []
    latent_corr = []
    if 'tidhy_mean_correlation' in comparison_results:
        models.append('TiDHy')
        latent_corr.append(comparison_results['tidhy_mean_correlation'])
    if 'slds_mean_correlation' in comparison_results:
        models.append('SLDS')
        latent_corr.append(comparison_results['slds_mean_correlation'])

    if models:
        axes[2].bar(models, latent_corr, color=['#1f77b4', '#ff7f0e'][:len(models)])
        axes[2].set_ylabel('Mean Correlation')
        axes[2].set_title('Latent-to-Ground Truth Correlation')
        axes[2].set_ylim([0, 1.0])
        for i, v in enumerate(latent_corr):
            axes[2].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ground truth comparison to {save_path}")

    return fig


def create_comparison_report(
    comparison_results: Dict[str, Any],
    output_dir: Path,
    ground_truth_timescales: Optional[np.ndarray] = None
):
    """
    Create a comprehensive comparison report with all figures.

    Args:
        comparison_results: Output from compare_models()
        output_dir: Directory to save all figures
        ground_truth_timescales: Optional ground truth timescales
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating comparison report in {output_dir}")

    # Reconstruction comparison
    plot_reconstruction_comparison(
        comparison_results,
        save_path=output_dir / 'reconstruction_comparison.png'
    )
    plt.close()

    # Dimensionality comparison
    plot_dimensionality_comparison(
        comparison_results,
        save_path=output_dir / 'dimensionality_comparison.png'
    )
    plt.close()

    # Timescale comparison
    plot_timescale_comparison(
        comparison_results,
        ground_truth_timescales=ground_truth_timescales,
        save_path=output_dir / 'timescale_comparison.png'
    )
    plt.close()

    # Ground truth comparison
    if 'slds_state_accuracy' in comparison_results or 'slds_latent_mse' in comparison_results:
        plot_ground_truth_comparison(
            comparison_results,
            save_path=output_dir / 'ground_truth_comparison.png'
        )
        plt.close()

    # Transition matrix
    if 'slds_timescales' in comparison_results:
        slds_analysis = comparison_results['slds_timescales']
        if 'state_durations' in slds_analysis:
            # Need to load transition matrix separately
            print("Note: Transition matrix plot requires loading SLDS results separately")

    print(f"Report created successfully in {output_dir}")
