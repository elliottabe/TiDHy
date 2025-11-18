"""
TiDHy vs SLDS Comprehensive Comparison Script

Compares TiDHy and SLDS models across multiple metrics:
- Reconstruction quality
- Latent space recovery
- Timescale extraction
- Interpretability measures

Author: Elliott Abe
"""

import argparse
from pathlib import Path
import json
import numpy as np
import jax.numpy as jnp
import h5py
from typing import Dict, Any, Optional, List
import sys
from natsort import natsorted

# Add TiDHy to path
sys.path.insert(0, str(Path(__file__).parent))

from TiDHy.utils.slds_analysis import (
    load_slds_results,
    analyze_slds_comprehensive,
    compare_slds_to_ground_truth,
)
from TiDHy.utils.analysis import (
    analyze_mixture_timescales,
    analyze_effective_timescales,
    analyze_weighted_mixture_timescales,
    analyze_latent_dimension,
    compare_discovered_to_ground_truth,
)
from TiDHy.utils.state_annotation_comparison import (
    analyze_state_annotation_correspondence,
    plot_confusion_matrix,
)
import TiDHy.utils.io_dict_to_hdf5 as ioh5


def load_tidhy_results(filepath: Path) -> Dict[str, Any]:
    """
    Load TiDHy results from HDF5 file.

    Args:
        filepath: Path to TiDHy results HDF5 file

    Returns:
        Dictionary with TiDHy outputs
    """
    results = {}
    results = ioh5.load(filepath)
    seq_len = natsorted(list(results.keys()))
    
    return results['{}'.format(seq_len[-1])]


def compute_reconstruction_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    model_name: str
) -> Dict[str, float]:
    """
    Compute reconstruction quality metrics.

    Args:
        predictions: Model predictions
        ground_truth: True observations
        model_name: Name for logging

    Returns:
        Dictionary of metrics
    """
    mse = np.mean((predictions - ground_truth) ** 2)
    rmse = np.sqrt(mse)

    # R-squared
    ss_res = np.sum((ground_truth - predictions) ** 2)
    ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Per-dimension metrics
    mse_per_dim = np.mean((predictions - ground_truth) ** 2, axis=0)

    return {
        f'{model_name}_mse': float(mse),
        f'{model_name}_rmse': float(rmse),
        f'{model_name}_r2': float(r2),
        f'{model_name}_mse_per_dim': mse_per_dim.tolist(),
    }


def compute_latent_recovery_metrics(
    inferred_latents: np.ndarray,
    true_latents: np.ndarray,
    model_name: str
) -> Dict[str, float]:
    """
    Compute latent space recovery metrics.

    Args:
        inferred_latents: Inferred continuous latents
        true_latents: True continuous latents
        model_name: Name for logging

    Returns:
        Dictionary of metrics
    """
    # MSE in latent space
    latent_mse = np.mean((inferred_latents - true_latents) ** 2)

    # Correlation per dimension
    T, D = true_latents.shape
    correlations = []
    for d in range(min(D, inferred_latents.shape[1])):
        corr = np.corrcoef(true_latents[:, d], inferred_latents[:, d])[0, 1]
        correlations.append(corr)

    mean_correlation = np.mean(correlations)

    return {
        f'{model_name}_latent_mse': float(latent_mse),
        f'{model_name}_mean_correlation': float(mean_correlation),
        f'{model_name}_correlations_per_dim': [float(c) for c in correlations],
    }


def compute_state_recovery_metrics(
    inferred_states: np.ndarray,
    true_states: np.ndarray,
    model_name: str,
    use_comprehensive: bool = False
) -> Dict[str, float]:
    """
    Compute discrete state recovery accuracy.

    Args:
        inferred_states: Inferred discrete states
        true_states: True discrete states
        model_name: Name for logging
        use_comprehensive: If True, use comprehensive state-annotation comparison

    Returns:
        Dictionary of metrics
    """
    if use_comprehensive:
        # Use new comprehensive analysis
        results = analyze_state_annotation_correspondence(
            states=inferred_states,
            annotations=true_states,
            behavior_names=None,
            verbose=False
        )

        # Extract key metrics with model_name prefix
        return {
            f'{model_name}_state_accuracy': results['clustering_metrics'].get('accuracy', 0.0),
            f'{model_name}_adjusted_rand_index': results['clustering_metrics']['adjusted_rand_index'],
            f'{model_name}_normalized_mutual_info': results['clustering_metrics']['normalized_mutual_info'],
            f'{model_name}_v_measure': results['clustering_metrics']['v_measure'],
            f'{model_name}_state_purity': results['purity_metrics']['purity'],
            f'{model_name}_annotation_purity': results['purity_metrics']['inverse_purity'],
            f'{model_name}_macro_f1': results['per_behavior_metrics']['macro_avg_f1'],
            f'{model_name}_full_analysis': results,  # Store full results for later
        }
    else:
        # Simple accuracy-based comparison
        accuracy = np.mean(inferred_states == true_states)

        # Confusion matrix (simplified)
        num_states = max(int(true_states.max()), int(inferred_states.max())) + 1
        confusion = np.zeros((num_states, num_states))
        for t_true, i_inf in zip(true_states, inferred_states):
            confusion[int(t_true), int(i_inf)] += 1

        return {
            f'{model_name}_state_accuracy': float(accuracy),
            f'{model_name}_confusion_matrix': confusion.tolist(),
        }


def compute_dimensionality_metrics(
    latent_trajectory: np.ndarray,
    model_name: str
) -> Dict[str, Any]:
    """
    Compute effective dimensionality of latent representation.

    Args:
        latent_trajectory: Latent trajectory [T, D]
        model_name: Name for logging

    Returns:
        Dictionary with dimensionality metrics
    """

    # Use the proper high-level function that handles eigenvalue computation
    analysis = analyze_latent_dimension(
        jnp.array(latent_trajectory),
        method='pca',
        variance_thresholds=(0.90, 0.95, 0.99)
    )

    return {
        f'{model_name}_eff_dim_90': int(analysis['effective_dim_90']),
        f'{model_name}_eff_dim_95': int(analysis['effective_dim_95']),
        f'{model_name}_eff_dim_99': int(analysis['effective_dim_99']),
        f'{model_name}_participation_ratio': float(analysis['participation_ratio']),
        f'{model_name}_nominal_dim': latent_trajectory.shape[1],
    }


def analyze_tidhy_timescales(
    tidhy_results: Dict[str, Any],
    dt: float = 1.0
) -> Dict[str, Any]:
    """
    Analyze TiDHy timescales using multiple methods.

    Args:
        tidhy_results: TiDHy results dictionary
        dt: Time step size

    Returns:
        Dictionary with timescale analyses
    """
    # This requires the TiDHy model object, which we need to load separately
    # For now, we'll return a placeholder
    # TODO: Implement full TiDHy timescale extraction from saved parameters

    return {
        'status': 'TiDHy timescale analysis requires model object',
        'note': 'Implement loading temporal matrices V_m from saved parameters',
    }


def compare_models(
    tidhy_path: Path,
    slds_path: Path,
    dt: float = 1.0,
    dataset_name: str = '',
    ground_truth_timescales: Optional[np.ndarray] = None,
    annotations: Optional[np.ndarray] = None,
    behavior_names: Optional[List[str]] = None,
    use_comprehensive_state_comparison: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive comparison between TiDHy and SLDS models.

    Args:
        tidhy_path: Path to TiDHy results HDF5
        slds_path: Path to SLDS results HDF5
        dt: Time step size
        dataset_name: Name of dataset for logging
        ground_truth_timescales: Optional ground truth timescales for comparison
        annotations: Optional behavioral annotations for state comparison (e.g., CalMS21)
        behavior_names: Optional list of behavior names for annotation indices
        use_comprehensive_state_comparison: If True, use comprehensive state-annotation analysis

    Returns:
        Comprehensive comparison dictionary
    """
    print(f"Comparing models on dataset: {dataset_name}")
    print(f"  TiDHy: {tidhy_path}")
    print(f"  SLDS:  {slds_path}")

    # Load results
    print("\nLoading results...")
    model_type='SLDS'
    tidhy_results = load_tidhy_results(tidhy_path)
    slds_results = load_slds_results(slds_path, model_type=model_type)

    comparison = {
        'dataset': dataset_name,
        'tidhy_path': str(tidhy_path),
        'slds_path': str(slds_path),
        'dt': dt,
    }

    # ===== Reconstruction Metrics =====
    print("\nComputing reconstruction metrics...")
    if 'I_hat' in tidhy_results and 'I' in tidhy_results:
        tidhy_recon = compute_reconstruction_metrics(
            tidhy_results['I_hat'],
            tidhy_results['I'],
            'tidhy'
        )
        comparison.update(tidhy_recon)

    slds_recon = compute_reconstruction_metrics(
        slds_results[f'{model_type}_emission'],
        slds_results['ground_truth_inputs'],
        'slds'
    )
    comparison.update(slds_recon)

    # ===== Latent Recovery Metrics =====
    print("Computing latent recovery metrics...")
    if 'ground_truth_states_x' in slds_results:
        # SLDS latent recovery
        slds_latent = compute_latent_recovery_metrics(
            slds_results[f'{model_type}_latents'],
            slds_results['ground_truth_states_x'],
            'slds'
        )
        comparison.update(slds_latent)

        # TiDHy latent recovery (using R_hat)
        if 'R_hat' in tidhy_results and 'ground_truth_states_x' in tidhy_results:
            tidhy_latent = compute_latent_recovery_metrics(
                tidhy_results['R_hat'],
                tidhy_results['ground_truth_states_x'],
                'tidhy'
            )
            comparison.update(tidhy_latent)

    # ===== State Recovery Metrics =====
    print("Computing state recovery metrics...")
    if 'ground_truth_states_z' in slds_results:
        # SLDS state recovery
        slds_state = compute_state_recovery_metrics(
            slds_results[f'{model_type}_states'],
            slds_results['ground_truth_states_z'],
            'slds',
            use_comprehensive=use_comprehensive_state_comparison
        )
        comparison.update(slds_state)

        # TiDHy doesn't have discrete states, but we could cluster r2
        # TODO: Implement r2 clustering to discrete states

    # ===== Behavioral Annotation Comparison =====
    if annotations is not None:
        print("Computing state-annotation comparison...")
        if f'{model_type}_states' in slds_results:
            print("  Analyzing SLDS states vs. behavioral annotations...")
            slds_annotation_results = analyze_state_annotation_correspondence(
                states=slds_results[f'{model_type}_states'],
                annotations=annotations,
                behavior_names=behavior_names,
                verbose=True
            )
            comparison['slds_annotation_comparison'] = slds_annotation_results

            # Save confusion matrix plot if possible
            try:
                from matplotlib import pyplot as plt
                fig = plot_confusion_matrix(
                    matched_states=slds_annotation_results['matching']['matched_states'],
                    annotations=annotations,
                    behavior_names=behavior_names,
                    normalize='true',
                    figsize=(10, 8)
                )
                confusion_path = Path(f'{dataset_name}_slds_confusion_matrix.png')
                fig.savefig(confusion_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  Confusion matrix saved to {confusion_path}")
                comparison['slds_confusion_matrix_path'] = str(confusion_path)
            except Exception as e:
                print(f"  Warning: Could not save confusion matrix: {e}")

    # ===== Dimensionality Metrics =====
    print("Computing dimensionality metrics...")
    slds_dim = compute_dimensionality_metrics(slds_results[f'{model_type}_latents'], 'slds')
    comparison.update(slds_dim)

    if 'R_hat' in tidhy_results:
        tidhy_r_dim = compute_dimensionality_metrics(tidhy_results['R_hat'], 'tidhy_r')
        comparison.update(tidhy_r_dim)
    if 'R2_hat' in tidhy_results:
        tidhy_r2_dim = compute_dimensionality_metrics(tidhy_results['R2_hat'], 'tidhy_r2')
        comparison.update(tidhy_r2_dim)

    # ===== Timescale Analysis =====
    print("Analyzing timescales...")

    # SLDS timescales
    slds_timescale_analysis = analyze_slds_comprehensive(slds_results, dt=dt)
    comparison['slds_timescales'] = slds_timescale_analysis

    # TiDHy timescales
    tidhy_timescale_analysis = analyze_tidhy_timescales(tidhy_results, dt=dt)
    comparison['tidhy_timescales'] = tidhy_timescale_analysis

    # Ground truth comparison
    if ground_truth_timescales is not None:
        print("Comparing to ground truth timescales...")
        slds_gt_comparison = compare_slds_to_ground_truth(
            slds_timescale_analysis,
            ground_truth_timescales,
            tolerance=0.2
        )
        comparison['slds_gt_comparison'] = slds_gt_comparison

        # TODO: Add TiDHy ground truth comparison once timescale extraction is implemented

    print("\nComparison complete!")
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description='Compare TiDHy and SLDS models across multiple metrics'
    )
    parser.add_argument(
        '--tidhy',
        type=Path,
        required=True,
        help='Path to TiDHy results HDF5 file'
    )
    parser.add_argument(
        '--slds',
        type=Path,
        required=True,
        help='Path to SLDS results HDF5 file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('comparison_results.json'),
        help='Output path for comparison JSON (default: comparison_results.json)'
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=0.03333333333333333,
        help='Time step size (default: 0.03333333333333333 for 30 Hz data)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='',
        help='Dataset name for logging'
    )
    parser.add_argument(
        '--ground-truth-timescales',
        type=str,
        default=None,
        help='Comma-separated list of ground truth timescales (e.g., "10.5,25.3,100.0")'
    )
    parser.add_argument(
        '--annotations',
        type=Path,
        default=None,
        help='Path to numpy file (.npy) containing behavioral annotations'
    )
    parser.add_argument(
        '--behavior-names',
        type=str,
        default=None,
        help='Comma-separated list of behavior names (e.g., "investigation,attack,mount")'
    )
    parser.add_argument(
        '--no-comprehensive-state-comparison',
        action='store_true',
        help='Use simple state accuracy instead of comprehensive comparison'
    )

    args = parser.parse_args()

    # Parse ground truth timescales if provided
    gt_timescales = None
    if args.ground_truth_timescales:
        gt_timescales = np.array([float(x) for x in args.ground_truth_timescales.split(',')])

    # Load annotations if provided
    annotations = None
    if args.annotations:
        print(f"Loading annotations from {args.annotations}")
        annotations = np.load(args.annotations)
        print(f"  Loaded {len(annotations)} annotation timesteps")

    # Parse behavior names if provided
    behavior_names = None
    if args.behavior_names:
        behavior_names = [name.strip() for name in args.behavior_names.split(',')]
        print(f"  Using behavior names: {behavior_names}")

    # Run comparison
    results = compare_models(
        tidhy_path=args.tidhy,
        slds_path=args.slds,
        dt=args.dt,
        dataset_name=args.dataset,
        ground_truth_timescales=gt_timescales,
        annotations=annotations,
        behavior_names=behavior_names,
        use_comprehensive_state_comparison=not args.no_comprehensive_state_comparison,
    )

    # Save results
    print(f"\nSaving results to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))

    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Dataset: {results['dataset']}")
    print(f"\nReconstruction Quality:")
    if 'tidhy_r2' in results:
        print(f"  TiDHy R²: {results.get('tidhy_r2', 'N/A')}")
    print(f"  SLDS R²:  {results.get('slds_r2', 'N/A')}")
    print(f"\nEffective Dimensionality:")
    if 'tidhy_r_participation_ratio' in results:
        print(f"  TiDHy (r):  {results.get('tidhy_r_participation_ratio', 'N/A'):.2f}")
    print(f"  SLDS:       {results.get('slds_participation_ratio', 'N/A'):.2f}")
    print(f"\nTimescale Discovery:")
    if 'slds_timescales' in results and 'summary' in results['slds_timescales']:
        summary = results['slds_timescales']['summary']
        print(f"  SLDS discovered: {summary.get('num_dynamics_timescales', 'N/A')} timescales")
        print(f"  Range: {summary.get('min_dynamics_timescale', 'N/A'):.2f} - {summary.get('max_dynamics_timescale', 'N/A'):.2f}")

    # Annotation comparison summary
    if 'slds_annotation_comparison' in results:
        print(f"\nState-Annotation Correspondence (SLDS):")
        annot_comp = results['slds_annotation_comparison']
        print(f"  Adjusted Rand Index: {annot_comp['clustering_metrics']['adjusted_rand_index']:.3f}")
        print(f"  Normalized Mutual Info: {annot_comp['clustering_metrics']['normalized_mutual_info']:.3f}")
        print(f"  V-measure: {annot_comp['clustering_metrics']['v_measure']:.3f}")
        if 'accuracy' in annot_comp['clustering_metrics']:
            print(f"  Accuracy (after matching): {annot_comp['clustering_metrics']['accuracy']:.3f}")
        print(f"  State Purity: {annot_comp['purity_metrics']['purity']:.3f}")
        print(f"  Macro-avg F1: {annot_comp['per_behavior_metrics']['macro_avg_f1']:.3f}")
        if 'slds_confusion_matrix_path' in results:
            print(f"  Confusion matrix: {results['slds_confusion_matrix_path']}")

    print("="*60)


if __name__ == '__main__':
    main()
