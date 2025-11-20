"""
State-Annotation Comparison for Behavioral Datasets

This module provides tools for comparing learned discrete states (from SLDS or
other models) with ground-truth behavioral annotations (e.g., from CalMS21).

Since discrete states are learned unsupervised, they don't correspond directly
to annotation labels. This module provides:
- Hungarian algorithm matching to align states with annotations
- Clustering quality metrics (ARI, NMI, V-measure)
- State purity and coverage analysis
- Per-behavior performance metrics
- Visualization utilities

Author: Elliott Abe
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


def match_states_to_annotations(
    states: np.ndarray,
    annotations: np.ndarray,
    return_contingency: bool = False
) -> Dict[str, Any]:
    """
    Match learned discrete states to ground-truth annotations using Hungarian algorithm.

    The Hungarian algorithm finds the optimal one-to-one assignment between states
    and annotations that maximizes their co-occurrence (overlap).

    Args:
        states: Array of shape [T] with discrete state indices
        annotations: Array of shape [T] with annotation/behavior indices
        return_contingency: If True, include contingency table in output

    Returns:
        Dictionary with:
        - state_to_annotation: Dict mapping state index -> annotation index
        - annotation_to_state: Dict mapping annotation index -> state index
        - matched_states: Array [T] with states remapped to annotation indices
        - contingency: [num_states, num_annotations] co-occurrence matrix (if requested)
        - num_states: Number of unique states
        - num_annotations: Number of unique annotations
    """
    # Ensure inputs are 1D arrays
    states = np.asarray(states).flatten()
    annotations = np.asarray(annotations).flatten()

    if len(states) != len(annotations):
        raise ValueError(
            f"States and annotations must have same length: "
            f"{len(states)} vs {len(annotations)}"
        )

    # Get number of unique states and annotations
    num_states = int(states.max()) + 1
    num_annotations = int(annotations.max()) + 1

    # Build contingency table (co-occurrence matrix)
    # contingency[i, j] = number of timesteps where state=i and annotation=j
    contingency = np.zeros((num_states, num_annotations), dtype=np.int64)
    for s, a in zip(states, annotations):
        contingency[int(s), int(a)] += 1

    # Use Hungarian algorithm to find optimal matching
    # We want to maximize overlap, so use negative contingency as cost
    row_ind, col_ind = linear_sum_assignment(-contingency)

    # Create bidirectional mapping
    state_to_annotation = {int(s): int(a) for s, a in zip(row_ind, col_ind)}
    annotation_to_state = {int(a): int(s) for s, a in zip(row_ind, col_ind)}

    # Remap states to matched annotation indices
    # States that weren't matched are mapped to -1
    matched_states = np.array([state_to_annotation.get(int(s), -1) for s in states])

    result = {
        'state_to_annotation': state_to_annotation,
        'annotation_to_state': annotation_to_state,
        'matched_states': matched_states,
        'num_states': num_states,
        'num_annotations': num_annotations,
    }

    if return_contingency:
        result['contingency'] = contingency

    return result


def compute_clustering_metrics(
    states: np.ndarray,
    annotations: np.ndarray,
    matched_states: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute clustering quality metrics comparing states to annotations.

    These metrics don't require aligned labels and measure clustering quality:
    - Adjusted Rand Index (ARI): Similarity between two clusterings, adjusted for chance
    - Normalized Mutual Information (NMI): Information shared between clusterings
    - V-measure: Harmonic mean of homogeneity and completeness
    - Accuracy: Only computed if matched_states is provided

    Args:
        states: Array [T] with original discrete state indices
        annotations: Array [T] with ground-truth annotation indices
        matched_states: Optional array [T] with states remapped to annotations
                       (from match_states_to_annotations). If provided, accuracy is computed.

    Returns:
        Dictionary with metric scores (all in range [0, 1], higher is better)
    """
    states = np.asarray(states).flatten()
    annotations = np.asarray(annotations).flatten()

    # Compute alignment-free metrics
    ari = adjusted_rand_score(annotations, states)
    nmi = normalized_mutual_info_score(annotations, states)
    v_measure = v_measure_score(annotations, states)

    metrics = {
        'adjusted_rand_index': float(ari),
        'normalized_mutual_info': float(nmi),
        'v_measure': float(v_measure),
    }

    # Compute accuracy if matched states provided
    if matched_states is not None:
        matched_states = np.asarray(matched_states).flatten()
        # Only compute accuracy on matched portions (exclude -1)
        valid_mask = matched_states >= 0
        if valid_mask.sum() > 0:
            accuracy = accuracy_score(
                annotations[valid_mask],
                matched_states[valid_mask]
            )
            metrics['accuracy'] = float(accuracy)
            metrics['matched_fraction'] = float(valid_mask.mean())

    return metrics


def compute_state_purity(
    states: np.ndarray,
    annotations: np.ndarray
) -> Dict[str, Any]:
    """
    Compute purity and coverage metrics for state-annotation correspondence.

    Purity measures how "pure" each state is (does it mainly contain one behavior?).
    Coverage measures how well states cover the different behaviors.

    Args:
        states: Array [T] with discrete state indices
        annotations: Array [T] with annotation indices

    Returns:
        Dictionary with:
        - purity: Overall purity score [0, 1]
        - per_state_purity: Array [num_states] with purity of each state
        - per_state_dominant_annotation: Array [num_states] with most common annotation per state
        - per_annotation_coverage: Array [num_annotations] with fraction covered by best state
        - inverse_purity: Overall inverse purity (annotation-to-state purity)
        - per_annotation_purity: Array [num_annotations] with purity of each annotation
    """
    states = np.asarray(states).flatten()
    annotations = np.asarray(annotations).flatten()

    num_states = int(states.max()) + 1
    num_annotations = int(annotations.max()) + 1

    # Build contingency table
    contingency = np.zeros((num_states, num_annotations), dtype=np.int64)
    for s, a in zip(states, annotations):
        contingency[int(s), int(a)] += 1

    # Compute state purity: for each state, fraction in most common annotation
    per_state_purity = np.zeros(num_states)
    per_state_dominant_annotation = np.zeros(num_states, dtype=np.int64)

    for s in range(num_states):
        counts = contingency[s]
        if counts.sum() > 0:
            per_state_purity[s] = counts.max() / counts.sum()
            per_state_dominant_annotation[s] = counts.argmax()

    # Overall purity: weighted average by state occupancy
    state_counts = np.array([np.sum(states == s) for s in range(num_states)])
    total_count = state_counts.sum()
    if total_count > 0:
        purity = np.sum(per_state_purity * state_counts) / total_count
    else:
        purity = 0.0

    # Compute annotation purity (inverse purity): for each annotation, fraction in most common state
    per_annotation_purity = np.zeros(num_annotations)
    per_annotation_dominant_state = np.zeros(num_annotations, dtype=np.int64)

    for a in range(num_annotations):
        counts = contingency[:, a]
        if counts.sum() > 0:
            per_annotation_purity[a] = counts.max() / counts.sum()
            per_annotation_dominant_state[a] = counts.argmax()

    # Overall inverse purity: weighted average by annotation occupancy
    annotation_counts = np.array([np.sum(annotations == a) for a in range(num_annotations)])
    if annotation_counts.sum() > 0:
        inverse_purity = np.sum(per_annotation_purity * annotation_counts) / annotation_counts.sum()
    else:
        inverse_purity = 0.0

    # Compute coverage: for each annotation, what fraction is covered by best state
    per_annotation_coverage = np.zeros(num_annotations)
    for a in range(num_annotations):
        if annotation_counts[a] > 0:
            per_annotation_coverage[a] = contingency[:, a].max() / annotation_counts[a]

    return {
        'purity': float(purity),
        'per_state_purity': per_state_purity,
        'per_state_dominant_annotation': per_state_dominant_annotation,
        'inverse_purity': float(inverse_purity),
        'per_annotation_purity': per_annotation_purity,
        'per_annotation_dominant_state': per_annotation_dominant_state,
        'per_annotation_coverage': per_annotation_coverage,
        'num_states': num_states,
        'num_annotations': num_annotations,
    }


def compute_per_behavior_metrics(
    matched_states: np.ndarray,
    annotations: np.ndarray,
    behavior_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute precision, recall, and F1-score for each behavior after state-annotation matching.

    Args:
        matched_states: Array [T] with states remapped to annotation indices
        annotations: Array [T] with ground-truth annotation indices
        behavior_names: Optional list of behavior names for each annotation index

    Returns:
        Dictionary with:
        - per_behavior_precision: Array [num_behaviors] with precision for each
        - per_behavior_recall: Array [num_behaviors] with recall for each
        - per_behavior_f1: Array [num_behaviors] with F1-score for each
        - per_behavior_support: Array [num_behaviors] with number of samples per behavior
        - macro_avg_precision: Macro-averaged precision
        - macro_avg_recall: Macro-averaged recall
        - macro_avg_f1: Macro-averaged F1-score
        - weighted_avg_precision: Weighted-averaged precision
        - weighted_avg_recall: Weighted-averaged recall
        - weighted_avg_f1: Weighted-averaged F1-score
        - behavior_names: List of behavior names (if provided)
    """
    matched_states = np.asarray(matched_states).flatten()
    annotations = np.asarray(annotations).flatten()

    # Only compute on valid matches (exclude -1)
    valid_mask = matched_states >= 0
    if valid_mask.sum() == 0:
        raise ValueError("No valid matched states found (all are -1)")

    matched_states_valid = matched_states[valid_mask]
    annotations_valid = annotations[valid_mask]

    # Get unique behavior labels
    unique_labels = np.unique(annotations)
    num_behaviors = len(unique_labels)

    # Compute precision, recall, f1 for each behavior
    precision, recall, f1, support = precision_recall_fscore_support(
        annotations_valid,
        matched_states_valid,
        labels=unique_labels,
        average=None,
        zero_division=0
    )

    # Compute macro and weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        annotations_valid,
        matched_states_valid,
        labels=unique_labels,
        average='macro',
        zero_division=0
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        annotations_valid,
        matched_states_valid,
        labels=unique_labels,
        average='weighted',
        zero_division=0
    )

    result = {
        'per_behavior_precision': precision,
        'per_behavior_recall': recall,
        'per_behavior_f1': f1,
        'per_behavior_support': support,
        'macro_avg_precision': float(macro_precision),
        'macro_avg_recall': float(macro_recall),
        'macro_avg_f1': float(macro_f1),
        'weighted_avg_precision': float(weighted_precision),
        'weighted_avg_recall': float(weighted_recall),
        'weighted_avg_f1': float(weighted_f1),
        'num_behaviors': num_behaviors,
    }

    if behavior_names is not None:
        result['behavior_names'] = behavior_names

    return result


def plot_confusion_matrix(
    matched_states: np.ndarray,
    annotations: np.ndarray,
    behavior_names: Optional[List[str]] = None,
    normalize: str = 'true',
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'Blues',
    fontsize: int = 13,
    model: str = 'TiDHy',
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot confusion matrix comparing matched states to annotations.

    Args:
        matched_states: Array [T] with states remapped to annotation indices
        annotations: Array [T] with ground-truth annotation indices
        behavior_names: Optional list of behavior names for labeling
        normalize: 'true' (row-normalize), 'pred' (col-normalize), 'all', or None
        figsize: Figure size (width, height)
        cmap: Colormap name
        ax: Optional matplotlib axes to plot on

    Returns:
        Matplotlib figure
    """
    matched_states = np.asarray(matched_states).flatten()
    annotations = np.asarray(annotations).flatten()

    # Only use valid matches
    valid_mask = matched_states >= 0
    if valid_mask.sum() == 0:
        raise ValueError("No valid matched states found (all are -1)")

    matched_states_valid = matched_states[valid_mask]
    annotations_valid = annotations[valid_mask]

    # Compute confusion matrix
    cm = confusion_matrix(annotations_valid, matched_states_valid)

    # Normalize if requested
    if normalize == 'true':
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
    elif normalize == 'pred':
        cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)
        fmt = '.2f'
    elif normalize == 'all':
        cm = cm.astype('float') / cm.sum()
        fmt = '.2f'
    else:
        fmt = 'd'

    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        ax=ax,
        xticklabels=behavior_names if behavior_names else 'auto',
        yticklabels=behavior_names if behavior_names else 'auto',
        vmin=0,
        vmax=1 if normalize else None,
        cbar_kws={'label': 'fraction' if normalize else 'count'}
    )
    
    ax.set_xlabel('predicted state', fontsize=fontsize-2)
    ax.set_ylabel('true annotation', fontsize=fontsize-2)
    ax.set_title(f'{model} confusion matrix', fontsize=fontsize)

    # plt.tight_layout()

    return fig


def analyze_state_annotation_correspondence(
    states: np.ndarray,
    annotations: np.ndarray,
    behavior_names: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive analysis of correspondence between learned states and annotations.

    This is the main analysis function that combines all metrics and provides
    a complete characterization of state-annotation alignment.

    Args:
        states: Array [T] with discrete state indices
        annotations: Array [T] with annotation indices
        behavior_names: Optional list of behavior names
        verbose: If True, print summary statistics

    Returns:
        Dictionary containing all analysis results:
        - matching: Results from match_states_to_annotations()
        - clustering_metrics: Results from compute_clustering_metrics()
        - purity_metrics: Results from compute_state_purity()
        - per_behavior_metrics: Results from compute_per_behavior_metrics()
    """
    # Step 1: Match states to annotations
    matching = match_states_to_annotations(states, annotations, return_contingency=True)

    # Step 2: Compute clustering metrics
    clustering_metrics = compute_clustering_metrics(
        states, annotations, matched_states=matching['matched_states']
    )

    # Step 3: Compute purity metrics
    purity_metrics = compute_state_purity(states, annotations)

    # Step 4: Compute per-behavior metrics
    per_behavior_metrics = compute_per_behavior_metrics(
        matching['matched_states'], annotations, behavior_names
    )

    # Compile results
    results = {
        'matching': matching,
        'clustering_metrics': clustering_metrics,
        'purity_metrics': purity_metrics,
        'per_behavior_metrics': per_behavior_metrics,
    }

    # Print summary if verbose
    if verbose:
        print("=" * 70)
        print("State-Annotation Correspondence Analysis")
        print("=" * 70)
        print(f"\nData Summary:")
        print(f"  Total timesteps: {len(states)}")
        print(f"  Number of learned states: {matching['num_states']}")
        print(f"  Number of annotation behaviors: {matching['num_annotations']}")

        print(f"\nClustering Quality Metrics:")
        print(f"  Adjusted Rand Index: {clustering_metrics['adjusted_rand_index']:.3f}")
        print(f"  Normalized Mutual Info: {clustering_metrics['normalized_mutual_info']:.3f}")
        print(f"  V-measure: {clustering_metrics['v_measure']:.3f}")
        if 'accuracy' in clustering_metrics:
            print(f"  Accuracy (after matching): {clustering_metrics['accuracy']:.3f}")
            print(f"  Matched fraction: {clustering_metrics['matched_fraction']:.3f}")

        print(f"\nPurity Metrics:")
        print(f"  State purity: {purity_metrics['purity']:.3f}")
        print(f"  Annotation purity (inverse): {purity_metrics['inverse_purity']:.3f}")

        print(f"\nPer-Behavior Performance (Macro-averaged):")
        print(f"  Precision: {per_behavior_metrics['macro_avg_precision']:.3f}")
        print(f"  Recall: {per_behavior_metrics['macro_avg_recall']:.3f}")
        print(f"  F1-score: {per_behavior_metrics['macro_avg_f1']:.3f}")

        if behavior_names is not None:
            print(f"\nPer-Behavior Breakdown:")
            for i, name in enumerate(behavior_names[:len(per_behavior_metrics['per_behavior_f1'])]):
                print(f"  {name}:")
                print(f"    Precision: {per_behavior_metrics['per_behavior_precision'][i]:.3f}")
                print(f"    Recall: {per_behavior_metrics['per_behavior_recall'][i]:.3f}")
                print(f"    F1: {per_behavior_metrics['per_behavior_f1'][i]:.3f}")
                print(f"    Support: {per_behavior_metrics['per_behavior_support'][i]}")

        print("=" * 70)

    return results
