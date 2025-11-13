import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Analysis Visualization Functions (Model Selection & Timescale Discovery)
# ============================================================================

def plot_eigenvalue_spectrum(
    eigenvalues,
    variance_explained=None,
    ax=None,
    title='Eigenvalue Spectrum',
    show_cumulative=True
):
    """
    Plot eigenvalue spectrum with optional cumulative variance.
    
    Scree plot showing eigenvalue magnitudes and cumulative variance explained.
    
    Args:
        eigenvalues: Array of eigenvalues (sorted descending)
        variance_explained: Cumulative variance explained (optional)
        ax: Matplotlib axis (creates new if None)
        title: Plot title
        show_cumulative: Whether to show cumulative variance line
    
    Returns:
        Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    n_eigs = len(eigenvalues)
    x = np.arange(1, n_eigs + 1)
    
    # Plot eigenvalues
    ax.bar(x, np.abs(eigenvalues), alpha=0.7, color='steelblue', label='Eigenvalues')
    ax.set_xlabel('Component')
    ax.set_ylabel('Eigenvalue Magnitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add cumulative variance if provided
    if show_cumulative and variance_explained is not None:
        ax2 = ax.twinx()
        ax2.plot(x, variance_explained * 100, 'r-', marker='o', 
                linewidth=2, markersize=4, label='Cumulative Variance')
        ax2.set_ylabel('Cumulative Variance Explained (%)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_ylim([0, 105])
        ax2.grid(False)
        
        # Add horizontal line at 95%
        ax2.axhline(95, color='r', linestyle='--', alpha=0.5, linewidth=1)
        ax2.text(n_eigs * 0.7, 96, '95%', color='r', fontsize=9)
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    
    return ax


def plot_timescale_distribution(
    timescales,
    ax=None,
    title='Timescale Distribution',
    log_scale=True,
    bins=50
):
    """
    Plot distribution of discovered timescales.
    
    Histogram/KDE of timescales, typically shown in log scale since
    timescales can span multiple orders of magnitude.
    
    Args:
        timescales: Array or dict of timescales
        ax: Matplotlib axis (creates new if None)
        title: Plot title
        log_scale: Whether to use log scale for x-axis
        bins: Number of histogram bins
    
    Returns:
        Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    # Handle dict input (from analyze_mixture_timescales)
    if isinstance(timescales, dict):
        if 'all_timescales' in timescales:
            timescales = timescales['all_timescales']
        else:
            raise ValueError("Dict must contain 'all_timescales' key")
    
    # Convert to numpy
    timescales = np.array(timescales)
    
    # Filter out infinite values
    finite_mask = np.isfinite(timescales)
    timescales = timescales[finite_mask]
    
    if len(timescales) == 0:
        ax.text(0.5, 0.5, 'No finite timescales', 
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Plot histogram
    ax.hist(timescales, bins=bins, alpha=0.7, color='steelblue', 
            edgecolor='black', density=True)
    
    # Add KDE if enough points
    if len(timescales) > 5:
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(np.log10(timescales) if log_scale else timescales)
            if log_scale:
                x_range = np.logspace(np.log10(timescales.min()), 
                                     np.log10(timescales.max()), 200)
                kde_vals = kde(np.log10(x_range))
            else:
                x_range = np.linspace(timescales.min(), timescales.max(), 200)
                kde_vals = kde(x_range)
            ax.plot(x_range, kde_vals, 'r-', linewidth=2, label='KDE', alpha=0.8)
        except:
            pass  # Skip KDE if scipy not available
    
    if log_scale:
        ax.set_xscale('log')
        ax.set_xlabel('Timescale τ (log scale)')
    else:
        ax.set_xlabel('Timescale τ')
    
    ax.set_ylabel('Probability Density')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if len(timescales) > 5:
        ax.legend()
    
    # Add statistics text
    stats_text = f'n = {len(timescales)}\n'
    stats_text += f'min = {timescales.min():.2f}\n'
    stats_text += f'max = {timescales.max():.2f}\n'
    stats_text += f'median = {np.median(timescales):.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return ax


def plot_complex_eigenvalues(
    eigenvalues,
    ax=None,
    title='Eigenvalue Spectrum (Complex Plane)',
    show_unit_circle=True,
    component_labels=None
):
    """
    Plot eigenvalues in the complex plane with unit circle.
    
    Eigenvalues inside the unit circle correspond to stable modes.
    The angle determines oscillation frequency, magnitude determines decay rate.
    
    Args:
        eigenvalues: Complex eigenvalues or list of eigenvalue arrays
        ax: Matplotlib axis (creates new if None)
        title: Plot title
        show_unit_circle: Whether to show unit circle (stability boundary)
        component_labels: Labels for different eigenvalue sets (if list provided)
    
    Returns:
        Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Handle single array vs list of arrays
    if not isinstance(eigenvalues, list):
        eigenvalues = [eigenvalues]
        if component_labels is None:
            component_labels = ['Eigenvalues']
    elif component_labels is None:
        component_labels = [f'Component {i+1}' for i in range(len(eigenvalues))]
    
    # Color cycle
    colors = plt.cm.tab10(np.linspace(0, 1, len(eigenvalues)))
    
    # Plot eigenvalues
    for eigs, label, color in zip(eigenvalues, component_labels, colors):
        eigs = np.array(eigs)
        ax.scatter(eigs.real, eigs.imag, alpha=0.7, s=50, 
                  label=label, color=color, edgecolor='black', linewidth=0.5)
    
    # Unit circle
    if show_unit_circle:
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=2, 
               alpha=0.5, label='Unit Circle (Stability)')
    
    # Axes
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
    
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return ax


def plot_hypernetwork_usage(
    W,
    ax=None,
    title='Hypernetwork Component Usage',
    threshold=0.01
):
    """
    Plot hypernetwork mixture component usage over time.
    
    Shows bar chart of mean weights and heatmap of temporal evolution.
    
    Args:
        W: Mixture weights - shape (T, mix_dim) or dict from analyze_hypernetwork_usage
        ax: Matplotlib axis or tuple of axes (creates new if None)
        title: Main plot title
        threshold: Threshold for highlighting active components
    
    Returns:
        Matplotlib axis or axes
    """
    # Handle dict input
    if isinstance(W, dict):
        mean_weights = W['mean_weights']
        usage_fraction = W['usage_fraction']
        mix_dim = W['mix_dim']
        W_matrix = None  # Don't have time series
    else:
        # Reshape if batched
        if W.ndim == 3:
            W = W.reshape(-1, W.shape[-1])
        mean_weights = np.mean(W, axis=0)
        usage_fraction = np.mean(W > threshold, axis=0)
        mix_dim = W.shape[1]
        W_matrix = W
    
    # Create subplots if needed
    if ax is None:
        if W_matrix is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
            ax2 = None
    elif isinstance(ax, (list, tuple, np.ndarray)) and len(ax) == 2:
        ax1, ax2 = ax
    else:
        ax1 = ax
        ax2 = None
    
    # Bar chart of mean weights
    x = np.arange(mix_dim)
    colors = ['steelblue' if w > threshold else 'lightgray' for w in mean_weights]
    ax1.bar(x, mean_weights, color=colors, edgecolor='black', linewidth=1)
    ax1.axhline(threshold, color='r', linestyle='--', linewidth=1, 
               alpha=0.5, label=f'Threshold ({threshold})')
    ax1.set_xlabel('Mixture Component')
    ax1.set_ylabel('Mean Weight')
    ax1.set_title(f'{title} - Average')
    ax1.set_xticks(x)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()
    
    # Add usage fraction as text
    for i, (w, uf) in enumerate(zip(mean_weights, usage_fraction)):
        ax1.text(i, w + 0.01, f'{uf*100:.0f}%', ha='center', 
                fontsize=8, rotation=0)
    
    # Heatmap of temporal evolution (if full matrix available)
    if ax2 is not None and W_matrix is not None:
        # Subsample if too many timesteps
        T = W_matrix.shape[0]
        if T > 1000:
            idx = np.linspace(0, T-1, 1000, dtype=int)
            W_plot = W_matrix[idx, :]
        else:
            W_plot = W_matrix
        
        im = ax2.imshow(W_plot.T, aspect='auto', cmap='viridis', 
                       interpolation='nearest', origin='lower')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Mixture Component')
        ax2.set_title(f'{title} - Temporal Evolution')
        ax2.set_yticks(np.arange(mix_dim))
        plt.colorbar(im, ax=ax2, label='Weight')
    
    plt.tight_layout()
    
    if ax2 is not None:
        return ax1, ax2
    else:
        return ax1


def plot_dimensionality_analysis(
    analysis_results,
    figsize=(15, 5)
):
    """
    Comprehensive plot of dimensionality analysis results.
    
    Creates a 3-panel figure showing:
    1. Eigenvalue spectrum with cumulative variance
    2. Participation ratio and effective dimensions
    3. Eigenvalue decay (log scale)
    
    Args:
        analysis_results: Dict from analyze_latent_dimension()
        figsize: Figure size
    
    Returns:
        Figure and axes
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    eigenvalues = analysis_results['eigenvalues']
    variance_explained = analysis_results['variance_explained']
    pr = analysis_results['participation_ratio']
    
    # Panel 1: Eigenvalue spectrum
    plot_eigenvalue_spectrum(eigenvalues, variance_explained, ax=ax1,
                            title='Eigenvalue Spectrum')
    
    # Panel 2: Summary statistics
    ax2.axis('off')
    stats_text = 'Dimensionality Analysis\n' + '='*30 + '\n\n'
    stats_text += f'Total dimensions: {analysis_results["total_dims"]}\n\n'
    stats_text += f'Participation Ratio: {pr:.2f}\n\n'
    
    for key in sorted(analysis_results.keys()):
        if key.startswith('effective_dim_'):
            threshold = key.split('_')[-1]
            value = analysis_results[key]
            stats_text += f'Effective dim ({threshold}%): {value}\n'
    
    stats_text += f'\nMethod: {analysis_results["method"]}'
    
    ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes,
            fontsize=11, verticalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Panel 3: Log-scale eigenvalue decay
    n_eigs = len(eigenvalues)
    ax3.semilogy(np.arange(1, n_eigs+1), np.abs(eigenvalues), 
                'o-', linewidth=2, markersize=4, color='steelblue')
    ax3.set_xlabel('Component')
    ax3.set_ylabel('Eigenvalue (log scale)')
    ax3.set_title('Eigenvalue Decay')
    ax3.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    return fig, (ax1, ax2, ax3)


def plot_effective_vs_basis_timescales(
    basis_results,
    effective_results=None,
    weighted_results=None,
    time_varying_results=None,
    figsize=(16, 10)
):
    """
    Compare timescales from different analysis methods.

    Creates a comprehensive visualization comparing:
    - Basis V_m timescales (standard analysis)
    - Effective V_t timescales (actual temporal matrices used)
    - Weighted timescales (V_m weighted by usage)
    - Time-varying timescales (optional, shows temporal evolution)

    Args:
        basis_results: Output from analyze_mixture_timescales()
        effective_results: Output from analyze_effective_timescales() (optional)
        weighted_results: Output from analyze_weighted_mixture_timescales() (optional)
        time_varying_results: Output from analyze_time_varying_timescales() (optional)
        figsize: Figure size (default: (16, 10))

    Returns:
        Matplotlib figure and axes
    """
    # Determine number of panels based on available data
    n_panels = 2  # Always have basis distribution and comparison
    if time_varying_results is not None:
        n_panels += 1

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Panel 1: Basis timescale distribution (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    basis_timescales = basis_results['all_timescales']
    if len(basis_timescales) > 0:
        ax1.hist(np.log10(basis_timescales), bins=30, alpha=0.7,
                 color='steelblue', edgecolor='black', label='Basis V_m')
        ax1.set_xlabel('log₁₀(Timescale)')
        ax1.set_ylabel('Count')
        ax1.set_title('Basis Timescale Distribution')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

    # Panel 2: Effective timescale distribution (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    if effective_results is not None and len(effective_results['all_timescales']) > 0:
        effective_timescales = effective_results['all_timescales']
        ax2.hist(np.log10(effective_timescales), bins=30, alpha=0.7,
                 color='coral', edgecolor='black', label='Effective V_t')
        ax2.set_xlabel('log₁₀(Timescale)')
        ax2.set_ylabel('Count')
        ax2.set_title('Effective Timescale Distribution')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No effective timescales\nprovided',
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Effective Timescales')

    # Panel 3: Comparison statistics (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')

    # Compile statistics
    stats_text = 'Timescale Comparison\n' + '='*30 + '\n\n'

    # Basis timescales
    if len(basis_timescales) > 0:
        stats_text += f'Basis (V_m) timescales:\n'
        stats_text += f'  Range: [{np.min(basis_timescales):.2f}, {np.max(basis_timescales):.2f}]\n'
        stats_text += f'  Mean: {np.mean(basis_timescales):.2f}\n'
        stats_text += f'  N unique: {len(basis_results["unique_timescales"])}\n\n'

    # Effective timescales
    if effective_results is not None and len(effective_results['all_timescales']) > 0:
        eff_timescales = effective_results['all_timescales']
        stats_text += f'Effective (V_t) timescales:\n'
        stats_text += f'  Range: [{effective_results["timescale_stats"]["min"]:.2f}, '
        stats_text += f'{effective_results["timescale_stats"]["max"]:.2f}]\n'
        stats_text += f'  Mean: {effective_results["timescale_stats"]["mean"]:.2f}\n'
        stats_text += f'  N analyzed: {effective_results["n_matrices_analyzed"]}\n\n'

    # Weighted timescales
    if weighted_results is not None and weighted_results['weighted_mean'] is not None:
        stats_text += f'Weighted timescales:\n'
        stats_text += f'  Weighted mean: {weighted_results["weighted_mean"]:.2f}\n'
        stats_text += f'  Unweighted mean: {weighted_results["comparison"]["unweighted_mean"]:.2f}\n'
        stats_text += f'  Usage entropy: {weighted_results["comparison"]["usage_entropy"]:.3f}\n'

    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
            fontsize=9, verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Panel 4: Overlay comparison (bottom left)
    ax4 = fig.add_subplot(gs[1, 0])
    if len(basis_timescales) > 0:
        ax4.hist(np.log10(basis_timescales), bins=30, alpha=0.5,
                 color='steelblue', label='Basis V_m', density=True)
    if effective_results is not None and len(effective_results['all_timescales']) > 0:
        ax4.hist(np.log10(effective_results['all_timescales']), bins=30, alpha=0.5,
                 color='coral', label='Effective V_t', density=True)
    if weighted_results is not None and len(weighted_results['weighted_timescales']) > 0:
        ax4.hist(np.log10(weighted_results['weighted_timescales']), bins=30, alpha=0.5,
                 color='green', label='Weighted', density=True)

    ax4.set_xlabel('log₁₀(Timescale)')
    ax4.set_ylabel('Density')
    ax4.set_title('Timescale Comparison (Normalized)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Panel 5: Component usage (bottom middle)
    ax5 = fig.add_subplot(gs[1, 1])
    if weighted_results is not None:
        component_usage = weighted_results['component_usage']
        n_components = len(component_usage)
        ax5.bar(range(n_components), component_usage, color='skyblue', edgecolor='black')
        ax5.set_xlabel('Mixture Component')
        ax5.set_ylabel('Average Usage Weight')
        ax5.set_title('Hypernetwork Component Usage')
        ax5.grid(True, alpha=0.3, axis='y')
    else:
        ax5.text(0.5, 0.5, 'No weighted analysis\nprovided',
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Component Usage')

    # Panel 6: Time-varying timescales (bottom right)
    ax6 = fig.add_subplot(gs[1, 2])
    if time_varying_results is not None:
        dominant = time_varying_results['dominant_timescales']
        time_indices = time_varying_results['time_indices']
        n_dominant = time_varying_results['n_dominant']

        for i in range(n_dominant):
            timescale_series = dominant[:, i]
            # Only plot valid (finite) values
            valid_mask = np.isfinite(timescale_series)
            if np.any(valid_mask):
                ax6.plot(time_indices[valid_mask], timescale_series[valid_mask],
                        'o-', markersize=3, label=f'τ_{i+1} (rank {i+1})', alpha=0.7)

        # Mark transitions
        transitions = time_varying_results['transitions']
        if len(transitions) > 0:
            for trans_t in transitions:
                ax6.axvline(trans_t, color='red', linestyle='--', alpha=0.5)

        ax6.set_xlabel('Time Step')
        ax6.set_ylabel('Timescale')
        ax6.set_title('Time-Varying Timescales')
        ax6.set_yscale('log')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3, which='both')
    else:
        ax6.text(0.5, 0.5, 'No time-varying\nanalysis provided',
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Time-Varying Timescales')

    return fig, (ax1, ax2, ax3, ax4, ax5, ax6)


def plot_discovered_vs_ground_truth(
    comparison_results,
    method_name='Discovered',
    figsize=(14, 10)
):
    """
    Visualize comparison between discovered and ground truth timescales.

    Creates a 4-panel figure showing:
    - Scatter plot: discovered vs ground truth (with y=x ideal line)
    - Bar chart: relative errors for matched pairs
    - Timescale spectrum: all discovered with ground truth marked
    - Metrics table: precision, recall, mean error

    Args:
        comparison_results: Output from compare_discovered_to_ground_truth() dict,
                          can also be dict with multiple methods:
                          {'basis': comparison, 'effective': comparison, ...}
        method_name: Name of method for title (default: 'Discovered')
        figsize: Figure size (default: (14, 10))

    Returns:
        Matplotlib figure and axes
    """
    # Check if we have multiple methods to compare
    if isinstance(comparison_results, dict) and 'matched_pairs' in comparison_results:
        # Single method
        comparisons = {method_name: comparison_results}
    else:
        # Multiple methods
        comparisons = comparison_results

    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel 1: Scatter plot (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    colors = plt.cm.tab10(np.linspace(0, 1, len(comparisons)))

    all_values = []
    for (name, comparison), color in zip(comparisons.items(), colors):
        if comparison is None:
            continue

        matched = comparison['matched_pairs']
        if len(matched) > 0:
            disc_vals = [pair[0] for pair in matched]
            true_vals = [pair[1] for pair in matched]
            ax1.scatter(true_vals, disc_vals, alpha=0.7, s=100,
                       label=name, color=color, edgecolors='black')
            all_values.extend(disc_vals + true_vals)

    if len(all_values) > 0:
        # Add y=x line (ideal)
        min_val = min(all_values) * 0.9
        max_val = max(all_values) * 1.1
        ax1.plot([min_val, max_val], [min_val, max_val],
                'k--', linewidth=2, label='Ideal (y=x)', alpha=0.5)

        # Add tolerance bounds
        first_comparison = next(iter(comparisons.values()))
        if first_comparison is not None:
            tolerance = first_comparison.get('tolerance', 0.2)
            ax1.fill_between([min_val, max_val],
                           [min_val * (1-tolerance), max_val * (1-tolerance)],
                           [min_val * (1+tolerance), max_val * (1+tolerance)],
                           alpha=0.1, color='gray', label=f'±{tolerance*100:.0f}% tolerance')

        ax1.set_xlabel('Ground Truth Timescale')
        ax1.set_ylabel('Discovered Timescale')
        ax1.set_title('Discovered vs Ground Truth')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')

    # Panel 2: Relative errors bar chart (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    bar_width = 0.8 / len(comparisons)
    x_base = np.arange(len(comparisons))

    for i, (name, comparison) in enumerate(comparisons.items()):
        if comparison is None or len(comparison['relative_errors']) == 0:
            continue

        errors = comparison['relative_errors']
        # One bar per matched pair
        n_matches = len(errors)
        x_positions = np.arange(n_matches) + i * bar_width

        ax2.bar(x_positions, [e * 100 for e in errors],
               bar_width, label=name, alpha=0.7)

    ax2.set_xlabel('Matched Pair Index')
    ax2.set_ylabel('Relative Error (%)')
    ax2.set_title('Relative Errors for Matched Timescales')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(20, color='red', linestyle='--', alpha=0.5, label='20% threshold')

    # Panel 3: Timescale spectrum with ground truth markers (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])

    for (name, comparison), color in zip(comparisons.items(), colors):
        if comparison is None:
            continue

        # Get all discovered timescales for this method
        # (need to access from parent analysis results)
        # For now, just plot matched + false positives
        all_disc = [pair[0] for pair in comparison['matched_pairs']] + comparison['false_positives']
        if len(all_disc) > 0:
            # Create histogram
            log_timescales = np.log10(all_disc)
            ax3.hist(log_timescales, bins=20, alpha=0.5, label=name, color=color)

    # Mark ground truth timescales with vertical lines
    first_comparison = next(iter(comparisons.values()))
    if first_comparison is not None:
        gt_timescales = []
        # Reconstruct ground truth from matched pairs and false negatives
        for comparison in comparisons.values():
            if comparison is not None:
                gt_timescales.extend([pair[1] for pair in comparison['matched_pairs']])
                gt_timescales.extend(comparison['false_negatives'])
                break  # All methods have same ground truth

        gt_unique = list(set(gt_timescales))
        for gt in gt_unique:
            ax3.axvline(np.log10(gt), color='red', linestyle='--',
                       linewidth=2, alpha=0.7)

        # Add legend entry for ground truth
        ax3.axvline(np.nan, color='red', linestyle='--',
                   linewidth=2, label='Ground Truth')

    ax3.set_xlabel('log₁₀(Timescale)')
    ax3.set_ylabel('Count')
    ax3.set_title('Timescale Spectrum with Ground Truth')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Metrics table (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # Build metrics table
    table_text = 'Recovery Metrics\n' + '='*40 + '\n\n'

    for name, comparison in comparisons.items():
        if comparison is None:
            table_text += f'{name}: No results\n\n'
            continue

        table_text += f'{name}:\n'
        table_text += f'  Recovery rate: {comparison["recovery_rate"]*100:.1f}%\n'
        table_text += f'  Precision: {comparison["precision"]*100:.1f}%\n'
        table_text += f'  Recall: {comparison["recall"]*100:.1f}%\n'

        if comparison['mean_relative_error'] is not None:
            table_text += f'  Mean rel. error: {comparison["mean_relative_error"]*100:.1f}%\n'

        table_text += f'  Matched: {comparison["n_matched"]}/{comparison["n_ground_truth"]}\n'
        table_text += f'  False positives: {len(comparison["false_positives"])}\n'
        table_text += f'  False negatives: {len(comparison["false_negatives"])}\n\n'

    ax4.text(0.05, 0.95, table_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Timescale Recovery Analysis', fontsize=14, fontweight='bold')

    return fig, (ax1, ax2, ax3, ax4)
