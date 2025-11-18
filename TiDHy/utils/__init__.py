#__init__.py

from .io_dict_to_hdf5 import *
# from .utils import *

# Analysis module for model selection and timescale discovery
from .analysis import (
    # Effective dimension analysis
    compute_effective_dimension,
    participation_ratio,
    analyze_latent_dimension,
    # Spectral timescale analysis
    compute_timescale_spectrum,
    analyze_mixture_timescales,
    cluster_timescales,
    # Hypernetwork-aware timescale analysis
    analyze_effective_timescales,
    analyze_time_varying_timescales,
    # Hypernetwork analysis
    analyze_hypernetwork_usage,
    compute_component_entropy,
    # Ground truth comparison (validation)
    compute_rossler_ground_truth_timescales,
    compare_discovered_to_ground_truth,
    analyze_hierarchical_rossler_recovery
)

# # Visualization functions
from .analysis_plotting import (
    plot_eigenvalue_spectrum,
    plot_timescale_distribution,
    plot_complex_eigenvalues,
    plot_hypernetwork_usage,
    plot_dimensionality_analysis,
    plot_effective_vs_basis_timescales,
    plot_discovered_vs_ground_truth
)


__version__ = '0.1.0'
__author__ = 'Elliott T. T. Abe'