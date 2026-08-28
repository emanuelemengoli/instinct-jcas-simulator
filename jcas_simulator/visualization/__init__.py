"""Optional visualization API for structured JCAS simulation results."""

from .non_captive_plots import plot_non_captive_estimation_comparison

from .association_plots import (
    plot_corr_scatter,
    plot_filter_queue_association_scatter,
    plot_interference_association_scatter,
    plot_sinr_association_scatter,
)

from .metric_plots import (
    plot_covariance_trace_kde,
    plot_sinr_kde,
    plot_workload_kde,
)
from .network_plots import plot_voronoi_network
from .trajectory_plots import animate_entity_trajectories, plot_entity_trajectories

__all__ = [
    "plot_non_captive_estimation_comparison",
    "plot_voronoi_network",
    "plot_sinr_kde",
    "plot_covariance_trace_kde",
    "plot_workload_kde",
    "plot_entity_trajectories",
    "animate_entity_trajectories",
    "plot_corr_scatter",
    "plot_interference_association_scatter",
    "plot_sinr_association_scatter",
    "plot_filter_queue_association_scatter",
]
