"""Phase 2 support-only compressed option graph baseline.

Phase 2 constructs compressed directed option graphs from offline trajectory
support only. It does not train policies, implement latent models, or run
environment rollouts.
"""

__all__ = [
    "node_selection",
    "edge_dataset",
    "option_graph",
    "planning",
    "compatibility",
    "evaluation",
    "plotting",
]

