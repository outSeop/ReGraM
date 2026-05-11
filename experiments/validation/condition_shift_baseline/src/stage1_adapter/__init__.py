"""Memory-based Stage 1 component proposal adapter."""

from stage1_adapter.descriptors import CandidateDescriptor, describe_candidate_masks
from stage1_adapter.prototypes import (
    ComponentPrototype,
    build_component_prototypes,
    score_candidate_against_prototypes,
    summarize_adapter_scores,
)

__all__ = [
    "CandidateDescriptor",
    "ComponentPrototype",
    "build_component_prototypes",
    "describe_candidate_masks",
    "score_candidate_against_prototypes",
    "summarize_adapter_scores",
]
