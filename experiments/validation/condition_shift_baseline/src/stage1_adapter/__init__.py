"""Memory-based Stage 1 component proposal adapter."""

from stage1_adapter.descriptors import CandidateDescriptor, describe_candidate_masks
from stage1_adapter.patch_graph import (
    PatchGraphConfig,
    PatchGraphProbeResult,
    build_reliable_patch_membership,
    run_masked_patch_prototype_probe,
    select_reliable_candidate_ids,
    summarize_patch_edges,
)
from stage1_adapter.prototypes import (
    ComponentPrototype,
    build_component_prototypes,
    score_candidate_against_prototypes,
    summarize_adapter_scores,
)

__all__ = [
    "CandidateDescriptor",
    "ComponentPrototype",
    "PatchGraphConfig",
    "PatchGraphProbeResult",
    "build_reliable_patch_membership",
    "build_component_prototypes",
    "describe_candidate_masks",
    "run_masked_patch_prototype_probe",
    "score_candidate_against_prototypes",
    "select_reliable_candidate_ids",
    "summarize_patch_edges",
    "summarize_adapter_scores",
]
