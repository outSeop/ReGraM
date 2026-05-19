"""Memory-based Stage 1 component proposal adapter."""

from stage1_adapter.candidate_masks import CandidateMaskNormalizationConfig, normalize_candidate_masks
from stage1_adapter.descriptors import CandidateDescriptor, describe_candidate_masks
from stage1_adapter.dino_patch_matching import (
    DinoPatchMatchResult,
    DinoRepresentationShiftResult,
    analyze_dino_representation_shift,
    match_dino_patch_grid,
    summarize_dino_patch_match,
    summarize_dino_representation_shift,
)
from stage1_adapter.memory_bank import (
    InstanceEntry,
    MemoryBank,
    PrototypeMemoryEntry,
    SpatialGraphEdge,
    build_memory_bank,
    memory_bank_summary,
)
from stage1_adapter.memory_ot_matcher import (
    AnomalyDecomposition,
    MatchingConfig,
    PrototypeMatchingResult,
    active_assignment_mask,
    decomposition_summary,
    match_query_against_memory,
    query_patch_grid_from_feature_map,
)
from stage1_adapter.patch_graph import (
    PatchGraphConfig,
    PatchGraphProbeResult,
    build_reliable_patch_membership,
    patch_region_masks,
    run_masked_patch_prototype_probe,
    select_reliable_candidate_ids,
    smooth_teacher_with_membership,
    summarize_patch_edge_metrics,
    summarize_patch_edges,
    summarize_teacher_quality,
)
from stage1_adapter.prototypes import (
    ComponentPrototype,
    build_component_prototypes,
    score_candidate_against_prototypes,
    summarize_adapter_scores,
)

__all__ = [
    "CandidateDescriptor",
    "CandidateMaskNormalizationConfig",
    "ComponentPrototype",
    "DinoPatchMatchResult",
    "DinoRepresentationShiftResult",
    "AnomalyDecomposition",
    "InstanceEntry",
    "MatchingConfig",
    "MemoryBank",
    "PatchGraphConfig",
    "PatchGraphProbeResult",
    "PrototypeMatchingResult",
    "PrototypeMemoryEntry",
    "SpatialGraphEdge",
    "active_assignment_mask",
    "analyze_dino_representation_shift",
    "build_reliable_patch_membership",
    "build_component_prototypes",
    "build_memory_bank",
    "decomposition_summary",
    "describe_candidate_masks",
    "match_query_against_memory",
    "match_dino_patch_grid",
    "memory_bank_summary",
    "normalize_candidate_masks",
    "patch_region_masks",
    "query_patch_grid_from_feature_map",
    "run_masked_patch_prototype_probe",
    "score_candidate_against_prototypes",
    "select_reliable_candidate_ids",
    "smooth_teacher_with_membership",
    "summarize_patch_edge_metrics",
    "summarize_patch_edges",
    "summarize_teacher_quality",
    "summarize_adapter_scores",
    "summarize_dino_patch_match",
    "summarize_dino_representation_shift",
]
