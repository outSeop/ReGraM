# Memory-Anchored OT Matching for Logical Anomaly Detection — W1 Spec

**Project:** ReGraM (master's thesis, condition-shift-robust logical AD on MVTec LOCO)
**Stage:** W1 — memory bank build + clean self-probe sanity check
**Target notebook:** `experiments/validation/condition_shift_baseline/notebook/09_memory_anchored_ot_matching.ipynb`

---

## 1. Background

기존 graph-matching 방식의 bottleneck은 query-side SAM/DINO 검출이 condition shift (blur/noise/position_shift/brightness)에 약하다는 점이었다.

새 접근의 핵심 idea:

- **Memory**는 clean normal에서 한 번 구축 (통제된 조건)
- **Query**는 raw DINO patch만 사용, query-side SAM 의존성 제거
- Memory의 prototype별 patch-bag과 query patch를 **unbalanced optimal transport**로 매칭
- 4가지 anomaly signal + spatial graph constraint로 logical anomaly 검출

핵심 design decision (확정됨):

- **Thing/stuff 구분 없음**: 모든 prototype이 uniform patch-bag으로 표현됨 (사과 = 작은 bag, granola = 큰 bag, 동일한 OT framework로 처리)
- **Instance identity 보존**: memory에서 instance별로 patches 분리 저장 → count anomaly 검출 가능
- **Stage A (W1-W5)**: prototype-level spatial graph (2D Gaussian over Δposition)
- **Stage B (W6+)**: position-stable category에서 instance-level graph activate (ablation)

---

## 2. Architecture Overview

```
memory:
  patches[prototype_id] = [
      instance_1: {patch_features: (n_p, D), patch_positions: (n_p, 2), centroid_pos: (2,)},
      instance_2: {...},
      ...
  ]
  prototype_centroid[p]: (D,)              # mean of instance-mean features
  prototype_within_var[p]: float           # within-cluster cosine sim variance
  expected_instance_count[p]: float        # per-image average instance count
  spatial_graph[(p_i, p_j)] = {
      delta_pos_mean: (2,),                # mean Δpos over all instance pairs in clean
      delta_pos_cov:  (2, 2),
      num_pairs: int
  }

query:
  raw DINO patches: (n_q, D) + positions: (n_q, 2)

inference:
  1. soft assignment:
       sim[i, p] = cosine(query_patch_i, prototype_centroid[p])
       default: active(i, p) = p is in top-k prototypes for query_patch_i (k=3)
       optional ablation: τ_p = quantile(within-cluster similarity distribution, soft_assign_quantile)
                         active(i, p) = (sim[i, p] > τ_p)

  2. per prototype p:
       Q_p = query patches with active(·, p), weights = sim[·, p]
       M_p = concat of all instance patches, with instance_id metadata preserved
       (plan, cost) = unbalanced_sinkhorn(Q_p, M_p, ε, τ_marginal)

       signal_1[p] = mass(Q_p) - matched_mass_to_M_p(plan)
       signal_2[p] = mass(M_p) - matched_mass_to_Q_p(plan)
       matched_instance_count[p] = # of memory instances with received_mass > min_matched_mass
       matched_per_image[p] = matched_instance_count[p] / num_occurring_images[p]
       signal_3[p] = |expected_instance_count[p] - matched_per_image[p]|
       matched_extent[p] = weighted_centroid(query_positions, plan)

  3. spatial constraint:
       for (p_i, p_j) in spatial_graph:
         observed_delta = matched_extent[p_j] - matched_extent[p_i]
         signal_4[(p_i, p_j)] = mahalanobis(observed_delta, spatial_graph[(p_i, p_j)])

  4. aggregate:
       anomaly_score = Σ_p (signal_1[p] + signal_2[p] + signal_3[p])
                     + Σ_pair signal_4[pair]
```

---

## 3. W1 Scope

W1은 다음 두 가지 deliverable:

**(a) Memory build pipeline** — clean image set에서 위 memory structure 전체 구축.

**(b) Clean self-probe** — clean image 1장을 query로 넣어 inference 전체 실행하고 anomaly score가 합리적으로 낮게 나오는지 확인.

W2 이후는 본 문서 §6에 요약. W1에서는 W2 이후 기능 구현 금지 (scope creep 방지).

---

## 4. Reuse from Existing Codebase

기존 `08_stage1_component_adapter_probe` 노트북에서 재사용 가능한 모듈:

- `stage1_adapter.CandidateMaskNormalizationConfig` — mask normalization 설정
- `stage1_adapter.normalize_candidate_masks` — mask 정규화 (thing/stuff cluster 포함)
- `stage1_adapter.describe_candidate_masks` — mask별 DINO descriptor 추출
- `stage1_adapter.build_component_prototypes` — clean descriptor → prototype 클러스터링
- `relation.grounding_mask_cluster.raw_masks_from_label_image` — grounding mask 로드
- `graph_probe.component_io.save_json` — JSON 저장
- DINO feature extraction 함수 (`extract_dino_feature_map`) — 08 notebook cell 8

기존 normalization config는 그대로 사용:

```python
MASK_NORMALIZATION_CONFIG = CandidateMaskNormalizationConfig(
    max_mask_area_ratio=0.30,
    min_mask_area_ratio=0.001,
    small_cluster_area_ratio=0.006,
    min_cluster_members=3,
    min_cluster_union_area_ratio=0.004,
    max_cluster_union_area_ratio=0.25,
    max_centroid_dist_ratio=0.10,
    max_bbox_gap_ratio=0.04,
)
```

---

## 5. Implementation Tasks

### 5.1 New module: `stage1_adapter/memory_bank.py`

새 데이터 클래스:

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class InstanceEntry:
    instance_id: str               # e.g. "img_000_mask_002"
    source_image_id: str
    source_mask_id: str
    prototype_id: str              # assigned prototype after clustering
    patch_features: np.ndarray     # (n_patches, D)
    patch_positions: np.ndarray    # (n_patches, 2), normalized [0,1]
    centroid_pos: np.ndarray       # (2,) mean of patch_positions
    mask_area_ratio: float
    instance_mean_feature: np.ndarray  # (D,) mean of patch_features

@dataclass
class PrototypeMemoryEntry:
    prototype_id: str
    instances: list[InstanceEntry]
    centroid: np.ndarray           # (D,) mean over instance_mean_feature
    within_variance: float         # var of cosine-sim between instance means and centroid
    within_sim_quantiles: dict[float, float]   # {0.10: ..., 0.20: ..., 0.50: ...}
    expected_instance_count: float # mean count per clean image where prototype present
    occurrence_rate: float         # fraction of clean images containing this prototype

@dataclass
class SpatialGraphEdge:
    prototype_i: str
    prototype_j: str
    delta_pos_mean: np.ndarray     # (2,) mean of (centroid_j - centroid_i) across all instance pairs
    delta_pos_cov: np.ndarray      # (2, 2)
    num_pairs: int                 # # of (instance_i, instance_j) pairs aggregated

@dataclass
class MemoryBank:
    category: str
    prototypes: dict[str, PrototypeMemoryEntry]
    spatial_graph: dict[tuple[str, str], SpatialGraphEdge]
    feature_backbone: str
    image_size: int
    num_source_images: int
    config_dict: dict              # 재현용 metadata
```

신규 함수:

```python
def build_memory_bank(
    clean_image_paths: list[Path],
    clean_mask_paths: list[Path],
    *,
    feature_extractor: Callable[[Path], np.ndarray],   # returns (H_grid, W_grid, D)
    raw_masks_loader: Callable[[Path, tuple[int, int]], list[dict]],
    image_size: int,
    feature_backbone: str,
    max_prototypes: int = 8,
    normalization_config: CandidateMaskNormalizationConfig,
    min_pairs_for_spatial_edge: int = 3,
) -> MemoryBank:
    """
    Build memory bank from clean images.

    Algorithm:
      1. For each clean image:
         a. Load raw masks (normalized via raw_masks_loader)
         b. Extract DINO feature map
         c. For each normalized mask:
            - Compute patch_features (DINO patches inside mask region)
            - Compute patch_positions (normalized [0,1] grid coords)
            - Compute centroid_pos
            - Build raw InstanceEntry (prototype_id pending)
         d. Also use describe_candidate_masks for prototype clustering input

      2. Cluster all instance descriptors using build_component_prototypes
         (reuses 08 notebook logic). max_prototypes=8 default.

      3. Assign each InstanceEntry to nearest prototype (hard assignment, cosine).

      4. For each prototype:
         - Compute centroid = mean of instance_mean_features
         - Compute within_sim distribution → variance + quantiles
         - Compute expected_instance_count = mean per-image instance count
           among images where prototype is present
         - occurrence_rate = images_with_prototype / total_images

      5. For each unordered prototype pair (p_i, p_j):
         - For each clean image, for each (instance_i ∈ p_i, instance_j ∈ p_j) cross product
           in that image:
             Δpos = centroid_j - centroid_i
         - Aggregate all Δpos across all images
         - If num_pairs >= min_pairs_for_spatial_edge:
             fit Gaussian (mean, cov) → SpatialGraphEdge
           else:
             skip (no edge stored, signal_4 will skip this pair at inference)

      6. Return MemoryBank with metadata.
    """
```

세부 결정 (이미 확정):

- **Instance → prototype assignment**: hard assignment (best cosine match). soft가 더 robust이지만 W1은 단순화.
- **patch_positions normalization**: image grid coord를 [0,1] × [0,1]로 정규화. position-shift robustness는 inference 시 relative position만 사용해서 확보 (memory에 absolute coord 저장하되 spatial_graph는 항상 Δ로 계산).
- **prototype 클러스터링**: `build_component_prototypes` 그대로 사용. max_prototypes=8.
- **spatial edge cardinality**: 한 image에서 p_i와 p_j 각각 여러 instance면 모든 cross-pair (cartesian product)를 수집. 한 image에 instance가 너무 많으면 edge weight overrepresentation 가능 — W1에서는 단순화하고 W4-W5에서 보정 검토.

### 5.2 New module: `stage1_adapter/memory_ot_matcher.py`

```python
@dataclass
class MatchingConfig:
    assignment_mode: str = "top_k"           # "top_k" default, "threshold" ablation
    top_k_prototypes: int = 3                # each query patch activates top-k prototypes
    min_assignment_similarity: float | None = 0.5  # top-k safety gate for weak prototypes
    soft_assign_quantile: float = 0.20       # only used when assignment_mode="threshold"
    ot_reg: float = 0.05                     # Sinkhorn ε
    ot_marginal_penalty: float = 1.0         # unbalanced reg_m (KL marginal)
    min_matched_mass_for_instance: float = 0.5  # mass threshold to count instance as "present"
    skip_signal_4_if_no_extent: bool = True  # prototype 매칭 mass 너무 작으면 signal_4 skip

@dataclass
class PrototypeMatchingResult:
    prototype_id: str
    query_mass: float
    memory_mass: float
    matching_cost: float
    unmatched_query_mass: float       # signal_1 contribution
    unmatched_memory_mass: float      # signal_2 contribution
    matched_instance_count: int          # total matched memory instances
    matched_instance_count_per_image: float # matched count normalized by clean occurrence images
    expected_instance_count: float          # expected count per occurring clean image
    instance_count_diff: float        # signal_3 contribution
    matched_extent: np.ndarray | None # (2,), None if query_mass too small
    transport_plan: np.ndarray        # (n_query_p, n_memory_p), debug용

@dataclass
class AnomalyDecomposition:
    per_prototype: dict[str, PrototypeMatchingResult]
    spatial_violations: dict[tuple[str, str], float]   # signal_4 per pair
    signal_1_total: float
    signal_2_total: float
    signal_3_total: float
    signal_4_total: float
    total: float
    debug: dict                       # any extra diagnostic info

def match_query_against_memory(
    query_patch_features: np.ndarray,    # (n_q, D)
    query_patch_positions: np.ndarray,   # (n_q, 2), normalized [0,1]
    memory: MemoryBank,
    config: MatchingConfig,
) -> AnomalyDecomposition:
    """
    Algorithm:
      1. Compute sim[i, p] = cosine(query_patch_i, memory.prototypes[p].centroid)
         for all (i, p).

      2. Build active assignment mask:
         if config.assignment_mode == "top_k":
             active(i, p) = p is in top-k prototypes for query patch i
         else:
             τ_p = memory.prototypes[p].within_sim_quantiles[config.soft_assign_quantile]
             active(i, p) = (sim[i, p] > τ_p)

      3. For each prototype p:
         active_mask = active[:, p]
         if active_mask.sum() == 0:
             # nothing matches this prototype — entire memory_mass becomes unmatched
             record PrototypeMatchingResult with matched_extent=None
             continue

         Q_p_features  = query_patch_features[active_mask]
         Q_p_positions = query_patch_positions[active_mask]
         Q_p_weights   = sim[active_mask, p]  # weighted by similarity strength

         M_p_features = concat of [inst.patch_features for inst in memory.prototypes[p].instances]
         M_p_instance_ids = parallel array marking which instance each row belongs to
         M_p_weights = uniform 1/len(M_p_features) (or per-instance normalized)

         # Cost matrix: cosine distance
         C = 1 - cosine_similarity_matrix(Q_p_features, M_p_features)  # shape (|Q_p|, |M_p|)

         # Unbalanced Sinkhorn (POT)
         plan = ot.unbalanced.sinkhorn_unbalanced(
             a=Q_p_weights / Q_p_weights.sum(),
             b=M_p_weights / M_p_weights.sum(),
             M=C,
             reg=config.ot_reg,
             reg_m=config.ot_marginal_penalty,
         )

         matching_cost = (plan * C).sum()
         marginal_Q = plan.sum(axis=1)  # how much each query patch got transported
         marginal_M = plan.sum(axis=0)  # how much each memory patch received

         unmatched_query = max(0, Q_p_weights.sum() - marginal_Q.sum())  # for unbalanced
         unmatched_memory = max(0, M_p_weights.sum() - marginal_M.sum())

         # Per-instance received mass
         per_instance_mass = group_sum_by(marginal_M, M_p_instance_ids)
         matched_instance_count = sum(1 for inst, m in per_instance_mass.items()
                                       if m > config.min_matched_mass_for_instance)
         matched_per_image = matched_instance_count / num_occurring_images[p]
         expected = memory.prototypes[p].expected_instance_count
         instance_count_diff = abs(matched_per_image - expected)

         # Matched extent: weighted centroid of query positions
         if marginal_Q.sum() > 0:
             matched_extent = (Q_p_positions * marginal_Q[:, None]).sum(axis=0) / marginal_Q.sum()
         else:
             matched_extent = None

         record PrototypeMatchingResult.

      3. Spatial violations:
         for (p_i, p_j), edge in memory.spatial_graph.items():
           e_i = per_prototype[p_i].matched_extent
           e_j = per_prototype[p_j].matched_extent
           if e_i is None or e_j is None:
               if config.skip_signal_4_if_no_extent:
                   continue
               else:
                   record large penalty (e.g., 10.0)
           observed_delta = e_j - e_i
           cov_inv = np.linalg.pinv(edge.delta_pos_cov + 1e-6 * I)
           diff = observed_delta - edge.delta_pos_mean
           mahal = sqrt(diff @ cov_inv @ diff)
           spatial_violations[(p_i, p_j)] = mahal

      4. Sum signals and return AnomalyDecomposition.
    """
```

세부 결정:

- **Cost function**: cosine distance (1 - cosine sim). DINO feature는 일반적으로 cosine geometry에서 잘 동작.
- **Weight normalization**: Q_p와 M_p의 weight를 각각 합 1로 normalize. unbalanced OT는 mass 차이를 marginal penalty로 처리.
- **Empty Q_p 처리**: 한 prototype에 query patch가 하나도 active하지 않으면 → 전체 memory_mass가 signal_2에 전부 들어감, signal_1 contribution 0, matched_extent None.
- **Empty extent 처리**: signal_4 계산 시 한쪽 prototype의 extent가 None이면 skip (W1 default). 나중에 missing prototype을 별도 anomaly로 가중치 부여 가능.

### 5.3 New notebook: `09_memory_anchored_ot_matching.ipynb`

08 notebook의 cell 패턴을 따름. Cell 구성:

| Cell | Type | Content |
|------|------|---------|
| 1 | markdown | Notebook 소개 (memory-anchored OT matching W1 self-probe) |
| 2 | markdown | Cell role: Repo setup |
| 3 | code | 08과 동일 repo setup |
| 4 | markdown | Cell role: Settings |
| 5 | code | `CATEGORY = 'breakfast_box'`, `MEMORY_SAMPLE_IDS = ['000.png','001.png','002.png','004.png','005.png']`, `PROBE_SAMPLE_ID = '003.png'` (memory에서 제외) |
| 6 | markdown | Cell role: DINO + helper setup |
| 7 | code | 08의 DINO 로직 재사용 |
| 8 | markdown | Cell role: Build memory bank |
| 9 | code | `memory = build_memory_bank(...)` 호출, prototype별 instance 수 / centroid quantiles / spatial_graph edge 수 표시 |
| 10 | markdown | Cell role: Visualize memory |
| 11 | code | prototype별 (id, instance_count, occurrence_rate, expected_count) 표 + spatial_graph edge mean Δpos를 화살표로 시각화 |
| 12 | markdown | Cell role: Self-probe inference |
| 13 | code | PROBE_SAMPLE_ID 이미지에서 raw DINO patch 추출 → `match_query_against_memory` 호출 → AnomalyDecomposition 표시 |
| 14 | markdown | Cell role: Interpretation |
| 15 | code | signal별 breakdown 표 + per-prototype matching_cost / matched_extent 화살표 시각화 + acceptance criteria 자동 체크 |

### 5.4 W1 acceptance criteria (cell 15에서 자동 체크)

Self-probe (clean image → 동일 카테고리의 clean memory)에서:

```python
checks = {
    "memory has at least 5 prototypes": len(memory.prototypes) >= 5,
    "every prototype has at least one instance": all(
        len(p.instances) >= 1 for p in memory.prototypes.values()
    ),
    "spatial graph covers >= 50% of prototype pairs":
        len(memory.spatial_graph) / max(1, C(len(memory.prototypes), 2)) >= 0.5,
    "signal_1 ratio < 0.20": result.signal_1_total / total_query_mass < 0.20,
    "signal_2 ratio < 0.20": result.signal_2_total / total_memory_mass < 0.20,
    "signal_3 per prototype avg < 0.5":
        result.signal_3_total / len(memory.prototypes) < 0.5,
    "signal_4 avg < 3.0":
        np.mean(list(result.spatial_violations.values())) < 3.0
        if result.spatial_violations else True,
    "no NaN in any signal": all_signals_finite(result),
}
```

모든 check가 PASS면 W1 완료. FAIL 항목이 있으면 디버깅:

- `signal_1 > 0.20`: top-k active patch가 OT에서 memory로 충분히 transport되지 않음 → OT marginal penalty/cost scale 확인. threshold ablation에서는 τ_p가 너무 strict할 수도 있음
- `signal_2 > 0.20`: memory의 너무 많은 patch가 query에서 매칭 안 됨 → ot_marginal_penalty 너무 strict, 또는 memory와 query 통계가 너무 다름
- `signal_3 > 0.5`: per-image instance count 매칭 실패 → min_matched_mass_for_instance 또는 weak prototype assignment 확인
- `signal_4 > 3.0`: spatial constraint 위배 → spatial_graph cov 추정 noisy (instance pair 수 부족), 또는 query patch 위치가 실제로 어긋남

---

## 6. Roadmap (W2 이후, 본 W1 작업에서는 구현 X)

- **W2**: query pipeline 시각화 강화 — prototype별 soft assignment heatmap, matched_extent overlay
- **W3**: 실제 anomaly image에 적용해서 signal 1, 2, 3이 anomaly와 normal을 구별하는지 확인
- **W4**: signal 4 (spatial graph) 통합 검증, 4-signal aggregation 방식 정교화
- **W5**: LOCO breakfast_box 전체 evaluation (clean test + condition-shifted), AUROC 측정
- **W6**: Stage B (position-stable category에서 instance-level spatial graph activate) ablation

---

## 7. Dependencies

기존:
- `dinov2_vits14` (torch.hub)
- numpy, pandas, matplotlib, PIL, torch

신규:
- `pot` (Python Optimal Transport): `pip install pot`
  - 사용 API: `ot.unbalanced.sinkhorn_unbalanced`
- (이미 있음) `scipy` for linalg.pinv

---

## 8. Out of Scope (W1에서 절대 하지 말 것)

- W2 이후 기능 (실제 anomaly image 평가, AUROC 계산, ablation 등) → 본 W1에서 구현 금지
- Instance-level spatial graph (Stage B) → W6
- 학습 가능한 head (현재 모두 deterministic) → 본 stage에서 도입 안 함
- Multi-category memory (breakfast_box에만 집중) → 카테고리 일반화는 W6
- 8-neighbor GAT (이전 논의에서 나왔던 alternative) → 본 OT 접근으로 결정됨, GAT는 후속 비교 baseline 또는 fallback으로만 검토

---

## 9. Narrative summary (paper writing용 memo)

> "Memory-anchored unbalanced optimal transport with prototype-level spatial constraints for logical anomaly detection under condition shift."

기존 work과의 차별화:

- ComAD/CSAD: query-side SAM mask + graph matching → shift에 query-side 검출 의존. 본 연구는 query-side raw patch only.
- Generic OT-AD: feature matching cost만 사용. 본 연구는 instance-aware mass tracking (signal 3) + spatial graph (signal 4)로 logical AD task fit.
- UniVAD: caption-based grounding → text 의존. 본 연구는 patch-feature memory only.

---

## 10. TODO (LLM 구현 순서)

1. `src/stage1_adapter/memory_bank.py` 작성 (데이터 클래스 + `build_memory_bank`)
2. `src/stage1_adapter/memory_ot_matcher.py` 작성 (`MatchingConfig`, `match_query_against_memory`)
3. `src/stage1_adapter/__init__.py`에 새 export 추가
4. `notebook/09_memory_anchored_ot_matching.ipynb` 작성
5. `requirements.txt` (또는 `setup.py`)에 `pot` 추가
6. Self-probe 실행해서 §5.4 acceptance criteria 모두 PASS 확인
7. PASS 안 되면 §5.4 디버깅 가이드 따라 hyperparameter 조정, design 가정은 건드리지 말 것

---

**참고:** 이 문서는 W1만을 위한 spec이다. W1 완료 후 별도 W2 spec을 작성하므로, W2-W6 관련 추측성 구현은 본 작업에서 하지 말 것.
