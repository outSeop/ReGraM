# Condition Shift Configs

이 디렉터리는 condition-shift baseline 실험의 설정 기준을 모아둔다. 현재 01 노트북은 `experiment_template.yaml`을 읽어서 모델 하이퍼파라미터와 sweep을 runner command로 변환한다.

## Files

| file | role |
| --- | --- |
| `experiment_template.yaml` | 새 실험 config를 만들 때 출발점으로 쓰는 템플릿. dataset, model hyperparameter, shift protocol, metric, output 정책을 한 곳에 적는다. |
| `baseline_registry.yaml` | 비교 후보 baseline 목록과 우선순위. 어떤 모델을 primary/secondary/internal로 볼지 정리한다. |
| `augmentation_protocol.yaml` | condition shift 종류와 severity별 augmentation parameter 정의. manifest 생성/해석 기준이다. |
| `mvtec_loco_patchcore.yaml` | 초기 PatchCore 중심 실험 preset. 현재 runner 설정의 최신 기준은 `experiment_template.yaml`을 우선한다. |

## Current Usage

현재 실행은 `notebook/01_run_orchestrator.ipynb`의 `EXPERIMENT_CONFIG_PATH`가 가리키는 YAML에서 모델 설정을 읽는다.
YAML 로딩에는 `PyYAML`이 필요하다. Colab에는 보통 포함되어 있고, 로컬에서 없으면 `pip install pyyaml` 후 실행한다.

```python
EXPERIMENT_CONFIG_PATH = EXP_ROOT / "configs" / "experiment_template.yaml"
EXPERIMENT_CONFIG = load_experiment_config(EXPERIMENT_CONFIG_PATH)
MODEL_CONFIG = model_config_from_experiment_config(EXPERIMENT_CONFIG)
MODEL_SWEEP = model_sweep_from_experiment_config(EXPERIMENT_CONFIG)
```

`build_baseline_specs(..., model_config=MODEL_CONFIG, model_sweep=MODEL_SWEEP)`가 이 값을 받아 각 runner의 CLI 인자로 바꾼다.

- UniVAD: `image_size`, `k_shot`, `round`, `export_heatmaps`, `heatmap_max_images`
- PatchCore: `resize`, `imagesize`, `batch_size`, `num_workers`, `sampler_percentage`, `export_heatmaps`, `heatmap_max_images`

예를 들어 UniVAD를 5-shot 단일 실행으로 돌리려면 YAML에서 다음 값만 바꾼다.

```yaml
models:
  univad:
    k_shot: 5
    round: 0
```

그 다음 설정 셀부터 다시 실행하면 command가 `--k-shot 5 --round 0`으로 생성된다.

여러 하이퍼파라미터 조합을 돌릴 때는 `models`에 list를 직접 넣지 말고 `sweep` 섹션을 켠다.

```yaml
sweep:
  enabled: true
  models:
    univad:
      k_shot: [1, 4]
      round: [0, 1, 2]
```

이 경우 실행 조합은 Cartesian product로 펼쳐진다.

```text
k_shot=1, round=0
k_shot=1, round=1
k_shot=1, round=2
k_shot=4, round=0
k_shot=4, round=1
k_shot=4, round=2
```

결과는 덮어쓰지 않도록 variant별 하위 폴더에 저장된다.

```text
reports/univad_manifest_shift/k_shot_1__round_0/breakfast_box_multi_high.json
reports/univad_manifest_shift/k_shot_4__round_2/breakfast_box_multi_high.json
```

현재 runner가 직접 소비하는 augmented split은 query normal이다. Clean logical/structural anomaly는 shifted normal과의 AUROC 비교 기준으로 사용한다. `query_logical_anomaly_augmented`, `query_structural_anomaly_augmented`는 anomaly 자체에 condition shift를 걸어보는 확장 실험을 위한 optional field다.

## Recommended Workflow

1. `experiment_template.yaml`을 복사해서 새 실험 config 이름으로 저장한다.
2. `question`, `dataset`, `models`, `shift_protocol`, `evaluation`을 실험 의도에 맞게 수정한다.
3. `sweep.enabled`가 `false`이면 `models` 단일값만 실행하고, `true`이면 `sweep.models` 조합을 실행한다.
4. 실행 후 report/summary를 볼 때 config 값을 같이 기록한다. 특히 `k_shot`, `round`, `severity_levels`, `kinds`, `categories`는 결과 해석에 직접 영향을 준다.

## Key Distinction

- `baseline_registry.yaml`: 어떤 baseline을 비교 대상으로 둘지 정하는 목록이다. 실행 hyperparameter를 넣는 곳이 아니다.
- `experiment_template.yaml`: 특정 실험의 dataset, model hyperparameter, shift, metric 정책을 적는 곳이다.
- `augmentation_protocol.yaml`: shift 자체의 정의와 severity별 parameter를 적는 곳이다.
- notebook `EXPERIMENT_CONFIG_PATH`: 현재 실제 runner command에 반영되는 YAML config 경로다.
