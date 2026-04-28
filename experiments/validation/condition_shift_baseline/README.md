# Condition Shift Baseline Validation

이 폴더는 `기존 logical AD baseline이 실제로 condition shift에 취약한가?`를 검증하기 위한 실험 스캐폴드다.

빠른 파일 역할 구분은 [docs/FILE_MAP.md](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/docs/FILE_MAP.md)를 먼저 본다.
실제로 실행되는 Python 소스만 보고 싶으면 [src/README.md](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/src/README.md)를 본다.

## 목적

- 기존 baseline이 `condition shift`로 증강된 normal 데이터를 anomaly처럼 오판하는지 확인
- `clean normal` 대비 `augmented normal`에서 score와 false positive가 얼마나 흔들리는지 기록
- baseline 간 비교 시 같은 split 계약과 결과 포맷 재사용

## 초기 비교군

- `PatchCore`
- `UniVAD`
- `CSAD`
- `Position-only baseline`

위 4개를 1차 비교군으로 둔다.

- `PatchCore`
  - patch memory 기반 normal-only baseline
- `UniVAD`
  - few-shot, training-free, component-level 비교군
- `CSAD`
  - logical anomaly/component segmentation 비교군
- `Position-only baseline`
  - 내부 단순 기준선

baseline 조사와 코드 출처는 [docs/baseline_survey.md](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/docs/baseline_survey.md)에 정리한다.

## 최소 실험 구성

- `reference_normal_clean`
  - few-shot normal reference
- `query_normal_clean`
  - clean normal query
- `manifests/*.jsonl` (repo-top [manifests/](/Users/song-inseop/연구/ReGraM/manifests))
  - 원본 경로와 augmentation 계약을 기록한 on-the-fly query manifest
  - **canonical location은 repo-top `manifests/` 하나**. `source_path_mode: "repo_relative"` 포맷으로 저장한다. 실험 하위에는 manifest를 두지 않는다.

필요 시 후속으로 아래 split을 추가한다.

- `query_logical_anomaly_clean`
- `query_logical_anomaly_augmented`

## 핵심 질문

1. normal clean 대비 normal shifted에서 false positive가 얼마나 증가하는가?
2. augmentation 이후 anomaly score 분포가 얼마나 이동하는가?
3. shift 종류별로 민감도가 다른가?

## 권장 폴더 사용 방식

- `configs/`
  - baseline별 실험 설정 파일
- `manifests/` (repo-top, 실험 하위에는 두지 않음)
  - on-the-fly augmentation query 정의
- `results/`
  - baseline별 실행 결과 요약
- `reports/`
  - 그래프, 표, 해석 메모
- `runs/legacy_file_backed/`
  - 이전 file-backed augmentation 방식으로 얻은 결과 보관
- `legacy/`
  - 현재 기본 흐름에서 빠진 예전 helper와 runner 보관

## 책임 분리 원칙

이 실험 자산은 아래 역할 분리 원칙을 따른다.

- `Notebook`
  - thin orchestrator
  - 환경 확인, 경로 확인, Python runner 호출, `summary.json` 로드, 간단 표/시각화만 담당
  - 긴 helper 구현은 보관하지 않고 `src/core/notebook_orchestration.py`, `src/core/dashboard_loader.py`, `src/univad/setup_runtime.py`를 호출한다
  - Colab 메모리 안정성을 위해 UniVAD manifest shift 기본값은 `--image-size 224`로 둔다
- `Python`
  - 실험 로직, 전처리, clean eval, corruption eval, summary 생성의 단일 책임 담당
- `Git`
  - 코드, 문서, 설정, manifest, small summary, 공식 해석 Markdown만 추적
- `wandb`
  - run tracking과 비교 보조 계층만 담당
- 최종 판단과 보고
  - repo 내부 Markdown을 source of truth로 유지

## Notebook 역할

- 노트북은 실험 핵심 로직을 직접 구현하지 않는다.
- 인라인 helper source를 보관하지 않는다.
- 100줄 이상 setup/run/report helper는 Python 모듈로 분리한다.
- Colab에서는 먼저 Git으로 repo를 clone 또는 pull 하고, 그 다음 dataset bootstrap이 필요하면 `colab/bootstrap_runtime.py` 같은 별도 Python 스크립트를 호출한다.
- runner 실패 시 traceback을 가공하지 않고, runner가 남긴 `summary.json` 또는 `log.txt` 경로를 그대로 보여준다.
- 현재 기본 노트북 진입점은 [notebook/01_run_orchestrator.ipynb](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/notebook/01_run_orchestrator.ipynb) 이다.
- 분석 전용은 [notebook/02_analysis_dashboard.ipynb](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/notebook/02_analysis_dashboard.ipynb),
  figure export 전용은 [notebook/03_figure_export.ipynb](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/notebook/03_figure_export.ipynb)을 사용한다.

## Git 기반 notebook 원칙

- notebook이 호출하는 `.py`는 Git으로 추적되는 repo 파일을 source of truth로 삼는다.
- Colab이나 서버 runtime에서는 먼저 repo를 clone 또는 pull 해서 코드 상태를 맞춘다.
- notebook은 Drive sync나 코드 문자열 내장을 통해 helper를 들고 있지 않는다.
- dataset이나 작은 보조 자산만 runtime으로 따로 복사한다.
- Colab 실행 순서는 `git pull` -> dataset bootstrap -> runtime setup/readiness -> runner 실행 -> dashboard 로 유지한다.
- notebook output은 커밋 전에 비운다. 코드, 설정, small summary, 공식 Markdown만 Git source of truth로 둔다.

## Python runner 출력 계약

모든 runner는 가능하면 아래 공통 출력 계약을 따른다.

- `summary.json`
  - 비교 가능한 정형 결과
- `log.txt`
  - 실행 로그 또는 실행 요약 로그
- optional `artifacts/`
  - 큰 figure, panel, 보조 시각 자산

현재 공통 summary schema는 `src/core/contracts.py`의 `condition_shift_baseline.summary.v1`을 기준으로 둔다.
top-level 공통 필드는 아래와 같다.

- `baseline`
- `dataset`
- `class_name`
- `eval_type`
- `device`
- `config`
- `metrics`
- `paths`
- `artifacts`
- `payload`

runner는 노트북 없이 CLI에서 독립 실행 가능해야 한다.
공통 manifest 해석, summary scaffold, wandb/log 조립은 `src/core/manifest_shift_common.py`에 둔다.

## PatchCore device 정책

- PatchCore runner 기본 device는 `cuda if available else cpu`
- 필요하면 `--device cpu` 또는 `--device cuda`로 명시적으로 강제한다

## Git 추적 원칙

- Git이 추적하는 것
  - 코드
  - 문서
  - 설정
  - manifest
  - small summary JSON
  - 결과 해석용 Markdown
- Git에서 제외하는 것
  - notebook outputs
  - 대형 panel/png
  - 대형 raw logs
  - runtime copy 자산
  - cache, checkpoint, dataset

실험 디렉터리 전용 ignore 규칙은 `.gitignore`에서 관리한다.

## wandb 운영 원칙

- wandb는 tracking layer이지 source of truth가 아니다.
- 기록 대상
  - run config
  - baseline / class / eval type / device
  - 핵심 metric
  - small summary JSON
  - `log.txt`
- 비기록 대상
  - 최종 결론 문서
  - 세션 판단 / 확정 / 보류 상태
- 공식 보고서 원본
- `.env.example`에는 placeholder만 둔다. 실제 `WANDB_API_KEY`는 repo root `.env` 또는 process env에만 둔다.

Markdown 보고서는 wandb 링크를 참고 링크로만 포함한다.

### PatchCore 최소 사용법

PatchCore manifest runner는 옵션으로 wandb tracking을 켤 수 있다.

- 기본값은 off
- 켤 때는 `--use-wandb`를 준다
- 여러 shift 비교용 기본 group은 `patchcore-manifest-shift`

예시:

```bash
python experiments/validation/condition_shift_baseline/src/core/run_patchcore_manifest_shift.py \
  --category breakfast_box \
  --manifest manifests/query_motion_blur.jsonl \
  --severities low \
  --use-wandb \
  --wandb-log-images \
  --wandb-max-images 2 \
  --wandb-project regram-condition-shift
```

기록되는 것:

- runner config
- `manifest_name`, `shift_family` 같은 run-level 비교 축
- severity를 run-level로 쪼갤 때는 `--severities low`처럼 단일 severity만 넘긴다
- severity의 실제 강도 값도 run-level config에 남긴다. 예: `severity_param_sigma=24.0`, `severity_param_kernel_size=13`
- clean metric
- augmentation/severity별 핵심 metric
- 선택적으로 `원본 | 증강본` preview 이미지 소량
- `summary.json`, `log.txt` artifact

권장 비교 단위:

- `manifest 하나 = shift 하나`
- run 하나 = `shift 하나 + severity 하나`
- 예: `query_motion_blur.jsonl + low`, `query_motion_blur.jsonl + high`
- 이 경우 wandb에서는 `shift_family`, `severity` 기준으로 run-level 필터링과 그룹화를 한다.

## 첫 실행 체크리스트

1. baseline 이름을 정한다.
2. baseline 후보와 코드 출처를 `docs/baseline_survey.md`와 `configs/baseline_registry.yaml`에서 확인한다.
3. reference/query split 경로를 `configs/experiment_template.yaml`에 채운다.
4. baseline별 실험 설정 파일을 `configs/baseline_registry.yaml` 기준으로 분기한다.
5. query manifest를 만든다.
6. 결과를 `results/result_template.md` 형식으로 남긴다.
7. threshold 기준을 `clean_max` 또는 `per_run_optimized` 중 하나로 고정한다.

### Refactor smoke check (권장)

리팩터링 이후 경로 호환을 빠르게 확인할 때:

```bash
python experiments/validation/condition_shift_baseline/scripts/smoke_import_paths.py
pytest -q experiments/validation/condition_shift_baseline/tests/test_import_compatibility.py
```

## 우선 지표

- `FPR(clean normal)`
- `FPR(augmented normal)`
- `delta_fpr = FPR(augmented normal) - FPR(clean normal)`
- `mean_score(clean normal)`
- `mean_score(augmented normal)`
- `score_shift = mean_score(augmented normal) - mean_score(clean normal)`
- `distribution_overlap(clean, augmented)`

## 해석 주의

- `AUROC` 하나만으로 condition 취약성을 판단하지 않는다.
- 핵심은 `정상인데 shift만 들어간 샘플`을 모델이 anomaly처럼 읽는지다.
- augmentation을 강하게 걸수록 score가 단조 증가하면 low-level feature dependence가 강하다는 신호다.

## 현재 정리 기준

- 기본 경로는 manifest 기반 on-the-fly 평가를 전제로 한다.
- `data/query_normal_augmented/`는 더 이상 기본 데이터셋으로 취급하지 않는다.
- 예전 file-backed 결과는 `runs/legacy_file_backed/` 아래로 분리 보관한다.
