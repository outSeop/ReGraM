# Condition Shift Baseline File Map

이 문서는 `condition_shift_baseline` 폴더에서

- 지금 실제로 쓰는 파일
- 보조 파일
- 과거 보관용 파일

을 빠르게 구분하기 위한 지도다.

## Start Here

처음 볼 때는 아래 순서만 따라가면 된다.

1. [README.md](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/README.md)
2. [notebook/01_run_orchestrator.ipynb](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/notebook/01_run_orchestrator.ipynb)
3. [src/data/build_query_manifest.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/src/data/build_query_manifest.py)
4. [src/runners/patchcore/run_manifest_shift.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/src/runners/patchcore/run_manifest_shift.py)
5. [reports/README.md](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/reports/README.md)

위 5개가 현재 기준 진입점이다.

참고: `src/core/*`에는 이전 경로 호환을 위한 wrapper가 남아 있다. 신규 코드는 namespaced 경로(`src/data`, `src/orchestration`, `src/runners`)를 우선 사용한다.

## Active Core

지금 기준으로 `실제로 자주 쓰는 핵심 파일`이다.

- [README.md](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/README.md)
  - 이 실험의 운영 원칙과 역할 분리 규칙
- [notebook/01_run_orchestrator.ipynb](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/notebook/01_run_orchestrator.ipynb)
  - thin orchestrator
  - Git checkout, dataset 준비, runner 실행, summary viewer
- [src/data/build_query_manifest.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/src/data/build_query_manifest.py)
  - on-the-fly augmentation manifest 생성
- [src/data/augmentation_runtime.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/src/data/augmentation_runtime.py)
  - manifest schema와 augmentation 적용 로직
- [src/core/manifest_shift_common.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/src/core/manifest_shift_common.py)
  - PatchCore/UniVAD 공통 manifest-shift run spec, summary, wandb/log helper
- [src/orchestration/notebook_orchestration.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/src/orchestration/notebook_orchestration.py)
  - notebook용 설정 조립, run config 생성, 실행 queue, display helper
- [src/orchestration/dashboard_loader.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/src/orchestration/dashboard_loader.py)
  - summary JSON 로드, dashboard table/plot 렌더링
- [src/data/manifest_paths.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/src/data/manifest_paths.py)
  - PatchCore/UniVAD와 무관한 manifest source path 해석
- [src/runners/patchcore/run_manifest_shift.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/src/runners/patchcore/run_manifest_shift.py)
  - PatchCore manifest 기반 thin runner
- [src/common/contracts.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/src/common/contracts.py)
  - summary/log 출력 계약

## Active Support

핵심은 아니지만 현재 경로에서 같이 쓰는 파일이다.

- [src/runners/patchcore/evaluate_clean.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/src/runners/patchcore/evaluate_clean.py)
  - PatchCore clean baseline 요약 생성
- [src/runners/patchcore/validate_manifest_identity.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/src/runners/patchcore/validate_manifest_identity.py)
  - identity manifest 검증
- [src/runners/patchcore/check_identity_repro.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/src/runners/patchcore/check_identity_repro.py)
  - PatchCore identity 재현 체크
- [src/analysis/render_augmentation_samples.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/src/analysis/render_augmentation_samples.py)
  - augmentation 샘플 panel 렌더링
- [tests/test_manifest_shift_common.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/tests/test_manifest_shift_common.py)
  - 공통 helper 회귀 테스트

## UniVAD Path

UniVAD 관련은 별도 축이다. PatchCore 흐름과 섞어 보지 않는 편이 좋다.

- [src/univad/prepare_mvtec_loco.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/src/univad/prepare_mvtec_loco.py)
  - raw LOCO를 UniVAD 입력 포맷으로 변환
- [src/univad/prepare_smoke_subset.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/src/univad/prepare_smoke_subset.py)
  - smoke subset 생성
- [src/univad/run_smoke_colab.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/src/univad/run_smoke_colab.py)
  - Colab smoke runner
- [src/univad/run_clean_eval.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/src/univad/run_clean_eval.py)
  - UniVAD clean eval summary 생성
- [src/univad/run_manifest_shift.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/src/univad/run_manifest_shift.py)
  - UniVAD manifest 기반 thin runner
- [src/univad/setup_runtime.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/src/univad/setup_runtime.py)
  - Colab UniVAD dependency, checkpoint, dataset, mask readiness/setup helper

## Colab / Runtime

런타임 준비 전용 파일이다.

- [colab/README.md](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/colab/README.md)
  - Colab 실행 순서 문서
- [colab/bootstrap_runtime.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/colab/bootstrap_runtime.py)
  - prepared dataset, small reports 복사
- [colab/raw_dataset_archive.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/colab/raw_dataset_archive.py)
  - raw dataset tar 생성 / 런타임 복원

## Config / Manifest / Output

이 폴더들은 코드가 아니라 실험 입력과 산출물이다.

- [configs](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/configs)
  - baseline registry, 템플릿, protocol 설정
- [manifests](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/manifests)
  - 실험 폴더 내부 manifest 예시
- [reports](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/reports)
  - 공식 보고와 small summary
- [results](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/results)
  - 결과 템플릿

루트의 [manifests](/Users/song-inseop/연구/ReGraM/manifests) 폴더도 현재 notebook이 우선 탐색하는 입력 후보다.

## Reference Docs

설계 이해를 보강하는 문서다. 실행 진입점은 아니다.

- [docs/baseline_survey.md](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/docs/baseline_survey.md)
  - baseline 출처와 메모
- [docs/augmentation_protocol.md](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/docs/augmentation_protocol.md)
  - augmentation 설계 문서

## Legacy

지금 기본 흐름에서 빼둔 파일이다.

- [legacy/README.md](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/legacy/README.md)
- [legacy/run_patchcore_query_shift.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/legacy/run_patchcore_query_shift.py)
- [legacy/generate_augmentations.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/legacy/generate_augmentations.py)
- [legacy/sync_to_drive.py](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/legacy/sync_to_drive.py)

기본 경로에서 이 파일들은 무시해도 된다.

## Practical Rule

헷갈릴 때는 아래처럼 보면 된다.

- 실험을 돌리고 싶다
  - `notebook/01_run_orchestrator.ipynb`
- manifest를 만들고 싶다
  - `src/data/build_query_manifest.py`
- PatchCore를 돌리고 싶다
  - `src/runners/patchcore/run_manifest_shift.py`
- runner 공통화 계층을 보고 싶다
  - `src/core/manifest_shift_common.py` (공통 helper는 유지)
- UniVAD를 돌리고 싶다
  - `prepare_univad_*`, `run_*`
- Colab 런타임을 준비하고 싶다
  - `colab/*`
- 과거 코드인지 확인하고 싶다
  - `legacy/*`
