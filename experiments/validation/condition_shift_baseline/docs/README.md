# Condition Shift Baseline Docs

이 디렉터리는 `condition_shift_baseline` 실험의 설계/운영 문서만 둔다. 실행 진입점은 상위 [README.md](../README.md)와 [FILE_MAP.md](FILE_MAP.md)를 먼저 본다.

## Operational Docs

- [FILE_MAP.md](FILE_MAP.md)
  - 현재 쓰는 파일, 보조 파일, legacy 파일 구분.
- [../notebook/01_run_orchestrator.ipynb](../notebook/01_run_orchestrator.ipynb)
  - runner 실행과 summary/log handoff.
- [../notebook/02_analysis_dashboard.ipynb](../notebook/02_analysis_dashboard.ipynb)
  - 저장된 summary 기반 분석 dashboard.
- [../notebook/03_figure_export.ipynb](../notebook/03_figure_export.ipynb)
  - 보고서/발표용 figure와 table export.
- [../src/README.md](../src/README.md)
  - 실제 실행되는 Python source layout.
- [../colab/README.md](../colab/README.md)
  - Colab CUDA runtime 실행 순서.
- [../reports/README.md](../reports/README.md)
  - 공식 보고서와 보조 artifact 구분.

## Reference Docs

- [baseline_survey.md](baseline_survey.md)
  - baseline 후보와 코드 출처 정리.
- [augmentation_protocol.md](augmentation_protocol.md)
  - condition shift augmentation 축, 강도, 평가 계약.

## Current Conventions

- manifest는 repo-top [manifests](../../../../manifests)가 canonical 위치다.
- 신규 실행 경로는 `src/data`, `src/orchestration`, `src/runners`, `src/univad`를 우선 사용한다.
- `src/core/*`의 대다수 파일은 이전 경로 호환 wrapper다.
- `src/core/manifest_shift_common.py`는 아직 PatchCore/UniVAD 공통 helper의 canonical 위치다.
