# Legacy Scripts

이 디렉터리는 현재 기본 실험 경로에서 제외된 스크립트를 보관한다.

## 이동 기준

아래 스크립트는 `manifest 기반 current flow` 또는 `Git 기반 notebook workflow`의 기본 경로에서 빠졌다.

- `run_patchcore_query_shift.py`
  - file-backed `data/query_normal_augmented` 흐름 전용
- `generate_augmentations.py`
  - query 이미지들을 물리적으로 복제해서 저장하는 예전 augmentation 생성기
- `sync_to_drive.py`
  - notebook이 Git clone/pull 대신 Drive sync를 중심으로 운영되던 시기의 보조 스크립트

## 현재 원칙

- 기본 경로는 `src/core/build_query_manifest.py` + `src/core/run_patchcore_manifest_shift.py`다.
- 노트북은 Git으로 repo를 가져오고, versioned `.py`를 직접 호출한다.
- dataset이나 작은 보고서 자산만 필요하면 `colab/bootstrap_runtime.py`로 별도 복사한다.

이 스크립트들은 당장 삭제하지 않고, 과거 결과 재현이나 비교가 필요할 때만 참고한다.
