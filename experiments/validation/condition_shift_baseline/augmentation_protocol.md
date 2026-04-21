# Augmentation Protocol / 2026-04-20 / condition shift

## 1. 목표

- `query_normal_augmented`를 생성하는 규칙을 고정한다.
- augmentation으로 인해 새로운 logical anomaly가 생기지 않도록 제약을 명시한다.
- baseline 간 비교 시 동일한 `clean vs augmented normal` 평가축을 재사용할 수 있게 한다.

## 2. 현재 이해

- [확정][index.md 기반] 이번 검증의 목적은 기존 baseline이 `condition/domain augmentation`만으로 얼마나 쉽게 흔들리는지 보는 것이다.
- [확정][index.md 기반] 따라서 augmentation은 논리 상태를 바꾸지 않고, 시각 조건만 바꿔야 한다.
- [확정][실험 필요] augmentation protocol이 고정되지 않으면 성능 하락의 원인이 모델 민감도인지, augmentation 과도성인지 해석할 수 없다.
- [확정][사용자 결정] `background_clutter`는 1차 비교축에서 제외한다.

## 3. 기본 원칙

- [확정][실험 필요] 모든 augmentation은 `semantic-preserving`이어야 한다.
- [확정][실험 필요] object count와 logical label semantics를 바꾸는 편집은 금지한다.
- [확정][실험 필요] clean 이미지와 augmented 이미지는 동일한 sample id를 공유하는 paired sample로 관리한다.
- [확정][실험 필요] 기본은 `single-factor augmentation`으로 유지한다.
- [확정][실험 필요] `position shift / rotation / scale`은 허용하되, object가 프레임 밖으로 잘리거나 조립 의미가 바뀌면 실패 샘플로 간주한다.
- [확정][사용자 결정] 1차 실행은 가장 일반적이고 대표성이 큰 소수 축만 남긴다.

## 4. 1차 augmentation 축

- `brightness`
  - 목적: 가장 기본적인 photometric shift 민감도 확인
- `gaussian_blur / motion_blur`
  - 목적: focus or motion degradation 민감도 확인
- `gaussian_noise`
  - 목적: 대표적인 sensor noise 민감도 확인
- `compression / low_resolution`
  - 목적: codec 및 해상도 저하 민감도 확인
- `position_shift`
  - 목적: 작은 spatial perturbation 민감도 확인
- `low_light`
  - 목적: 실제 현장에서 자주 나타나는 저조도 조건 민감도 확인

## 5. 강도 정의

### brightness

- `low`
  - factor: `0.80`
- `medium`
  - factor: `0.60`
- `high`
  - factor: `0.40`

### gaussian_blur

- `low`
  - radius: `2.0`
- `medium`
  - radius: `4.0`
- `high`
  - radius: `6.0`

### motion_blur

- `low`
  - kernel size: `5`
- `medium`
  - kernel size: `9`
- `high`
  - kernel size: `13`

### gaussian_noise

- `low`
  - sigma: `24`
- `medium`
  - sigma: `48`
- `high`
  - sigma: `72`

### compression

- `low`
  - JPEG quality: `45`
- `medium`
  - JPEG quality: `20`
- `high`
  - JPEG quality: `10`

### low_resolution

- `low`
  - downsample scale: `0.75`
- `medium`
  - downsample scale: `0.50`
- `high`
  - downsample scale: `0.33`

### position_shift

- `low`
  - max translation ratio: `0.03`
- `medium`
  - max translation ratio: `0.06`
- `high`
  - max translation ratio: `0.10`

### low_light

- `low`
  - brightness: `0.75`
  - contrast: `0.85`
- `medium`
  - brightness: `0.55`
  - contrast: `0.75`
- `high`
  - brightness: `0.40`
  - contrast: `0.65`

## 6. 금지 규칙

- object를 지우거나 추가하지 않는다.
- object count가 바뀌는 수준의 crop이나 occlusion은 허용하지 않는다.
- relation semantics를 바꾸는 큰 affine transform은 허용하지 않는다.
- `position_shift`는 small perturbation 범위에서만 사용한다.
- augmentation 후 사람이 보기에 logical label이 바뀐 것처럼 보이면 사용하지 않는다.

## 7. 저장 계약

- 각 augmented sample은 아래 메타데이터를 가진다.
  - `source_id`
  - `augmentation_type`
  - `severity`
  - `seed`
  - `output_path`
- 권장 경로 형식:
  - `data/query_normal_augmented/<augmentation_type>/<severity>/<category>/<source_id>.png`

## 8. 평가 계약

- clean과 augmented는 동일한 threshold policy를 사용한다.
- `clean normal`에서 정한 threshold를 augmented normal에도 그대로 적용한다.
- 1차 보고에서는 아래 항목을 반드시 기록한다.
  - `FPR(clean normal)`
  - `FPR(augmented normal)`
  - `score_shift`
  - augmentation type별 민감도 차이

## 9. 현재 상태

- 기존 `query_normal_augmented`는 이전 `background_clutter` 중심 프로토콜로 생성된 산출물이므로, 이 문서 기준으로는 구버전 데이터다.
- 새 프로토콜을 쓰려면 `query_normal_augmented`를 비우고 재생성해야 한다.

## 3줄 요약
1. 1차 condition shift 축은 `brightness`, `gaussian_blur`, `motion_blur`, `gaussian_noise`, `compression`, `low_resolution`, `position_shift`, `low_light`로 줄였다.
2. `background_clutter`와 중복성이 큰 세부 corruption들은 제외했고, 가장 일반적이고 대표적인 축만 남겼다.
3. 현재 증강 데이터는 구버전이므로 새 프로토콜 기준 재생성이 필요하다.
