# Baseline Survey / 2026-04-19 / condition shift

## 1. 목표

- `condition shift`가 기존 logical AD baseline 성능에 미치는 영향을 확인하기 위한 초기 비교군을 정리한다.
- 각 baseline의 역할과 코드 출처를 `official`, `dataset official`, `unofficial or not verified`로 구분한다.

## 2. index.md 기준 현재 이해

- [확정][index.md 기반] 첫 사전 검증은 `기존 logical AD baseline이 실제로 condition shift에 취약한가`를 확인하는 것이다.
- [확정][index.md 기반] 여기서 핵심은 평균 성능 그 자체보다 `정상인데 shift만 바뀐 경우의 false positive 증가`다.
- [확정][index.md 기반] 비교군은 patch 기반, component 기반, logical/component segmentation 기반, 내부 단순 baseline을 함께 포함해야 해석력이 생긴다.

## 3. 이번 세션의 판단

- [확정][외부 근거 필요] VID-AD는 다섯 capture condition을 사용한다.
  - `White BG`
  - `Cable BG`
  - `Mesh BG`
  - `Low-light CD`
  - `Blurry CD`
- [확정][외부 근거 필요] VID-AD Table 4에는 `EfficientAD`, `PatchCore`, `PaDiM`, `VAE`, `AnoGAN`, `CSAD`, `UniVAD`가 baseline으로 비교된다.
- [확정][외부 근거 필요] Table 4 기준 mean/std는 다음과 같다.
  - `EfficientAD`: `0.539 ± 0.057`
  - `PatchCore`: `0.445 ± 0.029`
  - `PaDiM`: `0.378 ± 0.032`
  - `VAE`: `0.452 ± 0.034`
  - `AnoGAN`: `0.494 ± 0.028`
  - `CSAD`: `0.662 ± 0.028`
  - `UniVAD`: `0.574 ± 0.011`
- [확정][실험 필요] 1차 실행 비교군은 아래 4개가 가장 현실적이다.
  - `PatchCore`
  - `UniVAD`
  - `CSAD`
  - `Position-only baseline`
- [제안][실험 필요] 2차 후보는 `EfficientAD`, `PaDiM`, `VAE`, `AnoGAN`으로 두되, 코드 관리 부담과 구현 신뢰도를 따져 후순위로 둔다.
- [제안][실험 필요] 내부 baseline은 아래 순서로 확장한다.
  - `Position-only`
  - `Node + Geometric Edge`
  - `Node + Geo + Gray-scale Union`
  - `+ Background Calibration`
  - `+ 2-hop consistency`

## 4. 코드 출처 정리

### official 또는 강하게 확인된 코드

- `PatchCore`
  - repo: `amazon-science/patchcore-inspection`
  - 상태: official implementation
  - 역할: patch memory 기반 normal-only 대표 baseline
- `UniVAD`
  - repo: `FantasticGNU/UniVAD`
  - 상태: official implementation
  - 역할: few-shot, training-free, component-level 직접 비교군
- `CSAD`
  - repo: `Tokichan/CSAD`
  - 상태: official implementation
  - 역할: logical anomaly / component segmentation 비교군
- `VID-AD`
  - repo: `nkthiroto/VID-AD`
  - 상태: 논문에서 dataset repository로 직접 지목됨
  - 역할: condition shift 벤치마크 데이터셋

### unofficial 또는 이번 조사에서 official 여부 미확인

- `EfficientAD`
  - 확인한 repo: `nelson1425/EfficientAD`, `rximg/EfficientAD`
  - 상태: 둘 다 unofficial이라고 명시
  - 메모: 이번 조사 범위에서는 official repo를 확인하지 못함
- `PaDiM`
  - 확인한 repo: `Lornatang/PaDiM`
  - 상태: unofficial implementation이라고 명시
  - 메모: 이번 조사 범위에서는 official repo를 확인하지 못함
- `AnoGAN`
  - 메모: 공개 구현은 다수 있으나, 이번 조사 범위에서는 industrial logical AD 비교용 표준 official repo를 확인하지 못함
- `VAE`
  - 메모: 구현 예시는 많지만 anomaly baseline용 단일 official repo로 고정하기 어렵다

## 5. 보류 또는 충돌

- 현재 저장소에는 외부 코드를 pull 하거나 서브모듈로 연결한 상태가 아니다.
- 따라서 이번 단계는 `비교군 정의`와 `코드 출처 문서화`까지 수행한다.
- `EfficientAD`, `PaDiM`, `AnoGAN`, `VAE`는 후속 단계에서 실제 재현 비용을 보고 채택 여부를 다시 판단한다.

## 6. 총괄 세션에 전달할 질문

- 1차 실험은 `VID-AD` 우선인가, 아니면 내부 synthetic shift benchmark를 먼저 만들 것인가?
- 외부 baseline은 official repository만 우선 채택할 것인가?
- `Position-only`를 제외한 내부 baseline도 동시에 만들 것인가, 아니면 외부 baseline 정리 후 순차적으로 갈 것인가?

## 7. 다음 액션 제안

- `configs/baseline_registry.yaml`에 1차/2차 baseline 후보와 코드 상태를 구조화한다.
- 다음 단계에서 `Position-only baseline` 입출력 계약을 별도 문서로 고정한다.
- 외부 코드를 실제로 붙일 때는 baseline별 실행 명령, 입력 포맷, 출력 score 포맷을 추가한다.

## 3줄 요약
1. 초기 비교군은 `PatchCore`, `UniVAD`, `CSAD`, `Position-only`가 가장 현실적이다.
2. `PatchCore`, `UniVAD`, `CSAD`는 공식 코드 확인이 가능했고, `VID-AD`는 논문에서 dataset repo를 직접 가리킨다.
3. `EfficientAD`, `PaDiM`, `AnoGAN`, `VAE`는 후순위 후보로 두되, 이번 조사 범위에서는 official 여부가 불분명하거나 unofficial 구현만 확인됐다.
