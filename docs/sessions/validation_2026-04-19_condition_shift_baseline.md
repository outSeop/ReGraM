# validation / 2026-04-19 / condition shift baseline

## 1. 목표

- [검증 필요][index.md 기반] 기존 logical AD baseline이 `condition shift`에 취약한지 검증하기 위한 실험 설계를 고정한다.
- [검증 필요][index.md 기반] 특히 기존 모델이 `condition shift`로 증강된 데이터를 clean normal과 얼마나 다르게 오판하는지 관찰할 수 있는 비교 구조를 만든다.

## 2. index.md 기준 현재 이해

- [확정][index.md 기반] 본 프로젝트의 문제의식은 `logical anomaly`와 `condition-induced discrepancy`를 구분하지 못하는 기존 방법의 한계에서 출발한다.
- [확정][index.md 기반] 첫 사전 검증 항목은 `기존 logical AD baseline이 실제로 condition shift에 취약한지` 확인하는 것이다.
- [확정][index.md 기반] 여기서 핵심 관찰치는 `정상인데 condition만 바뀐 query`를 기존 모델이 얼마나 anomaly처럼 오판하는지 여부다.
- [확정][index.md 기반] 본 검증 세션은 새 모델 제안을 하지 않고 baseline 취약성의 존재 여부와 강도를 측정하는 데 집중한다.

## 3. 이번 세션의 판단

- [확정][index.md 기반] baseline 취약성 검증의 1차 비교군은 최소 3개 split으로 구성한다.
  - `reference_normal_clean`
  - `query_normal_clean`
  - `query_normal_augmented`
- [확정][실험 필요] 핵심 비교 지표는 `normal clean -> normal augmented` 구간에서의 false positive 증가량과 anomaly score shift다.
- [제안][실험 필요] 부가 비교 지표는 `clean normal`과 `augmented normal`의 score distribution overlap이다.
- [제안][실험 필요] 가능하면 baseline별로 동일 reference set을 공유하고, query condition만 바꾸는 paired evaluation을 우선한다.
- [제안][실험 필요] shift 종류는 최소 `brightness`, `blur`, `background clutter` 세 축으로 분리해 기록한다.
- [검증 필요][실험 필요] baseline이 취약하다고 판단하는 기준은 아래 셋 중 하나 이상이 명확히 관찰되는 경우로 둔다.
  - 정상 augmented query에서 false positive rate가 clean 대비 일관되게 증가
  - anomaly score 평균이나 중앙값이 augmented normal에서 일관되게 상승
  - clean normal과 augmented normal의 score distribution이 유의하게 분리되어 augmentation 자체를 anomaly로 읽는 경향이 보임

## 4. 보류 또는 충돌

- 현재 저장소에는 데이터셋 로더, baseline 구현, 실험 실행 코드가 없다.
- 따라서 이번 단계에서는 `실험 폴더 구조`, `설정 템플릿`, `결과 기록 계약`까지만 고정한다.
- `logical anomaly` split은 1차 실험의 필수 조건이 아니다.
  - 지금 목적은 baseline의 `domain/condition sensitivity` 확인이다.
  - logical anomaly split은 후속 해석용으로만 추가한다.

## 5. 총괄 세션에 전달할 질문

- 1차 검증 대상 baseline을 몇 개까지 둘 것인가?
- `condition shift`는 synthetic corruption 우선인지, 실제 촬영 조건 변화 우선인지?
- baseline 비교의 1차 지표를 `normal FPR@fixed threshold` 중심으로 둘지, `score shift` 중심으로 둘지?

## 6. 다음 액션 제안

- `experiments/validation/condition_shift_baseline/` 아래에 데이터 계약, 실험 설정, 결과 기록 템플릿을 둔다.
- baseline별 실행 스크립트가 생기면 공통 입력/출력 스키마를 맞춘다.
- 이후 실제 구현 단계에서는 `clean normal vs augmented normal score shift`를 우선 시각화하는 리포트를 붙인다.

## 3줄 요약
1. 이번 검증의 핵심은 `condition shift`가 기존 logical AD baseline의 정상 오탐을 실제로 키우는지 확인하는 것이다.
2. 실험은 `reference normal`, `query normal clean`, `query normal augmented`를 기본 split으로 잡는다.
3. 현재 단계에서는 코드보다 먼저 실험 폴더 구조와 결과 기록 계약을 고정한다.
