# 프로젝트 인수인계 문서

## 1. 프로젝트 개요

본 프로젝트는 **normal-only few-shot setting**에서, 소수의 정상 reference 이미지로부터 **component 및 component 관계의 정상 규칙(rule)** 을 형성하고, query 이미지가 이 규칙과 얼마나 정합하는지를 기반으로 **logical anomaly**를 탐지하는 방향을 목표로 한다.

기존 anomaly detection은 정상 분포를 기준으로 query가 얼마나 벗어나는지를 보는 방식이 많다. 그러나 실제 산업 환경에서는 조명 변화, blur, background clutter 등 **visual condition shift**가 정상 분포 자체를 흔들 수 있으며, 이로 인해 정상 샘플이 anomaly로 오탐지될 수 있다. 특히 logical anomaly는 patch-level appearance 이상보다 **component 간 관계 위반**으로 나타나는 경우가 많다.

따라서 본 프로젝트는 anomaly를 단순 외형 이상이 아니라, **reference 기반 component relation rule violation**으로 재정의하는 방향을 취한다.

---

## 2. 핵심 문제 정의

### 해결하려는 문제
- visual distraction(밝기 변화, blur, background clutter 등) 하에서 기존 AD 방법의 성능 저하
- appearance 중심 방법이 **condition-induced discrepancy**와 **true logical anomaly**를 잘 구분하지 못하는 문제
- logical anomaly detection 연구의 상대적 부족
- normal-only 환경에서 anomaly data 없이도 robust한 logical anomaly detection을 수행하고자 함

### 연구 질문
- 소수의 정상 reference만으로 **정상 component relation rule**을 만들 수 있는가?
- query 이미지가 이 정상 rule과 얼마나 정합하는지를 통해 anomaly를 판단할 수 있는가?
- background feature를 anomaly의 직접 판단 대상이 아니라 **condition calibration signal**로 쓸 수 있는가?
- pairwise relation만으로 충분한가, 아니면 최소한의 graph-level consistency(현재는 **2-hop relational consistency**)가 필요한가?

---

## 3. 현재 채택된 전체 방향

### 기본 설정
- **Normal-only**
- **Few-shot**
- 가능하면 **training-free에 가까운 구조**
- backbone / foundation model 재학습 없이, reference 기반 rule memory를 생성
- query는 이 rule memory와 matching하여 anomaly score 산출

### 철학적 위치
- PatchCore처럼 정상 reference로부터 기준을 만든다는 철학은 유지
- 단, patch memory가 아니라 **component / relation rule memory**를 사용
- UniVAD의 component-level 접근을 토대로 하되, **reference-query rule matching**으로 확장
- 그래프는 full GNN 학습보다는 **graph-structured probabilistic rule matching**의 틀로 사용

---

## 4. 현재 구조 요약

### 4.1 Component Extraction
- UniVAD의 **C3**를 활용하여 이미지로부터 component mask 및 component feature 추출
- component 단위의 표현을 형성
- 배경 patch / background feature도 함께 추출
- 추후 C3 경량화나 대체는 가능하지만, 현재 1차 구조에선 **C3를 그대로 사용하는 방향**

### 4.2 Node 정의
각 component를 node로 정의한다.

node에 포함되는 정보:
- component feature
- component 위치
- component 크기 / shape
- component type 또는 prototype identity

즉 node는 **개별 component의 정상성**을 나타낸다.

### 4.3 Edge 정의
각 component pair를 edge로 정의한다.

현재 edge는 세 축으로 나눠 생각 중이다.

#### (a) Geometric relation
- normalized distance
- angle
- relative size
- contact / overlap
- 상대적 배치 정보

#### (b) Structural relation
- **gray-scale 기반 union feature**
- 두 component를 함께 포괄하는 union relation image에서 추출
- 색상 의존은 줄이고 구조 중심 정보를 반영
- edge/gradient/shape 중심 구조 표현 가능

#### (c) Appearance relation
- component 간 상대 색 차이
- 약한 pairwise compatibility
- 단, appearance는 structural branch보다 보조적 역할로 둘 가능성이 큼

---

## 5. Gray-scale union feature에 대한 현재 입장

현재 edge relation의 핵심 입력 중 하나는 **gray-scale 기반 union relation image**다.

의도:
- 두 component를 함께 포함하는 region은 pairwise relation을 직접 담고 있음
- raw RGB union은 배경/조명/blur 영향을 너무 많이 받음
- 따라서 color dependence를 낮추고 **구조 중심 relation representation**을 얻기 위해 gray-scale 또는 structure-oriented relation image를 사용

주의:
- “condition 제거”라기보다 **condition dependence 완화**로 표현하는 것이 적절
- 색상 정보는 완전히 버리지 않고, appearance relation branch에서 약하게 유지 가능

---

## 6. Rule Memory

본 프로젝트의 핵심은 학습된 classifier가 아니라, **reference 기반 rule memory**이다.

### 6.1 Node Rule
정상 reference로부터 각 component에 대해 형성
- 정상 component type / prototype
- 예상 등장 위치의 허용 범위
- 크기 허용 범위

즉,  
“이 component는 정상일 때 대략 이렇게 생기고 이 범위 안에 나타난다”

### 6.2 Edge Rule
정상 reference로부터 각 component pair에 대해 형성
- geometric relation rule
- structural relation rule
- appearance relation rule

즉,  
“이 두 component는 정상일 때 대략 이런 관계를 가진다”

### 6.3 Rule 표현 방식
현재 고려 중:
- Gaussian
- MoG (Mixture of Gaussians)
- 일부 relation descriptor에 대해 subspace modeling 가능성도 후순위로 존재

특히 geometric relation(거리, 각도)은 저차원이므로 Gaussian/MoG로 보기 적절함.

---

## 7. Matching 방식

현재 가장 적절한 matching 구조는 다음과 같다.

### Step 1. Node correspondence
- query component와 reference rule memory 간 대응을 먼저 정렬
- component feature, 위치, 크기를 기준으로 cost matrix 구성
- Hungarian matching 사용
- unmatched node는 missing / extra anomaly로 처리

### Step 2. Edge matching
- matched node 쌍에 대해 edge rule과 query edge를 비교
- geometric rule: 확률분포 기반 likelihood / distance
- structural rule: grayscale union feature 기반 distance
- appearance rule: 상대 색상 및 visual compatibility distance

### Step 3. 2-hop relational consistency
- pairwise relation만으로는 놓칠 수 있는 구조적 모순을 보기 위해 추가
- triad consistency라는 표현보다 **2-hop relational consistency**라는 표현이 더 적절하다고 판단
- direct edge만이 아니라, 두 단계 이웃 관계가 정상 구조와 일관적인지 평가

---

## 8. Background feature의 역할

배경은 anomaly 판단의 주연이 아니다.  
배경은 **condition shift estimator / calibration signal**이다.

사용 목적:
- reference와 query의 background feature를 비교
- cosine similarity 또는 distance 기반으로 background shift score 계산
- appearance-based relation score의 신뢰도를 조정
- 즉 배경은 anomaly 여부를 직접 결정하는 게 아니라, **condition-induced false positive를 줄이기 위한 score calibration**에 사용

현재 권장 구조:
- background shift가 크면 appearance relation score를 덜 믿음
- structural relation score는 상대적으로 유지

---

## 9. 그래프를 어떻게 위치시키는가

본 프로젝트는 full GNN 학습을 하는 그래프 모델이라기보다,
**graph-structured probabilistic rule matching framework**에 가깝다.

즉:
- 겉으로는 component relation graph
- 속으로는 reference-derived rule memory와 query의 matching
- 그래프를 살리는 핵심은 node/edge뿐 아니라 **2-hop relational consistency**까지 고려하는 데 있다

---

## 10. 현재 폐기된 아이디어

다음 아이디어들은 현재 구조에서는 폐기되었거나 채택하지 않기로 한 상태다.

### 폐기
- SALAD 등 기존 logical AD에 단순 데이터 증강 추가
- C3 전체 distillation
- full heterogeneous graph
- instance-level 세밀 비교
- raw RGB union feature 직접 사용
- background feature 직접 주입
- 위치 구역 기반 규칙
- pairwise relation classifier 직접 학습
- full graph matching / exact graph alignment

### 이유 요약
- 기술적 기여가 약함
- training-free / normal-only 방향과 충돌
- 구조가 지나치게 무거워짐
- relation보다 다른 요인을 더 보게 됨
- 현재 단계에서 복잡도 대비 이득이 불명확함

---

## 11. 후순위 / 고도화 아이디어

다음은 당장 핵심에는 넣지 않지만 확장 가능성이 있는 항목이다.

- structural / appearance disentanglement를 더 엄격히 모델링
- relation image encoder를 경량 학습하는 normal-only adaptation
- SubspaceAD식 relation descriptor subspace modeling
- calibration head 학습
- C3 경량화
- graph attention 유사 가중치(단, 현재는 rule-based weighting이 우선)
- 2-hop consistency를 더 정교한 graph consistency로 확장

---

## 12. 시작 전에 필요한 사전 검증

본 프로젝트를 실제로 진행하기 전에 반드시 확인할 항목들:

1. 기존 logical AD baseline이 실제로 condition shift에 취약한지  
   - VID-AD 주장만이 아니라 기존 데이터셋 변형에서도 확인 필요

2. C3가 condition shift 하에서도 안정적인 component를 추출하는지

3. few-shot reference만으로 node/edge rule이 실제로 형성 가능한지

4. node 위치만으로 충분한지, relation rule이 실제로 필요한지 baseline 비교

5. grayscale union feature가 edge relation 표현으로 실제로 유효한지

6. background feature가 calibration signal로 활용 가능한지

7. 2-hop relational consistency가 실제로 추가 설명력 / 성능 이득이 있는지

---

## 13. 현재 읽은 토대와 추가적으로 참고한 방향

핵심 토대:
- **UniVAD**: component extraction / few-shot training-free baseline
- **PatchCore**: normal-only memory bank 철학
- **SALAD**: logical anomaly, composition-level 문제의식
- **SubspaceAD**: normal variation을 단순한 수학적 구조로 모델링하는 예시

보강용 논리:
- pairwise relation representation
- cycle consistency / graph matching 계열의 구조 일관성 아이디어
- distribution shift robustness / calibration 관련 배경

---

## 14. 현재 구조의 핵심 한 줄 요약

> **소수의 정상 reference로부터 component relation graph 수준의 확률적 규칙을 형성하고, query가 이 규칙과 얼마나 정합하는지를 통해 visual distraction 하의 logical anomaly를 판단하는 normal-only few-shot framework**

---

## 15. 프로젝트 가안 이름

- **ReGraM**
- 의미: **Relation Graph Memory**

현재 프로젝트 가안은 **ReGraM**으로 확정.

---

## 16. 다음 언어모델이 이어서 해야 할 것

### 우선순위 높음
1. 현재 노션 초안을 **보고서형 문장**으로 다듬기
2. `문제 정의 → 아이디어 → 구조 → 기여점 → 검증 계획` 순으로 재배치
3. 사전 검증 항목을 실험 계획 수준으로 구체화
4. matching 및 score 구조를 수식 수준으로 더 정리
5. edge rule에서 geometric / structural / appearance를 어떻게 수식화할지 확정

### 우선순위 중간
6. 2-hop relational consistency를 구체적인 descriptor 수준으로 설계
7. Gaussian vs MoG vs subspace 중 어떤 rule 모델링이 더 적절한지 판단
8. background calibration score의 수식화

### 우선순위 낮음
9. relation encoder나 calibration head를 포함한 약한 학습형 확장 버전 검토
10. C3 대체 또는 경량화 방안 정리