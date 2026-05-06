이건 **아주 작게, 검증 실험으로 진행**하는 게 맞다. 목표는 “성능 좋은 새 모델”이 아니라:

> **position_shift에서 absolute-position 기반 score는 크게 흔들리지만, relative relation score는 덜 흔들리는가?**

이 질문 하나만 확인하면 된다.

---

# 1. 실험 단위부터 작게 잡기

처음 범위는 이렇게 제한하는 게 좋다.

```text
category: breakfast_box
corruption: position_shift
severity: high
reference: clean normal 1장
query:
  - clean normal 전체 또는 30장
  - position_shift normal 전체 또는 30장
  - clean anomaly 일부
```

처음부터 `screw_bag`, `gaussian_noise`, `gaussian_blur`까지 넣지 말고, **가장 명확한 position_shift만** 본다.

---

# 2. 비교할 score를 3개로 둔다

여기서 핵심은 relation score 하나만 만드는 게 아니라, 비교군을 같이 만드는 것이다.

## Score A. Absolute position score

이건 일부러 취약한 score다.

component centroid의 절대 좌표 차이를 본다.

```text
S_abs = mean_i || c_i^query - c_i^ref ||
```

position_shift에서는 이 값이 커져야 한다.

이 score는 “왜 absolute position prior가 위험한가”를 보여주는 대조군이다.

---

## Score B. Centered position score

전체 object/component graph의 중심을 빼고 비교한다.

```text
c_i_centered = c_i - mean(c_1, ..., c_N)

S_centered = mean_i || c_i_centered^query - c_i_centered^ref ||
```

전체 객체가 통째로 이동했다면, graph 중심을 뺀 뒤에는 차이가 작아져야 한다.

이게 1차로 보고 싶은 결과다.

---

## Score C. Pairwise relation score

component 간 상대 관계를 본다.

```text
r_ij = c_j - c_i

S_pair = mean_{i,j} || r_ij^query - r_ij^ref ||
```

더 안정적으로 하려면 scale normalization까지 넣는다.

```text
r_ij = (c_j - c_i) / mean_pairwise_distance
```

position_shift가 단순 translation이면 `S_pair`는 거의 변하지 않아야 한다.

---

# 3. component는 처음엔 reference-only로 고정

SAM을 query마다 돌리지 말고, 처음에는 **reference에서만 component를 뽑는 방식**이 낫다.

## 절차

1. clean reference image 1장 선택
2. SAM으로 component mask 추출
3. 너무 작은 mask, 너무 큰 mask 제거
4. top-k component만 선택
5. 각 component의 bbox/centroid/area 저장
6. query 이미지에는 reference bbox 또는 mask 위치를 그대로 사용

이러면 query-side segmentation이 흔들리는 문제를 피할 수 있다.

단, 여기서 중요한 점이 있다.

reference mask를 query에 그대로 얹으면 centroid가 똑같아서 relation score가 항상 0에 가까울 수 있다. 그러면 의미가 없다. 그래서 1차 실험에서는 두 가지 버전을 나눠야 한다.

---

# 4. 1차는 “known transform”을 활용하는 게 좋다

네가 만든 `position_shift`라면 shift가 어떻게 적용됐는지 알 가능성이 높다. 예를 들어 dx, dy만큼 이미지를 이동시켰다면, 이 정보를 사용할 수 있다.

## Version 1. Known-shift sanity check

reference component 좌표를 shift transform만큼 같이 이동시킨다.

```text
c_i_shifted_ref = c_i_ref + (dx, dy)
```

그다음 비교한다.

* absolute position score: `c_i_shifted_ref`와 `c_i_ref` 비교 → 큼
* centered score: 중심 제거 후 비교 → 작음
* pairwise score: component 간 벡터 비교 → 거의 0

이건 detection 실험은 아니고, **position_shift가 relative relation을 보존한다는 sanity check**다.

이 실험이 제일 먼저다.

---

# 5. 2차는 query에서 component를 다시 찾는 실험

sanity check가 됐으면, 그다음에 query-side SAM을 일부 샘플에만 돌린다.

## 절차

1. reference image에서 SAM mask 추출
2. shifted query image에서도 SAM mask 추출
3. component별 metadata 계산

   * centroid
   * bbox size
   * area
   * optional DINO pooled feature
4. reference component와 query component matching
5. matched component끼리 absolute / centered / pairwise score 계산

matching은 처음에는 단순하게 가면 된다.

```text
cost = alpha * centroid_distance
     + beta * area_ratio_distance
```

position_shift에서는 centroid_distance가 커질 수 있으니, 가능하면 **centered centroid distance**나 **relative layout matching**을 같이 써야 한다.

더 안전한 matching cost:

```text
cost(i, j) =
  alpha * DINO_feature_distance
+ beta  * area_ratio_distance
+ gamma * centered_centroid_distance
```

여기서 DINO feature는 matching용으로만 쓰고, relation score 자체에는 최소한으로만 쓰는 게 낫다.

---

# 6. 최종 비교표는 이렇게 만들면 됨

핵심 결과표는 이 형태면 충분하다.

```text
score_type        clean_normal_median   shifted_normal_median   median_shift   shifted_fpr
UniVAD score
PatchCore score
S_abs
S_centered
S_pair
```

기대하는 모양은 이거다.

```text
UniVAD score      shift 큼    FPR 높음
PatchCore score   shift 큼    FPR 높음
S_abs             shift 큼    FPR 높음
S_centered        shift 작음  FPR 낮음
S_pair            shift 작음  FPR 낮음
```

이렇게 나오면 네 주장이 아주 깔끔해진다.

> position_shift는 absolute coordinate 기반 판단을 흔들지만, graph-centered 또는 pairwise relative relation은 더 안정적이다.

---

# 7. anomaly까지는 2단계로 넣기

처음부터 anomaly detection 성능까지 욕심내면 꼬인다. 순서를 나누자.

## Step 1. Normal stability

먼저 정상만 본다.

```text
clean normal vs shifted normal
```

질문:

> 정상 구조가 유지된 shift에서 relation score가 안정적인가?

여기서 성공해야 다음으로 넘어간다.

---

## Step 2. Anomaly separability

그다음 anomaly를 넣는다.

```text
clean normal vs clean anomaly
shifted normal vs clean anomaly
```

질문:

> relation score가 실제 logical/structural anomaly에는 올라가는가?

이때 relation score가 모든 anomaly를 잘 잡을 필요는 없다.
특히 appearance anomaly는 못 잡아도 된다. 네가 보고 싶은 건 structural/logical 쪽이다.

따라서 anomaly type별로 나눠야 한다.

```text
normal
logical anomaly
structural anomaly
```

가능하면 relation score는 structural anomaly 쪽에서 더 잘 반응해야 한다.

---

# 8. 실험 결과 해석 기준

## 좋은 결과

```text
S_pair shifted_normal_fpr << UniVAD shifted_normal_fpr
S_pair median_score_shift << UniVAD median_score_shift
S_pair structural anomaly score > clean normal score
```

그러면 다음 주장 가능:

> relative relation score는 position_shift에는 안정적이면서, 구조적 이상에는 반응할 가능성이 있다.

---

## 애매한 결과

```text
S_pair는 shift에 안정적이지만 anomaly도 못 잡음
```

이 경우에도 완전히 실패는 아니다.

해석:

> 단순 pairwise geometry만으로는 anomaly separability가 약하지만, shift-invariant structural representation의 가능성은 확인했다.

다음 고도화:

* component count mismatch
* relative ordering
* area ratio
* 2-hop relation
* triad consistency
* query-side component matching

---

## 나쁜 결과

```text
S_pair도 position_shift에서 크게 흔들림
```

이 경우 원인을 봐야 한다.

대부분은 relation score 문제가 아니라:

* component extraction이 흔들림
* query-side SAM mask가 달라짐
* matching이 틀림
* position_shift가 단순 translation이 아니라 crop/occlusion을 유발함

이 중 하나일 가능성이 높다.

---

# 9. 코드 구조는 이렇게 짜면 됨

파일 단위로는 이 정도면 충분하다.

```text
experiments/relation_mvp/
  01_extract_reference_components.py
  02_compute_relation_scores.py
  03_evaluate_relation_scores.py
  04_visualize_relation_failure.py
```

## 저장할 metadata

```json
{
  "image_path": "...",
  "components": [
    {
      "id": 0,
      "bbox": [x1, y1, x2, y2],
      "centroid": [cx, cy],
      "area": 1234,
      "mask_path": "..."
    }
  ]
}
```

## relation vector

처음에는 이렇게만.

```python
relation_ij = [
    dx / W,
    dy / H,
    distance / mean_pairwise_distance,
    angle,
    area_i / area_j,
]
```

단, `dx, dy`는 translation에 영향을 안 받지만, scale/crop에는 영향이 있을 수 있다.
그래서 `distance / mean_pairwise_distance` 같은 scale normalization을 같이 넣는 게 좋다.

---

# 10. 지금 당장 할 순서

바로 할 일은 이거다.

```text
1. position_shift가 어떤 transform인지 기록
   - dx, dy translation인지
   - crop/padding 포함인지
   - object 내부 구조가 유지되는지

2. breakfast_box reference 1장 선택

3. reference component metadata 생성
   - SAM or manual/bbox 가능
   - top-k component 저장

4. known-shift sanity check
   - S_abs
   - S_centered
   - S_pair
   비교

5. query-side SAM 일부 샘플만 적용
   - shifted normal 10장
   - matching 후 S_pair 계산

6. UniVAD/PatchCore score shift와 S_pair score shift 비교
```

---

# 정리

진행 방식은 이렇게 잡으면 된다.

> 먼저 `position_shift`가 relative relation을 보존하는 shift인지 sanity check한다. 그다음 reference component를 기준으로 absolute position score, centered position score, pairwise relation score를 계산하고, PatchCore/UniVAD score와 비교한다. 최종적으로 `S_pair`가 shifted normal에서 덜 흔들리고 structural anomaly에는 어느 정도 반응하는지 확인한다.

처음부터 SAM + DINO clustering + GNN으로 가지 말고, **`S_abs` vs `S_centered` vs `S_pair` 비교**만 먼저 해라. 이게 네 아이디어의 가장 작은 검증이다.
