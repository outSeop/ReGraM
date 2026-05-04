# ReGraM — Claude Code 인수인계 문서

이 문서는 Claude Code가 이 레포지토리에서 작업을 이어받기 위한 컨텍스트와 작업 지침을 담고 있습니다. 작업 시작 전 반드시 전체를 읽어주세요.

---

## 1. 프로젝트 개요

- **프로젝트명**: ReGraM
- **연구 도메인**: Computer Vision / Anomaly Detection (AD)
- **데이터셋**: **MVTec LOCO AD** — 의도적 condition shift 평가가 핵심.
- **연구자**: Song Inseop (석사과정)
- **현재 단계**: 초기 실험 단계. `experiments/validation/condition_shift_baseline/`에 baseline 실험과 자체 `src/`가 들어 있음.
- **연구 방향**: MVTec LOCO 기반 의도적 condition shift 하에서 anomaly detection의 robustness / failure mode 분석.

> **TODO**: ReGraM 이름의 의미와 핵심 아이디어를 한 단락으로 정리해 README.md에 추가할 것. (graph + matching 계열로 추정되나 확정 필요)

### 1.1 데이터셋 컨텍스트 (중요)

본 프로젝트는 일반 MVTec AD가 아닌 **MVTec LOCO AD**를 사용합니다. LOCO는 logical anomaly와 structural anomaly를 분리하는 데이터셋이며, 본 연구는 여기에 **의도적 condition shift**(조명, 배경, 카메라 변화 등)를 가해 모델의 견고성을 평가합니다. 따라서:

- **Dataset 클래스**는 MVTec LOCO 포맷에 맞아야 하며, 추가로 condition shift 변형을 적용할 수 있는 인터페이스(transform / perturbation parameter)를 가져야 합니다.
- **Metric**은 logical / structural anomaly를 분리해 보고할 수 있어야 합니다 (LOCO 표준).
- `manifests/`는 condition shift 정의(어떤 perturbation을 어떤 강도로 적용했는지)를 담고 있을 가능성이 높음 — Phase 0에서 확인 필요.

---

## 2. 현재 레포 상태 (정직하게)

### 2.1 디렉터리 구조 (현재)

```
ReGraM/
├── .vscode/
├── docs/
├── experiments/
│   └── validation/
│       └── condition_shift_baseline/
│           └── src/              # ← 현재 src가 여기 안에 있음
├── manifests/
├── .env.example
├── .gitignore
└── index.md
```

### 2.2 코드 구성 비율

- **Jupyter Notebook: 97.4%**
- **Python: 2.6%** — 대부분 `experiments/validation/condition_shift_baseline/src/` 내부로 추정.

### 2.3 잘 되어 있는 것 (유지할 것)

- `.env.example` — 환경변수/경로 분리 의식이 있음.
- `experiments/validation/condition_shift_baseline/` — 실험을 의미 있는 이름으로 분리. `{stage}/{experiment_name}/` 네이밍 컨벤션 유지.
- `manifests/` — 데이터/실험 메타데이터 분리.
- `docs/` — 문서 디렉터리 분리.
- `.vscode/` — 에디터 설정 공유.
- 실험 폴더 안에 src를 둔 시도 자체는 **실험별 격리**라는 좋은 의도였음 — 단, 향후 재사용 우선 구조로 통합 예정.

### 2.4 부족하거나 비어 있는 것

- 공통 라이브러리 코드 (`src/regram/`) 없음. 현재 src는 실험 폴더에 종속됨.
- 학습/평가 진입점 (`scripts/train.py` 등) 없음.
- Config 관리 시스템 (`configs/`) 없음.
- 단위 테스트 (`tests/`) 없음.
- `README.md` 없음 (현재 `index.md`만 있음).
- `pyproject.toml` 또는 `requirements.txt` 명시 여부 불확실.

---

## 3. 작업 목표

이번 인수인계의 핵심 목표는 두 가지입니다:

1. **`experiments/.../src/`에 있는 코드를 공통 `src/regram/`으로 승격(promote)**하여 재사용 가능한 라이브러리로 만들기.
2. **노트북 중심 코드베이스를 모듈화된 연구 코드베이스로 점진적 리팩토링**.

> **구조 결정 (확정)**: 공통 `src/regram/`으로 통합합니다. 실험별 src는 두지 않습니다. 과거 실험의 재현성은 **git tag/branch**로 확보합니다 (예: 현재 baseline을 옮기기 전 `tag: pre-refactor-baseline` 부여).

### 3.1 최종 목표 디렉터리 구조

```
ReGraM/
├── configs/                    # NEW — Hydra config
│   ├── default.yaml
│   ├── model/
│   ├── dataset/                # MVTec LOCO + condition shift 정의
│   └── experiment/
├── src/regram/                 # NEW — 공통 라이브러리 (모든 실험이 import)
│   ├── __init__.py
│   ├── data/                   # MVTec LOCO Dataset, condition shift transform
│   ├── models/                 # 모델 아키텍처
│   ├── losses/
│   ├── trainers/
│   ├── evaluators/             # AUROC, AUPRO, logical/structural 분리 metric
│   ├── utils/
│   └── visualization/
├── scripts/                    # NEW — 실행 진입점
│   ├── train.py
│   ├── evaluate.py
│   └── prepare_data.py
├── notebooks/                  # 기존 노트북을 여기로 이동, 역할 재정의
│   ├── 01_eda_*.ipynb
│   ├── 02_failure_analysis_*.ipynb
│   └── 03_paper_figures_*.ipynb
├── experiments/                # 유지 — src는 더 이상 두지 않음
│   └── validation/
│       └── condition_shift_baseline/
│           ├── config.yaml         # 이 실험의 Hydra config snapshot
│           ├── checkpoints/
│           ├── logs/
│           └── results/            # metric, qualitative output
├── manifests/                  # 유지
├── docs/                       # 유지
├── tests/                      # NEW
├── .env.example
├── .gitignore
├── pyproject.toml              # NEW
├── CLAUDE.md                   # 이 문서
└── README.md                   # NEW
```

### 3.2 노트북의 새로운 역할 (엄격히 적용)

리팩토링 후 `notebooks/`에 들어가도 되는 것은 **딱 세 가지**:

1. **EDA** — MVTec LOCO 데이터셋 탐색, condition shift 적용 결과 시각 확인.
2. **분석/디버깅** — 학습 끝난 모델의 failure case 분석, attention/heatmap 시각화, condition별 성능 비교.
3. **논문 figure 생성** — qualitative result, table, plot.

들어가면 **안 되는 것**: training loop, 모델 정의, data loading, loss/metric 정의. 이런 코드는 즉시 `src/regram/`으로 이동하고 노트북에서는 `from regram.X import Y`로 import만.

---

## 4. 작업 단계 (순서대로 진행)

### Phase 0: 현황 파악 (반드시 먼저, 사용자 보고 필수)

작업 시작 전 다음을 확인하고 사용자(인섭)에게 보고:

1. **`experiments/validation/condition_shift_baseline/src/` 내부 구조와 모듈 목록** — 어떤 파일이 있고 각 파일의 역할은 무엇인지.
2. **모든 `.ipynb` 파일 목록과 역할 추정**. 각 노트북에서 다음이 어디에 있는지 매핑:
   - 모델 정의 / Dataset / Loss / Training loop / Evaluation / 시각화
3. **노트북과 `condition_shift_baseline/src/`의 의존 관계** — 노트북이 그 src를 import하는지, 아니면 독립적으로 코드를 갖고 있는지.
4. **의존성 파일** (`requirements.txt`, `pyproject.toml`, `environment.yml`) 존재 및 내용.
5. **`manifests/`의 역할** — condition shift 정의가 들어 있는지, 아니면 데이터 split 정의인지.
6. **`condition_shift_baseline/` 안의 비-코드 산출물** — checkpoint, log, 결과 파일 위치.

**보고 후 사용자 승인 받고 Phase 1로.** 임의로 파일 이동/삭제 절대 금지.

### Phase 1: 마이그레이션 전 안전장치 (가장 중요)

**현재 코드와 실험 결과를 보존한 채로 새 구조의 골격을 먼저 만듭니다.**

1. **현재 상태에 git tag 부여**: `git tag pre-refactor-baseline` — 리팩토링 전 baseline 상태를 영구 보존.
2. **새 브랜치에서 작업**: `git checkout -b refactor/promote-src-to-common`.
3. `README.md` 작성 (프로젝트 개요, 설치, 실행 방법 골격). `index.md` 처리 방침은 사용자에게 묻기.
4. `pyproject.toml` 작성. 노트북과 기존 src에서 import하는 라이브러리를 모두 스캔해 의존성 명시.
5. `.gitignore` 점검 — `outputs/`, `wandb/`, `__pycache__/`, `.ipynb_checkpoints/`, `data/`, `*.ckpt`, `*.pth`, `experiments/**/checkpoints/`, `experiments/**/logs/` 등.
6. **빈 패키지 골격 생성**: `src/regram/__init__.py`와 하위 디렉터리(`data/`, `models/`, `losses/`, `trainers/`, `evaluators/`, `utils/`, `visualization/`) 각각 빈 `__init__.py`.
7. `pip install -e .` 동작 확인 후 사용자에게 보고.

### Phase 2: src 마이그레이션 (실험 폴더 → 공통)

`experiments/validation/condition_shift_baseline/src/`의 코드를 `src/regram/`으로 옮깁니다. **한 모듈씩, 작은 단위로**.

권장 우선순위:

1. **utils** — seed, logger, checkpoint 등 사이드이펙트 적은 것부터.
2. **evaluators** — AUROC, AUPRO, LOCO의 logical/structural 분리 metric.
3. **data** — MVTec LOCO Dataset, condition shift transform.
4. **models** — 모델 아키텍처.
5. **losses**.
6. **trainers** — 마지막. 변동성이 가장 큼.

각 모듈 이동마다:
- 기존 `condition_shift_baseline/src/` 안의 import 경로를 `from regram.X import Y` 로 수정.
- 노트북도 동일하게 import 경로 수정.
- **노트북 한 번 실행**해 같은 결과가 나오는지 확인 요청 (output 비교).
- 한 모듈 = 한 commit.

이 단계가 끝나면 `experiments/validation/condition_shift_baseline/src/`는 비워지고, 그 위치에는 config snapshot과 결과물만 남습니다.

> **주의**: 이 단계에서는 **코드 동작은 그대로 두고 위치만 옮깁니다**. 함수 시그니처, 알고리즘, 결과 모두 동일해야 함. 리팩토링과 마이그레이션을 동시에 하지 말 것.

### Phase 3: 진입점 스크립트 작성

1. `scripts/train.py` — 가장 잘 돌아가는 노트북 기준으로 학습 진입점. 처음에는 argparse로 시작 가능.
2. `scripts/evaluate.py` — checkpoint 로드 후 평가. **condition shift 인자**를 받아 여러 조건에서 batch 평가 가능하게.
3. 두 스크립트가 노트북과 동일한 metric을 재현하는지 사용자와 함께 검증.

### Phase 4: Config 시스템 도입 (Hydra)

argparse → Hydra:

1. `configs/default.yaml`, `configs/model/`, `configs/dataset/` (LOCO category, condition shift 종류), `configs/experiment/` 구조.
2. `scripts/train.py`를 `@hydra.main`으로 래핑.
3. Hydra의 출력 디렉터리를 `experiments/{stage}/{name}/{run_id}/`로 매핑.
4. WandB 통합. `entity`/`project`는 `.env`에서 읽기.

### Phase 5: 테스트 추가

연구 코드라도 다음은 반드시:

- `evaluators/` — AUROC, AUPRO 계산이 sklearn 등 alternative 구현과 일치.
- `data/` — MVTec LOCO Dataset이 올바른 shape/dtype 반환, condition shift transform이 결정적(seed 고정 시).
- `utils/` — checkpoint save/load round-trip.

`pytest` 사용. `tests/` 구조는 `src/regram/`을 미러링.

### Phase 6: 노트북 정리

1. 기존 노트북을 `notebooks/`로 이동.
2. 4.2의 "들어가면 안 되는 것" 제거 — 이미 `src/`로 옮긴 코드는 import로 대체.
3. 파일명 통일: `{번호}_{역할}_{대상}.ipynb` (예: `01_eda_mvtec_loco.ipynb`, `02_failure_analysis_condition_shift.ipynb`).
4. 각 노트북 맨 위에 `%load_ext autoreload`, `%autoreload 2`.

---

## 5. 작업 중 지켜야 할 규칙

### 5.1 절대 하지 말 것

- **`experiments/`, `manifests/` 안의 결과물(checkpoint, log, metric 파일)을 임의로 삭제/이동하지 말 것.**
- **`condition_shift_baseline/src/`를 통째로 옮긴 뒤 원본을 즉시 삭제하지 말 것.** Phase 2에서는 *복사 → 새 위치에서 동작 확인 → 원본 삭제* 순서 준수. 한 commit에 다 하지 말고 분리.
- **노트북의 출력(output) 셀을 임의로 clear하지 말 것.** 사용자에게 먼저 묻기.
- **`.env` 파일을 commit하지 말 것.**
- **Phase 2 도중 함수 시그니처/알고리즘 변경 금지.** 위치 이동만.
- **사용자 확인 없이 의존성 버전 업데이트 금지.** 재현성 깨짐.

### 5.2 항상 할 것

- **Phase 0 → Phase 1 진입 전 사용자 승인 필수.** 자율 진행 금지.
- **Phase 2의 각 모듈 이동마다 동작 확인 요청.**
- 작은 단위 commit. 메시지 예: `refactor: move utils from experiments/.../src to src/regram`, `feat: add scripts/train.py`, `chore: add pyproject.toml`.
- 사용자는 한국어/영어 모두 가능하나 **기술 설명은 한국어 선호**. 코드 주석은 영어로.

### 5.3 의사소통

- 사용자 이름은 **인섭 (Inseop)**. "인서프" 아님.
- 모호하면 추측하지 말고 질문.
- 큰 결정(라이브러리 선택, 구조 변경)은 옵션 제시 후 선택을 받을 것.

---

## 6. 도메인 컨텍스트 (참고)

ReGraM은 MVTec LOCO 데이터셋에 의도적 condition shift를 가해 anomaly detection 모델의 robustness를 평가하는 연구 프로젝트입니다. 코드베이스가 잘 지원해야 하는 것:

- **Category별 평가** — LOCO category별 metric 분리.
- **Logical / structural anomaly 분리 metric** — LOCO 표준 평가 방식.
- **Condition shift batch 평가** — 같은 모델을 여러 perturbation 조건에서 일괄 평가. `evaluate.py`는 condition을 인자로 받아 결과를 정리된 형태로 저장해야 함.
- **Failure case 시각화 파이프라인** — heatmap overlay, GT mask 비교, condition별 정렬 grid.

참고할 만한 유사 코드베이스: [Anomalib (OpenVINO)](https://github.com/openvinotoolkit/anomalib) — 모듈 분리 구조 참고용. fork하지 말고 구조만 참고.

---

## 7. 시작 시 체크리스트

Claude Code가 이 레포에서 처음 작업을 시작할 때:

- [ ] 이 `CLAUDE.md` 전체를 읽었는가
- [ ] `git status` / `git log --oneline -20`으로 현재 브랜치와 최근 커밋 확인
- [ ] `experiments/validation/condition_shift_baseline/`의 산출물(checkpoint, log) 위치 점검 (수정 금지 대상)
- [ ] `experiments/validation/condition_shift_baseline/src/` 내부 구조 파악
- [ ] 모든 `.ipynb` 목록과 역할 추정
- [ ] 노트북과 기존 src의 import 관계 파악
- [ ] 의존성 파일 존재 여부 확인
- [ ] Phase 0 결과를 사용자(인섭)에게 보고하고 Phase 1 승인 요청

---

*문서 작성일: 2026-05-04*
*다음 업데이트는 Phase 1 완료 후 권장.*
