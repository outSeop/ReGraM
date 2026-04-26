# Colab CUDA Quickstart

`UniVAD`는 로컬 CPU보다 Colab GPU에서 smoke test를 먼저 통과시키는 편이 훨씬 낫다. 아래 순서는 `Git 기반 repo checkout + Drive의 prepared dataset 재사용`을 기준으로 한다.

## Recommended Flow

1. GPU runtime으로 Colab을 연다.
2. Drive를 mount한다.
3. `/content/ReGraM`에 repo를 Git clone 하거나 기존 checkout을 pull 한다.
4. 이미 준비된 `data/mvtec_loco_caption` / `data/mvtec_loco_caption_smoke`가 있으면 그것을 먼저 재사용한다.
5. 준비된 dataset이 없을 때만 raw LOCO에서 `data/mvtec_loco_caption`를 만든다.
6. `pretrained_ckpts`를 `external/UniVAD/pretrained_ckpts` 아래에 둔다.
7. notebook의 runtime setup/readiness 셀을 실행한다.
8. `restart_required`가 나오면 Colab runtime을 재시작하고 처음부터 다시 실행한다.
9. smoke subset이 없을 때만 만든다.
10. `run_univad_smoke_colab.py`로 `good` / `logical` 2장만 먼저 돌린다.
11. smoke가 통과하면 `run_univad_clean_eval.py` 또는 manifest runner로 평가와 summary JSON 저장까지 끝낸다.

## Minimal Commands

```bash
python - <<'PY'
from google.colab import drive
drive.mount('/content/drive')
PY

REPO_URL="https://github.com/outSeop/ReGraM.git"
REPO_DIR="/content/ReGraM"
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
```

prepared dataset만 runtime으로 복사하려면:

```bash
python experiments/validation/condition_shift_baseline/colab/bootstrap_runtime.py --dry-run
python experiments/validation/condition_shift_baseline/colab/bootstrap_runtime.py
```

가장 빠른 경로는 이미 준비된 dataset을 그대로 재사용하는 것이다.

이미 아래 둘이 있으면 raw dataset 준비 단계는 건너뛴다.

- `data/mvtec_loco_caption/meta.json`
- `data/mvtec_loco_caption_smoke/meta.json`

준비된 dataset이 없고 데이터가 원시 LOCO 포맷이면:

```bash
python experiments/validation/condition_shift_baseline/src/univad/prepare_mvtec_loco.py \
  --src-root /content/ReGraM/data/row/mvtec_loco_anomaly_detection \
  --dst-root data/mvtec_loco_caption
```

smoke subset 생성:

```bash
python experiments/validation/condition_shift_baseline/src/univad/prepare_smoke_subset.py \
  --src-root data/mvtec_loco_caption \
  --dst-root data/mvtec_loco_caption_smoke \
  --class-name breakfast_box
```

smoke inference:

```bash
python experiments/validation/condition_shift_baseline/src/univad/run_smoke_colab.py \
  --repo-root . \
  --data-root data/mvtec_loco_caption_smoke \
  --class-name breakfast_box \
  --output experiments/validation/condition_shift_baseline/reports/univad_smoke_colab/breakfast_box.json
```

결과 파일:

- `experiments/validation/condition_shift_baseline/reports/univad_smoke_colab/breakfast_box.json`

clean 전체 평가:

```bash
python experiments/validation/condition_shift_baseline/src/univad/run_clean_eval.py \
  --repo-root . \
  --data-root data/mvtec_loco_caption \
  --class-name breakfast_box \
  --output experiments/validation/condition_shift_baseline/reports/univad_clean_eval/breakfast_box.json
```

결과 파일:

- `experiments/validation/condition_shift_baseline/reports/univad_clean_eval/breakfast_box.json`
- `experiments/validation/condition_shift_baseline/reports/univad_clean_eval/mvtec_loco/log.txt`

## Notes

- notebook 기본 동작은 `prepared dataset 재사용`이다.
- notebook은 조작판과 viewer 역할만 한다. Colab/UniVAD dependency, checkpoint, mask setup은 `src/univad/setup_runtime.py`에서 관리한다.
- 권장 실행 순서는 `git pull` -> dataset bootstrap -> runtime setup/readiness -> runner 실행 -> dashboard다.
- runtime setup이 `numpy`, `opencv`, `transformers`, `torchao`, `torchmetrics` 상태를 수정하면 `restart_required`를 보고 runtime을 재시작한다.
- notebook의 UniVAD 기본 runner 옵션은 Colab T4/L4 메모리를 우선해서 `--image-size 224 --amp`로 둔다. 448 실험이 필요하면 notebook에서 `univad_extra_args`를 명시적으로 넘겨 override한다.
- `Drive Archive Build`는 기본 실행 경로가 아니다.
- 기존 `mvtec_loco_anomaly_detection.tar` 또는 `.tar.gz`가 있으면 다시 만들지 않는 편이 낫다.
- 이 smoke runner는 `train/good/000`, `test/good/000`, `test/logical_anomalies/000`만 사용한다.
- 목적은 `setup + forward`가 CUDA에서 끝까지 도는지 먼저 확인하는 것이다.
- clean eval wrapper는 공식 `test_univad.py`를 호출하고, 요약 metric만 별도 JSON으로 저장한다.
- smoke가 통과한 뒤에만 clean 전체 평가나 corruption 비교로 넘어간다.
- Drive 원본 repo 동기화보다 Git checkout을 기준으로 두는 편이 더 안전하다.
- Colab에서도 notebook이 호출하는 `.py`는 반드시 현재 checkout된 repo 버전을 사용해야 한다.
