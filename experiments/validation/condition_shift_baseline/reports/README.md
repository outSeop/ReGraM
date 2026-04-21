# Reports

이 디렉터리에는 아래 산출물을 둔다.

- 공식 해석용 Markdown
- small summary JSON
- clean vs shifted normal false positive 비교 표
- shift 종류별 qualitative 사례에 대한 작은 참조 자산

## Official Report vs Auxiliary Artifacts

이 디렉터리는 `공식 보고`와 `보조 산출물`을 구분해서 사용한다.

- 공식 보고
  - `report_*.md`
  - 작은 `summary.json`
  - 재현과 해석에 필요한 최소 표
- 보조 산출물
  - 대형 panel/png
  - 장문 raw log
  - runtime 복사용 임시 자산

원칙은 다음과 같다.

- 공식 판단과 결론은 Markdown에 남긴다.
- summary JSON은 비교 가능한 small reproducible state로만 둔다.
- 큰 시각 자산은 필요 시 `sample_panels/` 또는 외부 artifact store로 보낸다.
- wandb 링크가 있더라도 보고서 본문을 대체하지 않는다.

권장 파일명:

- `report_YYYY-MM-DD_baseline_name.md`
- `plot_YYYY-MM-DD_baseline_name.png`

현재 진행 요약:

- [report_2026-04-20_patchcore_univad_progress.md](/Users/song-inseop/연구/ReGraM/experiments/validation/condition_shift_baseline/reports/report_2026-04-20_patchcore_univad_progress.md)
