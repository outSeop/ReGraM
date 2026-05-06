# ReGraM Documentation Index

이 디렉터리는 프로젝트 전체 문서의 입구다. 연구 아이디어, 검증 실험, 세션 로그를 분리해서 본다.

## Start Here

1. [프로젝트 인수인계 문서](../index.md)
   - ReGraM의 문제 정의, 핵심 아이디어, 현재 채택한 구조.
2. [Condition Shift Baseline Validation](../experiments/validation/condition_shift_baseline/README.md)
   - 기존 logical AD baseline의 condition shift 취약성 검증 실험.
3. [Condition Shift 문서 지도](../experiments/validation/condition_shift_baseline/docs/README.md)
   - 해당 실험의 실행 문서, 설계 문서, 보고서 위치.

## Global Docs

- [session_common_prompt.md](session_common_prompt.md)
  - 세션 문서가 지켜야 할 공통 계약.
- [prompts/validation_session_prompt.md](prompts/validation_session_prompt.md)
  - validation 역할 세션용 확장 프롬프트.
- [sessions/validation_2026-04-19_condition_shift_baseline.md](sessions/validation_2026-04-19_condition_shift_baseline.md)
  - condition shift baseline 검증 세션 로그.
- [decisions/README.md](decisions/README.md)
  - 확정된 결정 기록을 둘 위치.

## Writing Rules

- 공식 판단과 결론은 Markdown에 남긴다.
- 실험 실행 경로는 각 실험 폴더의 `README.md`와 `docs/FILE_MAP.md`를 source of truth로 둔다.
- 로컬 절대경로 링크는 쓰지 않고, repo-relative 링크를 사용한다.
- 대형 산출물, notebook output, raw log는 문서 원본으로 취급하지 않는다.
