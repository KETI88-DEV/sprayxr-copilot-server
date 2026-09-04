# Task 10 report: σ 문턱 0.1 채택 — 실제 기본값 재측정

**Status: done.**

Task 10-A에서 `trainer/scripts/measured_score.py`의 `CORRECTION_SIGMA_THRESHOLD`를
**2.0 → 0.1**로 바꿨다(파일 수정, 런타임 주입 아님). 이 문서는 그 바뀐 **기본값
그대로** `improvement_study.py`를 돌려 잰 결과다. Task 7·8·9는 `CORRECTION_SIGMA_
THRESHOLD`를 실행 중에만 바꿔치기(monkeypatch)해서 쟀다 — 파일에 커밋된 값은
그때도 2.0이었다. 이 문서가 그 갭을 메운다: 실제로 커밋된 0.1로 코드를 그대로
실행하면 뭐가 나오는가.

## 표집 규칙 (숫자를 보기 전에 고정, Task 6·7·8·9와 동일)

- 코퍼스: `/data/sources/professionals`(479세션, 605 CSV, 64,098패스, 6작업자,
  worker=전형기).
- `random.Random(20260828)`로 전체 후보를 셔플해 비복원 200개를 질의로 뽑는다.
- 질의마다, 같은 우8 키를 가진 전형기 후보 중 `rng.choice`로 레퍼런스 하나를
  고른다. 후보가 없으면 `no_expert_reference`로 기록하고 버리지 않는다.
- `--pairs 200 --seed 20260828`, 코드에 먼저 고정.
- `CORRECTION_SIGMA_THRESHOLD`/`MIN_CORRECTION_CM`/`SIGMA_FLOOR_M`/`MAX_SIGMA`는
  전부 `measured_score.py`가 단일 출처다. **이 실행에서 아무것도 주입하지
  않았다** — 파일에 적힌 값(0.1)을 그대로 읽는다.

## 두 번 돌렸다 — 왜

10-A를 적용한 뒤 재측정을 시도하는 과정에서, Task 7·8·9가 참조 아티팩트로 쓴
`spread_reference_expert.json`(빌드 2026-08-28 11:26)이 월드 축 규약을 Y-up으로
바로잡은 커밋(`5383079`/`3ccc8cb`, 2026-09-02)보다 **먼저** 지어졌다는 것을
확인했다 — 즉 HEAD(`a77b8b0`)의 코드는 이미 Y-up 수정을 담고 있는데, Task 7·8·9가
쓴 산포 레퍼런스는 그 수정 이전 버전이다. `docs/measurements/2026-09-02-axis-fix-before-after.md`가
이미 이 문제를 겪고 처방을 남겼다: 그 태스크가 새로 지은
`spread_reference_expert_yup.json`(2026-09-03 05:29, 현재 코드로 재빌드)이 지금
HEAD와 일치하는 유일한 아티팩트다.

그래서 두 조건을 다 쟀다:

1. **교차검증(stale 아티팩트)**: Task 7·8·9가 실제로 쓴
   `spread_reference_expert.json`(구, Z-up 시절 빌드) + 그때 쓴 색인 캐시
   (`improvement_study_index_cache.json`)로 σ=0.1(파일에 커밋된 값 그대로)을
   돌린다. 이건 "Task 9의 주입 측정값과 내 코드 수정이 같은 결과를 내는가"만
   확인하는 용도다 — 코드가 컴파일러가 아니라 값이 같은지 실측으로 비트 단위
   확인하는 것이다.
2. **실제 기본값(현재 아티팩트)**: HEAD의 코드와 일치하는
   `spread_reference_expert_yup.json` + 그 코드로 다시 지은 색인 캐시
   (`improvement_study_index_cache_yup.json`, 이번 실행에서 재빌드됨 — 캐시
   지문이 CSV 트리와 안 맞아 자동으로 다시 지어졌다)로 σ=0.1을 돌린다. **이게
   진짜 "지금 이 저장소의 기본값을 실행하면 무엇이 나오는가"에 대한 답이다.**

## 명령

```
docker exec -w /workspace ship-paint-trainer python3 trainer/scripts/improvement_study.py \
  --root /data/sources/professionals \
  --reference /output/spread_reference_expert_yup.json \
  --pairs 200 --seed 20260828 \
  --index-cache /output/improvement_study_index_cache_yup.json \
  --out /output/improvement_study_task10_default.json --progress --log-level INFO

docker exec -w /workspace ship-paint-trainer python3 trainer/scripts/improvement_study.py \
  --root /data/sources/professionals \
  --reference /output/spread_reference_expert.json \
  --pairs 200 --seed 20260828 \
  --index-cache /output/improvement_study_index_cache.json \
  --out /output/improvement_study_task10_crosscheck_staleref.json --progress --log-level INFO
```

## 결과 1 — 교차검증(stale 아티팩트, Task 9와 동일 아티팩트)

| metric | 값 |
|---|---|
| status_counts | ok 179 / no_expert_reference 21 |
| split n / split_reason | 155 (`none`=155, `no_split_reference`=24) |
| split median / q1 / q3 | **0.828323** / 0.719421 / 0.903192 |
| split ≥0.70 | **76.77%**(119/155) |
| correction fire rate (split) | **100%**(155/155) |

Task 9의 주입 측정값(σ=0.1: median 0.8283, ≥0.70 76.8%, n=155, 발부율 100%)과
**비트 단위로 일치**한다(0.8283234174307934, 소수 첫 자리까지 완전히 같음). 이는
"실제 상수를 바꾸는 것"과 "실행 중 모듈 전역을 바꿔치기하는 것"이 이 코드에서
정확히 같은 동작을 낸다는 것을 확인한다 — 예상대로다(`right_axes`/`head_stance`
등이 모듈 전역을 호출 시점에 직접 읽으므로).

## 결과 2 — 실제 기본값(현재 아티팩트, HEAD의 코드와 일치)

| metric | 값 |
|---|---|
| status_counts | ok 179 / no_expert_reference 21 |
| split n / split_reason | **178**(`none`=178, `no_split_reference`=1) |
| split median / q1 / q3 | **0.826701** / 0.725516 / 0.904660 |
| split ≥0.70 | **79.21%**(141/178) |
| correction fire rate (split) | **100%**(178/178) |
| same median / ≥0.70 | 0.846849 / 86.59% |
| 입도별 n (uv8/uv4/uv2/uv_off/none) | 72/49/42/15/1 |

## 두 결과를 나란히 두면

| | Task 9 주입값(stale) | 교차검증(stale, 이 태스크) | 실제 기본값(현재 아티팩트) |
|---|---|---|---|
| split n | 155 | 155 | **178** |
| split median | 0.8283 | 0.828323 | **0.826701** |
| split ≥0.70 | 76.8% | 76.77% | **79.21%** |
| no_split_reference | 24 | 24 | **1** |
| uv8 커버리지(입도 n) | (안 보고) | 3 | **72** |

**두 값이 다르다 — 브리핑이 요구한 대로 그 차이 자체를 적는다.** split median은
0.8283 → 0.8267로 **0.0016 하락**(사실상 동일, 잡음 수준), ≥0.70 비율은
76.8% → 79.2%로 **오히려 상승**했다. 방향이나 크기가 뒤집히지 않았다 — Task
10-A의 채택 근거(σ=0.1에서 발부율 100%, split median ~0.83, ≥0.70 ~77~79%)는
어느 아티팩트로 재도 **그대로 선다**.

바뀐 것은 **분모다**: split이 정의되는 쌍이 155개(24개는 `no_split_reference`로
버려짐)에서 178개(1개만 버려짐)로 늘었다. 원인은 σ 문턱과 무관하다 — 참조로 쓴
산포 아티팩트가 Y-up 축 수정 **이전**(stale) 대 **이후**(현재 HEAD와 일치)인가의
차이다. stale 아티팩트에서는 우8 키 커버리지가 나빠(우8=3, 대부분
uv_off=80으로 밀림) 분리(split) 낼 레퍼런스 분포를 못 찾는 쌍이 많았다 — 이건
`docs/measurements/2026-09-02-axis-fix-before-after.md`가 이미 보고한 커버리지
개선(우8 25.0%→49.8%)과 같은 현상이 이번에도 나타난 것이다. **Task 7·8·9가
σ 문턱 자체에 대해 내린 결론(발부율·개선율의 방향과 대략적 크기)은 이 갭 때문에
틀리지 않았다** — 다만 그 숫자들의 정확한 소수점 값과 표본 구성은 그때 그
태스크들이 의도치 않게 stale 아티팩트를 썼기 때문에 나온 것이지, 지금 이
저장소를 그대로 돌리면 나오는 숫자와 정확히 같지는 않다.

## 결론 — 헤드라인 숫자 (실제 기본값 기준)

- **발부율: 47.1% → 100%**(이 문서가 재확인). 모든 split-정의 쌍(178/178)에서
  교정이 나갔다.
- **split median: 0.000 → 0.827**(Task 7의 σ=2.0 기준선 0.000 대비).
- **split ≥0.70: 2.6% → 79.2%**(Task 7의 σ=2.0 기준선 2.6% 대비, 이 문서의
  현재-아티팩트 값).
- 이 문서의 숫자는 `docs/measurements/2026-09-02-axis-fix-before-after.md`가
  이미 확립한 "현재 HEAD와 일치하는 유일한 아티팩트 세트"(yup) 위에서 나왔다 —
  Task 10-A의 상수 주석에 적은 표(2.0/0.5/0.1/0.0)는 Task 7·8·9의 stale 아티팩트
  기준이고, 이 문서의 결과 2가 그 표의 σ=0.1 행을 현재 아티팩트로 다시 확인한
  버전이다.

## 한계

- σ=0.1 한 점만 실제 기본값으로 재측정했다(Task 10의 요구 범위). σ 스윕
  전체(2.0/0.5/0.1/0.0)를 현재 아티팩트로 다시 도는 것은 이 태스크의 범위 밖이다.
- 이 측정도 Task 7·8·9와 같은 범위 제한을 그대로 물려받는다 — 전문가 대
  전문가(코퍼스가 숙련공 6명, 쌍이 전형기=전형기)이고, 훈련생 대 전문가는
  측정된 적이 없다.
- `spread_reference_expert.json`(stale)이 왜 재빌드되지 않은 채 Task 7·8·9에
  쓰였는지는 이 태스크가 조사하지 않았다 — 사실만 적는다.

## 산출물

- `/output/improvement_study_task10_default.json` — 실제 기본값, 현재(yup)
  아티팩트, 200쌍/seed 20260828.
- `/output/improvement_study_task10_crosscheck_staleref.json` — 교차검증, Task
  7·8·9와 동일한 stale 아티팩트.
- `/output/improvement_study_index_cache_yup.json` — 이번 실행에서 재빌드(CSV
  트리 지문 불일치로 자동 재생성, 64,098 후보/479세션, 이전 yup 캐시와 동일한
  후보 수).

## 테스트

`docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer mcp_servers -q` →
**702 passed**(기준선과 같은 수 — 8개 실패를 문턱 변경에 맞춰 픽스처를
고쳐 원래 성질을 유지한 채 다시 통과시켰다. 자세한 목록은
`.superpowers/sdd/2026-09-03-skeleton-view/task-10-report.md` 참고).
