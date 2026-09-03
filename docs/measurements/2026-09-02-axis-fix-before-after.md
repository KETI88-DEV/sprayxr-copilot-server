# Task 6 report: 재빌드와 재측정 — 전후 비교

**Status: done.**

재측정 자체는 코드를 바꾸지 않았다(아티팩트 2개 재빌드, 리포트 1개).
**다만 이 문서를 저장소에 남기는 커밋에서 코디네이터가 인덱스 캐시 지문에 축 규약을
추가하고 테스트를 하나 넣었다** — 그 커밋 이후 스위트는 673이다.

## Sampling rule (fixed before any number below)

Same rule as the pre-registered study, unchanged:

- Query pool: full corpus (all workers, levels 1–10, training-mode only), indexed via
  `CsvPassSource`, each pass tagged with its uv8 key (`motion_key.KeyParams()` default grid).
- `random.Random(20260828)` shuffles the full candidate list; the first 200 (non-repeating)
  become queries.
- For each query, the reference is `rng.choice` among 전형기 (expert) candidates sharing the
  query's uv8 key, excluding the query pass itself. No candidate → `status="no_expert_reference"`,
  kept, not dropped.
- `--pairs 200 --seed 20260828`, fixed in code before results were seen.

This is identical to the rule already baked into `trainer/scripts/improvement_study.py` and used
for both runs below — nothing about the sampling changed.

### Which index cache each run used

- **Before (Z-up, multi-axis-corrected)** — `/output/improvement_study_multiaxis.json`, already on
  disk from an earlier task, built 2026-08-31 06:02 against reference
  `/output/spread_reference_expert.json`. Its selection reused
  `/output/improvement_study_index_cache.json` (built 2026-08-28 13:26) — confirmed by checking the
  CSV-tree fingerprint recorded in that cache (`file_count=605, total_size=15526908501,
  max_mtime=1785292787.68`) against the current tree: **identical**, so the cache-reuse guard in
  `index_corpus_cached` would have accepted it silently. That cache's per-candidate motion keys
  reflect whatever `motion_key`/axis code was in the repo through 2026-08-31, i.e. Z-up with the
  prior multi-axis correction applied — not this task's Y-up fix.
- **After (Y-up, this task)** — `/output/improvement_study_yup.json`, built fresh in this session
  against reference `/output/spread_reference_expert_yup.json`, using a **new** index cache path
  `/output/improvement_study_index_cache_yup.json`. This cache was built from scratch in this run
  (64,098 candidates across 479 sessions, same CSV-tree fingerprint as above — the data on disk
  hasn't changed, only the code that turns a pass into a motion key has), so its keys reflect the
  current, corrected axis convention.

The two runs therefore differ only in the axis convention baked into the motion keys and the
spread reference — same 200-pair sampling procedure, same seed, same corpus.

### Commands run (foreground, held to completion — no external notification relied on)

```
docker exec -w /workspace ship-paint-trainer python3 trainer/scripts/build_spread_reference.py \
  --root /data/sources/professionals --worker 전형기 \
  --out /output/spread_reference_expert_yup.json --log-level INFO
# -> 79 sessions, 9571 passes accumulated

docker exec -w /workspace ship-paint-trainer python3 trainer/scripts/improvement_study.py \
  --root /data/sources/professionals \
  --reference /output/spread_reference_expert_yup.json \
  --pairs 200 --seed 20260828 \
  --index-cache /output/improvement_study_index_cache_yup.json \
  --out /output/improvement_study_yup.json --progress --log-level INFO
# -> 64098 candidates indexed, 200/200 pairs run, plus F5 sensitivity re-run
```

No pre-registered constant was touched (`CORRECTION_SIGMA_THRESHOLD`, `MIN_CORRECTION_CM`,
`SIGMA_FLOOR_M`, `MAX_SIGMA`, `GATE_RATIO`, `DEFAULT_ERROR_M`, `N_MIN`, `PAIR_SIGMA_FACTOR`,
`DEFAULT_WEIGHTS` — all read from `measured_score.py`/`service.py`/`spread_reference.py`, none
redefined here).

## Before/after table

| metric | before (Z-up, multi-axis-corrected) | after (Y-up, this task) |
|---|---|---|
| `split` median / q3 / ≥0.70 | 0.0 / 0.158 / 3 of 149 (2.0%) | 0.0 / 0.102 / 2 of 178 (1.1%) |
| `same` median / q3 / ≥0.70 | 0.0 / 0.195 / 0 of 157 (0.0%) | 0.0 / 0.147 / 2 of 179 (1.1%) |
| correction fire rate — `same` basis (any instruction) | 43.3% (68/157) | 36.3% (65/179) |
| correction fire rate — `split` basis (any instruction) | 39.6% (59/149) | 38.2% (68/178) |
| ceiling among fired, median / share ≥0.70 available | 0.846 / 76.5% | 0.819 / 86.2% |
| capture ratio median (same / split) | 0.294 / 0.329 | 0.223 / 0.167 |
| REQ-19 gate | 149 fail / 0 pass (8 undetermined) | 178 fail / 0 pass (1 undetermined) |
| coverage n≥20, uv8 / uv4 / uv2 / uv_off | 25.0 / 55.7 / 81.4 / 95.0% | 49.8 / 77.9 / 96.4 / 99.0% |
| `status_counts` (of 200 requested) | ok 157 / no_expert_reference 43 | ok 179 / no_expert_reference 21 |

`correction fire rate (any instruction)` in the pre-registered row header is ambiguous between the
`same` and `split` bases, which the module tracks separately (`correction_fire_rate.same` /
`correction_fire_rate.split`); the pre-registered before value 43.3% (68/157) matches the `same`
basis exactly, so that is the headline row — the `split`-basis row is added beside it for
completeness, not as a substitute.

Coverage is pass-weighted: for each granularity, `Σ(pass count in keys with n≥20) / Σ(all pass
count)`, computed directly from `spread_reference_expert*.json`. This formula reproduces the
pre-registered before values (25.0/55.7/81.4/95.0) exactly, confirming it's the right definition.

## Per-granularity pair counts (bucket-key distribution)

| granularity | before n | after n |
|---|---|---|
| uv8 | 38 | 72 |
| uv4 | 45 | 49 |
| uv2 | 45 | 42 |
| uv_off | 21 | 15 |
| none (no key resolved) | 8 | 1 |

## `split_reason` breakdown (of `ok` pairs)

| reason | before | after |
|---|---|---|
| none (split computed) | 149 | 178 |
| no_split_reference | 8 | 1 |

## Head/right firing overlap (F-basis, `right_fired`/`head_fired` individually)

| bucket | before n | after n |
|---|---|---|
| n_known | 157 | 179 |
| head_only | 6 | 13 |
| right_only | 39 | 38 |
| both | 23 | 14 |
| neither | 89 | 114 |

`same` improvement within each bucket (median, before → after):

| bucket | before median same | after median same |
|---|---|---|
| head_only | 0.202 | 0.189 |
| right_only | 0.265 | 0.193 |
| both | 0.170 | 0.182 |

## 70% goal

**The pre-registered 70% split-improvement goal is not met, before or after axis correction.**
`split ≥ 0.70` is 3 of 149 (2.0%) before and 2 of 178 (1.1%) after. Neither number is close to
70%, and the axis fix does not change that conclusion.

## Interpretation notes (side by side, no favored direction)

- **Down after the fix**: `split` and `same` q3 (0.158→0.102, 0.195→0.147), fire rate on the
  `same` basis (43.3%→36.3%), capture ratio median on both bases (0.294→0.223, 0.329→0.167), and
  the count of `both`-fired pairs (23→14, replaced mostly by more `neither` — 89→114).
- **Up after the fix**: `same` ≥0.70 count (0→2), ceiling-among-fired share ≥0.70 (76.5%→86.2%),
  fire rate on the `split` basis (39.6%→38.2%, roughly flat), all four coverage figures (uv8 nearly
  doubles, 25.0%→49.8%), and the count of pairs that found any expert reference at all (157→179 of
  200 attempted).
- Per the task's constraint, none of this is evidence the axis fix was right or wrong — that
  question is settled independently (three confirmations of Y-up plus the user's confirmation the
  capture is Unity), not by whether these numbers moved favorably. Both columns are measured on the
  identical 200-pair, seed-20260828 sample; only the axis convention underlying the motion keys and
  spread reference differs between them.

## Test summary

`docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer mcp_servers -q` →
재측정 시점 **672 passed**(이 태스크는 코드를 안 건드렸다). 위 지문 수정이 들어간 뒤 **673**이다.

## Artifacts written

- `/output/spread_reference_expert_yup.json` — new spread reference (Y-up), 79 sessions / 9571
  passes for 전형기, levels 1–10.
- `/output/improvement_study_index_cache_yup.json` — new index cache (Y-up motion keys), 64,098
  candidates across 479 sessions.
- `/output/improvement_study_yup.json` — full study output (200-pair, seed 20260828), including
  per-pair records and the F5 `PAIR_SIGMA_FACTOR=1.0` sensitivity re-run.

Previous artifacts (`/output/spread_reference_expert.json`,
`/output/improvement_study_index_cache.json`, `/output/improvement_study_multiaxis.json`,
`/output/improvement_study.json`) were left in place, untouched.
