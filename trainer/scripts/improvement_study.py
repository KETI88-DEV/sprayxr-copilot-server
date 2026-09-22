#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false
"""70% 목표(goal_year2)에 답하는 표를 낸다.

이 모듈의 산출물은 기능이 아니라 숫자와 결론이다. "한 스트로크를 전문가와 비교해
자세 유사도를 재고 교정 피드백을 낸다 — 그리고 교정 적용 후 자세 유사도 개선율이
70%에 도달해야 한다"는 계약 문구를 실데이터로 판정한다.

## same과 split

`service.compare`는 개선율을 두 잣대로 낸다.

- **same**: 교정을 고른 분포로 다시 채점한다. 교정된 관절이 정의상 그 분포의 평균에
  가까워지므로 향상이 **자기충족으로 보장**된다. 이것은 결과가 아니라 상한이다.
- **split**: 날짜를 반으로 갈라 A(교정 선택)와 B(채점)를 분리하고, 질의 자신의 날짜는
  A/B 양쪽에서 뺀다. 자기충족이 없는 정직한 숫자다. **70% 판정은 split으로만 낸다.**

## 표집 규칙 (숫자를 보기 전에 고정한다)

1. `trainer.pose_lab.csv_source.CsvPassSource`로 `--root` 아래 전수(모든 작업자,
   레벨 1~10, 학습모드만)를 색인한다 — 패스마다 우8(`motion_key.KeyParams()` 기본
   격자) 키를 `CsvPassSource.key_of`로 뽑는다. 이 값이 `service.compare`가 내부에서
   같은 기본 params로 다시 계산하는 키와 같아야 쌍이 "동작 키 불일치"로 버려지지
   않는다(둘 다 `stroke_frame.DEFAULT_SAMPLES` 리샘플 위에서 나온다).
2. `random.Random(--seed)`로 전체 후보를 셔플하고 앞 `--pairs`개를 질의로 뽑는다 —
   **비복원**(같은 패스가 질의로 두 번 뽑히지 않는다).
3. 질의마다 기준 패스 하나를 고른다. **기본은 쌍 규칙 N(`--pair-rule screen`)이다** —
   화면의 `service.reference_candidates`를 그대로 불러 후보 집합(전형기·같은 레벨·자신
   제외·동작 키 사다리의 첫 단)을 받고, 질의별 난수열로 그 안에서 하나를 고른다
   (`select_pairs_screen`, 사전등록 `docs/measurements/2026-09-22-same-level-pairs-d1-prereg.md`).
   옛 규칙 O(`--pair-rule legacy`, `select_pairs`)는 비교용으로만 남긴다 — 같은 우8 키를
   가진 전형기 후보 중에서(질의 자신 제외, **레벨은 가리지 않는다**) 공유 `rng.choice`로
   고른다. 옛 규칙의 쌍 179개 중 105개가 다른 레벨 짝이었다(측정 결함 M0).
   후보가 없으면 그 쌍은 `status="no_expert_reference"`로 기록하고 버리지 않는다 —
   분모에서 조용히 빼면 "레퍼런스가 없어 재지 못한 몫"이 숨는다.
4. `--pairs`(기본 200)와 `--seed`(기본 20260828)는 CLI 인자로 코드에 먼저 고정한다.
   결과를 본 뒤 유리한 절단·부분집합·문턱을 고르지 않는다.

## None은 0이 아니다

계산되지 않은 `split`(예: `insufficient_dates`/`insufficient_samples`)은 분포에서
제외한다. 0으로 채우면 "교정이 악화시켰다"로 잘못 읽힌다. `summarize()`가 이 규칙을
지킨다 — `n`은 실제로 값이 있는 표본 수이지 시도한 쌍 수가 아니다.

## Fix round 1(리뷰 반영)에서 바뀐 것

- **F1**: `same`/`split`이 100%에 못 미치는 이유를 "가중치 몫(1.0/1.55=64.5%)"으로
  주장했던 것을 철회한다. 그 수는 σ(가중 평균 W)의 몫이지 gap_closure의 몫이 아니다
  — S=exp(-W²/2)이 W에 비선형이라 W를 일부만 줄여도 gap_closure는 그보다 훨씬 크게
  오를 수 있다(실제로 이 보고서 자신의 데이터가 0.6452보다 큰 same/split 값을
  냈다 — 주장이 스스로의 데이터에 반박됐다). 대신 쌍마다 **지시 가능한 축을 전부
  완벽히 따랐을 때 낼 수 있는 최댓값**을 실제로 계산해 `ceiling` 필드로 남긴다
  (`_ceiling_gap_closure`). 살아남는 주장은 약한 것 하나뿐이다 — 왼손과 머리
  lateral을 절대 옮기지 않으므로 G<1.0은 항상 성립한다.
- **Task 6**: 머리가 교정 대상이 되면서(자세 문장) `_ceiling_gap_closure`의 정의를
  고쳤다. 예전에는 "오른손 σ만 완전 소거, 머리·왼손은 그대로"였다 — 이제 머리도
  일부(`STANCE_AXES` = wall/vertical) 지시 가능하므로 그 두 축도 소거하고, 만질 수
  없는 몫은 **왼손 전체와 머리의 `lateral`**만 남는다(`CORRECTABLE`,
  `ceiling_residual_sigma`). 가중 σ에서 남는 비중을 gap_closure의 남는 비중으로
  그대로 읽지 않는다 — `exp(-W²/2)`가 W에 비선형이라 잔차 σ로 W를 다시 계산한 뒤
  다시 `exp(-W²/2)`를 낸다.
2 - **F2**: `fired`(F/전체 레퍼런스 기준 발동 여부)를 split(A로 고르고 B로 채점) 모집단을
  가르는 데 써서 두 갈래의 발동 결정이 뒤섞였다. `split_fired`를 A 자신의 발동 결정으로
  따로 낸다(`_split_details`가 `spreads=`로 A/B를 각각 compare()에 직접 먹여
  service.py를 건드리지 않고 같은 계산을 재현한다).
- **F3**: 쌍마다의 원자료(질의·레퍼런스·상태·same·split·ceiling·발동 여부 등)를
  JSON 아티팩트에 `pairs`로 남긴다. `summarize()`에 `min`/`max`를 추가한다.
- **F4**: 색인 캐시에 파일 수·총 크기·최신 mtime 지문을 같이 저장하고, 재사용 시
  지금 트리의 지문과 맞지 않으면 조용히 쓰지 않고 다시 짓는다.
- **F5**: `PAIR_SIGMA_FACTOR=1.0` 민감도 비교를 표로 낸다(파일의 상수는 바꾸지 않고,
  실행 중에만 `service` 모듈의 전역을 되돌렸다가 끝나면 원복한다 — 파일에 커밋된
  값은 그대로다). `eval_pose_correction.MAX_ELIGIBLE_BEFORE` 가드를 적용했다면
  split이 어떻게 바뀌었을지도 계산해 공개하되(before_split_score를 A/B와 같은 방식
  으로 재현), 헤드라인에는 그 가드를 적용하지 않은 채로 둔다 — 결과를 보고 나서
  유리한 필터를 추가하는 것은 이 계획이 금지하는 사후 선택이기 때문이다.

## 사전등록 상수 (건드리지 않는다)

`CORRECTION_SIGMA_THRESHOLD`(2.0), `SIGMA_FLOOR_M`(0.02), `MAX_SIGMA`(6.0),
`GATE_RATIO`(0.5), `DEFAULT_ERROR_M`(0.144), `N_MIN`(20), `PAIR_SIGMA_FACTOR`(√2)는
전부 `measured_score.py`/`service.py`/`spread_reference.py`가 단일 출처다. 이 모듈은
그것들을 읽기만 하고 재정의하지 않는다 — F5의 민감도 비교도 파일을 고치지 않고
실행 중 모듈 전역을 잠깐 되돌렸다가 원복하는 방식으로만 낸다. 반쪽 두께
계층화(§half_thickness)의 문턱도 새로 고르지 않고 이미 사전등록된 `N_MIN`(20)을
그대로 재사용한다.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

motion_key = importlib.import_module("motion_key")
world_axes = importlib.import_module("world_axes")
eval_pose_correction = importlib.import_module("eval_pose_correction")
measured_score = importlib.import_module("measured_score")

from trainer.pose_lab import service  # noqa: E402
from trainer.pose_lab import csv_source as csv_source_module  # noqa: E402
from trainer.pose_lab.service import PassRef  # noqa: E402
from trainer.pose_lab.spread_reference import N_MIN, SpreadReference  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_PAIRS: Final = 200
DEFAULT_SEED: Final = 20260828
THRESHOLD: Final = 0.70
# 반쪽 두께 계층화 문턱. 새로 고르지 않는다 — build_spread_reference의 N_MIN을 그대로 쓴다.
THIN_HALF_THRESHOLD: Final = N_MIN
# F5 민감도 비교의 대체 배율. service.py의 상수(√2)는 파일에서 바꾸지 않는다 — 실행
# 중에만 모듈 전역을 이 값으로 되돌렸다가 끝나면 원복한다.
SENSITIVITY_PAIR_SIGMA_FACTOR: Final = 1.0
# eval_pose_correction.py가 이미 쓰는 문턱. 새로 고르지 않고 그대로 가져와 "적용했다면
# 어떻게 됐을지"만 공개용으로 계산한다 — 헤드라인에는 적용하지 않는다(F5).
MAX_ELIGIBLE_BEFORE: Final = eval_pose_correction.MAX_ELIGIBLE_BEFORE


# --- Step 1의 집계 함수 --------------------------------------------------------


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """선형보간 백분위(넘파이 기본 'linear' 방식과 같다). `median`도 이 함수로 낸다."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = fraction * (len(sorted_values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[int(index)]
    frac = index - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * frac


def summarize(values: Sequence[float | None], threshold: float) -> dict[str, float | int | None]:
    """분포 요약: n·중앙값·사분위·최소최대·문턱 이상 비율. `None`은 표본에서 뺀다
    (0으로 넣지 않는다)."""
    present = sorted(value for value in values if value is not None)
    n = len(present)
    if n == 0:
        return {
            "n": 0, "median": None, "q1": None, "q3": None,
            "min": None, "max": None, "at_or_above": None,
        }
    return {
        "n": n,
        "median": _percentile(present, 0.5),
        "q1": _percentile(present, 0.25),
        "q3": _percentile(present, 0.75),
        "min": present[0],
        "max": present[-1],
        "at_or_above": sum(1 for value in present if value >= threshold) / n,
    }


# --- 표집 ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candidate:
    ref: PassRef
    worker: str
    key: str


def index_corpus(root: Path, rows_cache: int = 8) -> tuple[csv_source_module.CsvPassSource, list[Candidate]]:
    """전수 색인. 우8 키(`compare()`의 기본 params와 같은 격자)로 후보를 만든다.

    `CsvPassSource.index_of`는 세션당 한 번만 CSV를 파싱하고, 색인 항목은 행이 아니라
    구간·키 재료만 들고 있는다 — 그래서 479세션 전수를 훑어도 메모리에 원본 행이
    누적되지 않는다.
    """
    source = csv_source_module.CsvPassSource(root, rows_cache=rows_cache)
    params = motion_key.KeyParams()
    candidates: list[Candidate] = []
    for item in source.sessions():
        for entry in source.index_of(item):
            ref = PassRef(
                worker=item.worker, date=item.date, level=item.level,
                session=item.session, pass_index=entry.pass_index,
            )
            candidates.append(Candidate(ref=ref, worker=item.worker, key=source.key_of(entry, params)))
    return source, candidates


def _candidate_to_json(candidate: Candidate) -> dict[str, Any]:
    ref = candidate.ref
    return {
        "worker": ref.worker, "date": ref.date, "level": ref.level,
        "session": ref.session, "pass_index": ref.pass_index,
        "cand_worker": candidate.worker, "key": candidate.key,
    }


def _candidate_from_json(payload: Mapping[str, Any]) -> Candidate:
    ref = PassRef(
        worker=str(payload["worker"]), date=str(payload["date"]), level=str(payload["level"]),
        session=str(payload["session"]), pass_index=int(payload["pass_index"]),
    )
    return Candidate(ref=ref, worker=str(payload["cand_worker"]), key=str(payload["key"]))


def _fingerprint(root: Path) -> dict[str, Any]:
    """캐시가 몰래 낡는 것을 막는 지문 — CSV 트리 **와** 키를 만든 규약.

    트리 쪽은 파일 수·총 바이트·최신 mtime이다. 20GB 가까운 CSV를 해시하지 않고도
    파일이 추가·교체됐는데 캐시가 예전 후보를 계속 내주는 것을 막기에 충분하고,
    내용을 한 바이트도 읽지 않는다 — `stat()`만 부른다.

    **규약 쪽이 왜 필요한가.** 캐시가 담는 것은 패스 구간과 **동작 키**인데, 동작 키는
    `world_axes.UP_INDEX`(어느 축이 수직인가)와 `KeyParams`(격자·대역 폭)에 의존한다.
    트리만 지문으로 삼으면 **옛 축으로 지어진 캐시가 트리가 그대로인 한 조용히 통과한다** —
    실제로 축 수정 직후 그 일이 일어났고, 재측정을 새 경로로 돌려서 우회했을 뿐이다.
    그러면 낡은 버킷에서 표집하게 되고 비교 전체가 소리 없이 무효가 된다.

    독스트링이 한때 이 함수를 "루트가 고정이면 같은 결과를 내는 순수 함수"라고 적었는데,
    코드에 대해서는 순수하지 않다. 그 차이가 이 필드들이 존재하는 이유다.
    """
    total_size = 0
    max_mtime = 0.0
    count = 0
    for path in sorted(root.glob("**/*.csv")):
        stat = path.stat()
        total_size += stat.st_size
        max_mtime = max(max_mtime, stat.st_mtime)
        count += 1
    params = motion_key.KeyParams()
    return {
        "file_count": count,
        "total_size": total_size,
        "max_mtime": max_mtime,
        "up_index": world_axes.UP_INDEX,
        "key_params": [params.uv_grid, params.lateral_deg, params.reach_m, params.uv_size],
        # M0(사전등록 2026-09-22 §2): 키 정의가 바뀌면(접두어 m1→m3 등) 격자·축이 그대로여도
        # 캐시에 든 키 문자열이 옛 정의다. 접두어를 지문에 넣어 옛 키 캐시가 조용히 통과하지 않게 한다.
        "key_prefix": motion_key.KEY_PREFIX,
    }


def index_corpus_cached(
    root: Path, cache_path: Path | None, rows_cache: int = 8
) -> tuple[csv_source_module.CsvPassSource, list[Candidate]]:
    """`index_corpus`의 결과를 파일로 캐시한다. 표본 규칙을 바꾸지 않는다 — 전수 색인은

    입력(루트 CSV 트리)이 고정이면 항상 같은 후보 목록을 내는 순수 함수이므로, 그
    출력을 디스크에 굳혀 재사용해도 어떤 패스가 후보가 되는지·어떤 키를 갖는지는
    전혀 달라지지 않는다. 느린 것은 479개 세션 CSV(약 15.5GB)의 1회 파싱뿐이고,
    그 결과(패스 구간·모션 키)만 캐시한다 — `select_pairs`의 seed·셔플·선정 로직은
    캐시 유무와 무관하게 그대로 돈다.

    F4: 캐시는 루트 경로뿐 아니라 `_fingerprint(root)`(파일 수·총 크기·최신 mtime)도
    같이 저장한다. 재사용 시 지금 트리의 지문과 다르면(파일이 추가·삭제·교체됐으면)
    조용히 낡은 후보를 내주지 않고 다시 짓는다.
    """
    source = csv_source_module.CsvPassSource(root, rows_cache=rows_cache)
    fingerprint = _fingerprint(root)
    if cache_path is not None and cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("root") == str(root) and payload.get("fingerprint") == fingerprint:
            candidates = [_candidate_from_json(item) for item in payload["candidates"]]
            logger.info(
                "loaded %d cached candidates from %s (fingerprint matched)", len(candidates), cache_path
            )
            return source, candidates
        logger.warning(
            "index cache %s is stale (root or CSV-tree fingerprint changed); rebuilding", cache_path
        )
    _, candidates = index_corpus(root, rows_cache=rows_cache)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "root": str(root),
                    "fingerprint": fingerprint,
                    "candidates": [_candidate_to_json(c) for c in candidates],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logger.info("wrote %d candidates to index cache %s", len(candidates), cache_path)
    return source, candidates


@dataclass(frozen=True, slots=True)
class Pair:
    query: PassRef
    reference: PassRef | None
    key: str
    # 아래는 쌍 규칙 N(`screen`)만 채운다. 옛 규칙(`legacy`)은 None으로 둔다.
    # 기준 단 = `reference_candidates.match_granularity`. `run_pair`가 compare의
    # `key_match.granularity`와 같은지 대조한다(사전등록 §6 F4).
    reference_granularity: str | None = None
    candidate_count: int | None = None
    truncated: bool | None = None
    # 후보 집합(`reference_candidates(...)["candidates"]`의 PassRef, 화면 순서 그대로).
    candidates: tuple[PassRef, ...] | None = None
    # `reference_candidates`가 status!="ok"를 냈을 때 그 reason(예: insufficient_frames).
    candidate_reason: str | None = None


PAIR_RULES: Final = ("screen", "legacy", "legacy_same_level", "screen_first")
DEFAULT_PAIR_RULE: Final = "screen"
DEFAULT_SEED_REF: Final = DEFAULT_SEED
# 화면(`static/app.js` fetchReference → `/api/reference`)이 넘기는 값과 같다.
SCREEN_SAMPLES: Final = 20
SCREEN_LIMIT: Final = 50


def select_pairs(
    candidates: Sequence[Candidate], expert_worker: str, pairs: int, seed: int
) -> list[Pair]:
    """모듈 독스트링 "표집 규칙"을 그대로 구현한다. 결과를 본 뒤 바꾸지 않는다."""
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    queries = shuffled[: max(0, pairs)]

    by_key_expert: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.worker == expert_worker:
            by_key_expert[candidate.key].append(candidate)

    result: list[Pair] = []
    for query in queries:
        options = [c for c in by_key_expert.get(query.key, []) if c.ref != query.ref]
        reference = rng.choice(options).ref if options else None
        result.append(Pair(query=query.ref, reference=reference, key=query.key))
    return result


def sample_queries(candidates: Sequence[Candidate], pairs: int, seed: int) -> list[Candidate]:
    """질의 표본 — `select_pairs`(옛 규칙)의 셔플과 **같은** 표본이다.

    셔플은 후보 목록의 순서에만 의존하고 키 문자열에는 의존하지 않는다. 그래서 어느 키로
    색인해도, 어느 쌍 규칙을 써도 질의 200개가 같다(사전등록 §1.2, §6 F2).
    """
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    return shuffled[: max(0, pairs)]


def _query_rng(seed_ref: int, query: PassRef) -> random.Random:
    """질의별 난수열. 문자열 시드(파이썬 버전 2 시딩, sha512)라 실행이 달라도 같다.

    옛 규칙처럼 난수열 하나를 모든 질의가 나눠 쓰면, 한 질의의 후보 수가 바뀔 때 뒤따르는
    모든 질의의 선택이 밀린다. 질의별로 두면 후보 집합이 같은 질의는 키를 바꿔도 같은
    기준을 받는다(사전등록 §1.2).
    """
    return random.Random(
        f"{seed_ref}|{query.worker}|{query.date}|{query.level}|{query.session}|{query.pass_index}"
    )


def _row_ref(row: Mapping[str, Any]) -> PassRef:
    return PassRef(
        worker=str(row["worker"]), date=str(row["date"]), level=str(row["level"]),
        session=str(row["session"]), pass_index=int(row["pass_index"]),
    )


def select_pairs_screen(
    source: Any,
    candidates: Sequence[Candidate],
    expert_worker: str,
    pairs: int,
    seed: int,
    seed_ref: int = DEFAULT_SEED_REF,
    choose: str = "random",
    candidates_fn: Any | None = None,
) -> list[Pair]:
    """쌍 규칙 N — 화면과 **같은 함수**로 후보 집합을 만든다(사전등록 §1.2).

    질의마다 `service.reference_candidates(source, query, KeyParams(), samples=20, limit=50)`를
    직접 부르고, 그 `candidates`를 후보 집합으로 쓴다. 전형기 고정·같은 레벨·대상 자신 제외·
    동작 키 사다리(uv8→uv4→uv2→uv_off, 후보가 나온 첫 단)는 전부 그 함수가 정한다 — 연구
    쪽에 규칙을 다시 쓰지 않는다. 다시 쓰면 화면과 연구가 조용히 갈릴 수 있다.

    `choose="random"`(N, 주 판정): 후보를 (date, session, pass_index)로 정렬하고 질의별
    난수열에서 하나. `choose="first"`(N0, 보고용): 화면이 처음 보이는 `candidates[0]`.
    """
    if choose not in ("random", "first"):
        raise ValueError(f"choose must be 'random' or 'first', got {choose!r}")
    if expert_worker != service.EXPERT_WORKER:
        # 화면은 전형기로 고정돼 있다. 아티팩트의 작업자가 다르면 연구와 화면이 다른 사람을 기준으로 삼는다.
        raise ValueError(
            f"reference artifact worker {expert_worker!r} != screen expert {service.EXPERT_WORKER!r}"
        )
    candidates_fn = candidates_fn or service.reference_candidates
    params = motion_key.KeyParams()
    result: list[Pair] = []
    for query in sample_queries(candidates, pairs, seed):
        found = candidates_fn(source, query.ref, params, samples=SCREEN_SAMPLES, limit=SCREEN_LIMIT)
        if found.get("status") != "ok":
            result.append(
                Pair(
                    query=query.ref, reference=None, key=query.key,
                    candidate_reason=str(found.get("reason")),
                )
            )
            continue
        refs = tuple(_row_ref(row) for row in found["candidates"])
        reference: PassRef | None = None
        if refs:
            if choose == "first":
                reference = refs[0]
            else:
                ordered = sorted(refs, key=lambda ref: (ref.date, ref.session, ref.pass_index))
                reference = _query_rng(seed_ref, query.ref).choice(ordered)
        result.append(
            Pair(
                query=query.ref, reference=reference, key=query.key,
                reference_granularity=found.get("match_granularity"),
                candidate_count=len(refs), truncated=bool(found.get("truncated")),
                candidates=refs,
            )
        )
    return result


def select_pairs_legacy_same_level(
    candidates: Sequence[Candidate], expert_worker: str, pairs: int, seed: int
) -> list[Pair]:
    """중간 규칙 O+L(보고용) — 옛 규칙에 **같은 레벨 필터만** 더한다. 사다리는 없다.

    공유 난수열·uv8 엄밀 키는 `select_pairs`와 같다. O와 N의 차이 중 레벨 필터의 몫과
    사다리의 몫을 나누는 데만 쓴다(사전등록 §1.3).
    """
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    queries = shuffled[: max(0, pairs)]

    by_key_level_expert: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.worker == expert_worker:
            by_key_level_expert[(candidate.key, candidate.ref.level)].append(candidate)

    result: list[Pair] = []
    for query in queries:
        options = [
            c for c in by_key_level_expert.get((query.key, query.ref.level), []) if c.ref != query.ref
        ]
        reference = rng.choice(options).ref if options else None
        result.append(Pair(query=query.ref, reference=reference, key=query.key))
    return result


def build_pairs(
    rule: str,
    source: Any,
    candidates: Sequence[Candidate],
    expert_worker: str,
    pairs: int,
    seed: int,
    seed_ref: int = DEFAULT_SEED_REF,
) -> list[Pair]:
    if rule == "legacy":
        return select_pairs(candidates, expert_worker, pairs, seed)
    if rule == "legacy_same_level":
        return select_pairs_legacy_same_level(candidates, expert_worker, pairs, seed)
    if rule == "screen":
        return select_pairs_screen(source, candidates, expert_worker, pairs, seed, seed_ref)
    if rule == "screen_first":
        return select_pairs_screen(source, candidates, expert_worker, pairs, seed, seed_ref, choose="first")
    raise ValueError(f"unknown pair rule {rule!r}; expected one of {PAIR_RULES}")


def assert_key_prefix(candidates: Sequence[Candidate], pairs: Sequence[Pair]) -> None:
    """사전등록 §6 F6: 캐시·쌍의 키가 모두 현행 접두어인가. 옛 산출물이 섞이면 실행이 무효다."""
    prefix = f"{motion_key.KEY_PREFIX}:"
    stale = [c.key for c in candidates if not c.key.startswith(prefix)]
    stale += [p.key for p in pairs if not p.key.startswith(prefix)]
    if stale:
        raise RuntimeError(f"{len(stale)} keys lack prefix {prefix!r} (e.g. {stale[0]!r})")


# --- 실행 --------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class Outcome:
    query: PassRef
    reference: PassRef | None
    key: str
    status: str  # "ok" | "no_expert_reference" | "unavailable"
    reason: str | None
    same: float | None
    split: float | None
    split_reason: str | None
    granularity: str | None
    reference_n: int | None
    split_n_a: int | None
    split_n_b: int | None
    gate: str | None
    gate_reason: str | None
    normalization: str | None
    # F/전체 레퍼런스 기준 발동 여부(same이 쓰는 잣대) — **어느 쪽이든 지시가 하나라도
    # 나갔는가**(오른손 축 또는 머리 자세). Fix round 1(F1) 전에는 `measured.finding`
    # (오른손뿐)만 봤다 — 그러면 머리만 발동한 6쌍이 "미발동"으로 잘못 분류돼
    # not_fired_summary에 0이 아닌 improvement가 섞였다(`max=0.2972`가 그 증거였다).
    # 이제 `measured.finding` OR `measured.stance`다.
    fired: bool | None
    # F2/F1: A(교정 선택 반쪽) 자신의 발동 여부 — 역시 오른손 OR 머리 어느 쪽이든.
    # split을 가르는 데는 이 값을 써야 한다 — `fired`(F 기준)로 나누면 두 갈래의
    # 결정이 뒤섞인다.
    split_fired: bool | None
    # Task 6: 머리 자세만의 발동 여부(F 기준). `measured.stance`가 출처다.
    head_fired: bool | None = None
    # F1: 오른손 문장만의 발동 여부(F 기준). `measured.finding`이 출처다 — `fired`가
    # 오른손·머리를 합친 값이 된 뒤에도 `fired_overlap`의 head-only/right-only/both
    # 분해에는 오른손 단독 신호가 따로 필요하다.
    right_fired: bool | None = None
    # F1: 오른손 σ를 완전히(관대하게) 0으로, 머리·왼손은 그대로 뒀을 때 낼 수 있는
    # gap_closure 최댓값. `same`이 실제로 낸 값(자기충족 상한)과 별개로, "교정이
    # 완벽했다면 얼마나 갈 수 있었는가"를 보여준다.
    ceiling: float | None
    # Fix round 2, F2: `ceiling`과 같은 계산이지만 B(채점 레퍼런스) 기준으로 낸
    # gap_closure 최댓값. `split`은 B 기준으로 채점되므로(before_split/after_split이
    # 둘 다 B 스프레드로 잰다) `split_over_ceiling`은 이 값으로 나눠야 한다 —
    # F(전체 레퍼런스) 기준 `ceiling`으로 나누면 잣대가 섞인다.
    ceiling_split: float | None = None
    # F5(가드 공개용): F 레퍼런스 기준 before 점수(=same의 S0)와, B 레퍼런스 기준
    # before 점수(=split의 S0). eval_pose_correction.MAX_ELIGIBLE_BEFORE 가드를
    # 적용했다면 무엇이 걸러졌을지 재현하는 데만 쓴다.
    before_same_score: float | None
    before_split_score: float | None
    # 쌍 규칙 N 기록(사전등록 §1.2). 옛 규칙에서는 None이다.
    reference_granularity: str | None = None
    candidate_count: int | None = None
    truncated: bool | None = None
    # compare가 두 키를 만나게 한 단(`key_match.granularity`). F4 대조에 쓴다.
    key_match_granularity: str | None = None


# 지시할 수 있는 축. 이것 밖은 완벽한 교정으로도 남는다.
# head는 measured_score.STANCE_AXES와 같아야 한다 — 갈라지면 천장이 틀린다.
CORRECTABLE: Final[dict[str, tuple[str, ...]]] = {
    "right": measured_score.AXIS_ORDER,
    "head": measured_score.STANCE_AXES,
    "left": (),  # 지시 대상이 아니다
}


def ceiling_residual_sigma(
    axis_sigmas: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """지시 가능한 축을 전부 완벽히 따랐을 때 각 점에 남는 sigma.

    `CORRECTABLE`에 없는 축은 그대로 남는다. 남는 것을 0으로 치면 천장이 실제보다
    높게 나오고 미달 폭이 작아 보인다.

    Fix round 2, F1: `measured_score.score_track`이 점의 결합 σ를
    `min(MAX_SIGMA, ·)`로 클립한다(`MAX_SIGMA=6.0`, 사전등록 상수, 여기서 정의하지
    않고 읽기만 한다). 이 잔차도 같은 클립을 적용해야 한다 — 안 하면 채점기가 절대
    낼 수 없는 스케일(예: 7.292) 위에 천장이 앉고, "실제로 낸 것/천장"이 산술적으로
    1.0을 넘을 수 있다. 실제로 이 표본에서 10쌍이 이 결함에 걸렸다(Fix round 1
    보고서의 "천장을 넘는 성과"는 이 클립 누락이 원인 중 하나였다).
    """
    residual: dict[str, float] = {}
    for point, axes in axis_sigmas.items():
        correctable = CORRECTABLE.get(point, ())
        total = sum(
            float(value) ** 2 for axis, value in axes.items() if axis not in correctable
        )
        residual[point] = min(measured_score.MAX_SIGMA, math.sqrt(total))
    return residual


def _ceiling_gap_closure(measured: Mapping[str, Any]) -> float | None:
    """지시 가능한 축(`CORRECTABLE`)을 전부 완벽히 따랐을 때의 gap_closure 최댓값.

    오른손은 세 축 다(`AXIS_ORDER`), 머리는 자세로 번역 가능한 두 축(`STANCE_AXES`
    — wall/vertical)만, 왼손은 전혀 지시 대상이 아니다. 남는 것은 왼손 전체와 머리의
    `lateral`이다 — 왼손은 인과가 약해 지시하지 않기로 했고(§7), 머리 lateral은
    작업자 요가 실측이 아니라 좌우를 판정할 기준이 없다(`service.ASSUMED_YAW`).
    `ceiling_residual_sigma`가 점마다 남는 σ를 실제 σ 제곱합에서 계산하고, 여기서는
    그 잔차로 `measured_score.score_track`과 같은 가중 평균·`exp(-W²/2)` 공식을 그대로
    쓴다. 가중 σ에서 몇 %가 남는지를 gap_closure의 몫으로 읽으면 틀린다 —
    gap_closure=1-exp(-W²/2)는 W에 비선형이라, 남는 σ의 비중을 직접 gap_closure의
    비중으로 전용할 수 없다. 그래서 잔차 σ로 W를 다시 계산하고 그 W로 다시
    exp(-W²/2)를 낸다 — 지름길로 몫만 곱하지 않는다. 응답에 이미 실린 `deltas`/
    `score`만 읽는다 — 새 σ를 만들지 않고 `service.py`/`measured_score.py`를 건드리지
    않는다.
    """
    deltas = measured.get("deltas")
    before_score = measured.get("score")
    if not deltas or before_score is None:
        return None
    weights = service.measured_score.DEFAULT_WEIGHTS
    axis_sigmas_by_point = {
        delta["point"]: delta.get("axis_sigmas") or {} for delta in deltas
    }
    residual = ceiling_residual_sigma(axis_sigmas_by_point)
    total_weight = sum(weights.get(point, 0.0) for point in residual)
    if total_weight <= 0.0:
        return None
    weighted_ceiling = sum(
        weights.get(point, 0.0) * sigma for point, sigma in residual.items()
    ) / total_weight
    ceiling_score = math.exp(-0.5 * weighted_ceiling * weighted_ceiling)
    return service.pose_similarity.gap_closure(before_score, ceiling_score)


def _split_details(
    source: csv_source_module.CsvPassSource,
    reference_spreads: SpreadReference,
    query: PassRef,
    reference: PassRef,
    measured: Mapping[str, Any],
) -> dict[str, Any]:
    """F2/F5: A(교정 선택)와 B(채점) 반쪽 각각의 배선을 `compare()`의 `spreads=` 우회
    경로로 재현한다 — `service.py`를 고치지 않고, 그 안에 이미 있는 로직만 두 번
    바깥에서 불러 쓴다.

    `compare()`가 내부에서 `before_a = score_track(query_track, reference_track,
    _scale_spreads(raw_a), weights, basis)`를 계산하지만 그 값도 발동 여부도 응답에
    싣지 않는다. 대신 `compare(source, query, reference, spreads=_scale_spreads(raw_a))`를
    한 번 더 부르면, 그 호출의 최상위 `before`가 정확히 같은 계산(같은 query_track·
    reference_track·weights·basis·spreads)이 되므로 `measured.finding`/`measured.stance`가
    곧 "A 기준으로 오른손/머리가 발동했는가"이고 `measured.score`가 곧 그 발동 여부를
    결정한 σ의 점수다. B도 같은 방식으로 `before_split_score`를 재현한다.

    F1: `a_fired`는 **오른손 OR 머리** — 어느 쪽이든 A 기준으로 지시가 하나라도 나갔는가
    다. 오른손(`finding`)만 보면 머리만 발동한 쌍이 "A도 미발동"으로 잘못 읽힌다 —
    `run_pair`의 F-기준 `fired`와 같은 결함이었다.

    Fix round 2, F2: B 호출의 `measured`에도 `deltas`/`score`가 그대로 실려 있으므로
    `_ceiling_gap_closure(b_measured)`를 그대로 재사용해 **B 기준 천장**
    (`ceiling_split`)을 낸다. `split`은 before/after를 둘 다 B 스프레드로 재므로
    (`service.improvement`의 독스트링), 그 천장도 F가 아니라 B 위에서 재야 잣대가
    맞는다 — F 기준 `ceiling`으로 `split`을 나누면(Fix round 1이 그렇게 했다)
    분자·분모가 서로 다른 레퍼런스에서 나와 비율이 1.0을 넘거나 왜곡될 수 있다.

    split_reason이 있으면(계산된 split이 없으면) A/B 반쪽 자체가 없으므로 빈 결과를
    낸다.
    """
    empty: dict[str, Any] = {"a_fired": None, "before_split_score": None, "ceiling_split": None}
    if measured.get("split_reason") is not None:
        return empty
    granularity = measured.get("granularity")
    split_dates_a = measured.get("split_dates_a") or []
    split_dates_b = measured.get("split_dates_b") or []
    if not granularity or not split_dates_a or not split_dates_b:
        return empty

    query_frames = service.pass_frames(source.load(query))
    query_inputs = service.key_inputs(query_frames)
    if query_inputs is None:
        return empty
    ladder_keys = service._reference_keys(query_inputs)  # noqa: SLF001 — 같은 패키지 안, T4/T5 선례와 같은 근거.
    resolved_key = ladder_keys.get(granularity)
    if resolved_key is None:
        return empty

    raw_a = reference_spreads.spreads_for(granularity, resolved_key, split_dates_a)
    raw_b = reference_spreads.spreads_for(granularity, resolved_key, split_dates_b)

    a_fired: bool | None = None
    before_split_score: float | None = None
    ceiling_split: float | None = None
    if raw_a is not None:
        body_a = service.compare(source, query, reference, spreads=service._scale_spreads(raw_a), include_view=False)  # noqa: SLF001
        if body_a.get("status") == "ok":
            a_measured = body_a["measured"]
            a_fired = bool(a_measured.get("finding")) or bool(a_measured.get("stance"))
    if raw_b is not None:
        body_b = service.compare(source, query, reference, spreads=service._scale_spreads(raw_b), include_view=False)  # noqa: SLF001
        if body_b.get("status") == "ok":
            b_measured = body_b["measured"]
            before_split_score = b_measured["score"]
            ceiling_split = _ceiling_gap_closure(b_measured)
    return {
        "a_fired": a_fired,
        "before_split_score": before_split_score,
        "ceiling_split": ceiling_split,
    }


def run_pair(
    source: csv_source_module.CsvPassSource,
    reference_spreads: SpreadReference,
    pair: Pair,
    compute_extras: bool = True,
) -> Outcome:
    rule_fields: dict[str, Any] = {
        "reference_granularity": pair.reference_granularity,
        "candidate_count": pair.candidate_count,
        "truncated": pair.truncated,
    }
    if pair.reference is None:
        if pair.candidate_reason is not None:
            reason = pair.candidate_reason
        elif pair.candidates is not None:
            reason = "no_screen_candidate"
        else:
            reason = "no_same_key_expert_pass"
        return Outcome(
            query=pair.query, reference=None, key=pair.key, status="no_expert_reference",
            reason=reason, same=None, split=None, split_reason=None,
            granularity=None, reference_n=None, split_n_a=None, split_n_b=None,
            gate=None, gate_reason=None, normalization=None, fired=None, split_fired=None,
            ceiling=None, before_same_score=None, before_split_score=None, **rule_fields,
        )
    # 채점 필드만 받는다 — 표시 블록(스켈레톤·메시·각도표)의 비용·예외가 개선율 측정을
    # 흔들지 않게 한다(`service.compare` 독스트링).
    body = service.compare(
        source, pair.query, pair.reference, reference_spreads=reference_spreads, include_view=False
    )
    key_match = body.get("key_match")
    key_match_granularity = key_match.get("granularity") if isinstance(key_match, Mapping) else None
    rule_fields["key_match_granularity"] = key_match_granularity
    reason_unavailable: str | None = None
    if body.get("status") != "ok":
        reason_unavailable = str(body.get("reason"))
    elif pair.reference_granularity is not None and key_match_granularity != pair.reference_granularity:
        # 사전등록 §6 F4: 후보를 고른 단과 compare가 허락한 단이 갈렸다. 판정에서 빼지 않고
        # unavailable로 둔다 — 화면에서도 같은 모순이 난다는 뜻이라 결함으로 따로 보고한다.
        reason_unavailable = "ladder_mismatch"
    if reason_unavailable is not None:
        return Outcome(
            query=pair.query, reference=pair.reference, key=pair.key, status="unavailable",
            reason=reason_unavailable, same=None, split=None, split_reason=None,
            granularity=None, reference_n=None, split_n_a=None, split_n_b=None,
            gate=None, gate_reason=None, normalization=None, fired=None, split_fired=None,
            ceiling=None, before_same_score=None, before_split_score=None, **rule_fields,
        )
    measured = body["measured"]
    improvement = measured["improvement"]
    # F1(Fix round 1): 오른손 문장(`finding`)과 머리 자세(`stance`)는 각자 따로 두고,
    # "발동했다"(`fired`)는 **둘 중 하나라도** 났을 때다. 오른손만 보면(예전처럼)
    # 머리만 발동한 쌍이 "미발동"으로 잘못 분류된다 — not_fired 표본에 improvement가
    # 0이 아닌 쌍이 섞이는 형태로 드러난다(그 쌍은 실제로는 지시를 받았기 때문이다).
    right_fired = bool(measured.get("finding"))
    head_fired = bool(measured.get("stance"))
    fired = right_fired or head_fired
    ceiling = _ceiling_gap_closure(measured)

    split_fired: bool | None = None
    before_split_score: float | None = None
    ceiling_split: float | None = None
    if compute_extras:
        details = _split_details(source, reference_spreads, pair.query, pair.reference, measured)
        split_fired = details["a_fired"]
        before_split_score = details["before_split_score"]
        ceiling_split = details["ceiling_split"]

    return Outcome(
        query=pair.query, reference=pair.reference, key=pair.key, status="ok", reason=None,
        same=improvement.get("same"), split=improvement.get("split"),
        split_reason=measured.get("split_reason"), granularity=measured.get("granularity"),
        reference_n=measured.get("reference_n"), split_n_a=measured.get("split_n_a"),
        split_n_b=measured.get("split_n_b"), gate=measured.get("gate"),
        gate_reason=measured.get("gate_reason"), normalization=measured.get("normalization"),
        fired=fired, split_fired=split_fired, head_fired=head_fired, right_fired=right_fired,
        ceiling=ceiling, ceiling_split=ceiling_split,
        before_same_score=measured.get("score"), before_split_score=before_split_score,
        **rule_fields,
    )


def run_study(
    root: Path,
    reference_path: Path,
    pairs: int,
    seed: int,
    progress: bool = False,
    index_cache: Path | None = None,
    compute_extras: bool = True,
    pair_rule: str = DEFAULT_PAIR_RULE,
    seed_ref: int = DEFAULT_SEED_REF,
) -> tuple[list[Outcome], str, csv_source_module.CsvPassSource, SpreadReference, list[Pair]]:
    reference_spreads = SpreadReference.load(reference_path)
    expert_worker = reference_spreads.worker
    source, candidates = index_corpus_cached(root, index_cache)
    logger.info("indexed %d candidate passes across %d sessions", len(candidates), len(source.sessions()))
    selected = build_pairs(pair_rule, source, candidates, expert_worker, pairs, seed, seed_ref)
    assert_key_prefix(candidates, selected)
    outcomes: list[Outcome] = []
    for index, pair in enumerate(selected):
        outcomes.append(run_pair(source, reference_spreads, pair, compute_extras=compute_extras))
        if progress and (index + 1) % 20 == 0:
            print(f"  {index + 1}/{len(selected)}", file=sys.stderr, flush=True)
    return outcomes, expert_worker, source, reference_spreads, selected


def run_sensitivity(
    source: csv_source_module.CsvPassSource,
    reference_spreads: SpreadReference,
    pairs: Sequence[Pair],
    factor: float,
    progress: bool = False,
) -> list[Outcome]:
    """F5: `PAIR_SIGMA_FACTOR`를 이 값으로 바꿨다면 어땠을지.

    `service.py` 파일의 상수는 그대로 둔다 — 여기서는 이미 로드된 `service` 모듈
    객체의 전역 이름만 실행 중에 잠깐 바꾼다. `_scale_spreads`가 그 이름을 호출
    시점에 전역에서 찾아 쓰므로(닫힘이 아니라) 이 패치가 실제로 적용되고,
    `finally`에서 원래 값으로 되돌리므로 이 함수가 끝나면 흔적이 남지 않는다. 같은
    200쌍(같은 `Pair` 목록)을 그대로 재사용한다 — 표집을 다시 하지 않는다.
    """
    original = service.PAIR_SIGMA_FACTOR
    outcomes: list[Outcome] = []
    try:
        service.PAIR_SIGMA_FACTOR = factor
        for index, pair in enumerate(pairs):
            outcomes.append(run_pair(source, reference_spreads, pair, compute_extras=False))
            if progress and (index + 1) % 20 == 0:
                print(f"  [sensitivity] {index + 1}/{len(pairs)}", file=sys.stderr, flush=True)
    finally:
        service.PAIR_SIGMA_FACTOR = original
    return outcomes


# --- 집계·보고 ----------------------------------------------------------------


def _label(value: object) -> str:
    return "none" if value is None else str(value)


def _pass_ref_json(ref: PassRef | None) -> dict[str, Any] | None:
    if ref is None:
        return None
    return {
        "worker": ref.worker, "date": ref.date, "level": ref.level,
        "session": ref.session, "pass_index": ref.pass_index,
    }


def _outcome_to_json(outcome: Outcome) -> dict[str, Any]:
    return {
        "query": _pass_ref_json(outcome.query),
        "reference": _pass_ref_json(outcome.reference),
        "key": outcome.key,
        "status": outcome.status,
        "reason": outcome.reason,
        "same": outcome.same,
        "split": outcome.split,
        "split_reason": outcome.split_reason,
        "granularity": outcome.granularity,
        "reference_n": outcome.reference_n,
        "split_n_a": outcome.split_n_a,
        "split_n_b": outcome.split_n_b,
        "gate": outcome.gate,
        "gate_reason": outcome.gate_reason,
        "normalization": outcome.normalization,
        "fired": outcome.fired,
        "split_fired": outcome.split_fired,
        "head_fired": outcome.head_fired,
        "right_fired": outcome.right_fired,
        "ceiling": outcome.ceiling,
        "ceiling_split": outcome.ceiling_split,
        "before_same_score": outcome.before_same_score,
        "before_split_score": outcome.before_split_score,
        "reference_granularity": outcome.reference_granularity,
        "candidate_count": outcome.candidate_count,
        "truncated": outcome.truncated,
        "key_match_granularity": outcome.key_match_granularity,
    }


def _fire_rate_block(
    fired: Sequence[Outcome], not_fired: Sequence[Outcome],
    unknown: Sequence[Outcome], value_of: Any,
) -> dict[str, Any]:
    """발동률의 분모는 **발동 여부를 실제로 안 건수**다.

    전체 건수로 나누면 아무것도 재지 않았을 때(민감도 실행은 compute_extras=False라
    split_fired가 전부 None이다) 0/149 = 0.0이 나온다. "재보니 하나도 안 발동했다"와
    "재지 않았다"는 다른 말인데 그 0.0은 앞엣것으로 읽힌다 — 옆에 unknown=149가 적혀
    있어도 그렇다. 아는 것이 없으면 None이다.
    """
    measured = len(fired) + len(not_fired)
    return {
        "fired": len(fired),
        "not_fired": len(not_fired),
        "unknown": len(unknown),
        "share_fired": (len(fired) / measured) if measured else None,
        "fired_summary": summarize([value_of(o) for o in fired], THRESHOLD),
        "not_fired_summary": summarize([value_of(o) for o in not_fired], THRESHOLD),
    }


_SAMPLING_TEXT: Final[dict[str, str]] = {
    "legacy": (
        "질의: 전수 후보(모든 작업자, 우8 키 부착)를 seed로 셔플해 앞 N개를 "
        "비복원으로 뽑는다. 레퍼런스: 질의와 같은 우8 키를 가진 전형기(레퍼런스 "
        "아티팩트의 worker) 후보 중 질의 자신을 뺀 나머지에서 rng.choice로 하나. "
        "후보가 없으면 status=no_expert_reference로 기록하고 버리지 않는다. "
        "--pairs와 --seed는 결과를 보기 전에 코드에 고정했다."
    ),
    "legacy_same_level": (
        "O+L(보고용): legacy와 같되 전형기 후보를 질의와 같은 레벨로만 거른다. 사다리 없음."
    ),
    "screen": (
        "N(사전등록 2026-09-22 §1.2): 질의는 legacy와 같은 셔플 표본. 후보 집합은 "
        "service.reference_candidates(source, query, KeyParams(), samples=20, limit=50)의 "
        "candidates(전형기·같은 레벨·자신 제외·uv8→uv4→uv2→uv_off 첫 단) 그대로. "
        "(date, session, pass_index)로 정렬한 뒤 질의별 문자열 시드 난수열로 하나를 고른다."
    ),
    "screen_first": (
        "N0(보고용): N과 같은 후보 집합에서 화면이 처음 보이는 candidates[0]을 고른다."
    ),
}


def build_report(
    outcomes: Sequence[Outcome], pairs: int, seed: int, expert_worker: str, root: Path, reference_path: Path,
    include_pairs: bool = True,
    pair_rule: str = DEFAULT_PAIR_RULE,
    seed_ref: int = DEFAULT_SEED_REF,
) -> dict[str, Any]:
    ok = [item for item in outcomes if item.status == "ok"]
    status_counts = Counter(item.status for item in outcomes)

    granularity_counts = Counter(item.granularity for item in ok)
    per_layer = {
        _label(layer): {
            "n": count,
            "same": summarize([item.same for item in ok if item.granularity == layer], THRESHOLD),
            "split": summarize([item.split for item in ok if item.granularity == layer], THRESHOLD),
        }
        for layer, count in granularity_counts.items()
    }

    split_reason_counts = Counter(item.split_reason for item in ok)

    # F1(Fix round 1): "발동"은 오른손 축 OR 머리 자세 — 어느 쪽이든 지시가 하나라도
    # 나갔는가다. same은 F(전체 레퍼런스) 기준 발동 여부(`fired`)로, split은
    # A(교정 선택 반쪽) 자신의 발동 여부(`split_fired`)로 각자 갈라야 한다 — 두 갈래의
    # 결정이 다를 수 있다(실제로 이 표본에서도 다르다). 오른손만 보던 예전 버전은
    # 머리만 발동한 쌍을 "미발동"에 잘못 넣었다 — `not_fired_summary`에 0이 아닌
    # improvement가 섞여 있었던 것이 그 증거다(아래 검증 참고).
    same_fired = [item for item in ok if item.fired]
    same_not_fired = [item for item in ok if item.fired is False]
    same_unknown = [item for item in ok if item.fired is None]

    split_population = [item for item in ok if item.split is not None]
    split_fired = [item for item in split_population if item.split_fired]
    split_not_fired = [item for item in split_population if item.split_fired is False]
    split_unknown = [item for item in split_population if item.split_fired is None]

    correction_fire_rate = {
        "same": _fire_rate_block(same_fired, same_not_fired, same_unknown, lambda o: o.same),
        "split": _fire_rate_block(
            split_fired, split_not_fired, split_unknown, lambda o: o.split
        ),
    }

    # F1: not_fired 버킷의 모든 값이 정확히 0.0이어야 한다 — 아무 지시도 안 나갔으면
    # `apply_measured`가 아무것도 옮기지 않으므로 before==after이고 gap_closure는
    # 0이다. 전수(표본이 아니라)를 확인하고 결과를 report에 남긴다.
    # Fix round 2, F5: 버킷이 비어 있으면(`compute_extras=False`인 민감도 실행처럼
    # `split_fired`가 전부 None이라 `split_not_fired`가 아예 빈 경우) `all(...)`이
    # 공허하게 `True`를 낸다 — "검증했더니 전부 0이었다"로 잘못 읽힌다. `share_fired`가
    # 이미 쓰는 규칙과 같게, 잰 것이 없으면 `None`을 낸다.
    correction_fire_rate["same"]["not_fired_all_zero"] = (
        all(item.same == 0.0 for item in same_not_fired) if same_not_fired else None
    )
    correction_fire_rate["split"]["not_fired_all_zero"] = (
        all(item.split == 0.0 for item in split_not_fired) if split_not_fired else None
    )

    # Task 6/§10: 오른손 문장(`right_fired`)과 머리 자세(`head_fired`)가 같은 자세
    # 문제의 두 증상이면 겹칠 것이고(둘 다 발동이 큰 몫), 서로 독립이면
    # head-only/right-only가 크다. 둘 다 F(전체 레퍼런스) 기준이라 같은 잣대로 조합할
    # 수 있다. `right_fired`(오른손 단독 신호)를 쓴다 — 합친 `fired`를 쓰면 이 분해
    # 자체가 성립하지 않는다. 어느 쪽이든 발동 여부를 모르면(None) 이 표에서 뺀다 —
    # 0으로 채우지 않는다.
    both_known = [item for item in ok if item.right_fired is not None and item.head_fired is not None]
    head_only = [item for item in both_known if item.head_fired and not item.right_fired]
    right_only = [item for item in both_known if item.right_fired and not item.head_fired]
    both_fired = [item for item in both_known if item.right_fired and item.head_fired]
    neither_fired = [item for item in both_known if not item.right_fired and not item.head_fired]
    fired_overlap = {
        "basis": "F(전체 레퍼런스) 기준, 오른손/머리 각자의 단독 신호(`right_fired`/`head_fired`) — 합친 `fired`(correction_fire_rate가 쓰는 것)와는 다른 잣대다",
        "n_known": len(both_known),
        "head_only": len(head_only),
        "right_only": len(right_only),
        "both": len(both_fired),
        "neither": len(neither_fired),
        "same_by_bucket": {
            "head_only": summarize([item.same for item in head_only], THRESHOLD),
            "right_only": summarize([item.same for item in right_only], THRESHOLD),
            "both": summarize([item.same for item in both_fired], THRESHOLD),
        },
    }

    thickness_known = [
        item for item in ok if item.split_n_a is not None and item.split_n_b is not None
    ]
    thin = [item for item in thickness_known if min(item.split_n_a, item.split_n_b) < THIN_HALF_THRESHOLD]
    thick = [item for item in thickness_known if min(item.split_n_a, item.split_n_b) >= THIN_HALF_THRESHOLD]

    gate_counts = Counter(item.gate for item in ok)
    gate_by_layer: dict[str, Counter] = defaultdict(Counter)
    for item in ok:
        gate_by_layer[_label(item.granularity)][_label(item.gate)] += 1
    gate_determined = gate_counts.get("pass", 0) + gate_counts.get("fail", 0)

    # F1: 상한(오른손 σ 완전 교정, 머리·왼손 그대로). same이 실제로 발동한(fired) 표본
    # 위에서 보고한다 — "발동했다면 이론상 어디까지 갈 수 있었는가"가 질문이기 때문이다.
    ceiling = {
        "all_ok": summarize([item.ceiling for item in ok], THRESHOLD),
        "same_fired_only": summarize([item.ceiling for item in same_fired], THRESHOLD),
    }

    # 발동한 쌍마다 "낼 수 있었던 것(ceiling) 중 실제로 낸 것(same/split)의 몫"을
    # 잰다. `same_fired`(F 기준, 오른손 OR 머리 — F1로 넓어진 발동 모집단) 위에서
    # 잰다. Fix round 2, F2: `same`은 F(전체 레퍼런스) 기준으로 채점되므로 F 기준
    # `ceiling`으로 나누지만, `split`은 B(채점 레퍼런스) 기준으로 채점되므로
    # (`service.improvement`가 before_split/after_split을 둘 다 B 스프레드로 잰다)
    # 같은 B 기준으로 낸 `ceiling_split`으로 나눈다 — 분자·분모가 다른 레퍼런스에서
    # 나오면(Fix round 1이 그렇게 했다) 비율이 산술적으로 왜곡된다. 분모가 없거나
    # 0 이하면(메울 격차가 없거나 그 basis에서 값을 못 낸 쌍) 그 항목만 뺀다 —
    # 0으로 채우지 않는다.
    def _capture_values(attr: str, ceiling_attr: str, population: Sequence[Outcome]) -> list[float]:
        values: list[float] = []
        for item in population:
            value = getattr(item, attr)
            ceiling_value = getattr(item, ceiling_attr)
            if value is None or ceiling_value is None or ceiling_value <= 0.0:
                continue
            values.append(value / ceiling_value)
        return values

    def _capture_block(attr: str, ceiling_attr: str, population: Sequence[Outcome]) -> dict[str, Any]:
        values = _capture_values(attr, ceiling_attr, population)
        summary: dict[str, Any] = summarize(values, THRESHOLD)
        # Fix round 2: 클립·basis 결함이 고쳐졌다면 이 값은 어느 버킷에서도 0이어야
        # 한다 — "천장을 넘는 성과"가 실제로는 계산 결함이었다는 것을 재확인한다.
        summary["n_above_one"] = sum(1 for value in values if value > 1.0)
        return summary

    capture = {
        "note": (
            "발동한(오른손 OR 머리, F 기준) 쌍마다 same/ceiling(F 기준)과 "
            "split/ceiling_split(B 기준) — 그 쌍이 낼 수 있었던 gap_closure 중 실제로 "
            "낸 몫. 분모가 없거나 0 이하인 쌍은 그 잣대에서만 뺀다(나눗셈 미정의). "
            "≥0.70 열은 '이 쌍이 자신의 천장의 70% 이상을 실제로 회수했다'는 뜻이지 "
            "종합 개선율의 70% 목표와는 다른 잣대다 — 참고용으로만 낸다."
        ),
        "n_fired": len(same_fired),
        "same_over_ceiling": _capture_block("same", "ceiling", same_fired),
        "split_over_ceiling": _capture_block("split", "ceiling_split", same_fired),
        "by_bucket": {
            "head_only": {
                "n_fired": len(head_only),
                "same_over_ceiling": _capture_block("same", "ceiling", head_only),
                "split_over_ceiling": _capture_block("split", "ceiling_split", head_only),
            },
            "right_only": {
                "n_fired": len(right_only),
                "same_over_ceiling": _capture_block("same", "ceiling", right_only),
                "split_over_ceiling": _capture_block("split", "ceiling_split", right_only),
            },
            "both": {
                "n_fired": len(both_fired),
                "same_over_ceiling": _capture_block("same", "ceiling", both_fired),
                "split_over_ceiling": _capture_block("split", "ceiling_split", both_fired),
            },
        },
    }

    # F5: eval_pose_correction.MAX_ELIGIBLE_BEFORE 가드를 적용했다면 split이 어떻게
    # 바뀌었을지 — 공개용으로만 계산한다. 헤드라인(`split` 키)에는 이 가드를 적용하지
    # 않는다. eval_pose_correction.py의 `eligible()`은 "before"를 채점에 쓴 것과
    # 같은 레퍼런스로 잰다 — split의 자기 자신의 분모(1-before_split_score)가 불안정한
    # 경우를 거르려는 것이 원래 취지이므로, `before_split_score`(B 기준)로 거르는 쪽이
    # 그 취지에 더 가깝다. 다만 `before_same_score`(F 기준, "같은 패스가 얼마나
    # 어려운가"의 대체 지표)로 걸러도 같은 방향(70%에 한참 못 미침)이 나오므로, 어느
    # 쪽으로 재현했는지가 뒤섞이지 않도록 **둘 다** 낸다 — 하나만 골라 싣지 않는다.
    def _eligible(attr: str) -> list[Outcome]:
        return [
            item for item in split_population
            if getattr(item, attr) is not None and getattr(item, attr) <= MAX_ELIGIBLE_BEFORE
        ]

    eligible_by_split_before = _eligible("before_split_score")
    eligible_by_same_before = _eligible("before_same_score")
    eligibility_guard_disclosure = {
        "max_eligible_before": MAX_ELIGIBLE_BEFORE,
        "source": "trainer/scripts/eval_pose_correction.py::MAX_ELIGIBLE_BEFORE",
        "applied_to_headline": False,
        "split_population_n": len(split_population),
        "filtered_by_before_split_score": {
            "n_with_known_value": sum(
                1 for item in split_population if item.before_split_score is not None
            ),
            "n_excluded_by_guard": sum(
                1 for item in split_population
                if item.before_split_score is not None and item.before_split_score > MAX_ELIGIBLE_BEFORE
            ),
            "split_with_guard": summarize([item.split for item in eligible_by_split_before], THRESHOLD),
        },
        "filtered_by_before_same_score": {
            "n_with_known_value": sum(
                1 for item in split_population if item.before_same_score is not None
            ),
            "n_excluded_by_guard": sum(
                1 for item in split_population
                if item.before_same_score is not None and item.before_same_score > MAX_ELIGIBLE_BEFORE
            ),
            "split_with_guard": summarize([item.split for item in eligible_by_same_before], THRESHOLD),
        },
    }

    report: dict[str, Any] = {
        "selection_rule": {
            "root": str(root),
            "reference_artifact": str(reference_path),
            "expert_worker": expert_worker,
            "pairs_requested": pairs,
            "seed": seed,
            "sampling": _SAMPLING_TEXT[pair_rule],
            "pair_rule": pair_rule,
            "seed_ref": seed_ref if pair_rule in ("screen", "screen_first") else None,
            "key_prefix": motion_key.KEY_PREFIX,
        },
        # 쌍 규칙 N의 기록(사전등록 §1.2, §6 F4·F5). 옛 규칙에서는 전부 none/0이다.
        "reference_granularity_counts": {
            _label(key): value for key, value in Counter(item.reference_granularity for item in outcomes).items()
        },
        "ladder_mismatch": sum(1 for item in outcomes if item.reason == "ladder_mismatch"),
        "truncated_count": sum(1 for item in outcomes if item.truncated),
        "cross_level_pairs": sum(
            1 for item in outcomes if item.reference is not None and item.reference.level != item.query.level
        ),
        "status_counts": {_label(key): value for key, value in status_counts.items()},
        "n_attempted": len(outcomes),
        "n_ok": len(ok),
        "same": summarize([item.same for item in ok], THRESHOLD),
        "split": summarize([item.split for item in ok], THRESHOLD),
        "granularity_counts": {_label(key): value for key, value in granularity_counts.items()},
        "per_layer": per_layer,
        "split_reason_counts": {_label(key): value for key, value in split_reason_counts.items()},
        "correction_fire_rate": correction_fire_rate,
        "fired_overlap": fired_overlap,
        "ceiling": ceiling,
        "capture": capture,
        "half_thickness": {
            "thin_threshold_n_min": THIN_HALF_THRESHOLD,
            "n_with_known_thickness": len(thickness_known),
            "thin_n": len(thin),
            "thick_n": len(thick),
            "split_thin": summarize([item.split for item in thin], THRESHOLD),
            "split_thick": summarize([item.split for item in thick], THRESHOLD),
        },
        "gate": {
            "counts": {_label(key): value for key, value in gate_counts.items()},
            "pass_rate_of_determined": (
                gate_counts.get("pass", 0) / gate_determined if gate_determined else None
            ),
            "by_layer": {layer: dict(counter) for layer, counter in gate_by_layer.items()},
        },
        "eligibility_guard_disclosure": eligibility_guard_disclosure,
    }
    if include_pairs:
        report["pairs"] = [_outcome_to_json(item) for item in outcomes]
    return report


def _fmt(value: float | int | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_summary(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    rule = report["selection_rule"]
    lines.append(
        f"쌍 {rule['pairs_requested']}개 요청, seed={rule['seed']}, "
        f"전형기={rule['expert_worker']}, 쌍 규칙={rule.get('pair_rule')}, seed_ref={rule.get('seed_ref')}"
    )
    lines.append(
        f"기준 단: {report.get('reference_granularity_counts')} "
        f"ladder_mismatch={report.get('ladder_mismatch')} truncated={report.get('truncated_count')} "
        f"다른 레벨 짝={report.get('cross_level_pairs')}"
    )
    lines.append(f"상태: {report['status_counts']}")
    for name in ("same", "split"):
        entry = report[name]
        lines.append(
            f"{name}: n={entry['n']} median={_fmt(entry['median'])} "
            f"q1={_fmt(entry['q1'])} q3={_fmt(entry['q3'])} "
            f"≥{THRESHOLD:.0%}={_fmt(entry['at_or_above'])}"
        )
    lines.append(f"입도별 n: {report['granularity_counts']}")
    lines.append(f"split_reason: {report['split_reason_counts']}")
    for name in ("same", "split"):
        fire = report["correction_fire_rate"][name]
        lines.append(
            f"{name} 발동률: {fire['share_fired']} "
            f"(발동 {fire['fired']} / 미발동 {fire['not_fired']} / 불명 {fire['unknown']})"
        )
    ceiling = report["ceiling"]["same_fired_only"]
    lines.append(
        f"F1 상한(발동 표본만): median={_fmt(ceiling['median'])} max={_fmt(ceiling['max'])} "
        f"≥{THRESHOLD:.0%}={_fmt(ceiling['at_or_above'])}"
    )
    gate = report["gate"]
    lines.append(f"REQ-19 게이트: {gate['counts']} pass_rate={gate['pass_rate_of_determined']}")
    return "\n".join(lines)


# --- CLI ---------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=DEFAULT_PAIRS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--index-cache", type=Path, default=None,
        help="전수 색인(패스 구간·모션 키·CSV 트리 지문) 결과를 이 경로에 캐시하고 "
        "재사용한다. 표본 규칙을 바꾸지 않는다 — 트리가 바뀌면(F4) 자동으로 다시 짓는다.",
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--skip-sensitivity", action="store_true",
        help="F5 민감도(PAIR_SIGMA_FACTOR=1.0) 재실행을 건너뛴다(빠른 드라이런용).",
    )
    parser.add_argument(
        "--skip-extras", action="store_true",
        help="F2/F5용 추가 compare() 호출(A/B 반쪽 재현)을 건너뛴다(빠른 드라이런용).",
    )
    parser.add_argument(
        "--pair-rule", default=DEFAULT_PAIR_RULE, choices=PAIR_RULES,
        help="기준 패스 선택 규칙. screen(기본, 화면과 같은 후보 집합)·legacy(옛 규칙, 레벨 무시)·"
        "legacy_same_level(O+L, 보고용)·screen_first(N0, 보고용).",
    )
    parser.add_argument(
        "--seed-ref", type=int, default=DEFAULT_SEED_REF,
        help="screen 규칙에서 질의별 기준 선택 난수열의 시드(귀무 재추첨은 이것만 바꾼다).",
    )
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    if not args.root.is_dir():
        logger.error("root not found: %s", args.root)
        return 1
    if not args.reference.is_file():
        logger.error("reference artifact not found: %s", args.reference)
        return 1

    outcomes, expert_worker, source, reference_spreads, selected = run_study(
        args.root, args.reference, args.pairs, args.seed, args.progress,
        args.index_cache, compute_extras=not args.skip_extras,
        pair_rule=args.pair_rule, seed_ref=args.seed_ref,
    )
    report = build_report(
        outcomes, args.pairs, args.seed, expert_worker, args.root, args.reference,
        pair_rule=args.pair_rule, seed_ref=args.seed_ref,
    )
    print(render_summary(report))

    if not args.skip_sensitivity:
        if args.progress:
            print("민감도(PAIR_SIGMA_FACTOR=1.0) 재실행 중...", file=sys.stderr, flush=True)
        sensitivity_outcomes = run_sensitivity(
            source, reference_spreads, selected, SENSITIVITY_PAIR_SIGMA_FACTOR, args.progress
        )
        sensitivity_report = build_report(
            sensitivity_outcomes, args.pairs, args.seed, expert_worker, args.root, args.reference,
            include_pairs=False, pair_rule=args.pair_rule, seed_ref=args.seed_ref,
        )
        report["sensitivity"] = {
            "pair_sigma_factor": SENSITIVITY_PAIR_SIGMA_FACTOR,
            "note": (
                "service.py 파일의 PAIR_SIGMA_FACTOR(√2)는 바꾸지 않았다 — 실행 중에만 "
                "모듈 전역을 이 값으로 되돌렸다가 끝나면 원복하고, 같은 200쌍을 다시 "
                "돌렸다(F5)."
            ),
            "status_counts": sensitivity_report["status_counts"],
            "same": sensitivity_report["same"],
            "split": sensitivity_report["split"],
            "correction_fire_rate": sensitivity_report["correction_fire_rate"],
            "gate": sensitivity_report["gate"],
        }
        print(
            f"[민감도] same≥70%={_fmt(sensitivity_report['same']['at_or_above'])} "
            f"split≥70%={_fmt(sensitivity_report['split']['at_or_above'])} "
            f"발동률(same)={sensitivity_report['correction_fire_rate']['same']['share_fired']} "
            f"게이트={sensitivity_report['gate']['counts']}"
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
