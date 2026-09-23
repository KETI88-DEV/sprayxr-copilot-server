#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

builder = importlib.import_module("build_spread_reference")
motion_key = importlib.import_module("motion_key")
spread_stats = importlib.import_module("spread_stats")
measured_score = importlib.import_module("measured_score")

KEY = "m1:horizontal:right:1x2:center-near-high"


def _buckets(by_date):
    return {"uv8": {KEY: by_date}, "uv4": {}, "uv2": {}, "uv_off": {}}


def test_granularity_ladder_is_fine_to_coarse_ordered() -> None:
    assert [name for name, _grid in builder.GRANULARITIES] == ["uv8", "uv4", "uv2", "uv_off"]


def test_n_min_is_inherited_not_reinvented() -> None:
    """이전 연구에서 사전등록한 값이다. 이번 수치를 보고 정하지 않는다."""
    assert builder.N_MIN == 20


def test_artifact_has_every_layer_and_carries_dates() -> None:
    payload = builder.build_payload(
        worker="전형기", levels=["10"], sessions=3,
        buckets=_buckets({"20250901": {"right": {"wall": spread_stats.accumulate([0.10, 0.12])}}}),
    )
    assert payload["worker"] == "전형기"
    assert payload["n_min"] == 20
    assert set(payload["granularities"]) == {"uv8", "uv4", "uv2", "uv_off"}
    bucket = payload["granularities"]["uv8"][KEY]
    assert bucket["dates"] == ["20250901"]
    assert bucket["n"] == 2
    assert "by_date" in bucket
    assert bucket["points"]["right"]["wall"]["spread"] >= 0.0


def test_payload_round_trips_through_json() -> None:
    payload = builder.build_payload(
        worker="전형기", levels=["10"], sessions=1,
        buckets=_buckets({"20250901": {"right": {"wall": spread_stats.accumulate([0.1, 0.2])}}}),
    )
    restored = json.loads(json.dumps(payload, ensure_ascii=False))
    assert restored["granularities"]["uv8"][KEY]["n"] == 2


def test_two_dates_merge_into_one_bucket() -> None:
    payload = builder.build_payload(
        worker="전형기", levels=["10"], sessions=2,
        buckets=_buckets({
            "20250901": {"right": {"wall": spread_stats.accumulate([0.10, 0.12])}},
            "20250904": {"right": {"wall": spread_stats.accumulate([0.14])}},
        }),
    )
    bucket = payload["granularities"]["uv8"][KEY]
    assert bucket["dates"] == ["20250901", "20250904"]
    assert bucket["n"] == 3


def test_pass_axis_values_uses_the_correction_axes() -> None:
    """산포 축은 교정 축과 같아야 sigma로 축을 고를 수 있다."""
    class _Frame:
        normal = (1.0, 0.0, 0.0)
        tangent = (0.0, 1.0, 0.0)
        up = (0.0, 0.0, 1.0)

    tracks = {"right": [(0.3, 0.1, 0.2)] * 4, "head": [(0.5, 0.0, 0.0)] * 4, "left": []}
    values = builder.pass_axis_values(tracks, _Frame())
    assert set(values["right"]) == set(measured_score.AXIS_ORDER)
    assert "left" not in values


def test_pass_axis_values_keeps_axes_below_the_correction_threshold() -> None:
    """산포는 '문장으로 낼 만큼 크다'는 질문이 아니라 σ의 재료다. `_correction_candidates`를
    썼다면 `MIN_CORRECTION_CM`(1cm) 미만인 wall 성분이 후보에서 통째로 빠졌을 것이고,
    그러면 그 패스는 wall 축 산포에 기여하지 못해 산포가 큰 패스 쪽으로만 부풀었을
    것이다. `_axis_magnitudes_cm`은 임계 없이 세 축을 전부 낸다."""
    class _Frame:
        normal = (1.0, 0.0, 0.0)
        tangent = (0.0, 1.0, 0.0)
        up = (0.0, 0.0, 1.0)

    tiny_wall = 0.005  # 0.5cm, measured_score.MIN_CORRECTION_CM(1.0cm) 미만이지만 0은 아니다.
    tracks = {"right": [(tiny_wall, 0.1, 0.2)] * 4, "head": [], "left": []}
    values = builder.pass_axis_values(tracks, _Frame())
    assert set(values["right"]) == set(measured_score.AXIS_ORDER)
    assert math.isclose(values["right"]["wall"], tiny_wall, abs_tol=1e-12)


def test_key_without_uv_replaces_the_cell_component() -> None:
    assert builder.key_without_uv(KEY) == "m1:horizontal:right:na:center-near-high"


def test_thin_bucket_is_kept_in_the_artifact() -> None:
    """n<20이어도 아티팩트에는 남긴다. 폴백 판정은 읽는 쪽이 한다."""
    payload = builder.build_payload(
        worker="전형기", levels=["10"], sessions=1,
        buckets=_buckets({"20250901": {"right": {"wall": spread_stats.accumulate([0.1])}}}),
    )
    assert payload["granularities"]["uv8"][KEY]["n"] == 1


# --- 산포 빌더의 키 UV 경로(§8, 사전등록 2026-09-23) --------------------------------
#
# D1(m3)의 적용 지점은 셋이다: 색인(`csv_source`), 비교(`service.compare`), 그리고 **산포 빌더**.
# 앞의 둘은 `test_midpoint_uv_key.py`가 지킨다. 빌더만 검사 없이 서 있었다 — 빌더의 키 재료
# 호출을 `key_inputs(frames, raw_frames=frames)`로 되돌려도 전체 스위트가 통과했다.
# 되돌아가면 산포 버킷이 **시작점 칸**으로 매겨지고, 화면은 m3 키로 후보를 고르면서 채점만
# m1 칸의 산포로 하게 된다 — 조용히. 이 검사가 그 경로를 지킨다.

REAL_ROOT = Path("/data/sources/professionals")
# 시작 칸과 호장 중점 칸이 갈리는 패스가 실제로 들어 있는 세션(`test_midpoint_uv_key.SESSIONS`와 같은 것).
SPLIT_SESSION = ("김경호", "20250831", "1", "1_1000389_1.csv")


def _one_session_root(tmp_path: Path) -> Path:
    """세션 CSV 하나만 든 임시 루트. 빌더가 훑을 범위를 그 하나로 줄인다."""
    worker, date, _level, filename = SPLIT_SESSION
    source = REAL_ROOT / f"{worker}_{date}" / f"{worker}_{date}(Sequences)" / filename
    if not source.is_file():
        pytest.fail(f"실데이터가 없다({source}) — 이 검사는 skip하지 않는다")
    sequences = tmp_path / f"{worker}_{date}" / f"{worker}_{date}(Sequences)"
    sequences.mkdir(parents=True)
    (sequences / filename).symlink_to(source)
    return tmp_path


def test_builder_buckets_by_the_midpoint_cell_not_the_start_cell(tmp_path) -> None:
    """§8: 빌더의 버킷 키 = `service.pass_key_inputs` → `service.key_from`의 키이고,
    **시작 칸으로 매긴 키와는 다르다**.

    뒤의 단언이 없으면, 두 칸이 같은 입력에서는 이 검사가 아무것도 지키지 못한다(F16).
    """
    from trainer.pose_lab import csv_source, service

    worker, date, level, _filename = SPLIT_SESSION
    root = _one_session_root(tmp_path)
    payload = builder.build_from_root(root, worker, [level])
    built = set(payload["granularities"]["uv8"])
    assert built, "빌더가 uv8 버킷을 하나도 만들지 못했다 — 아래 대조가 공허하다"

    params = motion_key.KeyParams(uv_grid=8)
    source = csv_source.CsvPassSource(root)
    expected: set[str] = set()
    start_cell_keys: set[str] = set()
    split = 0
    for item in source.sessions():
        for entry in source.index_of(item):
            ref = service.PassRef(
                worker=item.worker, date=item.date, level=item.level,
                session=item.session, pass_index=entry.pass_index,
            )
            raw = source.load(ref)
            _frames, inputs = service.pass_key_inputs(raw)
            if inputs is None:
                continue
            expected.add(service.key_from(inputs, params))
            # 구멍이 뚫렸을 때 빌더가 만들게 되는 키: UV만 **첫 프레임**의 것으로 바꾼다.
            start_uv = service.spray_frames(raw)[0].uv
            start_key = motion_key.motion_key(
                displacement=inputs.displacement, yaw=service.yaw_of(inputs),
                head=inputs.head, hit=inputs.hit, uv=start_uv, params=params,
            )
            start_cell_keys.add(start_key)
            split += start_key != service.key_from(inputs, params)

    assert split >= 3, f"시작 칸과 중점 칸이 갈린 패스가 {split}개뿐 — 이 검사가 되돌림을 가려내지 못한다"
    assert built == expected, "빌더의 버킷 키가 비교 경로의 키와 다르다"
    assert built != start_cell_keys, "빌더가 시작점 칸으로 버킷을 매겼다 — D1(m3)이 산포에서 되돌아갔다"
