#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false
"""채점 격리: 표시 전용 변경이 점수·교정·개선율을 한 비트도 옮기지 않았는가.

기준선은 코드 변경 전 트리(git archive b5f7341)에서 뜬 `fixtures/scoring_baseline_b5f7341.json`이다.
합성 FakeSource 한 건과 실데이터 두 쌍(김경호 L1 1000389_1 #0 대 전형기 1000564_1 #17,
김경호 L3 1000391_1 #0 대 전형기 1000444_1 #10)을 JSON 문자열(sort_keys)로 비교한다.

실데이터가 없으면 **skip하지 않고 실패한다** — skip하면 이 비교가 실패할 수 없는 단언이 된다.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

measured_score = importlib.import_module("measured_score")

from trainer.pose_lab import composite, instructions, orientation, service  # noqa: E402
from trainer.pose_lab.service import PassRef  # noqa: E402
from trainer.pose_lab.tests.test_app import QUERY, REFERENCE, FakeSource  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "scoring_baseline_b5f7341.json"
DATA_ROOT = Path("/data/sources/professionals")
SPREADS = Path("/output/spread_reference_expert_yup.json")

# b5f7341 이후 표시 전용으로 더한 필드. 이것만 빼면 기준선과 같아야 한다.
NEW_MEASURED_KEYS = frozenset({"improvement_labels", "display", "deltas_display", "orientation"})
NEW_AXIS_KEYS = frozenset(
    {
        "n",
        "text",
        "text_reason",
        "text_mode",
        "figure",
        "sigma_display",
        "trace",
        "surface_angle_deg",
        "surface_angle_display",
        "surface_reason",
    }
)
LEGACY_CORRECTION_KEYS = ("point", "axis", "magnitude_cm", "direction", "sigma", "unit", "from", "to")


def _cases() -> list[dict[str, Any]]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


CASE_NAMES = [case["name"] for case in _cases()]


def _dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


_REAL: dict[str, Any] = {}


def _source_for(case: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    if case["name"].startswith("synthetic"):
        return FakeSource(), {}
    if not DATA_ROOT.is_dir() or not SPREADS.is_file():
        pytest.fail(f"실데이터가 없다({DATA_ROOT}, {SPREADS}) — 이 비교는 skip하지 않는다")
    if "source" not in _REAL:
        from trainer.pose_lab import csv_source
        from trainer.pose_lab.spread_reference import SpreadReference

        _REAL["source"] = csv_source.CsvPassSource(DATA_ROOT)
        _REAL["spreads"] = SpreadReference.load(SPREADS)
    return _REAL["source"], {"reference_spreads": _REAL["spreads"]}


def _run(case: dict[str, Any], include_view: bool) -> dict[str, Any]:
    source, kwargs = _source_for(case)
    body = service.compare(
        source, PassRef(**case["query"]), PassRef(**case["reference"]), include_view=include_view, **kwargs
    )
    assert body["status"] == "ok"
    return body


def legacy_measured(measured: dict[str, Any]) -> dict[str, Any]:
    out = {key: value for key, value in measured.items() if key not in NEW_MEASURED_KEYS}
    out["axes"] = [{k: v for k, v in axis.items() if k not in NEW_AXIS_KEYS} for axis in measured["axes"]]
    return out


def legacy_view(view: dict[str, Any]) -> dict[str, Any]:
    return {
        "view_reference": view["layers"]["reference"],
        "view_corrected": view["layers"]["corrected"],
        "view_corrections_legacy": [
            {key: arrow[key] for key in LEGACY_CORRECTION_KEYS} for arrow in view["corrections"]
        ],
        "weights": {key: view["weights"][key] for key in measured_score.DEFAULT_WEIGHTS},
    }


def assert_scoring_matches_baseline(body: dict[str, Any], case: dict[str, Any]) -> None:
    assert _dump(legacy_measured(body["measured"])) == _dump(case["measured"])
    assert _dump(body["joint"]) == _dump(case["joint"])


@pytest.mark.parametrize("name", CASE_NAMES)
def test_scoring_bit_identical_without_view(name: str) -> None:
    """`include_view=False`의 measured는 **신규 필드 없이** 기준선과 문자열로 같다."""
    case = next(c for c in _cases() if c["name"] == name)
    body = _run(case, include_view=False)
    assert body["view"] is None
    assert _dump(body["measured"]) == _dump(case["measured"])
    assert _dump(body["joint"]) == _dump(case["joint"])


@pytest.mark.parametrize("name", CASE_NAMES)
def test_scoring_bit_identical_with_view(name: str) -> None:
    """`include_view=True`에서도 신규 필드를 뺀 measured·기존 view 필드가 기준선과 같다."""
    case = next(c for c in _cases() if c["name"] == name)
    body = _run(case, include_view=True)
    assert_scoring_matches_baseline(body, case)
    expected = {key: case[key] for key in ("view_reference", "view_corrected", "view_corrections_legacy")}
    expected["weights"] = {key: case["weights"][key] for key in measured_score.DEFAULT_WEIGHTS}
    assert _dump(legacy_view(body["view"])) == _dump(expected)
    for name_, value in body["view"]["weights"].items():
        if name_ not in measured_score.DEFAULT_WEIGHTS:
            assert value == 0.0, name_


def _raise(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("display block boom")


DISPLAY_TARGETS = {
    "skeleton": (service, "skeleton_block"),
    "mesh": (service, "placed_mesh"),
    "instructions": (instructions, "axis_display"),
    "orientation": (orientation, "orientation_block"),
    "composite": (composite, "composite_block"),
}


@pytest.mark.parametrize("block", sorted(DISPLAY_TARGETS))
def test_compare_survives_display_errors(block: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """표시 블록 하나가 던져도 채점 필드는 기준선 그대로 나가고, 그 블록이 display_errors에 적힌다."""
    module, attribute = DISPLAY_TARGETS[block]
    monkeypatch.setattr(module, attribute, _raise)
    case = next(c for c in _cases() if c["name"] == "synthetic_fake_source")
    body = service.compare(FakeSource(), PassRef(**QUERY), PassRef(**REFERENCE))
    assert body["status"] == "ok"
    assert_scoring_matches_baseline(body, case)
    assert {"block": block, "error_type": "RuntimeError"} in body["view"]["display_errors"]
    if block == "mesh":
        assert body["view"]["mesh"]["reason"] == "display_error"
        assert body["view"]["mesh"]["reason"] != "unverified_alignment"


def test_improvement_study_calls_compare_without_view(monkeypatch: pytest.MonkeyPatch) -> None:
    """개선율 측정 도구의 compare 호출 세 곳이 전부 `include_view=False`다(스파이)."""
    improvement_study = importlib.import_module("improvement_study")
    calls: list[dict[str, Any]] = []

    def spy(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "unavailable", "reason": "spy"}

    monkeypatch.setattr(improvement_study.service, "compare", spy)
    query = PassRef(**QUERY)
    reference = PassRef(**REFERENCE)
    pair = improvement_study.Pair(query=query, reference=reference, key="k")
    improvement_study.run_pair(FakeSource(), object(), pair, compute_extras=False)

    class Spreads:
        def spreads_for(self, *_args: Any) -> dict[str, dict[str, float]]:
            return measured_score.uniform_spreads(0.05)

    frames = service.pass_frames(FakeSource().load(query))
    granularity = "uv8"
    assert service.key_inputs(frames) is not None
    measured = {"split_reason": None, "granularity": granularity, "split_dates_a": ["d1"], "split_dates_b": ["d2"]}
    improvement_study._split_details(FakeSource(), Spreads(), query, reference, measured)  # noqa: SLF001
    assert len(calls) == 3
    assert all(call.get("include_view") is False for call in calls), calls


def _point_deltas(measured: dict[str, Any]) -> list[Any]:
    return [
        measured_score.PointDelta(
            point=d["point"],
            sigma=d["sigma"],
            local_delta=tuple(d["delta"]),
            samples=d["samples"],
            axis_sigmas=dict(d["axis_sigmas"]),
        )
        for d in measured["deltas"]
    ]


def _axis_tuple(entry: Any) -> tuple[Any, ...]:
    get = (lambda k: entry[k]) if isinstance(entry, dict) else (lambda k: getattr(entry, k))
    return (get("point"), get("axis"), get("magnitude_cm"), get("direction"), get("sigma"))


@pytest.mark.parametrize("name", ["synthetic_fake_source", "real_L1_kkh_1000389_1_0"])
def test_issued_axes_single_list_matches_apply_measured(name: str) -> None:
    """`measured.axes`, n으로 짝지은 `view.corrections`, `apply_measured`가 옮기는 이동이 한 목록이다."""
    case = next(c for c in _cases() if c["name"] == name)
    body = _run(case, include_view=True)
    measured = body["measured"]
    source, _kwargs = _source_for(case)
    basis = service.representative_basis(service.pass_frames(source.load(PassRef(**case["query"]))))
    deltas = _point_deltas(measured)
    issued = measured_score.issued_axes(deltas, basis)
    assert issued, "이 표본은 교정이 발부돼야 한다"

    # (1) measured.axes 순서 == 다시 부른 issued_axes
    assert [_axis_tuple(a) for a in measured["axes"]] == [_axis_tuple(c) for c in issued]
    assert [a["n"] for a in measured["axes"]] == list(range(1, len(issued) + 1))
    # (2) view.corrections를 n으로 짝지으면 같은 튜플
    for arrow in body["view"]["corrections"]:
        assert _axis_tuple(arrow) == _axis_tuple(issued[arrow["n"] - 1]), arrow["n"]
    # (3) 점마다 Σ unit·magnitude/100 == apply_measured가 쓰는 issued_shifts(월드로 되돌림)
    shifts = measured_score.issued_shifts(deltas, basis)
    for point, local in shifts.items():
        world = [
            local[0] * basis.normal[i] + local[1] * basis.tangent[i] + local[2] * basis.up[i] for i in range(3)
        ]
        summed = [0.0, 0.0, 0.0]
        for entry in measured["axes"]:
            if entry["point"] != point:
                continue
            unit = next(c.unit for c in issued if _axis_tuple(c) == _axis_tuple(entry))
            for i in range(3):
                summed[i] += unit[i] * entry["magnitude_cm"] / 100.0
        assert all(math.isclose(summed[i], world[i], abs_tol=1e-12) for i in range(3)), (point, summed, world)
