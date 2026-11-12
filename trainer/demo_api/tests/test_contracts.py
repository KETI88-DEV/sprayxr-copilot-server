# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError
import pytest

from trainer.demo_api.contracts import SequenceSnapshot, SequenceUpdate, SequenceZone
from trainer.demo_api.v1_contracts import ReportDetail


def test_sequence_update_rejects_duplicate_next_zones() -> None:
    with pytest.raises(ValidationError):
        SequenceUpdate(
            type="update_sequence",
            level_id="TR-L01",
            current_zone_id="A-1",
            next_zone_ids=["A-2", "A-2"],
            completed_zone_ids=["A-1"],
        )


def test_snapshot_requires_current_zone_to_exist() -> None:
    with pytest.raises(ValidationError):
        SequenceSnapshot(
            level_id="TR-L01",
            label="Training Level 01",
            zones=[SequenceZone(zone_id="A-1", label="상부", status="pending")],
            current_zone_id="A-2",
            next_zone_ids=[],
            completed_zone_ids=[],
        )


def test_zone_status_enumerates_allowed_values() -> None:
    zone = SequenceZone(zone_id="A-1", label="상부", status="completed")
    assert zone.status == "completed"


# --- 세션 종합 리포트: 모양의 정본은 예시 파일이다 -----------------------------
#
# `trainer/demo_api/report_sample.json` 이 `GET /api/v1/reports/{report_id}` 의 모양을
# 정한다. 계약과 예시가 갈리면 어느 쪽이 맞는지 알 수 없어지므로 여기서 묶는다.

_REPORT_SAMPLE = Path(__file__).resolve().parents[1] / "report_sample.json"


def _sample() -> dict[str, Any]:
    if not _REPORT_SAMPLE.exists():
        pytest.skip(f"예시 파일이 없다: {_REPORT_SAMPLE}")
    return json.loads(_REPORT_SAMPLE.read_text(encoding="utf-8"))


def _paths(value: Any, prefix: str = "") -> set[str]:
    """중첩 키를 점 경로로 편다. 배열은 첫 원소만 본다 — 원소끼리 모양이 같다."""
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            found.add(prefix + key)
            found |= _paths(child, prefix + key + ".")
    elif isinstance(value, list) and value:
        found |= _paths(value[0], prefix)
    return found


def test_the_report_example_is_valid_json() -> None:
    """예시가 파싱되지 않으면 아무도 그 모양을 확인할 수 없다.

    2026-10-08 에 이 파일이 U+00A0 들여쓰기와 `//` 주석 때문에 JSON 이 아니었다.
    """
    sample = _sample()

    assert sample["status"] == "SUCCESS"
    assert isinstance(sample["data"]["report"], dict)


def test_report_detail_accepts_the_example_as_written() -> None:
    """Given the example, When parsed by the contract, Then it validates.

    깨뜨리는 법: 예시에 있는 필드를 계약에서 빼면 `extra="forbid"` 에 걸려 실패한다.
    """
    detail = ReportDetail.model_validate(_sample()["data"])

    assert detail.report.report_status == "complete"
    assert detail.report.pose_report.analysis_status == "available"


def test_the_contract_covers_every_field_the_example_uses() -> None:
    """예시의 모든 경로가 우리 출력에도 있어야 한다. 반대 방향은 한 건만 허용한다."""
    data = _sample()["data"]
    produced = ReportDetail.model_validate(data).model_dump(mode="json")

    assert _paths(data) - _paths(produced) == set(), "예시에 있는데 우리가 안 내는 필드"
    # 명세서 p.16 은 `direction_adherence_rate` 를 sequence_report 의 최상위 필드로
    # 적고 예시는 direction_summary 안에 둔다. 둘 다 싣되 한 번 계산해 같은 값을 쓴다.
    assert _paths(produced) - _paths(data) == {
        "report.sequence_report.direction_adherence_rate",
        "report.sequence_report.reference_session_count",
        "report.sequence_report.reference_flows",
        "report.sequence_report.position_distribution",
        "report.sequence_report.best_reference_similarity",
        "report.sequence_report.position_similarity",
        "report.coating_report.worst_zones.uv_bin",
        "report.coating_report.worst_zones.zone_class",
        "report.coating_report.worst_zones.action_reachable",
        "report.pose_report.confidence",
        "report.pose_report.limited_sequence_count",
        "report.pose_report.reason_codes",
    }


def test_report_sample_coating_matches_a_real_report() -> None:
    """예시의 coating_report 는 실제 리포트와 열쇠 집합이 같고 예시 안의 수가 서로 맞는다."""
    from trainer.demo_api.tests.test_session_coating import _EXPERT_THICKNESS, _png, _profile, _thickness
    import tempfile

    from trainer.demo_api.coating_progress import CoatingProgressEngine
    from trainer.demo_api.session_coating import analyze_session_coating

    with tempfile.TemporaryDirectory() as root:
        _profile(Path(root))
        real = analyze_session_coating(
            scenario_id="3", bodies=[{"sequence_setting": {"tool_type": "spray_gun"}}],
            engine=CoatingProgressEngine(Path(root)), height_map=_png(_thickness(_EXPERT_THICKNESS)),
            height_map_name="Map01_x_Height.png",
        ).report
    example = _sample()["data"]["report"]["coating_report"]

    # 계약 필드지만 아직 값을 못 내는 세 열쇠는 예시에 null 로 남긴다.
    assert _paths(example) - _paths(real) == {"coating_distribution_index", "reachability", "root_cause"}
    assert _paths(real) <= _paths(example)
    comparison = example["expert_comparison"]
    assert comparison["zone_count"] == comparison["normal"] + comparison["under_app"] + comparison["over_app"]
    assert comparison["zone_count"] == len(comparison["zones"])
    assert sum(example["zone_summary"].values()) == pytest.approx(1.0, abs=0.01)
