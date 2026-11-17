"""3D 모션 뷰어 재생은 첫 분사부터 시작한다(2026-11-17 사용자 원칙).

세션 시작부터 담긴 변환 기록은 앞부분이 건을 들기 전 대기·추적 초기화 구간이다. 그 첫 행을 0:00으로 그리면
건이 놓여 있던 자리에 찍히고 시점 중심이 틀어졌다(운영 학습모드 4 대표 기록).
"""
from __future__ import annotations

from pathlib import Path

from trainer.demo_api import motion_viewer as mv

HEADER = "시간(초),분사,카메라 Position X,카메라 Position Y,카메라 Position Z,한길숏건 Position X,한길숏건 Position Y,한길숏건 Position Z\n"


def _csv(path: Path, rows: list[tuple[float, str, tuple[float, float, float], tuple[float, float, float]]]) -> Path:
    path.write_text(HEADER + "".join(f"{t},{spray},{h[0]},{h[1]},{h[2]},{g[0]},{g[1]},{g[2]}\n" for t, spray, h, g in rows),
                    encoding="utf-8")
    return path


def test_playback_starts_at_the_first_spray_and_alert_clock_matches(tmp_path: Path) -> None:
    parked, head, hand = (-3.63, 5.22, -0.27), (0.1, 1.6, 0.0), (0.4, 1.2, 0.3)
    rows = [(0.0, "Off", (0.1, -0.06, 0.0), parked), (0.5, "Off", head, parked), (3.0, "Off", head, hand)]
    rows += [(3.97 + 0.1 * i, "On", head, hand) for i in range(20)]
    path = _csv(tmp_path / "converted_Height.csv", rows)
    frames = mv._frames(path, 10)
    assert frames[0]["spraying"] is True and frames[0]["t"] == 0.0
    assert frames[0]["tracked"]["right_hand"] == list(hand)  # 놓여 있던 건 자리가 아니다
    assert len(frames) == 10 and frames[-1]["t"] == round(0.1 * 19, 6)
    assert mv._clock_offset(path, 10) == 3.97  # 알림 시각을 CSV 첫 행이 아니라 첫 분사에 맞춘다


def test_recording_without_spray_keeps_the_first_row(tmp_path: Path) -> None:
    rows = [(float(i), "Off", (0.1, 1.6, 0.0), (0.4, 1.2, 0.3)) for i in range(5)]
    path = _csv(tmp_path / "idle.csv", rows)
    frames = mv._frames(path, 5)
    assert len(frames) == 5 and frames[0]["t"] == 0.0
    assert mv._clock_offset(path, 5) == 0.0
