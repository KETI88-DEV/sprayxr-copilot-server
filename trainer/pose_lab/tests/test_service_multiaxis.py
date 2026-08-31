#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false
"""다축 교정이 응답까지 배선됐는지. 선택 규칙 자체는 test_measured_score_multiaxis가 본다."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

measured_score = importlib.import_module("measured_score")

from trainer.pose_lab import service

FRAME = measured_score.WORLD_FRAME


def _deltas() -> list[object]:
    return [
        measured_score.PointDelta(
            point="right",
            sigma=4.0,
            local_delta=(0.08, 0.0, -0.06),
            samples=30,
            axis_sigmas={"wall": 3.5, "vertical": 2.6, "lateral": 0.1},
        ),
        measured_score.PointDelta(
            point="head",
            sigma=3.4,
            local_delta=(0.18, 0.0, 0.0),
            samples=30,
            axis_sigmas={"wall": 3.4, "vertical": 0.2, "lateral": 0.1},
        ),
    ]


def test_axes_block_lists_every_issued_axis_with_its_own_sigma() -> None:
    block = service.axes_block(_deltas(), FRAME)
    assert block, "발부된 축이 있어야 한다"
    sigmas = [entry["sigma"] for entry in block]
    assert sigmas == sorted(sigmas, reverse=True)
    for entry in block:
        assert set(entry) == {"point", "axis", "magnitude_cm", "direction", "sigma"}
        assert entry["point"] in {"right", "head"}


def test_axes_block_never_lists_the_left_hand() -> None:
    left = measured_score.PointDelta(
        point="left",
        sigma=5.0,
        local_delta=(0.30, 0.30, 0.30),
        samples=30,
        axis_sigmas={"wall": 5.0, "vertical": 5.0, "lateral": 5.0},
    )
    assert service.axes_block([left], FRAME) == []


def test_stance_block_is_none_when_the_head_is_quiet() -> None:
    quiet = [
        measured_score.PointDelta(
            point="head",
            sigma=0.5,
            local_delta=(0.01, 0.0, 0.0),
            samples=30,
            axis_sigmas={"wall": 0.2, "vertical": 0.1, "lateral": 0.1},
        )
    ]
    assert service.stance_block(quiet, FRAME) is None


def test_stance_block_carries_the_sentence_and_the_exact_centimetres() -> None:
    """문장은 정성적이고 숫자는 정량적이다. 시뮬레이션은 숫자를 쓴다."""
    block = service.stance_block(_deltas(), FRAME)
    assert block is not None
    assert set(block) == {"text", "axis", "magnitude_cm", "sigma"}
    assert isinstance(block["text"], str) and block["text"]
    assert "앉아" not in block["text"]
    assert abs(block["magnitude_cm"]) >= measured_score.MIN_CORRECTION_CM
