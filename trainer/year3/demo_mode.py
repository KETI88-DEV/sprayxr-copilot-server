"""3차년도 교정(P2, 관절 값 6개 이하)을 8130 교정 시연 화면에 **숨김 방식**으로 올린다.

pose_lab은 이 모듈이 있으면 불러 `register()`를 부른다(`trainer/pose_lab/app.py`). 숨김이라 시연 화면의 목록에는
나오지 않고 주소 `demo.html?mode=six_values`로만 열린다(사용자 2026-10-01: 3차년도 내용은 숨겨 둔 기능).
"""

from __future__ import annotations

from trainer.year3 import correction_policies

MODE: str = "six_values"


def register() -> None:
    from trainer.pose_lab import correction_demo

    correction_demo.register_mode(
        MODE, "관절 값 6개 이하 교정", correction_policies.POLICIES[correction_policies.PRIMARY], hidden=True
    )
