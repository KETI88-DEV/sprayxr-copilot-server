#!/usr/bin/env python3
"""캡처 좌표를 월드(레벨 메시가 저작된) 좌표로 옮긴다. 검증된 정렬만 쓴다.

**최초 계획은 틀렸다.** task-0-brief.md는 "런타임 월드 데이터가 올 때까지 `to_world`는
항등이다"라고 적었지만, 실제 사용자 지시는 "월드좌표로 변환해서 적용해보고"였다 —
지금 시도해 보라는 것이다. 항등 스텁은 `if IS_WORLD: return v / return v`처럼 두 분기가
같은 값을 내는, 이 저장소가 거듭 낸 결함(아무것도 강제하지 않는 구조)의 축소판이었다.

**실제로 검증된 변환들이 있다**: `trainer/data/level_alignment.json`의 "solved"
정렬이다(지금 레벨 3~10, 8개). `level_mesh.solve_alignment_uv`가 관측 UV로 만든
대응에 12자유도 아핀을 폐형식으로 맞추고 직교화한 결과이고,
`level_mesh.place(mesh, alignment)`가 그 변환으로 레벨 FBX(모델 공간, Unity에서
저작된 좌표)를 캡처 좌표계로 옮긴다 — 옮긴 뒤 홀드아웃 착탄점이 메시 표면에서
median ≤ 0.0003 mm, 절대 UV 잔차 p95 ≤ 0.0007 텍셀로 들어맞는다
(`trainer/scripts/tests/test_level_mesh.py`의 게이트 대조·변이 배터리 참고).
이 저장소에서 "월드 좌표 해석이 맞다"고 외부 증거로 확인된 지점이다.

**회전은 반사다(det = -1).** 여기서 쓰는 역행렬이 전치라는 사실은 그대로 옳다 —
반사도 직교행렬이라 `R⁻¹ = Rᵀ`다. 다만 `alignment.rotation`을 쿼터니언이나 오일러로
바꾸는 경로를 여기에 들이면 det < 0에서 조용히 망가진다
(`test_to_world_round_trips_through_a_reflection`이 그것을 고정한다).

**`to_world`는 그 변환의 역이다.** `place`는 `capture = scale·(R @ raw) + T`다.
역은 `raw = R⁻¹ @ ((capture - T) / scale)`이지만, 이 파일은 `scale`로 나누지
**않는다** — 그 이유가 이 함수의 핵심이다.

## 왜 스케일을 나누지 않는가

`alignment.scale`(≈0.0254)은 "레벨 FBX의 원시 단위 하나가 몇 미터인가"
그 자체다. 위 세 검증(ICP·FBX 선언값 0.0254·텍셀 밀도, `test_committed_scale_is_
within_the_texel_density_band`)이 독립으로 이 값에 수렴한다. 그래서 레벨 FBX의
원시 정점을 미터로 옮기려면 `raw * alignment.scale`이면 된다 — 그리고 `place`의
정의를 대입하면:

    capture = scale·(R @ raw) + T = R @ (scale·raw) + T = R @ world_m + T

(스칼라 `scale`은 행렬곱과 교환된다.) 그러므로

    world_m = R⁻¹ @ (capture - T) = Rᵀ @ (capture - T)      (R은 정규직교라 R⁻¹=Rᵀ)

**스케일 나눗셈이 대수적으로 통째로 사라진다** — 나누고 다시 곱하는 게 아니라 애초에
등장하지 않는다. 실측으로 확인했다(레벨 1 착탄점 2,365개; 그 레벨은 이제 "unverifiable"이지만 이
대수적 유도 자체는 정렬과 무관하다): 이 식으로 얻은
`world_m` 점과, 레벨 메시 정점을 `scale`로만 미터로 바꾼 메시(`level_mesh.
closest_distance`로 잰) 사이의 거리는 median 2.07e-10 m — `place()`로 얻은 캡처
좌표계 잔차와 소수점 다섯 자리까지 동일하다. 이 파일은 그 식만 구현한다.

## 단위: 미터로 확인했다, 가정하지 않았다

(레벨 1 메시를 이 식으로 미터로 옮긴 바운딩 박스는 (1.42 m, 0.025 m, 2.12 m) —
사람 키 절반쯤 되는 벽 패널에 2.5cm 두께, 선박 도장 훈련 레벨의 물리적 크기로
말이 된다. FBX 원시 단위 그대로였다면 (55.9, 0.98, 83.5) — 이 크기의 벽이라기엔
너무 크다(83 "미터"짜리 벽은 없다). `alignment.declared_scale`(FBX의 `UnitScaleFactor`
에서 읽은, 1인치=0.0254m와 정확히 같은 값)이 `alignment.scale`과 0.0077% 안에서
일치한다는 사실 자체가 "FBX 원시 단위는 인치이고, 그 원시 정점에 `scale`을 곱하면
미터가 된다"는 것의 세 번째 독립 증거다. 채택된 8개 레벨은 전부 `UnitScaleFactor`
2.54 → 0.0254와 1e-5 안에서 일치한다 — 그것이 지금 게이트 G4가 재는 값이다.)

## 검증되지 않은 레벨은 조용히 통과하지 않는다

`level_alignment.json`에 항목이 없는 레벨로 `to_world`를 부르면
`level_mesh.load_alignment`가 `LookupError`를 낸다 — 캡처 좌표를 월드라고 속여
돌려주지 않는다. 항목이 있어도 `provenance`가 "solved"가 아니면 `ValueError`를
낸다. 세 번째 상태가 있다: `"unverifiable"`은 **데이터가 검증을 받치지 못한다**는
뜻이다(레벨 1·2가 그렇다 — 레벨 1은 착탄점이 정확히 평면이라 12자유도가 결정되지
않고, 레벨 2는 아틀라스가 표면적의 6.5%에서 단사가 아니다). "항목이 없다"와
"검증할 수 없다"는 화면이 다르게 말해야 하는 서로 다른 상태다.

축 규약(어느 인덱스가 수직인가)은 여기서 다루지 않는다 — `world_axes.py`가 유일한
출처다. `to_world`가 내는 `world_m`은 레벨 메시가 저작된 좌표계 그대로이고, 그
축 배열이 `world_axes`가 이미 다루는 캡처 축 배열과 같다(둘 다 유니티 좌표를
그대로 읽는다) — 여기서 축을 다시 정하지 않는다.
"""

from __future__ import annotations

import level_mesh
import rotation_math

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]


def _rotation_matrix(rotation: tuple[float, ...]) -> Mat3:
    """정렬 레코드의 9개 행 우선(row-major) 값을 3x3 행렬로. `level_mesh._rotation_matrix`
    와 같은 배치다 — 그 함수는 비공개라 여기서 `Alignment`의 공개 필드에서 다시 만든다."""
    r = rotation
    return ((r[0], r[1], r[2]), (r[3], r[4], r[5]), (r[6], r[7], r[8]))


def to_world(point: Vec3, level: int) -> Vec3:
    """캡처 좌표 하나를 레벨 `level`의 월드(미터) 좌표로 옮긴다.

    검증된("solved") 정렬이 없는 레벨은 예외로 멈춘다 — 입력을 그대로 돌려주는
    잠정값은 없다. 지금은 레벨 1만 검증돼 있다.

    Raises:
        LookupError: `level_alignment.json`에 그 레벨 항목이 없다
            (`level_mesh.load_alignment`가 낸다).
        ValueError: 항목은 있지만 `provenance`가 "solved"가 아니다(검증되지 않았다).
    """
    alignment = level_mesh.load_alignment(level)
    if alignment.provenance != "solved":
        raise ValueError(
            f"capture_frame.to_world: level {level} alignment has provenance "
            f"{alignment.provenance!r}, not 'solved' — declared 변환은 검증되지 않았다 "
            "(레벨 1의 declared 변환은 실측과 3.68m 어긋났다, level_mesh.py 참고). "
            "'solved' 항목만 신뢰한다."
        )
    rotation = _rotation_matrix(alignment.rotation)
    rotation_t = rotation_math.transpose(rotation)
    delta: Vec3 = (
        point[0] - alignment.translation[0],
        point[1] - alignment.translation[1],
        point[2] - alignment.translation[2],
    )
    return rotation_math.matvec(rotation_t, delta)
