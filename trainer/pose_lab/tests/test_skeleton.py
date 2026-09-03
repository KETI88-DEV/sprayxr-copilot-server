import math
from trainer.pose_lab import skeleton


def _frame(head, right, left):
    class F:
        pass
    f = F()
    f.head, f.right, f.left, f.hit, f.yaw = head, right, left, None, 0.0
    return f


def test_kabsch_recovers_a_known_rotation():
    """알려진 90도 회전을 넣고 그것을 되찾는지 본다."""
    src = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)]
    # Y축 둘레 90도: (x,y,z) -> (z, y, -x)
    dst = [(0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    rotation, translation = skeleton.kabsch(src, dst)
    for s, d in zip(src, dst):
        moved = skeleton.apply(rotation, translation, s)
        assert all(abs(a - b) < 1e-9 for a, b in zip(moved, d))


def test_place_skeleton_moves_every_joint_not_just_the_anchors():
    """앵커가 아닌 관절도 같은 강체 변환을 받아야 한다."""
    local = {
        "Head": (0.0, 1.6, 0.0),
        "RightHand": (0.3, 1.2, 0.0),
        "LeftHand": (-0.3, 1.2, 0.0),
        "LeftToe": (-0.1, 0.0, 0.1),
    }
    shifted = _frame((0.0, 1.6, 5.0), (0.3, 1.2, 5.0), (-0.3, 1.2, 5.0))
    placed, residual = skeleton.place_skeleton(local, shifted)
    assert residual < 1e-9
    assert all(abs(a - b) < 1e-9 for a, b in zip(placed["LeftToe"], (-0.1, 0.0, 5.1)))


def test_residual_reports_the_real_disagreement():
    """리그가 낸 손이 실측과 어긋나면 잔차가 그 크기를 낸다."""
    local = {"Head": (0.0, 1.6, 0.0), "RightHand": (0.3, 1.2, 0.0), "LeftHand": (-0.3, 1.2, 0.0)}
    # 오른손만 10cm 벌린 목표 — 강체로는 맞출 수 없다
    stretched = _frame((0.0, 1.6, 0.0), (0.4, 1.2, 0.0), (-0.3, 1.2, 0.0))
    _, residual = skeleton.place_skeleton(local, stretched)
    assert 0.01 < residual < 0.1


def test_missing_left_hand_falls_back_to_two_anchors():
    """왼손이 None이어도 머리+오른손으로 배치한다."""
    local = {"Head": (0.0, 1.6, 0.0), "RightHand": (0.3, 1.2, 0.0), "LeftHand": (-0.3, 1.2, 0.0)}
    f = _frame((0.0, 1.6, 2.0), (0.3, 1.2, 2.0), None)
    placed, residual = skeleton.place_skeleton(local, f)
    assert placed is not None
    assert residual < 1e-6
