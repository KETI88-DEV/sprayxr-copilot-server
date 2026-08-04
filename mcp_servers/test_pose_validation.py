from fastapi.testclient import TestClient

from mcp_servers.main import app


def test_pose_recommend_requires_block_context() -> None:
    response = TestClient(app).post(
        "/pose_estimation/compare",
        json={"mode": "recommend"},
    )

    assert response.status_code == 422


def test_pose_compare_requires_current_pose() -> None:
    response = TestClient(app).post(
        "/pose_estimation/compare",
        json={"mode": "compare", "block_id": "BLK-A"},
    )

    assert response.status_code == 422
