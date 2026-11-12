from __future__ import annotations

import base64
import copy
import hashlib
from pathlib import Path

import anyio
import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.corrected_router import create_corrected_router
from trainer.demo_api.engine_contracts import (
    CONTRACT_VERSION,
    EVALUATION_MODE,
    EngineAssessment,
    EngineEventAck,
    SessionCompleteRequest,
    SessionEndRequest,
    SessionStartRequest,
    ZoneObservationRequest,
    canonical_sha256,
    engine_event_from_request,
)
from trainer.demo_api.engine_gateway import EngineGateway, ExplanationGateway, GatewayError
from trainer.demo_api.engine_store import CorrectedStore, CorrectedStoreConflict
from trainer.demo_api.engine_worker import CorrectedOutboxWorker


def _start() -> SessionStartRequest:
    return SessionStartRequest.model_validate(
        {
            "event_id": "evt-start",
            "session_id": "session-1",
            "contract_version": CONTRACT_VERSION,
            "user_id": "user-1",
            "scenario_id": "scenario-1",
            "started_at": "2026-11-06T00:00:00Z",
            "evaluation_mode": EVALUATION_MODE,
            "target_thickness_um": 100.0,
            "target_source": "simulator_session_target",
            "model_mapping": {
                "provider_model_key": "provider-model",
                "work_id": "work-1",
                "session_family_id": "family-1",
                "round_index": 1,
                "round_id": "round-1",
                "geometry_model_key": "geometry-1",
            },
            "zone_manifest": {
                "manifest_id": "tsb-zones-20261015-v1",
                "sha256": "a" * 64,
                "model_key": "geometry-1",
                "map_nums": [1],
            },
        }
    )


def _observation() -> ZoneObservationRequest:
    raw = b"synthetic-height"
    return ZoneObservationRequest.model_validate(
        {
            "event_id": "evt-zone",
            "session_id": "session-1",
            "contract_version": CONTRACT_VERSION,
            "observation_id": "observation-1",
            "face_id": "face-1",
            "map_num": 1,
            "zone_id": "1",
            "round_id": "round-1",
            "snapshot_id": "snapshot-1",
            "snapshot_revision": 1,
            "captured_at": "2026-11-06T00:00:02Z",
            "height_png": {
                "data_base64": base64.b64encode(raw).decode("ascii"),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "encoding": "rg16_b8",
                "basis": "current_round",
                "width": 2,
                "height": 2,
                "origin": "top_left",
                "crop": {
                    "x0": 0,
                    "y0": 0,
                    "width": 2,
                    "height": 2,
                    "atlas_width": 2,
                    "atlas_height": 2,
                },
            },
        }
    )


def _assessment() -> dict[str, object]:
    return {
        "assessment_id": "assessment-zone-1",
        "session_id": "session-1",
        "revision": 2,
        "assessment_type": "zone",
        "status": "available",
        "evidence_ids": ["evidence-quality-1", "evidence-behavior-1"],
        "metrics": {
            "observed_statistics": {
                "painted_mean_um": 80.0,
                "painted_sum_um": 160.0,
                "painted_pixel_count": 2,
                "unpainted_pixel_count": 2,
                "six_band_counts": {
                    "lt_0_9t": 2,
                    "from_0_9t_to_lt_1_0t": 0,
                    "from_1_0t_to_lt_1_5t": 0,
                    "from_1_5t_to_lt_2_0t": 0,
                    "from_2_0t_to_lt_2_5t": 0,
                    "gte_2_5t": 0,
                },
            },
            "coverage": {
                "zone_pixel_count": 4,
                "observed_zone_pixel_count": 4,
                "fraction": 1.0,
                "full_zone": True,
            },
            "provider_official_summary": None,
            "behavior_statistics": {
                "speed_mm_s": {
                    "count": 1,
                    "mean": 400.0,
                    "minimum": 400.0,
                    "maximum": 400.0,
                    "median": 400.0,
                },
                "distance_mm": None,
            },
        },
        "coaching": {
            "analysis_status": "available",
            "reason_code": None,
            "causes": [
                {
                    "cause_id": "cause-speed-1",
                    "statement": "관측된 속도가 저도막 규칙 후보와 함께 나타났습니다.",
                    "variable": "speed_mm_s",
                    "evidence_ids": ["evidence-behavior-1"],
                    "evidence_kind": "quality_rule_candidate",
                    "causal_status": "candidate_not_established",
                }
            ],
            "actions": [
                {
                    "action_id": "action-speed-1",
                    "statement": "다음 구역에서는 분사 속도를 낮춰 확인하세요.",
                    "variable": "speed_mm_s",
                    "direction": "decrease",
                    "priority": 1,
                    "evidence_ids": ["evidence-behavior-1"],
                    "evidence_kind": "quality_rule_candidate",
                    "target_range": None,
                    "expected_effect": None,
                }
            ],
            "primary_action_id": "action-speed-1",
        },
        "quality": {
            "analysis_status": "available",
            "reason_codes": [],
            "grade": "below",
            "target_thickness_um": 100.0,
            "painted_mean_um": 80.0,
            "painted_pixel_count": 2,
            "zone_pixel_count": 4,
            "unpainted_fraction": 0.5,
            "six_band_counts": {
                "lt_0_9t": 2,
                "from_0_9t_to_lt_1_0t": 0,
                "from_1_0t_to_lt_1_5t": 0,
                "from_1_5t_to_lt_2_0t": 0,
                "from_2_0t_to_lt_2_5t": 0,
                "gte_2_5t": 0,
            },
            "statistics_scope": "whole_zone",
        },
        "evidence": [
            {
                "evidence_id": "evidence-quality-1",
                "evidence_type": "quality_measurement",
                "statement": "도포부 평균은 80.0 um입니다.",
                "value": 80.0,
                "unit": "um",
                "source_event_ids": ["evt-zone"],
                "contract_version": CONTRACT_VERSION,
                "revision": 2,
            },
            {
                "evidence_id": "evidence-behavior-1",
                "evidence_type": "behavior_observation",
                "statement": "분사 프레임의 속도 관측이 있습니다.",
                "value": "observed",
                "unit": None,
                "source_event_ids": ["evt-zone"],
                "contract_version": CONTRACT_VERSION,
                "revision": 2,
            },
        ],
        "zone_key": {
            "model_key": "geometry-1",
            "face_id": "face-1",
            "map_num": 1,
            "zone_id": "1",
        },
        "provenance": {
            "engine_version": "corrected-zone-engine/v1",
            "criterion_version": "tsb-dv10-zone-quality/v1",
            "coaching_rule_version": "candidate-coaching/v1",
            "contract_version": CONTRACT_VERSION,
            "manifest_id": "tsb-zones-20261015-v1",
            "zone_asset_manifest_sha256": "a" * 64,
            "label_map_sha256": "b" * 64,
            "mask_sha256": "c" * 64,
            "source_event_ids": ["evt-start", "evt-zone"],
            "assessment_input_manifest_sha256": "d" * 64,
            "snapshot_id": "snapshot-1",
            "snapshot_revision": 1,
            "captured_at": "2026-11-06T00:00:02Z",
        },
        "created_at": "2026-11-06T00:00:03Z",
    }


def _session_assessment() -> EngineAssessment:
    value = copy.deepcopy(_assessment())
    value.update(
        {
            "assessment_id": "assessment-session-1",
            "revision": 1,
            "assessment_type": "session",
            "zone_key": None,
            "zone_assessments": [
                {
                    "assessment_id": "assessment-zone-1",
                    "revision": 2,
                    "zone_key": _assessment()["zone_key"],
                    "status": "available",
                    "quality": _assessment()["quality"],
                }
            ],
            "missing_or_pending_zones": {"missing": [], "pending": []},
        }
    )
    value["metrics"]["behavior_statistics"] = None  # type: ignore[index]
    value["quality"]["grade"] = None  # type: ignore[index]
    provenance = value["provenance"]
    provenance.update(  # type: ignore[union-attr]
        {
            "label_map_sha256": None,
            "mask_sha256": None,
            "snapshot_id": None,
            "snapshot_revision": None,
            "captured_at": "2026-11-06T00:00:05Z",
        }
    )
    return EngineAssessment.model_validate(value)


def test_worker_persists_assessment_and_qwen_explanation_then_query_reuses_revision(
    tmp_path: Path,
) -> None:
    assessment = _assessment()

    def engine(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=assessment)
        body = __import__("json").loads(request.content)
        assessment_id = "assessment-zone-1" if body["event_type"] == "zone_observation" else None
        return httpx.Response(
            200,
            json={
                "status": "accepted",
                "event_id": body["event_id"],
                "event_type": body["event_type"],
                "session_id": body["session_id"],
                "payload_hash": canonical_sha256(body),
                "assessment_id": assessment_id,
                "session_revision": 2 if assessment_id else 1,
                "processing_status": "processed",
                "evidence_ids": assessment["evidence_ids"] if assessment_id else [],
            },
        )

    def explain(request: httpx.Request) -> httpx.Response:
        body = __import__("json").loads(request.content)
        assert body["assessment"]["assessment_id"] == "assessment-zone-1"
        return httpx.Response(
            200,
            json={
                "status": "generated",
                "text": "저도막 관측 근거에 따라 다음 구역에서 분사 속도를 낮춰 확인하세요.",
                "model": {
                    "name": "qwen3:4b",
                    "digest": "sha256:model",
                    "backend": "ollama",
                    "quantization": "Q4_K_M",
                },
                "evidence_ids": ["evidence-quality-1", "evidence-behavior-1"],
                "selected_action_ids": ["action-speed-1"],
                "selected_cause_ids": ["cause-speed-1"],
                "validation": {
                    "passed": True,
                    "reason_codes": [],
                    "renderer": "qwen_selection_evidence_bound_templates_v1",
                    "free_form_prose_generated": False,
                },
                "inference": {"done": True, "eval_count": 12},
                "audit_id": "audit-1",
                "assessment_id": "assessment-zone-1",
                "revision": 2,
                "kind": "zone",
            },
        )

    store = CorrectedStore(str(tmp_path / "sequences.db"))
    worker = CorrectedOutboxWorker(
        store,
        EngineGateway("http://engine.test", transport=httpx.MockTransport(engine)),
        ExplanationGateway(
            "http://explanation.test", transport=httpx.MockTransport(explain)
        ),
    )
    start = _start()
    observation = _observation()
    store.accept(start, engine_event_from_request(start, source_instance_id="api-instance-1"))
    store.accept(
        observation,
        engine_event_from_request(observation, source_instance_id="api-instance-1"),
    )

    async def drain() -> None:
        assert await worker.run_once()  # start
        assert await worker.run_once()  # observation + assessment/outbox atomically
        assert await worker.run_once()  # Qwen explanation
        assert not await worker.run_once()
        await worker.stop()

    anyio.run(drain)

    saved = store.get_assessment("assessment-zone-1")
    rendered = store.get_explanation("assessment-zone-1")
    assert saved is not None and saved.revision == 2
    assert rendered is not None and rendered.model.name == "qwen3:4b"

    app = FastAPI()
    app.include_router(
        create_corrected_router(
            store=store, worker=worker, source_instance_id="api-instance-1"
        )
    )
    client = TestClient(app)
    response = client.post(
        "/api/v2/copilot/query",
        json={
            "session_id": "session-1",
            "assessment_id": "assessment-zone-1",
            "revision": 2,
            "question": "다음 구역에서 무엇을 바꿔야 하나요?",
        },
    )
    assert response.status_code == 200
    assert response.json()["assessment_id"] == "assessment-zone-1"
    assert response.json()["revision"] == 2
    assert response.json()["answer_status"] == "generated"
    assert "속도" in response.json()["answer"]
    store.close()


def test_engine_gateway_rejects_acknowledgement_for_different_event_content() -> None:
    event = engine_event_from_request(_start(), source_instance_id="api-instance-1")

    def wrong_hash(request: httpx.Request) -> httpx.Response:
        body = __import__("json").loads(request.content)
        return httpx.Response(
            200,
            json={
                "status": "accepted",
                "event_id": body["event_id"],
                "event_type": body["event_type"],
                "session_id": body["session_id"],
                "payload_hash": "0" * 64,
                "assessment_id": None,
                "session_revision": 0,
                "processing_status": "processed",
                "evidence_ids": [],
            },
        )

    async def run() -> None:
        gateway = EngineGateway(
            "http://engine.test", transport=httpx.MockTransport(wrong_hash)
        )
        try:
            with pytest.raises(GatewayError, match="payload hash mismatch") as caught:
                await gateway.post_event(event)
            assert caught.value.retryable is False
        finally:
            await gateway.close()

    anyio.run(run)


@pytest.mark.parametrize(
    ("revision", "evidence_ids", "message", "provide_assessment"),
    [
        (0, [], "requires a zone assessment", False),
        (3, ["evidence-quality-1", "evidence-behavior-1"], "revision mismatch", True),
        (2, ["evidence-quality-1"], "evidence_ids mismatch", True),
    ],
)
def test_store_rejects_acknowledgement_assessment_mismatch(
    tmp_path: Path,
    revision: int,
    evidence_ids: list[str],
    message: str,
    provide_assessment: bool,
) -> None:
    store = CorrectedStore(str(tmp_path / "assessment-binding.db"))
    start = _start()
    observation = _observation()
    for body in (start, observation):
        store.accept(
            body,
            engine_event_from_request(body, source_instance_id="api-instance-1"),
        )

    start_item = store.claim_engine_event()
    assert start_item is not None
    store.commit_engine_success(
        start_item.event.event_id,
        ack=EngineEventAck(
            status="accepted",
            event_id=start_item.event.event_id,
            event_type=start_item.event.event_type,
            session_id=start_item.event.session_id,
            payload_hash=canonical_sha256(start_item.event.model_dump(mode="json")),
            assessment_id=None,
            session_revision=0,
            processing_status="processed",
            evidence_ids=[],
        ),
        assessment=None,
        engine_session={"session_id": "session-1"},
    )

    zone_item = store.claim_engine_event()
    assert zone_item is not None
    assessment = EngineAssessment.model_validate(_assessment())
    bad_ack = EngineEventAck(
        status="accepted",
        event_id=zone_item.event.event_id,
        event_type=zone_item.event.event_type,
        session_id=zone_item.event.session_id,
        payload_hash=canonical_sha256(zone_item.event.model_dump(mode="json")),
        assessment_id=assessment.assessment_id if provide_assessment else None,
        session_revision=revision,
        processing_status="processed",
        evidence_ids=evidence_ids,
    )
    with pytest.raises(CorrectedStoreConflict, match=message):
        store.commit_engine_success(
            zone_item.event.event_id,
            ack=bad_ack,
            assessment=assessment if provide_assessment else None,
            engine_session={"session_id": "session-1"},
        )
    store.close()


def test_latest_assessment_follows_event_order_not_cross_scope_revision(tmp_path: Path) -> None:
    store = CorrectedStore(str(tmp_path / "ordered.db"))
    start = _start()
    observation = _observation()
    end = SessionEndRequest(
        event_id="evt-end",
        session_id="session-1",
        contract_version=CONTRACT_VERSION,
        ended_at="2026-11-06T00:00:04Z",
    )
    complete = SessionCompleteRequest.model_validate(
        {
            "event_id": "evt-complete",
            "session_id": "session-1",
            "contract_version": CONTRACT_VERSION,
            "completed_at": "2026-11-06T00:00:05Z",
            "provider_official_summary": {
                "avg_thickness_um": 80.0,
                "source": "simulator",
                "unit": "um",
                "progress_value": 100.0,
                "progress_unit": "percent_0_100",
            },
        }
    )
    for body in (start, observation, end, complete):
        store.accept(
            body,
            engine_event_from_request(body, source_instance_id="api-instance-1"),
        )

    zone = EngineAssessment.model_validate(_assessment())
    session = _session_assessment()
    expected = {
        "evt-start": (0, None),
        "evt-zone": (2, zone),
        "evt-end": (2, None),
        "evt-complete": (1, session),
    }
    for event_id, (revision, assessment) in expected.items():
        item = store.claim_engine_event()
        assert item is not None and item.event.event_id == event_id
        ack = EngineEventAck(
            status="accepted",
            event_id=event_id,
            event_type=item.event.event_type,
            session_id="session-1",
            payload_hash=canonical_sha256(item.event.model_dump(mode="json")),
            assessment_id=assessment.assessment_id if assessment else None,
            session_revision=revision,
            processing_status="processed",
            evidence_ids=assessment.evidence_ids if assessment else [],
        )
        store.commit_engine_success(
            event_id,
            ack=ack,
            assessment=assessment,
            engine_session={"session_id": "session-1"},
        )

    status = store.session_status("session-1")
    assert status is not None
    assert status["latest_assessment_id"] == "assessment-session-1"
    assert status["latest_assessment_revision"] == 1
    assert status["session_revision"] == 1
    store.close()
