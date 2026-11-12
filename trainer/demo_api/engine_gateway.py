"""HTTP clients for the corrected engine and revision-bound Qwen explanation service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import httpx
from pydantic import ValidationError

from trainer.demo_api.engine_contracts import (
    EngineAssessment,
    EngineEventAck,
    EngineEventEnvelope,
    ExplanationResponse,
    canonical_sha256,
)


@dataclass(frozen=True)
class GatewayError(RuntimeError):
    message: str
    retryable: bool
    status_code: int | None = None

    def __str__(self) -> str:
        return self.message


def _http_error(service: str, response: httpx.Response) -> GatewayError:
    status = response.status_code
    retryable = status in {408, 425, 429} or status >= 500
    # The peer may include untrusted or very large details.  Persist only a
    # bounded status summary in the outbox audit column.
    return GatewayError(
        f"{service} returned HTTP {status}",
        retryable=retryable,
        status_code=status,
    )


class EngineGateway:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not base_url or not base_url.startswith(("http://", "https://")):
            raise ValueError("CORRECTED_ENGINE_URL must be an absolute HTTP URL")
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s),
            transport=transport,
            follow_redirects=False,
            trust_env=False,
        )

    async def post_event(self, event: EngineEventEnvelope) -> EngineEventAck:
        try:
            response = await self._client.post(
                "/internal/v1/events", json=event.model_dump(mode="json")
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise GatewayError(f"engine transport failure: {type(exc).__name__}", True) from exc
        if response.status_code >= 400:
            raise _http_error("engine", response)
        try:
            ack = EngineEventAck.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise GatewayError("engine returned an invalid event acknowledgement", False) from exc
        if (ack.event_id, ack.event_type, ack.session_id) != (
            event.event_id,
            event.event_type,
            event.session_id,
        ):
            raise GatewayError("engine acknowledgement identity mismatch", False)
        if ack.payload_hash != canonical_sha256(event.model_dump(mode="json")):
            raise GatewayError("engine acknowledgement payload hash mismatch", False)
        if ack.processing_status != "processed":
            raise GatewayError("engine acknowledgement is not processed", False)
        return ack

    async def get_assessment(self, assessment_id: str) -> EngineAssessment:
        try:
            response = await self._client.get(f"/internal/v1/assessments/{assessment_id}")
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise GatewayError(f"engine transport failure: {type(exc).__name__}", True) from exc
        if response.status_code >= 400:
            raise _http_error("engine", response)
        try:
            assessment = EngineAssessment.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise GatewayError("engine returned an invalid assessment", False) from exc
        if assessment.assessment_id != assessment_id:
            raise GatewayError("engine assessment identity mismatch", False)
        return assessment

    async def get_session(self, session_id: str) -> dict[str, Any]:
        try:
            response = await self._client.get(f"/internal/v1/sessions/{session_id}")
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise GatewayError(f"engine transport failure: {type(exc).__name__}", True) from exc
        if response.status_code >= 400:
            raise _http_error("engine", response)
        value = response.json()
        if not isinstance(value, dict) or value.get("session_id") != session_id:
            raise GatewayError("engine returned an invalid session status", False)
        return cast(dict[str, Any], value)

    async def health(self) -> dict[str, Any]:
        try:
            response = await self._client.get("/internal/v1/health")
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise GatewayError(f"engine transport failure: {type(exc).__name__}", True) from exc
        if response.status_code >= 400:
            raise _http_error("engine", response)
        value = response.json()
        if not isinstance(value, dict):
            raise GatewayError("engine returned an invalid health response", False)
        return cast(dict[str, Any], value)

    async def close(self) -> None:
        await self._client.aclose()


class ExplanationGateway:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not base_url or not base_url.startswith(("http://", "https://")):
            raise ValueError("CORRECTED_EXPLANATION_URL must be an absolute HTTP URL")
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s),
            transport=transport,
            follow_redirects=False,
            trust_env=False,
        )

    async def explain(
        self,
        assessment: EngineAssessment,
        *,
        kind: str,
    ) -> ExplanationResponse:
        if kind not in {"zone", "session"} or kind != assessment.assessment_type:
            raise GatewayError("explanation kind does not match assessment type", False)
        try:
            response = await self._client.post(
                "/explain",
                json={"assessment": assessment.model_dump(mode="json"), "kind": kind},
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise GatewayError(
                f"explanation transport failure: {type(exc).__name__}", True
            ) from exc
        if response.status_code >= 400:
            raise _http_error("explanation", response)
        try:
            explanation = ExplanationResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise GatewayError("explanation service returned an invalid response", False) from exc
        unknown = set(explanation.evidence_ids) - set(assessment.evidence_ids)
        if unknown:
            raise GatewayError("explanation referenced evidence outside the assessment", False)
        if (
            explanation.assessment_id != assessment.assessment_id
            or explanation.revision != assessment.revision
            or explanation.kind != kind
        ):
            raise GatewayError("explanation assessment binding mismatch", False)
        if explanation.status == "generated" and not explanation.validation.passed:
            raise GatewayError("generated explanation failed service validation", False)
        return explanation

    async def close(self) -> None:
        await self._client.aclose()
