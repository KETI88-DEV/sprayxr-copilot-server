"""Single ordered worker that drains corrected engine and explanation outboxes."""

from __future__ import annotations

import asyncio
import logging

from anyio import to_thread

from trainer.demo_api.engine_gateway import EngineGateway, ExplanationGateway, GatewayError
from trainer.demo_api.engine_store import CorrectedStore, CorrectedStoreError


logger = logging.getLogger(__name__)


class CorrectedOutboxWorker:
    def __init__(
        self,
        store: CorrectedStore,
        engine: EngineGateway,
        explanations: ExplanationGateway,
        *,
        poll_interval_s: float = 0.25,
    ) -> None:
        self._store = store
        self._engine = engine
        self._explanations = explanations
        self._poll_interval_s = max(0.05, poll_interval_s)
        self._wake = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="corrected-v2-outbox")

    def notify(self) -> None:
        self._wake.set()

    async def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._task is not None:
            await self._task
            self._task = None
        await self._engine.close()
        await self._explanations.close()

    async def _run(self) -> None:
        while not self._stop.is_set():
            did_work = await self.run_once()
            if did_work:
                continue
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self._poll_interval_s)
            except TimeoutError:
                pass

    async def run_once(self) -> bool:
        # A completed Zone assessment should trigger its automatic explanation
        # before later lifecycle events race ahead.  Jobs are revision-bound, so
        # prioritizing them does not reorder physical engine input.
        job = await to_thread.run_sync(self._store.claim_explanation)
        if job is not None:
            try:
                explanation = await self._explanations.explain(job.assessment, kind=job.kind)
                await to_thread.run_sync(
                    lambda: self._store.commit_explanation(
                        job.assessment.assessment_id, explanation
                    )
                )
            except GatewayError as exc:
                await to_thread.run_sync(
                    lambda: self._store.mark_explanation_failure(
                        job.assessment.assessment_id,
                        error=str(exc),
                        retryable=exc.retryable,
                    )
                )
                logger.warning(
                    "corrected explanation failed assessment=%s retryable=%s: %s",
                    job.assessment.assessment_id,
                    exc.retryable,
                    exc,
                )
            except CorrectedStoreError as exc:
                await to_thread.run_sync(
                    lambda: self._store.mark_explanation_failure(
                        job.assessment.assessment_id,
                        error=str(exc),
                        retryable=False,
                    )
                )
                logger.error(
                    "corrected explanation persistence rejected assessment=%s: %s",
                    job.assessment.assessment_id,
                    exc,
                )
            except Exception as exc:  # noqa: BLE001 - keep the durable job retryable
                await to_thread.run_sync(
                    lambda: self._store.mark_explanation_failure(
                        job.assessment.assessment_id,
                        error=f"unexpected_worker_error:{type(exc).__name__}",
                        retryable=True,
                    )
                )
                logger.exception(
                    "unexpected corrected explanation worker failure assessment=%s",
                    job.assessment.assessment_id,
                )
            return True

        event = await to_thread.run_sync(self._store.claim_engine_event)
        if event is not None:
            try:
                ack = await self._engine.post_event(event.event)
                assessment = (
                    await self._engine.get_assessment(ack.assessment_id)
                    if ack.assessment_id is not None
                    else None
                )
                engine_session = await self._engine.get_session(event.event.session_id)
                await to_thread.run_sync(
                    lambda: self._store.commit_engine_success(
                        event.event.event_id,
                        ack=ack,
                        assessment=assessment,
                        engine_session=engine_session,
                    )
                )
            except GatewayError as exc:
                await to_thread.run_sync(
                    lambda: self._store.mark_engine_failure(
                        event.event.event_id, error=str(exc), retryable=exc.retryable
                    )
                )
                logger.warning(
                    "corrected engine event failed event=%s retryable=%s: %s",
                    event.event.event_id,
                    exc.retryable,
                    exc,
                )
            except CorrectedStoreError as exc:
                await to_thread.run_sync(
                    lambda: self._store.mark_engine_failure(
                        event.event.event_id, error=str(exc), retryable=False
                    )
                )
                logger.error("corrected engine result persistence rejected event=%s: %s", event.event.event_id, exc)
            except Exception as exc:  # noqa: BLE001 - retain work and retry after an unforeseen local failure
                await to_thread.run_sync(
                    lambda: self._store.mark_engine_failure(
                        event.event.event_id,
                        error=f"unexpected_worker_error:{type(exc).__name__}",
                        retryable=True,
                    )
                )
                logger.exception("unexpected corrected engine worker failure event=%s", event.event.event_id)
            return True

        return False
