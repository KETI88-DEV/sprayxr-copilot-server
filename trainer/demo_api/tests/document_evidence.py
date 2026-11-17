"""Explicit document-backed fixtures for routing tests, not missing-evidence tests."""
from typing import Any

from trainer.demo_api.tests.viewer_record import RecordingPipeline


DOCUMENT_ID = "DOC-routing-fixture-C0001"


def supported_document_result(response: str) -> dict[str, Any]:
    """Give a routing fixture a real cited passage matching its supplied answer.

    Callers deliberately choose this fixture when testing the document route.
    Empty-body/error/evidence-absence fixtures must keep their original result.
    """
    body = response.removeprefix("[답변]").strip()
    if not body:
        raise ValueError("An empty answer is not a supported document fixture")
    return {
        "response": response.rstrip() + "\n[근거] " + DOCUMENT_ID,
        "rag_chunks": [{"chunk_id": DOCUMENT_ID, "text": body, "rerank_score": 9.0}],
    }


class DocumentRecordingPipeline(RecordingPipeline):
    """RecordingPipeline with explicit answer-matching document evidence."""

    def run(self, **kwargs: object) -> dict[str, Any]:
        result = super().run(**kwargs)
        return result if result["rag_chunks"] else supported_document_result(result["response"])
