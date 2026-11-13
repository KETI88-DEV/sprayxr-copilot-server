#!/usr/bin/env python3
"""색인 파이프라인 보수(2026-11-13 점검 결함): 짧은 구역 보존, 글자 단위 자르기, 깨진 글자,
빈 문서 중복 처리, 덮어쓰기 거부, 질문 접두어, 모델·색인 짝 검사."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_index  # noqa: E402
import dedup_docs  # noqa: E402
import search_index  # noqa: E402


class WordTokenizer:
    """공백으로 나뉜 낱말 하나를 토큰 하나로 센다. 글자 위치(offset)도 돌려준다."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [0] * len(text.split())

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False) -> dict[str, Any]:
        return {"offset_mapping": [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]}


TOK = WordTokenizer()
CONFIG = {"chunking": {"min_tokens": 5, "max_tokens": 20, "overlap_fraction": 0.15, "min_chunk_tokens": 8}}


def words(text: str) -> set[str]:
    return set(text.split())


def test_short_sections_join_their_neighbours_instead_of_vanishing() -> None:
    assert build_index.merge_short_sections(["# 저도막", "원인 " * 10, "끝 맺음"], TOK, 8) == [
        "# 저도막\n\n" + "원인 " * 10 + "\n\n끝 맺음",
    ]
    assert build_index.merge_short_sections(["짧은 문서"], TOK, 8) == ["짧은 문서"]


def test_chunk_document_keeps_every_word_of_short_sections_and_tables() -> None:
    content = "\n".join([
        "# 저도막",
        "원인은 희석 과다 이다",
        "# 과도막",
        " ".join(f"과도막{i}" for i in range(30)),
        "# 핀홀",
        "기포 원인",
    ])
    doc = {"doc_id": "doc_x", "content": content, "tables": ["| 등급 | Sa 2 |"]}
    chunks = build_index.chunk_document(doc, TOK, CONFIG)

    got = set().union(*(words(c["text"]) for c in chunks))
    assert words(content) | {"|", "등급", "Sa", "2"} <= got
    assert all(c["tokens"] <= CONFIG["chunking"]["max_tokens"] for c in chunks)
    assert [c["chunk_id"] for c in chunks] == [f"DOC-doc_x-C{i:04d}" for i in range(1, len(chunks) + 1)]


def test_force_split_cuts_on_character_offsets_without_losing_the_tail() -> None:
    text = " ".join(f"도막{i}" for i in range(45))
    pieces = build_index.force_split_oversized([{"text": text, "tokens": 45, "doc_id": "d"}], TOK, 20)
    assert [p["tokens"] for p in pieces] == [20, 20, 5]
    assert " ".join(p["text"] for p in pieces) == text


def test_normalize_glyphs_restores_known_symbols_and_removes_the_rest() -> None:
    assert build_index.normalize_glyphs("정도는 Sa 2  , 조도") == "정도는 Sa 2½, 조도"
    assert build_index.normalize_glyphs(" DFT 40C") == "• DFT 40°C"
    assert build_index.normalize_glyphs("Surface���14") == "Surface 14"
    assert build_index.normalize_glyphs("사이 + 0.5 mm 부식") == "사이  + 0.5 mm  부식"


def test_repair_spacing_respaces_only_glued_korean() -> None:
    pytest.importorskip("kiwipiepy")
    chunks = [{"text": "도막두께가부족하면부식이빨리진행된다", "tokens": 1}, {"text": "도막 두께", "tokens": 2}]
    assert build_index.repair_spacing(chunks, TOK) == 1
    assert chunks[0]["spacing_repaired"] is True and " " in chunks[0]["text"]
    assert chunks[1] == {"text": "도막 두께", "tokens": 2}


def test_save_outputs_refuses_to_overwrite_an_existing_build(tmp_path: Path) -> None:
    existing = tmp_path / "chunks.jsonl"
    existing.write_text("old\n", encoding="utf-8")
    config = {"paths": {"chunks_jsonl": str(existing)}}
    with pytest.raises(FileExistsError):
        build_index.save_outputs([], None, None, config)
    assert existing.read_text(encoding="utf-8") == "old\n"


def _record(doc_id: str, content: str) -> dedup_docs.DocRecord:
    return dedup_docs.DocRecord(path=Path(f"{doc_id}.json"), payload={"doc_id": doc_id, "content": content, "dedup_log": {
        "stage_1_exact": {"removed": False, "duplicate_of": None},
    }})


def test_empty_documents_are_reported_not_deduplicated_into_each_other() -> None:
    records = [_record("doc_0020", ""), _record("doc_0023", "  \n"), _record("doc_0030", "본문")]
    kept, empty = dedup_docs.drop_empty_documents(records)
    assert empty == ["doc_0020", "doc_0023"]
    assert [r.payload["doc_id"] for r in kept] == ["doc_0030"]
    assert dedup_docs.stage_exact_dedup(kept)[1] == 0


def _metadata(tmp_path: Path, model: str) -> str:
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"embedding_model": model, "id_map": []}), encoding="utf-8")
    return str(path)


def test_load_metadata_rejects_an_index_built_with_another_embedder(tmp_path: Path) -> None:
    path = _metadata(tmp_path, "BAAI/bge-m3")
    assert search_index.load_metadata(path, "/models/bge-m3")["embedding_model"] == "BAAI/bge-m3"
    assert search_index.load_metadata(path)["id_map"] == []
    with pytest.raises(ValueError, match="KURE-v3"):
        search_index.load_metadata(path, "/models/KURE-v3")


def test_retrieve_encodes_the_question_with_the_query_prompt() -> None:
    class Embedder:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def encode_query(self, query: str, **_kwargs: Any) -> np.ndarray:
            self.queries.append(query)
            return np.ones(2, dtype=np.float32)

    class Index:
        def search(self, vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
            assert vec.shape == (1, 2)
            return np.array([[0.9]]), np.array([[0]])

    embedder = Embedder()
    id_map = [{"chunk_id": "DOC-a-C0001", "doc_id": "a", "text": "본문", "tokens": 1}]
    results = search_index.retrieve("저도막 원인", Index(), id_map, embedder, top_k=1)
    assert embedder.queries == ["저도막 원인"]
    assert results[0]["chunk_id"] == "DOC-a-C0001"
