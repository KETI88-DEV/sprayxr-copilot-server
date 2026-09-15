"""deploy/spark/scripts의 번들 준비·매니페스트·품질 게이트·롤백 설정을 합성 디렉터리로 검증한다."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "deploy" / "spark" / "scripts"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
DIGEST = "sha256:" + "ab" * 32


def _load(name: str) -> ModuleType:
    module_name = f"spark_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


bm = _load("bundle_manifest")
prepare = _load("prepare_bundle")
gate = _load("quality_gate")


def _snapshot_file(repo: Path, commit: str, rel: str, content: bytes) -> None:
    blob = repo / "blobs" / hashlib.sha256(content).hexdigest()
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(content)
    link = repo / "snapshots" / commit / rel
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(blob, link.parent))


def _ref(repo: Path, revision: str, commit: str) -> None:
    path = repo / "refs" / revision
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(commit)


def _make_sources(root: Path) -> dict[str, Path]:
    hub = root / "hub"
    m3 = hub / "models--BAAI--bge-m3"
    for rel, content in {
        "config.json": b'{"model_type": "xlm-roberta"}',
        "modules.json": b"[]",
        "tokenizer.json": b"{}",
        "pytorch_model.bin": b"PICKLE" * 64,
        "1_Pooling/config.json": b'{"pooling_mode_cls_token": true}',
    }.items():
        _snapshot_file(m3, "main0001", rel, content)
    _ref(m3, "main", "main0001")
    _snapshot_file(m3, "pr130001", "model.safetensors", b"SAFETENSORS-M3")
    _ref(m3, "refs/pr/130", "pr130001")

    reranker = hub / "models--BAAI--bge-reranker-v2-m3"
    for rel, content in {"config.json": b"{}", "tokenizer.json": b"{}", "model.safetensors": b"SAFETENSORS-RR"}.items():
        _snapshot_file(reranker, "rr000001", rel, content)
    _ref(reranker, "main", "rr000001")

    fp8 = root / "fp8"
    fp8.mkdir()
    files = {
        "config.json": b'{"quantization_config": {"quant_method": "fp8"}}',
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
        "chat_template.jinja": b"{{ messages }}",
        "layers-0.safetensors": b"L0" * 100,
        "layers-1.safetensors": b"L1" * 100,
        "model.safetensors.index.json": json.dumps(
            {"weight_map": {"a.weight": "layers-0.safetensors", "b.weight": "layers-1.safetensors"}}
        ).encode(),
    }
    for name, content in files.items():
        (fp8 / name).write_bytes(content)
    (fp8 / "fp8_manifest.json").write_text(
        json.dumps(
            {
                "created_utc": "2026-09-15T11:42:04+00:00",
                "input_merge_manifest_sha256": "d1" * 32,
                "quantized_tensors": 2,
                "total_tensors": 2,
                "verification": {"passed": True},
                "files_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
            }
        )
    )

    index = root / "indexes"
    index.mkdir()
    (index / "faiss.index").write_bytes(b"FAISS" * 10)
    (index / "metadata.json").write_text('{"id_map": []}')
    (index / "faiss.index.bak_prerepair").write_bytes(b"OLD")
    return {"hub": hub, "fp8": fp8, "index": index}


def _prepare_args(sources: dict[str, Path], bundle: Path, *extra: str) -> list[str]:
    return [
        "--bundle-dir", str(bundle),
        "--fp8-dir", str(sources["fp8"]),
        "--hf-hub-cache", str(sources["hub"]),
        "--index-dir", str(sources["index"]),
        "--pose-reference", str(bundle.parent / "absent_pose_reference.json"),
        *extra,
    ]  # fmt: skip


@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    sources = _make_sources(tmp_path / "src")
    target = tmp_path / "bundle"
    assert prepare.main(_prepare_args(sources, target, "--link")) == 0
    return target


def _create(bundle: Path, *extra: str) -> int:
    return bm.main(["create", "--bundle-dir", str(bundle), "--code-commit", COMMIT, "--jobs", "2", *extra])


def _verify(bundle: Path, *extra: str) -> int:
    return bm.main(["verify", "--bundle-dir", str(bundle), "--jobs", "2", *extra])


def test_prepare_builds_an_offline_loadable_layout_without_symlinks(bundle: Path) -> None:
    m3 = bundle / "models" / "bge-m3"
    sources = json.loads((bundle / "bundle_sources.json").read_text())

    assert (m3 / "model.safetensors").read_bytes() == b"SAFETENSORS-M3"
    assert not (m3 / "pytorch_model.bin").exists()
    assert (m3 / "1_Pooling" / "config.json").is_file()
    assert (bundle / "models" / "bge-reranker-v2-m3" / "model.safetensors").read_bytes() == b"SAFETENSORS-RR"
    assert sorted(p.name for p in (bundle / "output" / "indexes").iterdir()) == ["faiss.index", "metadata.json"]
    assert (bundle / "config").is_dir()
    assert not [p for p in bundle.rglob("*") if p.is_symlink()]
    assert (bundle / bm.FP8_MODEL_ROOT / "layers-0.safetensors").stat().st_nlink >= 2
    assert sources["sources"]["embedding"]["excluded"] == ["pytorch_model.bin"]
    assert sources["sources"]["embedding"]["overrides"]["model.safetensors"]["snapshot"] == "pr130001"
    assert any("mcp_pose_reference" in item for item in sources["absent_optional"])


def test_prepare_refuses_a_non_empty_bundle_dir(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path / "src")
    target = tmp_path / "bundle"
    target.mkdir()
    (target / "stale").write_text("x")

    assert prepare.main(_prepare_args(sources, target)) == 1


def test_prepare_fails_when_the_safetensors_revision_is_not_cached(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path / "src")
    (sources["hub"] / "models--BAAI--bge-m3" / "refs" / "refs" / "pr" / "130").unlink()

    assert prepare.main(_prepare_args(sources, tmp_path / "bundle")) == 1
    assert not (tmp_path / "bundle").exists()


def test_manifest_round_trip_records_lineage_and_verifies(bundle: Path) -> None:
    assert _create(bundle, "--image", f"vllm=vllm/vllm-openai:v0.29.0@{DIGEST}", "--image", "web=shipcoat/web:v6.1") == 0
    manifest = json.loads((bundle / bm.MANIFEST_NAME).read_text())

    assert manifest["code_commit"] == COMMIT
    assert manifest["images"]["vllm"] == {"ref": "vllm/vllm-openai:v0.29.0", "digest": DIGEST}
    assert manifest["images"]["web"]["digest"] is None
    assert manifest["provenance"]["model_fp8"]["compared"] == 7
    assert manifest["provenance"]["model_fp8"]["input_merge_manifest_sha256"] == "d1" * 32
    assert manifest["groups"]["embedding"]["files"] == 5
    assert manifest["total_bytes"] == sum(item["bytes"] for item in manifest["files"])
    assert _verify(bundle) == 0
    assert _verify(bundle, "--image-digest", f"vllm={DIGEST}", "--code-commit", COMMIT) == 0


def test_verify_catches_same_size_corruption_only_in_full_mode(bundle: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _create(bundle) == 0
    shard = bundle / bm.FP8_MODEL_ROOT / "layers-1.safetensors"
    data = bytearray(shard.read_bytes())
    data[10] ^= 0xFF
    shard.unlink()  # break the hardlink so the source stays intact
    shard.write_bytes(bytes(data))
    capsys.readouterr()

    assert _verify(bundle, "--quick") == 0
    capsys.readouterr()
    assert _verify(bundle) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["problems"] == [f"sha256 differs: {bm.FP8_MODEL_ROOT}/layers-1.safetensors"]


def test_verify_reports_missing_and_resized_files(bundle: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _create(bundle) == 0
    (bundle / "output" / "indexes" / "metadata.json").unlink()
    faiss_index = bundle / "output" / "indexes" / "faiss.index"
    faiss_index.unlink()
    faiss_index.write_bytes(b"short")
    capsys.readouterr()

    assert _verify(bundle, "--quick") == 1
    problems = json.loads(capsys.readouterr().out)["problems"]
    assert "missing: output/indexes/metadata.json" in problems
    assert any(p.startswith("size differs: output/indexes/faiss.index") for p in problems)


def test_verify_requires_the_selected_model_group(bundle: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _create(bundle) == 0
    capsys.readouterr()

    assert _verify(bundle, "--model", "bf16", "--quick") == 1
    assert "required asset group absent: model_bf16" in json.loads(capsys.readouterr().out)["problems"]


def test_verify_checks_image_digest_and_code_commit(bundle: Path) -> None:
    assert _create(bundle, "--image", "web=shipcoat/web:v6.1") == 0

    assert _verify(bundle, "--quick", "--image-digest", f"web={DIGEST}") == 1
    assert _verify(bundle, "--quick", "--code-commit", "fedcba9") == 1


def test_create_refuses_a_model_that_disagrees_with_its_source_manifest(
    bundle: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = bundle / bm.FP8_MODEL_ROOT / "fp8_manifest.json"
    data = json.loads(manifest_path.read_text())
    data["files_sha256"]["layers-0.safetensors"] = "0" * 64
    manifest_path.unlink()
    manifest_path.write_text(json.dumps(data))
    capsys.readouterr()

    assert _create(bundle) == 1
    problems = json.loads(capsys.readouterr().out)["problems"]
    assert problems == ["model_fp8: layers-0.safetensors differs from fp8_manifest.json"]
    assert not (bundle / bm.MANIFEST_NAME).exists()


def test_create_refuses_a_missing_shard_named_by_the_index(bundle: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (bundle / bm.FP8_MODEL_ROOT / "layers-1.safetensors").unlink()
    capsys.readouterr()

    assert _create(bundle) == 1
    problems = json.loads(capsys.readouterr().out)["problems"]
    assert f"model_fp8: shard {bm.FP8_MODEL_ROOT}/layers-1.safetensors named by the index is missing" in problems


def test_create_refuses_symlinks(bundle: Path) -> None:
    (bundle / "output" / "indexes" / "alias.index").symlink_to("faiss.index")

    assert _create(bundle) == 1


def test_extra_files_warn_and_generated_config_is_ignored(bundle: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _create(bundle) == 0
    (bundle / "config" / "rag_config_spark_vllm_bf16.yaml").write_text("llm: {}")
    (bundle / "notes.txt").write_text("operator note")
    capsys.readouterr()

    assert _verify(bundle, "--quick") == 0
    assert json.loads(capsys.readouterr().out)["warnings"] == ["not in manifest: notes.txt"]


def _write_runs(root: Path, run_totals: list[list[int]]) -> None:
    for offset, totals in enumerate(run_totals):
        target = root / f"seed_{42 + offset}" / "eval"
        target.mkdir(parents=True)
        with (target / gate.RESULTS_CSV).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["q_id", "category", "difficulty", "total", "note"])
            for index, total in enumerate(totals, start=1):
                writer.writerow([f"AQ{index:03d}", "tool_calling", "easy", total, ""])


def _totals_with_sum(total: int, questions: int = 30) -> list[int]:
    base, remainder = divmod(total, questions)
    return [base + (1 if i < remainder else 0) for i in range(questions)]


@pytest.mark.parametrize(
    ("candidate_sum", "expected_code", "verdict"),
    [(204, 0, "pass"), (203, 1, "fail")],  # 6.800 is exactly baseline 7.0 - 0.2; 6.767 is below it
)
def test_quality_gate_threshold_is_baseline_mean_minus_delta(
    tmp_path: Path, candidate_sum: int, expected_code: int, verdict: str
) -> None:
    _write_runs(tmp_path / "baseline", [_totals_with_sum(210)] * 8)
    _write_runs(tmp_path / "candidate", [_totals_with_sum(candidate_sum)] * 8)

    code, report = gate.evaluate(tmp_path / "baseline", tmp_path / "candidate", delta=0.2, expect_runs=8, expect_questions=30)

    assert (code, report["verdict"]) == (expected_code, verdict)
    assert report["baseline"]["mean"] == pytest.approx(7.0)
    assert report["candidate"]["mean"] == pytest.approx(candidate_sum / 30)


def test_quality_gate_refuses_incomplete_runs(tmp_path: Path) -> None:
    _write_runs(tmp_path / "baseline", [_totals_with_sum(210)] * 8)
    _write_runs(tmp_path / "candidate", [_totals_with_sum(240)] * 7)

    code, report = gate.evaluate(tmp_path / "baseline", tmp_path / "candidate", delta=0.2, expect_runs=8, expect_questions=30)

    assert code == 2
    assert report["problems"] == ["candidate: 7 runs, expected 8"]


def test_rollback_config_changes_only_the_served_name_and_tokenizer(tmp_path: Path) -> None:
    render = _load("render_rollback_config")
    source = REPO / "trainer" / "configs" / "rag_config_spark_vllm.yaml"
    output = tmp_path / "rag_config_spark_vllm_bf16.yaml"

    assert render.main(["--source", str(source), "--output", str(output)]) == 0
    original = yaml.safe_load(source.read_text(encoding="utf-8"))
    rendered = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert rendered["llm"]["served_model_name"] == "qwen36-27b-shipcoat-v6-1-bf16"
    assert rendered["llm"]["tokenizer_path"] == "/output/merged/qwen36_27b_shipcoat_v6_1_bf16"
    for section, key in render.CHANGED_KEYS:
        del rendered[section][key]
        del original[section][key]
    assert rendered == original
