"""deploy/spark/scripts의 번들 준비·매니페스트·품질 게이트·롤백 설정을 합성 디렉터리로 검증한다."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
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


def _write_runs(root: Path, run_totals: list[list[int]], *, first_question: int = 1) -> None:
    for offset, totals in enumerate(run_totals):
        target = root / f"seed_{42 + offset}" / "eval"
        target.mkdir(parents=True)
        with (target / gate.RESULTS_CSV).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["q_id", "category", "difficulty", "total", "note"])
            for index, total in enumerate(totals, start=first_question):
                writer.writerow([f"AQ{index:03d}", "tool_calling", "easy", total, ""])


def _totals_with_sum(total: int, questions: int = 30) -> list[int]:
    base, remainder = divmod(total, questions)
    return [base + (1 if i < remainder else 0) for i in range(questions)]


def _write_run_sums(root: Path, sums: list[int], *, first_question: int = 1) -> None:
    """실행마다 30문항 총점 합계를 정한다. 실행 평균은 합계/30이다."""
    _write_runs(root, [_totals_with_sum(total) for total in sums], first_question=first_question)


# 실제 NF4 기준선(/output/eval_results/qwen36_27b_v6_1_nf4_baseline_965549e) seed 42–49의 30문항 합계.
# 실행 평균 6.704, SD 0.2149.
NF4_BASELINE_SUMS = [204, 193, 199, 209, 210, 197, 194, 203]


def _gate(tmp_path: Path, **overrides: object) -> tuple[int, dict]:
    kwargs: dict = {"delta": 0.2, "min_runs": 8, "target_runs": 18, "expect_questions": 30}
    kwargs.update(overrides)
    return gate.evaluate(tmp_path / "baseline", tmp_path / "candidate", **kwargs)


@pytest.mark.parametrize(
    ("df", "expected"),
    [
        (1, 12.706), (2, 4.303), (3, 3.182), (4, 2.776), (5, 2.571), (7, 2.365),
        (10, 2.228), (14, 2.145), (30, 2.042), (100, 1.984), (math.inf, 1.960),
    ],
)
def test_student_t_975_quantile_matches_table(df: float, expected: float) -> None:
    assert round(gate.t_quantile_975(df), 3) == expected


def test_student_t_quantile_is_monotone_for_fractional_welch_df() -> None:
    assert gate.t_quantile_975(7) > gate.t_quantile_975(7.5) > gate.t_quantile_975(8)


def test_quality_gate_passes_when_ci_lower_bound_is_above_minus_delta(tmp_path: Path) -> None:
    _write_run_sums(tmp_path / "baseline", [210, 211, 209, 210, 210, 211, 209, 210])
    _write_run_sums(tmp_path / "candidate", [209, 210, 208, 209, 209, 210, 208, 209])

    code, report = _gate(tmp_path)

    assert (code, report["verdict"]) == (0, "pass")
    assert report["ci_low"] > -0.2
    assert report["difference"] == pytest.approx(-1 / 30)
    assert (report["stage"], report["n_baseline"], report["n_candidate"]) == (1, 8, 8)


def test_quality_gate_fails_when_ci_upper_bound_is_below_minus_delta(tmp_path: Path) -> None:
    _write_run_sums(tmp_path / "baseline", [210, 211, 209, 210, 210, 211, 209, 210])
    _write_run_sums(tmp_path / "candidate", [195, 196, 194, 195, 195, 196, 194, 195])

    code, report = _gate(tmp_path)

    assert (code, report["verdict"]) == (1, "fail")
    assert report["ci_high"] < -0.2


def test_quality_gate_is_inconclusive_where_the_old_point_estimate_rule_passed(tmp_path: Path) -> None:
    candidate_sums = [total - 3 for total in NF4_BASELINE_SUMS]  # 평균 0.1 하락, 같은 산포
    _write_run_sums(tmp_path / "baseline", NF4_BASELINE_SUMS)
    _write_run_sums(tmp_path / "candidate", candidate_sums)

    code, report = _gate(tmp_path)

    # 옛 규칙(후보 평균 ≥ 기준선 평균 − Δ)이면 합격이었다.
    assert report["candidate"]["mean"] >= report["baseline"]["mean"] - 0.2
    assert (code, report["verdict"]) == (3, "inconclusive")
    assert report["ci_low"] < -0.2 < report["ci_high"]
    assert report["df"] == pytest.approx(14.0)
    assert report["t_crit"] == pytest.approx(2.1448, abs=1e-3)
    se = math.sqrt(2 * statistics.variance([s / 30 for s in NF4_BASELINE_SUMS]) / 8)
    assert report["se"] == pytest.approx(se)
    assert report["ci_low"] == pytest.approx(-0.1 - report["t_crit"] * se)
    assert "18" in report["decision"] and "--target-runs" in report["decision"]


def test_required_runs_estimate_for_the_measured_nf4_spread_is_about_18(tmp_path: Path) -> None:
    _write_run_sums(tmp_path / "baseline", NF4_BASELINE_SUMS)
    _write_run_sums(tmp_path / "candidate", NF4_BASELINE_SUMS)

    _, report = _gate(tmp_path)

    assert report["baseline"]["sd"] == pytest.approx(0.215, abs=5e-4)
    assert 17 <= report["required_runs_estimate"] <= 19


def test_quality_gate_accepts_unequal_run_counts_with_welch_df(tmp_path: Path) -> None:
    candidate_sums = [203, 200, 206] * 6
    _write_run_sums(tmp_path / "baseline", NF4_BASELINE_SUMS)
    _write_run_sums(tmp_path / "candidate", candidate_sums)

    code, report = _gate(tmp_path)

    assert code != 2, report["problems"]
    assert (report["stage"], report["n_baseline"], report["n_candidate"]) == (1, 8, 18)
    wb = statistics.variance([s / 30 for s in NF4_BASELINE_SUMS]) / 8
    wc = statistics.variance([s / 30 for s in candidate_sums]) / 18
    assert report["df"] == pytest.approx((wb + wc) ** 2 / (wb**2 / 7 + wc**2 / 17))
    assert report["se"] == pytest.approx(math.sqrt(wb + wc))


def test_quality_gate_stage_two_inconclusive_is_left_to_the_user(tmp_path: Path) -> None:
    _write_run_sums(tmp_path / "baseline", [195, 207] * 9)
    _write_run_sums(tmp_path / "candidate", [192, 204] * 9)

    code, report = _gate(tmp_path)

    assert (code, report["verdict"], report["stage"]) == (3, "inconclusive", 2)
    assert "판정 불가 — 사용자 결정" in report["decision"]


def test_quality_gate_refuses_fewer_than_min_runs(tmp_path: Path) -> None:
    _write_runs(tmp_path / "baseline", [_totals_with_sum(210)] * 8)
    _write_runs(tmp_path / "candidate", [_totals_with_sum(240)] * 7)

    code, report = _gate(tmp_path)

    assert (code, report["verdict"]) == (2, "incomplete")
    assert report["problems"] == ["candidate: 7 runs, expected at least 8"]


def test_quality_gate_refuses_different_question_sets(tmp_path: Path) -> None:
    _write_run_sums(tmp_path / "baseline", NF4_BASELINE_SUMS)
    _write_run_sums(tmp_path / "candidate", NF4_BASELINE_SUMS, first_question=2)

    code, report = _gate(tmp_path)

    assert (code, report["problems"]) == (2, ["question ids differ between runs"])


def test_quality_gate_cli_exit_code_and_report_fields(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_run_sums(tmp_path / "baseline", NF4_BASELINE_SUMS)
    _write_run_sums(tmp_path / "candidate", [total - 3 for total in NF4_BASELINE_SUMS])

    code = gate.main(["--baseline-root", str(tmp_path / "baseline"), "--candidate-root", str(tmp_path / "candidate")])

    report = json.loads(capsys.readouterr().out)
    assert code == 3
    assert {
        "stage", "n_baseline", "n_candidate", "difference", "se", "df", "t_crit",
        "ci_low", "ci_high", "delta", "verdict", "required_runs_estimate", "per_question",
    } <= report.keys()
    assert len(report["per_question"]) == 30


def _run_gate_script(tmp_path: Path, env_extra: dict[str, str]) -> list[str]:
    """가짜 docker로 run_fp8_gate.sh를 돌리고 생성 명령이 받은 seed 목록을 돌려준다."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        'echo "$*" >> "$FAKE_LOG"\n'
        "shift 3\n"  # compose -f <file>
        "shift 3\n"  # exec -T agent-api
        'case "$1" in\n'
        "  printenv) echo 0123456789abcdef ;;\n"
        '  test) exec test "$2" "$FAKE_ROOT$3" ;;\n'
        "esac\n"
        "exit 0\n"
    )
    docker.chmod(0o755)
    (tmp_path / "root" / "data" / "eval").mkdir(parents=True)
    (tmp_path / "root" / "data" / "eval" / "eval_set_agentic.jsonl").write_text("{}\n")
    log = tmp_path / "docker.log"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_ROOT": str(tmp_path / "root"),
        "FAKE_LOG": str(log),
        "RUN_NAME": "gate_run",
        **env_extra,
    }
    result = subprocess.run(["bash", str(SCRIPTS / "run_fp8_gate.sh")], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    seeds = []
    for line in log.read_text().splitlines():
        if "generate_agentic_responses.py" in line:
            args = line.split()
            seeds.append(args[args.index("--seed") + 1])
    return seeds


def test_run_fp8_gate_defaults_to_eight_runs(tmp_path: Path) -> None:
    assert _run_gate_script(tmp_path, {}) == [str(seed) for seed in range(42, 50)]


def test_run_fp8_gate_extends_an_existing_run_skipping_scored_seeds(tmp_path: Path) -> None:
    for seed in range(42, 50):
        scored = tmp_path / "root" / "output" / "eval_results" / "gate_run" / f"seed_{seed}" / "eval"
        scored.mkdir(parents=True)
        (scored / gate.RESULTS_CSV).write_text("q_id,total\n")

    assert _run_gate_script(tmp_path, {"RUNS": "18"}) == [str(seed) for seed in range(50, 60)]


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
