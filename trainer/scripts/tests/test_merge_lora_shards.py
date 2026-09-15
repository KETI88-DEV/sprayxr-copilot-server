#!/usr/bin/env python3
# pyright: reportAny=false
"""merge_lora_shards: 합성 2샤드 체크포인트로 병합·검증·실패 경로를 확인한다(실제 모델 불필요)."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

mls = importlib.import_module("merge_lora_shards")
SCRIPT = Path(__file__).resolve().parents[1] / "merge_lora_shards.py"

RANK = 4
ALPHA = 8  # scaling 2.0
HIDDEN = 8
INTER = 12
VOCAB = 16

SHARD1 = "model-00001-of-00002.safetensors"
SHARD2 = "model-00002-of-00002.safetensors"
# (base 키, 어댑터 모듈 이름) — 실제 Qwen3.6 과 같은 이름 변환(model.layers → model.language_model.layers)
TARGETS = {
    "model.language_model.layers.0.self_attn.q_proj.weight": "model.layers.0.self_attn.q_proj",
    "model.language_model.layers.0.mlp.down_proj.weight": "model.layers.0.mlp.down_proj",
    "model.language_model.layers.1.self_attn.q_proj.weight": "model.layers.1.self_attn.q_proj",
    "model.language_model.layers.1.mlp.down_proj.weight": "model.layers.1.mlp.down_proj",
}


def _base_tensors(gen: torch.Generator) -> dict[str, dict[str, torch.Tensor]]:
    def r(*shape: int) -> torch.Tensor:
        return (torch.randn(*shape, generator=gen) * 0.05).to(torch.bfloat16)

    shard1 = {
        "model.language_model.embed_tokens.weight": r(VOCAB, HIDDEN),
        "model.language_model.layers.0.input_layernorm.weight": r(HIDDEN),
        "model.language_model.layers.0.self_attn.q_proj.weight": r(HIDDEN, HIDDEN),
        "model.language_model.layers.0.mlp.down_proj.weight": r(HIDDEN, INTER),
        "model.visual.blocks.0.attn.qkv.weight": r(6, 6),
        "model.visual.merger.norm.weight": r(6),
    }
    shard2 = {
        "model.language_model.layers.1.input_layernorm.weight": r(HIDDEN),
        "model.language_model.layers.1.self_attn.q_proj.weight": r(HIDDEN, HIDDEN),
        "model.language_model.layers.1.mlp.down_proj.weight": r(HIDDEN, INTER),
        "model.language_model.norm.weight": r(HIDDEN),
        "lm_head.weight": r(VOCAB, HIDDEN),
        "mtp.fc.weight": r(HIDDEN, 2 * HIDDEN),
        "mtp.norm.weight": r(HIDDEN),
    }
    return {SHARD1: shard1, SHARD2: shard2}


def make_base(root: Path, seed: int = 0) -> Path:
    base = root / "base"
    base.mkdir()
    gen = torch.Generator().manual_seed(seed)
    shards = _base_tensors(gen)
    weight_map: dict[str, str] = {}
    total = 0
    for name, tensors in shards.items():
        save_file(tensors, str(base / name), metadata={"format": "pt"})
        for k, t in tensors.items():
            weight_map[k] = name
            total += t.numel() * t.element_size()
    (base / mls.INDEX_NAME).write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}))
    (base / "config.json").write_text(json.dumps({"architectures": ["FakeForConditionalGeneration"]}))
    for aux in mls.AUX_FILES_REQUIRED:
        if not (base / aux).exists():
            (base / aux).write_text(f"aux {aux}\n")
    (base / "README.md").write_text("base card\n")
    return base


def make_adapter(
    root: Path,
    *,
    seed: int = 1,
    config_overrides: dict[str, Any] | None = None,
    drop: str | None = None,
    extra: dict[str, torch.Tensor] | None = None,
    targets: dict[str, str] = TARGETS,
    bad_shape_module: str | None = None,
) -> Path:
    adapter = root / "adapter"
    adapter.mkdir()
    gen = torch.Generator().manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}
    for base_key, module in targets.items():
        out_f, in_f = (HIDDEN, INTER) if "down_proj" in base_key else (HIDDEN, HIDDEN)
        if module == bad_shape_module:
            in_f += 1
        tensors[f"base_model.model.{module}.lora_A.weight"] = torch.randn(RANK, in_f, generator=gen) * 0.1
        tensors[f"base_model.model.{module}.lora_B.weight"] = torch.randn(out_f, RANK, generator=gen) * 0.1
    if drop is not None:
        del tensors[drop]
    tensors.update(extra or {})
    save_file(tensors, str(adapter / mls.ADAPTER_WEIGHTS), metadata={"format": "pt"})
    cfg: dict[str, Any] = {
        "peft_type": "LORA",
        "r": RANK,
        "lora_alpha": ALPHA,
        "bias": "none",
        "fan_in_fan_out": False,
        "use_dora": False,
        "use_rslora": False,
        "rank_pattern": {},
        "alpha_pattern": {},
        "modules_to_save": None,
        "target_modules": ["q_proj", "down_proj"],
    }
    cfg.update(config_overrides or {})
    (adapter / mls.ADAPTER_CONFIG).write_text(json.dumps(cfg))
    return adapter


def _raw(path: Path, key: str) -> bytes:
    info = mls.read_header(path).tensors[key]
    with path.open("rb") as f:
        f.seek(info.start)
        return f.read(info.nbytes)


def _merge(tmp_path: Path, **kwargs: Any) -> tuple[Path, Path, Path, dict[str, Any]]:
    base = make_base(tmp_path)
    adapter = make_adapter(tmp_path)
    out = tmp_path / "merged" / "out"
    manifest = mls.run_merge(base, adapter, out, transformers_check=False, threads=2, log=lambda _m: None, **kwargs)
    return base, adapter, out, manifest


def test_merge_then_verify_passes(tmp_path: Path) -> None:
    base, _adapter, out, manifest = _merge(tmp_path)

    assert out.is_dir()
    assert not list(out.parent.glob(".out.partial-*"))
    assert sorted(p.name for p in out.glob("*.safetensors")) == [SHARD1, SHARD2]
    assert (out / mls.MANIFEST_NAME).is_file()
    assert not (out / "README.md").exists()
    assert manifest["scaling"] == 2.0
    assert manifest["merge_dtype_path"] == "fp32->bf16"
    assert manifest["changed_tensor_count"] == 4
    assert manifest["changed_tensors"] == sorted(TARGETS)
    v = manifest["verification"]
    assert v["passed"] is True
    assert v["b_bytes"]["changed"] == 4 and v["b_bytes"]["unchanged_identical"] == 9
    assert v["c_recompute"]["bit_identical"] == 4
    assert v["a_headers"]["prefix_counts"]["mtp."] == 2
    assert v["a_headers"]["prefix_counts"]["model.visual."] == 2
    # manifest 의 sha256 이 실제 파일과 맞는다
    for name, meta in manifest["files"].items():
        assert mls.sha256_file(out / name) == meta["sha256"]
    for shard in (SHARD1, SHARD2):
        assert mls.read_header(out / shard).metadata == {"format": "pt"}
        assert set(mls.read_header(out / shard).tensors) == set(mls.read_header(base / shard).tensors)

    # 독립 재검증도 통과하고 manifest 에 기록된다
    report = mls.run_verify_only(base, tmp_path / "adapter", out, transformers_check=False, threads=2, log=lambda _m: None)
    assert report["passed"] is True and report["manifest_sha256_mismatch"] == []
    assert len(json.loads((out / mls.MANIFEST_NAME).read_text())["verify_history"]) == 1


def test_formula_bit_identical_and_differs_from_base(tmp_path: Path) -> None:
    base, adapter, out, _ = _merge(tmp_path)
    for base_key, module in TARGETS.items():
        shard = SHARD1 if ".layers.0." in base_key else SHARD2
        with safe_open(str(base / shard), "pt") as fb, safe_open(str(adapter / mls.ADAPTER_WEIGHTS), "pt") as fa, safe_open(
            str(out / shard), "pt"
        ) as fo:
            w = fb.get_tensor(base_key)
            a = fa.get_tensor(f"base_model.model.{module}.lora_A.weight")
            b = fa.get_tensor(f"base_model.model.{module}.lora_B.weight")
            expected = (w.float() + 2.0 * (b.float() @ a.float())).to(torch.bfloat16)
            got = fo.get_tensor(base_key)
        assert got.dtype == torch.bfloat16
        assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8))
        assert not torch.equal(got.view(torch.uint8), w.view(torch.uint8))


def test_non_target_bytes_preserved(tmp_path: Path) -> None:
    base, _adapter, out, _ = _merge(tmp_path)
    for shard in (SHARD1, SHARD2):
        for key in mls.read_header(base / shard).tensors:
            if key in TARGETS:
                continue
            assert _raw(out / shard, key) == _raw(base / shard, key), key


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    base = make_base(tmp_path)
    adapter = make_adapter(tmp_path)
    out = tmp_path / "merged" / "out"
    summary = mls.run_dry(base, adapter, out, expect=mls.Expectations(lora_pairs=4, key_count=13, scaling=2.0))
    assert summary["lora_pairs"] == 4 and summary["untouched_tensors"] == 9
    assert summary["module_counts"] == {"down_proj": 2, "q_proj": 2}
    assert not (tmp_path / "merged").exists()


def test_cli_dry_run_and_merge(tmp_path: Path) -> None:
    base = make_base(tmp_path)
    adapter = make_adapter(tmp_path)
    out = tmp_path / "merged" / "out"
    common = [sys.executable, str(SCRIPT), "--base", str(base), "--adapter", str(adapter), "--output", str(out)]
    dry = subprocess.run([*common, "--dry-run", "--expect", "none"], capture_output=True, text=True, check=False)
    assert dry.returncode == 0, dry.stderr
    assert json.loads(dry.stdout)["lora_pairs"] == 4
    # 기본 기대값(Qwen3.6-27B)은 합성 체크포인트에서 실패해야 한다
    strict = subprocess.run([*common, "--dry-run"], capture_output=True, text=True, check=False)
    assert strict.returncode == 1 and "기대값 불일치" in strict.stderr
    run = subprocess.run(
        [*common, "--expect", "none", "--threads", "2", "--skip-transformers-check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert json.loads((out / mls.MANIFEST_NAME).read_text())["verification"]["passed"] is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"use_dora": True},
        {"use_rslora": True},
        {"rank_pattern": {"q_proj": 8}},
        {"alpha_pattern": {"q_proj": 16}},
        {"modules_to_save": ["lm_head"]},
        {"fan_in_fan_out": True},
        {"bias": "all"},
        {"bias": "lora_only"},
    ],
)
def test_unsupported_config_fails(tmp_path: Path, overrides: dict[str, Any]) -> None:
    base = make_base(tmp_path)
    adapter = make_adapter(tmp_path, config_overrides=overrides)
    with pytest.raises(mls.MergeError, match="지원하지 않는 adapter_config"):
        mls.build_plan(base, adapter)


def test_missing_half_of_pair_fails(tmp_path: Path) -> None:
    base = make_base(tmp_path)
    adapter = make_adapter(tmp_path, drop="base_model.model.model.layers.1.mlp.down_proj.lora_B.weight")
    with pytest.raises(mls.MergeError, match="불완전"):
        mls.build_plan(base, adapter)


def test_unconsumed_adapter_tensor_fails(tmp_path: Path) -> None:
    base = make_base(tmp_path)
    adapter = make_adapter(tmp_path, extra={"base_model.model.lm_head.modules_to_save.weight": torch.zeros(VOCAB, HIDDEN)})
    with pytest.raises(mls.MergeError, match="A/B 형식이 아닌"):
        mls.build_plan(base, adapter)


def test_pair_without_base_target_fails(tmp_path: Path) -> None:
    base = make_base(tmp_path)
    targets = {**TARGETS, "model.language_model.layers.1.self_attn.k_proj.weight": "model.layers.1.self_attn.k_proj"}
    adapter = make_adapter(tmp_path, targets=targets, config_overrides={"target_modules": ["q_proj", "k_proj", "down_proj"]})
    with pytest.raises(mls.MergeError, match="정확히 1개"):
        mls.build_plan(base, adapter)


def test_shape_mismatch_fails(tmp_path: Path) -> None:
    base = make_base(tmp_path)
    adapter = make_adapter(tmp_path, bad_shape_module="model.layers.0.self_attn.q_proj")
    with pytest.raises(mls.MergeError, match="shape 불일치"):
        mls.build_plan(base, adapter)


def test_expectation_mismatch_fails(tmp_path: Path) -> None:
    base = make_base(tmp_path)
    adapter = make_adapter(tmp_path)
    plan = mls.build_plan(base, adapter)
    mls.check_expectations(plan, mls.Expectations(lora_pairs=4, prefix_counts={"mtp.": 2, "model.visual.": 2}))
    with pytest.raises(mls.MergeError, match="기대값 불일치"):
        mls.check_expectations(plan, mls.QWEN36_27B_V6_1)


def test_wrong_merge_is_caught_and_output_not_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """병합 함수가 틀리면 (c)가 잡고, 최종 경로도 임시 디렉터리도 남지 않는다."""
    base = make_base(tmp_path)
    adapter = make_adapter(tmp_path)
    out = tmp_path / "merged" / "out"

    def wrong(w: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scaling: float) -> torch.Tensor:
        return w + (scaling * (b @ a)).to(torch.bfloat16)  # BF16 에서 직접 더함

    monkeypatch.setattr(mls, "merge_weight", lambda w, a, b, s: wrong(w, a, b, s * 1.5))
    with pytest.raises(mls.VerificationError) as exc_info:
        mls.run_merge(base, adapter, out, transformers_check=False, threads=2, log=lambda _m: None)
    assert exc_info.value.report["c_recompute"]["bit_identical"] == 0
    assert exc_info.value.report["b_bytes"]["passed"] is True
    assert not out.exists()
    assert not list((tmp_path / "merged").glob(".out.partial-*"))


def test_verify_detects_tampered_non_target(tmp_path: Path) -> None:
    base, adapter, out, _ = _merge(tmp_path)
    info = mls.read_header(out / SHARD2).tensors["mtp.norm.weight"]
    with (out / SHARD2).open("r+b") as f:
        f.seek(info.start)
        first = f.read(1)
        f.seek(info.start)
        f.write(bytes([first[0] ^ 0xFF]))
    with pytest.raises(mls.VerificationError) as exc_info:
        mls.run_verify_only(base, adapter, out, transformers_check=False, threads=2, log=lambda _m: None)
    report = exc_info.value.report
    assert report["b_bytes"]["passed"] is False
    assert SHARD2 in report["manifest_sha256_mismatch"]
    assert json.loads((out / mls.MANIFEST_NAME).read_text())["verify_history"][-1]["passed"] is False


def test_existing_output_refused(tmp_path: Path) -> None:
    base = make_base(tmp_path)
    adapter = make_adapter(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(mls.MergeError, match="이미 있다"):
        mls.run_merge(base, adapter, out, transformers_check=False, threads=2, log=lambda _m: None)


def test_classify_model_keys_separates_ignored_mtp() -> None:
    model = {"model.a.weight": (2, 2), "lm_head.weight": (3, 2)}
    ckpt = {"model.a.weight": (2, 2), "lm_head.weight": (3, 2), "mtp.fc.weight": (2, 4), "mtp.norm.weight": (2,)}
    ok = mls.classify_model_keys(model, ckpt, ignore_unexpected=[r"^mtp.*"])
    assert ok["passed"] is True
    assert ok["missing"] == [] and ok["unexpected_not_ignored"] == []
    assert ok["unexpected_ignored_by_prefix"] == {"mtp": 2}

    no_ignore = mls.classify_model_keys(model, ckpt)
    assert no_ignore["passed"] is False and len(no_ignore["unexpected_not_ignored"]) == 2

    missing = mls.classify_model_keys({**model, "model.visual.x": (1,)}, ckpt, ignore_unexpected=[r"^mtp.*"])
    assert missing["passed"] is False and missing["missing"] == ["model.visual.x"]

    bad_shape = mls.classify_model_keys({**model, "lm_head.weight": (4, 2)}, ckpt, ignore_unexpected=[r"^mtp.*"])
    assert bad_shape["passed"] is False and bad_shape["shape_mismatches"][0]["key"] == "lm_head.weight"
