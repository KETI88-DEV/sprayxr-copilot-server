#!/usr/bin/env python3
# pyright: reportAny=false

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

qfp8 = importlib.import_module("quantize_fp8_block")

QCFG = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
    "modules_to_not_convert": ["model.norm", "lm_head"],
}


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# ------------------------------------------------------------------ numerics


def test_block_split_non_multiple_matches_manual_blocks() -> None:
    w = torch.randn(300, 200, generator=_gen()) * 0.02
    q, s = qfp8.quantize_block_fp8(w)
    assert q.dtype == torch.float8_e4m3fn and q.shape == (300, 200)
    assert s.shape == (3, 2)
    for i in range(3):
        for j in range(2):
            block = w[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128]
            assert s[i, j].item() == pytest.approx(block.abs().max().item() / 448.0, rel=1e-6)
            codes = q[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128].to(torch.float32)
            assert codes.abs().max().item() == 448.0  # the block max hits full scale
    # the last partial block (rows 256..299, cols 128..199) is dequantized with its own scale
    dq = qfp8.dequantize_block_fp8(q, s)
    assert dq.shape == w.shape
    assert torch.allclose(dq[256:, 128:], q[256:, 128:].float() * s[2, 1], rtol=0, atol=0)


def test_exactly_representable_weights_round_trip_bit_exact() -> None:
    g = _gen(1)
    raw = torch.randint(0, 256, (256, 384), generator=g, dtype=torch.uint8)
    values = raw.view(torch.float8_e4m3fn).float()
    values[values.isnan()] = 0.0  # 0x7f/0xff are NaN in e4m3fn
    values[::128, ::128] = 448.0  # full scale in every block
    codes = values.to(torch.float8_e4m3fn)
    scale = 2.0 ** torch.randint(-12, -2, (2, 3), generator=g).float()  # exact in BF16
    w = qfp8.dequantize_block_fp8(codes, scale)
    q, s = qfp8.quantize_block_fp8(w)
    assert torch.equal(s, scale)
    assert torch.equal(q.float(), codes.float())


def test_dequant_error_bounded_by_fp8_grid() -> None:
    w = torch.randn(512, 640, generator=_gen(2)) * 0.03
    w[5, 7] = 0.9  # outlier stretches one block's scale
    q, s = qfp8.quantize_block_fp8(w)
    dq = qfp8.dequantize_block_fp8(q, s.to(torch.bfloat16))
    s_exp = s.repeat_interleave(128, 0).repeat_interleave(128, 1)
    # the coarsest e4m3 step is 32 codes (256..448) -> half-step 16; BF16 scale rounding < 0.4% of 448
    assert bool(((w - dq).abs() <= 18.0 * s_exp).all())
    assert qfp8.relative_error(w, dq) < 0.04
    # direct cast must beat integer rounding (which throws away the sub-integer grid)
    int_round = torch.round(w / s_exp).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s_exp
    assert qfp8.relative_error(w, dq) < qfp8.relative_error(w, int_round)


def test_zero_block_is_safe() -> None:
    w = torch.randn(200, 256, generator=_gen(3))
    w[:128, :128] = 0.0
    q, s = qfp8.quantize_block_fp8(w)
    assert s[0, 0].item() == 1.0 and bool(torch.isfinite(s).all())
    assert bool((q[:128, :128].float() == 0).all())
    dq = qfp8.dequantize_block_fp8(q, s)
    assert bool(torch.isfinite(dq).all()) and bool((dq[:128, :128] == 0).all())
    qz, sz = qfp8.quantize_block_fp8(torch.zeros(5, 5))
    assert bool((qz.float() == 0).all()) and sz.tolist() == [[1.0]]


def test_nonfinite_weight_rejected() -> None:
    w = torch.ones(4, 4)
    w[1, 1] = float("nan")
    with pytest.raises(qfp8.ConversionError):
        _ = qfp8.quantize_block_fp8(w)


# ------------------------------------------------------------------ tiny checkpoint


def _tiny_model(root: Path, *, drop: str | None = None, extra: str | None = None, manifest: object = None):
    g = _gen(4)
    tensors = {
        "model.embed_tokens.weight": torch.randn(50, 16, generator=g).to(torch.bfloat16),
        "model.layers.0.mlp.up_proj.weight": torch.randn(300, 130, generator=g).to(torch.bfloat16),
        "model.layers.0.mlp.down_proj.weight": torch.randn(130, 300, generator=g).to(torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.randn(130, generator=g).to(torch.bfloat16),
        "model.layers.0.linear_attn.A_log": torch.randn(8, generator=g).to(torch.float32),
        "model.norm.weight": torch.randn(130, generator=g).to(torch.bfloat16),
        "lm_head.weight": torch.randn(50, 16, generator=g).to(torch.bfloat16),
    }
    targets = ["model.layers.0.mlp.up_proj.weight", "model.layers.0.mlp.down_proj.weight"]
    ref_map = {k: ("layers-0.safetensors" if ".layers.0." in k else "outside.safetensors") for k in tensors}
    for t in targets:
        ref_map[t + "_scale_inv"] = ref_map[t]

    inp = root / "in"
    inp.mkdir()
    if drop:
        del tensors[drop]
    if extra:
        tensors[extra] = torch.zeros(3, dtype=torch.bfloat16)
    names = sorted(tensors)
    shards = {"model-00001-of-00002.safetensors": names[::2], "model-00002-of-00002.safetensors": names[1::2]}
    weight_map = {}
    for fname, keys in shards.items():
        save_file({k: tensors[k] for k in keys}, str(inp / fname), metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(keys, fname))
    _ = (inp / qfp8.INDEX_NAME).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    input_cfg = {"architectures": ["Tiny"], "model_type": "tiny", "hidden_size": 130}
    _ = (inp / "config.json").write_text(json.dumps(input_cfg))
    _ = (inp / "tokenizer.json").write_text('{"tok": 1}')
    if manifest is not None:
        _ = (inp / qfp8.MERGE_MANIFEST_NAME).write_text(json.dumps(manifest))

    ref_index = root / "ref_index.json"
    _ = ref_index.write_text(json.dumps({"metadata": {}, "weight_map": ref_map}))
    ref_cfg = root / "ref_config.json"
    _ = ref_cfg.write_text(json.dumps({"architectures": ["Tiny"], "quantization_config": QCFG}))
    return inp, ref_index, ref_cfg, input_cfg, targets


def _cli(inp: Path, out: Path, ref_index: Path, ref_cfg: Path, *extra: str) -> int:
    return qfp8.main(
        ["--input", str(inp), "--output", str(out), "--reference-index", str(ref_index),
         "--reference-config", str(ref_cfg), "--threads", "2", *extra]
    )


def test_missing_input_key_fails(tmp_path: Path) -> None:
    inp, ref_index, ref_cfg, _, _ = _tiny_model(tmp_path, drop="model.layers.0.mlp.down_proj.weight")
    with pytest.raises(qfp8.ConversionError, match="missing from input.*down_proj"):
        _ = qfp8.build_plan(inp, ref_index, ref_cfg)


def test_extra_input_key_fails(tmp_path: Path) -> None:
    inp, ref_index, ref_cfg, _, _ = _tiny_model(tmp_path, extra="model.layers.0.unexpected.bias")
    with pytest.raises(qfp8.ConversionError, match="not in reference.*unexpected"):
        _ = qfp8.build_plan(inp, ref_index, ref_cfg)


def test_target_in_modules_to_not_convert_fails(tmp_path: Path) -> None:
    inp, ref_index, ref_cfg, _, _ = _tiny_model(tmp_path)
    cfg = json.loads(ref_cfg.read_text())
    cfg["quantization_config"]["modules_to_not_convert"].append("model.layers.0.mlp.up_proj")
    _ = ref_cfg.write_text(json.dumps(cfg))
    with pytest.raises(qfp8.ConversionError, match="modules_to_not_convert"):
        _ = qfp8.build_plan(inp, ref_index, ref_cfg)


def test_convert_end_to_end(tmp_path: Path) -> None:
    inp, ref_index, ref_cfg, input_cfg, targets = _tiny_model(tmp_path, manifest={"verification": {"passed": True}})
    out = tmp_path / "out"
    assert _cli(inp, out, ref_index, ref_cfg) == 0
    assert not (tmp_path / ".out.tmp").exists()

    ref_map = json.loads(ref_index.read_text())["weight_map"]
    out_index = json.loads((out / qfp8.INDEX_NAME).read_text())
    assert out_index["weight_map"] == ref_map  # key names and file layout follow the reference
    assert sorted(p.name for p in out.glob("*.safetensors")) == ["layers-0.safetensors", "outside.safetensors"]

    in_t = qfp8.load_input_tensors(inp)
    out_t: dict[str, object] = {}
    for f in ("layers-0.safetensors", "outside.safetensors"):
        out_t.update(qfp8.read_header(out / f))
    assert out_index["metadata"]["total_size"] == sum(t.nbytes for t in out_t.values())
    for name, src in in_t.items():
        dst = out_t[name]
        if name in targets:
            assert (dst.dtype, dst.shape) == ("F8_E4M3", src.shape)
            sc = out_t[name + "_scale_inv"]
            assert sc.dtype == "BF16" and sc.shape == (-(-src.shape[0] // 128), -(-src.shape[1] // 128))
        else:
            assert (dst.dtype, dst.shape) == (src.dtype, src.shape)
            assert b"".join(qfp8.iter_file_range(dst.file, dst.start, dst.end)) == b"".join(
                qfp8.iter_file_range(src.file, src.start, src.end)
            )

    out_cfg = json.loads((out / "config.json").read_text())
    assert out_cfg == input_cfg | {"quantization_config": QCFG}
    assert (out / "tokenizer.json").read_text() == '{"tok": 1}'

    manifest = json.loads((out / qfp8.MANIFEST_NAME).read_text())
    assert manifest["verification"]["passed"] is True
    assert manifest["input_merge_manifest_sha256"] == qfp8.sha256_file(inp / qfp8.MERGE_MANIFEST_NAME)
    assert manifest["files_sha256"]["layers-0.safetensors"] == qfp8.sha256_file(out / "layers-0.safetensors")
    assert manifest["created_utc"].endswith("+00:00")
    assert set(manifest["verification"]["per_layer"]) == {"model.layers.0"}

    assert _cli(inp, out, ref_index, ref_cfg, "--verify") == 0


def test_verify_detects_changed_nontarget_byte(tmp_path: Path) -> None:
    inp, ref_index, ref_cfg, _, _ = _tiny_model(tmp_path, manifest={"verified": True})
    out = tmp_path / "out"
    assert _cli(inp, out, ref_index, ref_cfg) == 0
    info = qfp8.read_header(out / "outside.safetensors")["model.norm.weight"]
    with (out / "outside.safetensors").open("r+b") as fh:
        _ = fh.seek(info.start)
        b = fh.read(1)
        _ = fh.seek(info.start)
        _ = fh.write(bytes([b[0] ^ 0xFF]))
    plan = qfp8.build_plan(inp, ref_index, ref_cfg)
    report = qfp8.verify_output(plan, out, log=lambda _m: None)
    assert report["passed"] is False
    assert any("model.norm.weight: non-target bytes differ" in f for f in report["failures"])


@pytest.mark.parametrize("manifest", [None, {"verification": {"passed": False}}, {"verified": True, "passed": False}, {}])
def test_merge_gate_refuses_unverified_input(tmp_path: Path, manifest: object) -> None:
    inp, ref_index, ref_cfg, _, _ = _tiny_model(tmp_path, manifest=manifest)
    out = tmp_path / "out"
    assert _cli(inp, out, ref_index, ref_cfg) == 2
    assert not out.exists() and not (tmp_path / ".out.tmp").exists()
