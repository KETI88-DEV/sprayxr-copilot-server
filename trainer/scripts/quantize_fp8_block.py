#!/usr/bin/env python3
# pyright: reportAny=false
"""Convert a BF16 checkpoint to the official Qwen/Qwen3.6-27B-FP8 block-128 format.

How the format was confirmed (2026-09-15):
- Official shard headers (read via HTTP Range, all 66 files): target weights are F8_E4M3,
  weight_scale_inv is BF16 [ceil(out/128), ceil(in/128)], the other 792 tensors are BF16.
  Files follow the reference index weight_map (layers-N / mtp / outside).
- vLLM main bb55077, fp8_utils.requant_weight_ue8m0_inplace:
  ``w_dq = w_q.to(torch.float32) * s_exp`` -> W ≈ Wq * weight_scale_inv (block-expanded).
- vLLM main bb55077, deep_gemm.per_block_cast_to_fp8:
  ``sf = x_amax / fp8_max`` and ``(x_view * (1.0 / sf)).to(fp8_dtype)`` -> rounds to the
  nearest representable FP8 value, not to integers.
- Official layers.3 k_proj/gate_proj, first 256 rows: this convention (float32 scale, then
  direct cast) reproduces the official code bytes at 99.9% and the BF16 scale at 100%.
  Rounding to integers before the cast matches only 85%.

Modes: conversion (default), --dry-run, --verify (checks an existing output), --report (D1
error comparison: FP8 vs NF4).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)  # 448.0
DEFAULT_BLOCK = (128, 128)
SCALE_SUFFIX = "_scale_inv"
TARGET_DTYPE = "F8_E4M3"
SCALE_DTYPE = "BF16"
INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
MANIFEST_NAME = "fp8_manifest.json"
MERGE_MANIFEST_NAME = "merge_manifest.json"
DEFAULT_MAX_REL_ERR = 0.05
MAX_THREADS = 8
CHUNK = 64 * 1024 * 1024
DEFAULT_REPORT_LAYERS = (2, 3, 34, 35, 62, 63)

CONVENTION: dict[str, Any] = {
    "scale_name": "weight_scale_inv",
    "dequant": "W ≈ Wq.float32 * weight_scale_inv (block-expanded, cropped)",
    "scale": "amax(|W_block|) / finfo(float8_e4m3fn).max (=448), computed in float32",
    "scale_storage_dtype": SCALE_DTYPE,
    "quant": "clamp(W / scale, -448, 448) -> cast to float8_e4m3fn (nearest representable)",
    "zero_block": "amax == 0 -> scale 1.0, all codes 0",
    "block_size": list(DEFAULT_BLOCK),
    "sources": [
        "https://huggingface.co/Qwen/Qwen3.6-27B-FP8 shard headers (HTTP Range)",
        "vllm@bb55077 vllm/model_executor/layers/quantization/utils/fp8_utils.py"
        " requant_weight_ue8m0_inplace",
        "vllm@bb55077 vllm/utils/deep_gemm.py per_block_cast_to_fp8",
    ],
}

_ST_ITEMSIZE = {
    "F8_E4M3": 1, "F8_E5M2": 1, "BOOL": 1, "U8": 1, "I8": 1,
    "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "F32": 4, "I32": 4, "U32": 4, "F64": 8, "I64": 8, "U64": 8,
}
_FLOAT_INPUT_DTYPES = {"BF16", "F16", "F32"}
_LAYER_RE = re.compile(r"^(?P<prefix>.*?layers)\.(?P<idx>\d+)\.")


class ConversionError(RuntimeError):
    pass


# --------------------------------------------------------------------------- numerics


def block_shape(shape: tuple[int, ...] | torch.Size, block: tuple[int, int] = DEFAULT_BLOCK) -> tuple[int, int]:
    return (math.ceil(shape[0] / block[0]), math.ceil(shape[1] / block[1]))


def quantize_block_fp8(
    weight: torch.Tensor, block: tuple[int, int] = DEFAULT_BLOCK
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (float8_e4m3fn codes, float32 scale_inv [ceil(r/br), ceil(c/bc)])."""
    if weight.dim() != 2:
        raise ConversionError(f"2D weight required, got shape {tuple(weight.shape)}")
    w = weight.to(torch.float32)
    if not bool(torch.isfinite(w).all()):
        raise ConversionError("weight contains NaN/Inf")
    rows, cols = w.shape
    br, bc = block
    nr, nc = block_shape(w.shape, block)
    padded = torch.nn.functional.pad(w, (0, nc * bc - cols, 0, nr * br - rows))
    view = padded.view(nr, br, nc, bc)
    amax = view.abs().amax(dim=(1, 3))
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    scaled = (view / scale[:, None, :, None]).clamp_(-FP8_MAX, FP8_MAX)
    codes = scaled.view(nr * br, nc * bc)[:rows, :cols].to(FP8_DTYPE).contiguous()
    return codes, scale


def dequantize_block_fp8(
    codes: torch.Tensor, scale_inv: torch.Tensor, block: tuple[int, int] = DEFAULT_BLOCK
) -> torch.Tensor:
    rows, cols = codes.shape
    if tuple(scale_inv.shape) != block_shape(codes.shape, block):
        raise ConversionError(f"scale shape {tuple(scale_inv.shape)} != {block_shape(codes.shape, block)}")
    s = scale_inv.to(torch.float32)
    expanded = s.repeat_interleave(block[0], dim=0)[:rows].repeat_interleave(block[1], dim=1)[:, :cols]
    return codes.to(torch.float32) * expanded


def relative_error(reference: torch.Tensor, approx: torch.Tensor) -> float:
    ref = reference.to(torch.float64)
    denom = float(torch.linalg.vector_norm(ref))
    diff = float(torch.linalg.vector_norm(ref - approx.to(torch.float64)))
    if denom == 0.0:
        return 0.0 if diff == 0.0 else math.inf
    return diff / denom


# ------------------------------------------------------------------------ safetensors IO


@dataclass(frozen=True)
class TensorInfo:
    file: Path
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int  # absolute byte offset in file
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


def read_header(path: Path) -> dict[str, TensorInfo]:
    with path.open("rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise ConversionError(f"{path}: truncated safetensors header")
        n = struct.unpack("<Q", raw_len)[0]
        if n > 100_000_000:
            raise ConversionError(f"{path}: implausible header length {n}")
        header = json.loads(fh.read(n))
    base = 8 + n
    header.pop("__metadata__", None)
    out: dict[str, TensorInfo] = {}
    for name, meta in header.items():
        s, e = meta["data_offsets"]
        out[name] = TensorInfo(path, name, meta["dtype"], tuple(meta["shape"]), base + s, base + e)
    return out


def iter_file_range(path: Path, start: int, end: int) -> Iterator[bytes]:
    with path.open("rb") as fh:
        fh.seek(start)
        remaining = end - start
        while remaining > 0:
            chunk = fh.read(min(CHUNK, remaining))
            if not chunk:
                raise ConversionError(f"{path}: unexpected EOF")
            remaining -= len(chunk)
            yield chunk


def tensor_bytes(t: torch.Tensor) -> memoryview:
    flat = t.detach().to("cpu").contiguous().reshape(-1)
    return memoryview(flat.view(torch.uint8).numpy())


@dataclass
class OutEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    chunks: Callable[[], Iterator[bytes | memoryview]] = field(repr=False)


def write_safetensors(path: Path, entries: list[OutEntry]) -> None:
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for e in entries:
        header[e.name] = {"dtype": e.dtype, "shape": list(e.shape), "data_offsets": [offset, offset + e.nbytes]}
        offset += e.nbytes
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    with path.open("wb") as fh:
        _ = fh.write(struct.pack("<Q", len(raw)))
        _ = fh.write(raw)
        for e in entries:
            written = 0
            for chunk in e.chunks():
                written += fh.write(chunk)
            if written != e.nbytes:
                raise ConversionError(f"{e.name}: wrote {written} bytes, expected {e.nbytes}")


def load_tensor(info: TensorInfo) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(str(info.file), framework="pt") as fh:
        return fh.get_tensor(info.name)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------- plan


@dataclass
class Plan:
    input_dir: Path
    reference_index: dict[str, str]
    reference_config: dict[str, Any]
    input_config: dict[str, Any]
    input_tensors: dict[str, TensorInfo]
    targets: list[str]
    block: tuple[int, int]
    out_meta: dict[str, tuple[str, tuple[int, ...], int]]
    out_files: dict[str, list[str]]
    reference_headers: dict[str, dict[str, Any]] | None = None

    @property
    def total_size(self) -> int:
        return sum(m[2] for m in self.out_meta.values())


def _fmt_keys(keys: set[str] | list[str], limit: int = 10) -> str:
    items = sorted(keys)
    more = f" ... (+{len(items) - limit})" if len(items) > limit else ""
    return ", ".join(items[:limit]) + more


def load_input_tensors(input_dir: Path) -> dict[str, TensorInfo]:
    index_path = input_dir / INDEX_NAME
    if index_path.is_file():
        weight_map: dict[str, str] = _load_json(index_path)["weight_map"]
        files = sorted(set(weight_map.values()))
    else:
        single = input_dir / "model.safetensors"
        if not single.is_file():
            raise ConversionError(f"{input_dir}: neither {INDEX_NAME} nor model.safetensors exists")
        weight_map = {}
        files = [single.name]
    tensors: dict[str, TensorInfo] = {}
    for fname in files:
        for name, info in read_header(input_dir / fname).items():
            if name in tensors:
                raise ConversionError(f"duplicate tensor {name} in input shards")
            tensors[name] = info
    if weight_map:
        not_in_files = set(weight_map) - set(tensors)
        if not_in_files:
            raise ConversionError(f"input index lists tensors missing from shards: {_fmt_keys(not_in_files)}")
        wrong_file = [k for k, f in weight_map.items() if tensors[k].file.name != f]
        if wrong_file:
            raise ConversionError(f"input index file mismatch: {_fmt_keys(wrong_file)}")
        unindexed = set(tensors) - set(weight_map)
        if unindexed:
            raise ConversionError(f"input shards contain tensors not in index: {_fmt_keys(unindexed)}")
    return tensors


def check_reference_quant_config(ref_config: dict[str, Any]) -> tuple[int, int]:
    q = ref_config.get("quantization_config")
    if not isinstance(q, dict):
        raise ConversionError("reference config has no quantization_config")
    expected = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic"}
    for k, v in expected.items():
        if q.get(k) != v:
            raise ConversionError(f"reference quantization_config.{k}={q.get(k)!r}, expected {v!r}")
    bs = q.get("weight_block_size")
    if not (isinstance(bs, list) and len(bs) == 2 and all(isinstance(x, int) and x > 0 for x in bs)):
        raise ConversionError(f"reference weight_block_size invalid: {bs!r}")
    return (bs[0], bs[1])


def build_plan(
    input_dir: Path,
    reference_index_path: Path,
    reference_config_path: Path,
    reference_headers_path: Path | None = None,
) -> Plan:
    ref_map: dict[str, str] = _load_json(reference_index_path)["weight_map"]
    ref_config: dict[str, Any] = _load_json(reference_config_path)
    block = check_reference_quant_config(ref_config)
    ref_headers = _load_json(reference_headers_path)["tensors"] if reference_headers_path else None

    scale_keys = [k for k in ref_map if k.endswith(SCALE_SUFFIX)]
    targets = sorted(k[: -len(SCALE_SUFFIX)] for k in scale_keys)
    if not targets:
        raise ConversionError("reference index has no weight_scale_inv keys")
    plain_ref = {k for k in ref_map if not k.endswith(SCALE_SUFFIX)}
    missing_target_weights = [t for t in targets if t not in ref_map]
    if missing_target_weights:
        raise ConversionError(f"reference scale without weight: {_fmt_keys(missing_target_weights)}")
    split = [t for t in targets if ref_map[t] != ref_map[t + SCALE_SUFFIX]]
    if split:
        raise ConversionError(f"reference puts weight and scale in different files: {_fmt_keys(split)}")

    not_convert = set(ref_config["quantization_config"].get("modules_to_not_convert") or [])
    excluded_targets = [t for t in targets if t.removesuffix(".weight") in not_convert]
    if excluded_targets:
        raise ConversionError(f"targets listed in modules_to_not_convert: {_fmt_keys(excluded_targets)}")

    input_config: dict[str, Any] = _load_json(input_dir / CONFIG_NAME)
    tensors = load_input_tensors(input_dir)
    input_keys = set(tensors)
    missing = plain_ref - input_keys
    extra = input_keys - plain_ref
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing from input ({len(missing)}): {_fmt_keys(missing)}")
        if extra:
            parts.append(f"not in reference ({len(extra)}): {_fmt_keys(extra)}")
        raise ConversionError("input key set != reference key set; " + "; ".join(parts))

    out_meta: dict[str, tuple[str, tuple[int, ...], int]] = {}
    target_set = set(targets)
    for name in sorted(plain_ref):
        info = tensors[name]
        if info.dtype not in _ST_ITEMSIZE:
            raise ConversionError(f"{name}: unsupported dtype {info.dtype}")
        if name in target_set:
            if info.dtype not in _FLOAT_INPUT_DTYPES or len(info.shape) != 2:
                raise ConversionError(f"target {name}: need 2D float, got {info.dtype} {info.shape}")
            numel = info.shape[0] * info.shape[1]
            out_meta[name] = (TARGET_DTYPE, info.shape, numel)
            bshape = block_shape(info.shape, block)
            out_meta[name + SCALE_SUFFIX] = (SCALE_DTYPE, bshape, bshape[0] * bshape[1] * _ST_ITEMSIZE[SCALE_DTYPE])
        else:
            out_meta[name] = (info.dtype, info.shape, info.nbytes)

    if ref_headers is not None:
        diffs = []
        for name, (dtype, shape, _) in out_meta.items():
            ref = ref_headers.get(name)
            if ref is None or ref["dtype"] != dtype or tuple(ref["shape"]) != tuple(shape):
                diffs.append(f"{name}: planned {dtype}{list(shape)} vs reference {ref and (ref['dtype'], ref['shape'])}")
        if diffs:
            raise ConversionError(f"planned dtype/shape differ from reference headers ({len(diffs)}): " + "; ".join(diffs[:5]))

    out_files: dict[str, list[str]] = {}
    for name in sorted(ref_map):
        out_files.setdefault(ref_map[name], []).append(name)
    return Plan(
        input_dir=input_dir,
        reference_index=ref_map,
        reference_config=ref_config,
        input_config=input_config,
        input_tensors=tensors,
        targets=targets,
        block=block,
        out_meta=out_meta,
        out_files=out_files,
        reference_headers=ref_headers,
    )


def build_output_config(plan: Plan) -> dict[str, Any]:
    cfg = copy.deepcopy(plan.input_config)
    cfg["quantization_config"] = copy.deepcopy(plan.reference_config["quantization_config"])
    return cfg


# ------------------------------------------------------------------------ merge gate


def merge_manifest_status(input_dir: Path) -> tuple[bool, str]:
    """True only if merge_manifest.json exists and records a passed verification
    (at least one explicit True, no explicit False)."""
    path = input_dir / MERGE_MANIFEST_NAME
    if not path.is_file():
        return False, f"{path} missing"
    try:
        data = _load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"{path} unreadable: {exc}"
    if not isinstance(data, dict):
        return False, f"{path}: top level is not an object"
    flags: list[Any] = [data.get("verified"), data.get("verification_passed"), data.get("passed")]
    for key in ("verification", "verify", "validation"):
        v = data.get(key)
        if isinstance(v, dict):
            flags += [v.get("passed"), v.get("ok"), v.get("verified")]
        elif isinstance(v, bool):
            flags.append(v)
    status = data.get("status")
    if isinstance(status, str):
        flags.append(True if status.lower() in {"verified", "passed", "ok"} else None)
    explicit = [f for f in flags if isinstance(f, bool)]
    if explicit and all(explicit):
        return True, f"{path} records verification passed"
    return False, f"{path} does not record a passed verification (flags={explicit})"


# --------------------------------------------------------------------------- convert


def _plain_entry(plan: Plan, name: str) -> OutEntry:
    info = plan.input_tensors[name]
    return OutEntry(name, info.dtype, info.shape, info.nbytes, lambda: iter_file_range(info.file, info.start, info.end))


def write_shards(plan: Plan, out_dir: Path, log: Callable[[str], None] = print) -> None:
    target_set = set(plan.targets)
    done = 0
    t0 = time.time()
    for fname, names in sorted(plan.out_files.items()):
        cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        entries: list[OutEntry] = []
        for name in names:
            dtype, shape, nbytes = plan.out_meta[name]
            if name in target_set:

                def codes_chunks(n: str = name) -> Iterator[memoryview]:
                    w = load_tensor(plan.input_tensors[n])
                    q, s = quantize_block_fp8(w, plan.block)
                    del w
                    cache[n] = (q, s)
                    yield tensor_bytes(q)

                entries.append(OutEntry(name, dtype, shape, nbytes, codes_chunks))
            elif name.endswith(SCALE_SUFFIX) and name[: -len(SCALE_SUFFIX)] in target_set:

                def scale_chunks(n: str = name[: -len(SCALE_SUFFIX)]) -> Iterator[memoryview]:
                    _, s = cache.pop(n)
                    yield tensor_bytes(s.to(torch.bfloat16))

                entries.append(OutEntry(name, dtype, shape, nbytes, scale_chunks))
            else:
                entries.append(_plain_entry(plan, name))
        write_safetensors(out_dir / fname, entries)
        done += 1
        log(f"[write] {fname} ({len(names)} tensors) {done}/{len(plan.out_files)} {time.time() - t0:.0f}s")


def copy_aux_files(input_dir: Path, out_dir: Path) -> list[str]:
    skip = {INDEX_NAME, CONFIG_NAME, MERGE_MANIFEST_NAME, MANIFEST_NAME}
    copied = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix != ".safetensors" and p.name not in skip:
            _ = shutil.copy2(p, out_dir / p.name)
            copied.append(p.name)
    return copied


def _write_json(path: Path, obj: Any) -> None:
    _ = path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def convert(
    plan: Plan,
    output_dir: Path,
    *,
    max_rel_err: float = DEFAULT_MAX_REL_ERR,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    input_dir = plan.input_dir.resolve()
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ConversionError("output must not be inside the input directory")
    if output_dir.exists():
        raise ConversionError(f"{output_dir} already exists")
    tmp = output_dir.parent / f".{output_dir.name}.tmp"
    if tmp.exists():
        raise ConversionError(f"stale temp dir {tmp} exists; inspect and remove it first")
    tmp.mkdir(parents=True)

    write_shards(plan, tmp, log)
    _write_json(tmp / CONFIG_NAME, build_output_config(plan))
    _write_json(
        tmp / INDEX_NAME,
        {"metadata": {"total_size": plan.total_size}, "weight_map": {k: plan.reference_index[k] for k in sorted(plan.reference_index)}},
    )
    aux = copy_aux_files(input_dir, tmp)

    report = verify_output(plan, tmp, max_rel_err=max_rel_err, log=log)
    if not report["passed"]:
        _write_json(tmp / "fp8_verify_failed.json", report)
        raise ConversionError(f"verification failed; output left at {tmp}: {report['failures'][:5]}")

    merge_manifest = input_dir / MERGE_MANIFEST_NAME
    files = {p.name: sha256_file(p) for p in sorted(tmp.iterdir()) if p.is_file()}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "input_merge_manifest": str(merge_manifest) if merge_manifest.is_file() else None,
        "input_merge_manifest_sha256": sha256_file(merge_manifest) if merge_manifest.is_file() else None,
        "output_dir": str(output_dir),
        "convention": CONVENTION | {"block_size": list(plan.block)},
        "quantized_tensors": len(plan.targets),
        "total_tensors": len(plan.reference_index),
        "total_size": plan.total_size,
        "aux_files_copied": aux,
        "verification": report,
        "files_sha256": files,
    }
    _write_json(tmp / MANIFEST_NAME, manifest)
    os.rename(tmp, output_dir)
    log(f"[done] {output_dir}")
    return manifest


# ---------------------------------------------------------------------------- verify


def _layer_label(key: str) -> str:
    m = _LAYER_RE.match(key)
    return f"{m['prefix']}.{m['idx']}" if m else key


def _ranges_equal(a: TensorInfo, b: TensorInfo) -> bool:
    if a.nbytes != b.nbytes:
        return False
    for ca, cb in zip(iter_file_range(a.file, a.start, a.end), iter_file_range(b.file, b.start, b.end), strict=True):
        if ca != cb:
            return False
    return True


def verify_output(
    plan: Plan,
    out_dir: Path,
    *,
    max_rel_err: float = DEFAULT_MAX_REL_ERR,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    failures: list[str] = []
    checks: dict[str, Any] = {}

    out_index = _load_json(out_dir / INDEX_NAME)
    out_map: dict[str, str] = out_index["weight_map"]
    checks["index_keys"] = len(out_map)
    if set(out_map) != set(plan.reference_index):
        failures.append(
            f"index key set != reference: missing {_fmt_keys(set(plan.reference_index) - set(out_map))};"
            f" extra {_fmt_keys(set(out_map) - set(plan.reference_index))}"
        )
    layout_diff = [k for k in out_map if plan.reference_index.get(k) != out_map[k]]
    if layout_diff:
        failures.append(f"index file layout != reference: {_fmt_keys(layout_diff)}")

    out_tensors: dict[str, TensorInfo] = {}
    for fname in sorted(set(out_map.values())):
        hdr = read_header(out_dir / fname)
        expected = {k for k, f in out_map.items() if f == fname}
        if set(hdr) != expected:
            failures.append(f"{fname}: header keys != index keys")
        out_tensors.update(hdr)
    size_on_disk = sum(t.nbytes for t in out_tensors.values())
    if out_index.get("metadata", {}).get("total_size") != size_on_disk:
        failures.append(f"index total_size {out_index.get('metadata')} != tensor bytes {size_on_disk}")

    target_set = set(plan.targets)
    per_layer: dict[str, dict[str, Any]] = {}
    n_bytes_checked = 0
    for name in sorted(plan.reference_index):
        out = out_tensors.get(name)
        if out is None:
            failures.append(f"{name}: absent from output shards")
            continue
        ref_h = plan.reference_headers.get(name) if plan.reference_headers else None
        if ref_h is not None and (ref_h["dtype"] != out.dtype or tuple(ref_h["shape"]) != out.shape):
            failures.append(f"{name}: {out.dtype}{list(out.shape)} != reference {ref_h['dtype']}{ref_h['shape']}")
        if name.endswith(SCALE_SUFFIX):
            base = name[: -len(SCALE_SUFFIX)]
            want = block_shape(plan.input_tensors[base].shape, plan.block)
            if out.dtype != SCALE_DTYPE or out.shape != want:
                failures.append(f"{name}: {out.dtype}{list(out.shape)} expected {SCALE_DTYPE}{list(want)}")
            continue
        src = plan.input_tensors[name]
        if name in target_set:
            if out.dtype != TARGET_DTYPE or out.shape != src.shape:
                failures.append(f"{name}: {out.dtype}{list(out.shape)} expected {TARGET_DTYPE}{list(src.shape)}")
                continue
            scale_info = out_tensors.get(name + SCALE_SUFFIX)
            if scale_info is None:
                continue
            w = load_tensor(src)
            dq = dequantize_block_fp8(load_tensor(out), load_tensor(scale_info), plan.block)
            if not bool(torch.isfinite(dq).all()):
                failures.append(f"{name}: dequantized weight has NaN/Inf")
            err = relative_error(w, dq)
            del w, dq
            label = _layer_label(name)
            row = per_layer.setdefault(label, {"tensors": 0, "max_rel_err": 0.0, "sum_rel_err": 0.0, "by_module": {}})
            row["tensors"] += 1
            row["max_rel_err"] = max(row["max_rel_err"], err)
            row["sum_rel_err"] += err
            row["by_module"][name.removeprefix(label + ".")] = round(err, 6)
            if not err <= max_rel_err:
                failures.append(f"{name}: dequant relative error {err:.4f} > {max_rel_err}")
        else:
            if out.dtype != src.dtype or out.shape != src.shape:
                failures.append(f"{name}: {out.dtype}{list(out.shape)} != input {src.dtype}{list(src.shape)}")
            elif not _ranges_equal(src, out):
                failures.append(f"{name}: non-target bytes differ from input")
            n_bytes_checked += src.nbytes

    for row in per_layer.values():
        row["mean_rel_err"] = round(row.pop("sum_rel_err") / row["tensors"], 6)
        row["max_rel_err"] = round(row["max_rel_err"], 6)
    quantized = sum(r["tensors"] for r in per_layer.values())
    if quantized != len(plan.targets):
        failures.append(f"quantized tensors verified {quantized} != targets {len(plan.targets)}")

    out_cfg = _load_json(out_dir / CONFIG_NAME)
    if out_cfg.get("quantization_config") != plan.reference_config["quantization_config"]:
        failures.append("config quantization_config != reference")
    if {k: v for k, v in out_cfg.items() if k != "quantization_config"} != {
        k: v for k, v in plan.input_config.items() if k != "quantization_config"
    }:
        failures.append("config (excluding quantization_config) != input config")
    ref_cfg_rest = {k: v for k, v in plan.reference_config.items() if k != "quantization_config"}
    cfg_rest = {k: v for k, v in out_cfg.items() if k != "quantization_config"}
    checks["config_keys_differing_from_reference"] = sorted(
        k for k in set(ref_cfg_rest) | set(cfg_rest) if ref_cfg_rest.get(k) != cfg_rest.get(k)
    )

    checks.update(
        quantized_tensors=quantized,
        nontarget_bytes_compared=n_bytes_checked,
        worst_rel_err=max((r["max_rel_err"] for r in per_layer.values()), default=None),
        max_rel_err_limit=max_rel_err,
    )
    report = {"passed": not failures, "failures": failures, "checks": checks, "per_layer": dict(sorted(per_layer.items()))}
    log(f"[verify] passed={report['passed']} failures={len(failures)} worst_rel_err={checks['worst_rel_err']}")
    return report


# ---------------------------------------------------------------------------- report


def measure_d1(
    plan: Plan,
    layers: tuple[int, ...],
    *,
    device: str = "cuda",
    batch: int = 64,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Block-128 FP8 vs bitsandbytes NF4 (blocksize 64, double quant) on the same tensors."""
    import bitsandbytes.functional as bnbf

    cuda = device.startswith("cuda")
    if cuda:
        free, _ = torch.cuda.mem_get_info()
        if free < 3 * 2**30:
            raise ConversionError(f"GPU free memory {free / 2**30:.1f} GiB < 3 GiB; refusing to measure")
        torch.cuda.reset_peak_memory_stats()
    wanted = {f".layers.{i}." for i in layers}
    names = [t for t in plan.targets if t.startswith("model.language_model.") and any(w in t for w in wanted)]
    rows = []
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for name in names:
        # FP8 path and all error arithmetic on CPU (the same quantizer the conversion uses);
        # only NF4 quantize/dequantize touches the GPU, keeping its peak well under 2 GiB.
        w16 = load_tensor(plan.input_tensors[name])
        w = w16.to(torch.float32)
        x = torch.randn(batch, w.shape[1], generator=gen)
        y = x @ w.T
        q, s = quantize_block_fp8(w)
        dq = dequantize_block_fp8(q, s.to(torch.bfloat16))
        del q
        fp8_w = relative_error(w, dq)
        fp8_y = relative_error(y, x @ dq.T)
        del dq
        packed, state = bnbf.quantize_4bit(
            w16.to(device), blocksize=64, compress_statistics=True, quant_type="nf4"
        )
        dq = bnbf.dequantize_4bit(packed, state).to("cpu", torch.float32)
        del packed, state
        if cuda:
            torch.cuda.empty_cache()
        nf4_w = relative_error(w, dq)
        nf4_y = relative_error(y, x @ dq.T)
        del dq, w, w16, x, y
        row = {
            "tensor": name.removeprefix("model.language_model."),
            "shape": list(plan.input_tensors[name].shape),
            "fp8_weight_rel_err": fp8_w,
            "nf4_weight_rel_err": nf4_w,
            "weight_ratio": fp8_w / nf4_w,
            "fp8_output_rel_err": fp8_y,
            "nf4_output_rel_err": nf4_y,
            "output_ratio": fp8_y / nf4_y,
        }
        rows.append(row)
        log(f"[report] {row['tensor']} fp8 {fp8_w:.4f}/{fp8_y:.4f} nf4 {nf4_w:.4f}/{nf4_y:.4f}")
    peak = torch.cuda.max_memory_allocated() if cuda else 0
    # Criterion fixed before measuring: holds iff FP8 error <= 0.5 x NF4 error on every tensor,
    # for both the weight and the random-input output.
    worst_w = max(r["weight_ratio"] for r in rows)
    worst_y = max(r["output_ratio"] for r in rows)
    return {
        "layers": list(layers),
        "tensors": len(rows),
        "batch": batch,
        "seed": seed,
        "gpu_peak_allocated_bytes": peak,
        "criterion": "fp8_rel_err <= 0.5 * nf4_rel_err for every tensor (weight and output)",
        "worst_weight_ratio": worst_w,
        "worst_output_ratio": worst_y,
        "premise_holds": worst_w <= 0.5 and worst_y <= 0.5,
        "rows": rows,
    }


# ------------------------------------------------------------------------------- CLI


def _parse_layers(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _ = p.add_argument("--input", type=Path, required=True, help="BF16 checkpoint directory")
    _ = p.add_argument("--output", type=Path, help="output directory (conversion / --verify)")
    _ = p.add_argument("--reference-index", type=Path, required=True)
    _ = p.add_argument("--reference-config", type=Path, required=True)
    _ = p.add_argument("--reference-headers", type=Path, help="official tensor dtype/shape JSON (optional)")
    _ = p.add_argument("--dry-run", action="store_true", help="validate and print the plan; write nothing")
    _ = p.add_argument("--verify", action="store_true", help="verify an existing --output")
    _ = p.add_argument("--report", action="store_true", help="D1 comparison: FP8 vs NF4 on --input")
    _ = p.add_argument("--report-layers", type=_parse_layers, default=DEFAULT_REPORT_LAYERS)
    _ = p.add_argument("--report-device", default="cuda")
    _ = p.add_argument("--max-rel-err", type=float, default=DEFAULT_MAX_REL_ERR)
    _ = p.add_argument("--threads", type=int, default=MAX_THREADS)
    _ = p.add_argument(
        "--allow-unverified-input",
        action="store_true",
        help="skip the merge_manifest gate (tests / experiments only, never for release output)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.set_num_threads(max(1, min(args.threads, MAX_THREADS)))
    try:
        plan = build_plan(args.input, args.reference_index, args.reference_config, args.reference_headers)
        summary = {
            "targets": len(plan.targets),
            "reference_keys": len(plan.reference_index),
            "output_files": len(plan.out_files),
            "total_size": plan.total_size,
            "block": list(plan.block),
        }
        if args.report:
            print(json.dumps(measure_d1(plan, args.report_layers, device=args.report_device), indent=2))
            return 0
        if args.dry_run:
            gate_ok, gate_msg = merge_manifest_status(args.input)
            print(json.dumps(summary | {"merge_manifest_ok": gate_ok, "merge_manifest": gate_msg}, indent=2))
            return 0
        if args.output is None:
            raise ConversionError("--output is required")
        if args.verify:
            report = verify_output(plan, args.output, max_rel_err=args.max_rel_err)
            print(json.dumps(report, indent=2))
            return 0 if report["passed"] else 1
        gate_ok, gate_msg = merge_manifest_status(args.input)
        if not gate_ok and not args.allow_unverified_input:
            raise ConversionError(f"merge gate: {gate_msg}")
        print(json.dumps(summary, indent=2))
        _ = convert(plan, args.output, max_rel_err=args.max_rel_err)
        return 0
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
