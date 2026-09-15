#!/usr/bin/env python3
"""LoRA 어댑터를 BF16 기반 체크포인트에 샤드 단위로 병합하고 무결성을 검증한다.

transformers 의 merge_and_unload/save_pretrained 경로는 쓰지 않는다. Qwen3.6-27B 에서
ForCausalLM 은 mtp.*·model.visual.* 를 버리고 config 를 텍스트 전용으로 바꾸며,
ForConditionalGeneration + PeftModel 은 어댑터 키 992개 중 0개를 매칭해 LoRA 를 아예
적용하지 않는다. 그래서 safetensors 샤드를 직접 읽고 LoRA 대상 텐서만 바꿔 같은 이름의
샤드로 다시 쓴다. 나머지 텐서는 바이트 그대로 옮긴다.

병합식은 W' = (W.float() + scaling * (B.float() @ A.float())).to(torch.bfloat16) 이다.
BF16 에서 바로 더하면 반올림이 달라 비트가 달라진다.

출력은 같은 부모 아래 임시 디렉터리에 쓰고, 검증을 통과해야 최종 이름으로 rename 한다.
검증(--verify, 병합 직후 자동 실행):
  (a) 샤드별 헤더 키 집합·dtype·shape 가 base 와 같고 mtp.*·model.visual.* 가 남아 있다.
  (b) 텐서 원시 바이트를 비교해 달라진 텐서 집합이 LoRA 대상 집합과 정확히 같다.
  (c) LoRA 대상 전부를 원본에서 다시 계산해 출력과 비트 단위로 같다.
  (d) meta device 에서 transformers 모델을 만들어 missing/unexpected 키를 가른다.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import re
import resource
import shutil
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Mapping

INDEX_NAME: Final = "model.safetensors.index.json"
ADAPTER_WEIGHTS: Final = "adapter_model.safetensors"
ADAPTER_CONFIG: Final = "adapter_config.json"
MANIFEST_NAME: Final = "merge_manifest.json"
MERGE_DTYPE_PATH: Final = "fp32->bf16"
FORMULA: Final = "W' = (W.float() + scaling * (B.float() @ A.float())).to(torch.bfloat16)"
MAX_THREADS: Final = 8
CHUNK_BYTES: Final = 64 * 1024 * 1024

# base 폴더에서 그대로 복사하는 비가중치 파일. 하나라도 없으면 실패한다.
AUX_FILES_REQUIRED: Final[tuple[str, ...]] = (
    "config.json",
    INDEX_NAME,
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "LICENSE",
)
# 있으면 복사한다. configuration.json 은 ModelScope 프레임워크/태스크 메타라 병합 뒤에도 참이다.
AUX_FILES_OPTIONAL: Final[tuple[str, ...]] = ("configuration.json",)
# 복사하지 않는 것과 이유. manifest 에 그대로 적는다.
AUX_EXCLUDED: Final[dict[str, str]] = {
    "README.md": "base 모델 카드다. 병합 모델을 base 로 서술하므로 옮기면 출처가 틀린다.",
    ".cache/": "다운로드 도구 캐시다. 모델 산출물이 아니다.",
    ".gitattributes": "허브 LFS 설정이다. 모델 산출물이 아니다.",
}

PREFIX_GROUPS: Final[tuple[str, ...]] = ("model.language_model.", "model.visual.", "mtp.", "lm_head.")

ADAPTER_KEY_RE: Final = re.compile(r"^base_model\.model\.(?P<module>.+)\.lora_(?P<ab>[AB])\.weight$")


class MergeError(RuntimeError):
    """병합 전제·대응이 깨졌다. 아무것도 최종 경로에 남기지 않는다."""


class VerificationError(RuntimeError):
    """병합 산출물이 검증을 통과하지 못했다."""

    def __init__(self, message: str, report: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.report = dict(report)


# ---------------------------------------------------------------------------
# 기대값(실측 2026-09-15). 스크립트가 다시 단언한다.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Expectations:
    architecture: str | None = None
    shard_count: int | None = None
    key_count: int | None = None
    prefix_counts: Mapping[str, int] | None = None
    total_size: int | None = None
    adapter_tensor_count: int | None = None
    lora_pairs: int | None = None
    scaling: float | None = None
    rank: int | None = None
    # transformers 가 무시하는 unexpected 키의 접두사(정상).
    ignored_unexpected_prefix: str | None = None


QWEN36_27B_V6_1: Final = Expectations(
    architecture="Qwen3_5ForConditionalGeneration",
    shard_count=15,
    key_count=1199,
    prefix_counts={"model.language_model.": 850, "model.visual.": 333, "mtp.": 15, "lm_head.": 1},
    total_size=55_562_855_904,
    adapter_tensor_count=992,
    lora_pairs=496,
    scaling=2.0,
    rank=16,
    ignored_unexpected_prefix="mtp.",
)
EXPECTATION_PRESETS: Final[dict[str, Expectations]] = {
    "qwen36-27b-v6.1": QWEN36_27B_V6_1,
    "none": Expectations(),
}


def _require(cond: bool, message: str, exc: type[Exception] = MergeError) -> None:
    if not cond:
        raise exc(message)


# ---------------------------------------------------------------------------
# safetensors 헤더
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    start: int  # 파일 안 절대 오프셋
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Header:
    tensors: dict[str, TensorInfo]
    metadata: dict[str, str]


def read_header(path: Path) -> Header:
    with path.open("rb") as f:
        raw_len = f.read(8)
        _require(len(raw_len) == 8, f"safetensors 헤더 길이를 읽지 못했다: {path}")
        (n,) = struct.unpack("<Q", raw_len)
        header = json.loads(f.read(n))
    metadata = header.pop("__metadata__", None) or {}
    base = 8 + n
    tensors = {
        k: TensorInfo(
            dtype=v["dtype"],
            shape=tuple(int(x) for x in v["shape"]),
            start=base + int(v["data_offsets"][0]),
            end=base + int(v["data_offsets"][1]),
        )
        for k, v in header.items()
    }
    return Header(tensors=tensors, metadata=dict(metadata))


def prefix_counts(keys: Iterable[str]) -> dict[str, int]:
    counts = {p: 0 for p in PREFIX_GROUPS}
    counts["other"] = 0
    for k in keys:
        for p in PREFIX_GROUPS:
            if k.startswith(p):
                counts[p] += 1
                break
        else:
            counts["other"] += 1
    return counts


# ---------------------------------------------------------------------------
# adapter_config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoraConfig:
    rank: int
    alpha: float
    scaling: float
    target_modules: tuple[str, ...] | None


def load_lora_config(adapter_dir: Path) -> LoraConfig:
    cfg = json.loads((adapter_dir / ADAPTER_CONFIG).read_text(encoding="utf-8"))
    problems: list[str] = []

    def bad(name: str, cond: bool) -> None:
        if cond:
            problems.append(f"{name}={cfg.get(name)!r}")

    bad("peft_type", cfg.get("peft_type") != "LORA")
    bad("use_dora", bool(cfg.get("use_dora")))
    bad("use_rslora", bool(cfg.get("use_rslora")))
    bad("use_qalora", bool(cfg.get("use_qalora")))
    bad("rank_pattern", bool(cfg.get("rank_pattern")))
    bad("alpha_pattern", bool(cfg.get("alpha_pattern")))
    bad("modules_to_save", bool(cfg.get("modules_to_save")))
    bad("fan_in_fan_out", bool(cfg.get("fan_in_fan_out")))
    bad("bias", cfg.get("bias", "none") != "none")
    bad("lora_bias", bool(cfg.get("lora_bias")))
    bad("target_parameters", bool(cfg.get("target_parameters")))
    bad("trainable_token_indices", bool(cfg.get("trainable_token_indices")))
    bad("layer_replication", bool(cfg.get("layer_replication")))
    bad("alora_invocation_tokens", bool(cfg.get("alora_invocation_tokens")))
    if problems:
        raise MergeError("지원하지 않는 adapter_config 옵션: " + ", ".join(problems))

    rank = cfg.get("r")
    alpha = cfg.get("lora_alpha")
    _require(isinstance(rank, int) and rank > 0, f"r 이 양의 정수가 아니다: {rank!r}")
    _require(isinstance(alpha, (int, float)) and alpha > 0, f"lora_alpha 가 양수가 아니다: {alpha!r}")
    targets = cfg.get("target_modules")
    target_modules = tuple(targets) if isinstance(targets, list) else None
    return LoraConfig(rank=rank, alpha=float(alpha), scaling=float(alpha) / rank, target_modules=target_modules)


# ---------------------------------------------------------------------------
# 대응 계획
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoraPair:
    base_key: str
    a_key: str
    b_key: str
    shard: str
    out_features: int
    in_features: int


@dataclass
class MergePlan:
    base_dir: Path
    adapter_dir: Path
    lora: LoraConfig
    weight_map: dict[str, str]
    total_size: int
    shard_headers: dict[str, Header]
    adapter_header: Header
    pairs: dict[str, LoraPair]  # base_key -> pair
    architecture: str | None = None
    module_counts: dict[str, int] = field(default_factory=dict)

    @property
    def shards(self) -> list[str]:
        return sorted(self.shard_headers)

    def pairs_in_shard(self, shard: str) -> dict[str, LoraPair]:
        return {k: p for k, p in self.pairs.items() if p.shard == shard}


def _resolve_base_key(module: str, base_keys: set[str]) -> str:
    candidates = [f"{module}.weight"]
    if module.startswith("model.") and not module.startswith("model.language_model."):
        candidates.append("model.language_model." + module[len("model.") :] + ".weight")
    found = [c for c in candidates if c in base_keys]
    _require(len(found) == 1, f"LoRA 모듈 {module!r} 의 base 키 후보 {candidates} 중 존재 {found} — 정확히 1개여야 한다")
    return found[0]


def build_plan(base_dir: Path, adapter_dir: Path) -> MergePlan:
    lora = load_lora_config(adapter_dir)
    index = json.loads((base_dir / INDEX_NAME).read_text(encoding="utf-8"))
    weight_map: dict[str, str] = dict(index["weight_map"])
    total_size = int(index.get("metadata", {}).get("total_size", -1))

    shard_headers = {s: read_header(base_dir / s) for s in sorted(set(weight_map.values()))}
    header_keys: dict[str, str] = {}
    for shard, h in shard_headers.items():
        for k in h.tensors:
            _require(k not in header_keys, f"키 {k} 가 샤드 {header_keys.get(k)} 와 {shard} 에 중복된다")
            header_keys[k] = shard
    _require(header_keys == weight_map, "base index 의 weight_map 이 샤드 헤더와 다르다")
    actual_size = sum(t.nbytes for h in shard_headers.values() for t in h.tensors.values())
    _require(actual_size == total_size, f"index total_size {total_size} != 샤드 텐서 바이트 합 {actual_size}")

    config_path = base_dir / "config.json"
    architecture = None
    if config_path.exists():
        archs = json.loads(config_path.read_text(encoding="utf-8")).get("architectures") or []
        architecture = archs[0] if archs else None

    adapter_header = read_header(adapter_dir / ADAPTER_WEIGHTS)
    grouped: dict[str, dict[str, str]] = collections.defaultdict(dict)
    unmatched: list[str] = []
    for key in adapter_header.tensors:
        m = ADAPTER_KEY_RE.match(key)
        if m is None:
            unmatched.append(key)
            continue
        grouped[m.group("module")][m.group("ab")] = key
    _require(not unmatched, f"LoRA A/B 형식이 아닌 어댑터 키 {len(unmatched)}개: {sorted(unmatched)[:5]}")

    base_keys = set(weight_map)
    pairs: dict[str, LoraPair] = {}
    consumed: list[str] = []
    module_counts: collections.Counter[str] = collections.Counter()
    for module in sorted(grouped):
        ab = grouped[module]
        _require(set(ab) == {"A", "B"}, f"LoRA 쌍이 불완전하다: {module} 에 {sorted(ab)} 만 있다")
        base_key = _resolve_base_key(module, base_keys)
        _require(base_key not in pairs, f"base 키 {base_key} 에 LoRA 쌍이 두 번 대응한다")
        leaf = module.rsplit(".", 1)[-1]
        if lora.target_modules is not None:
            _require(leaf in lora.target_modules, f"{module} 의 모듈 {leaf} 가 target_modules 에 없다")

        shard = weight_map[base_key]
        w = shard_headers[shard].tensors[base_key]
        a = adapter_header.tensors[ab["A"]]
        b = adapter_header.tensors[ab["B"]]
        _require(w.dtype == "BF16", f"{base_key} dtype {w.dtype} — BF16 만 병합한다")
        _require(a.dtype == "F32" and b.dtype == "F32", f"{module} LoRA dtype A={a.dtype} B={b.dtype} — F32 만 받는다")
        _require(len(w.shape) == 2, f"{base_key} shape {w.shape} 가 2차원이 아니다")
        out_f, in_f = w.shape
        _require(
            a.shape == (lora.rank, in_f) and b.shape == (out_f, lora.rank),
            f"{module} shape 불일치: W={w.shape} A={a.shape} B={b.shape} r={lora.rank}",
        )
        pairs[base_key] = LoraPair(base_key, ab["A"], ab["B"], shard, out_f, in_f)
        consumed.extend((ab["A"], ab["B"]))
        module_counts[leaf] += 1

    # 모든 어댑터 텐서가 정확히 한 번 소비됐는지.
    _require(len(consumed) == len(set(consumed)), "어댑터 텐서가 두 번 소비됐다")
    leftover = set(adapter_header.tensors) - set(consumed)
    _require(not leftover, f"소비되지 않은 어댑터 텐서 {len(leftover)}개: {sorted(leftover)[:5]}")
    _require(len(consumed) == 2 * len(pairs), "소비 텐서 수가 쌍 수의 두 배가 아니다")

    return MergePlan(
        base_dir=base_dir,
        adapter_dir=adapter_dir,
        lora=lora,
        weight_map=weight_map,
        total_size=total_size,
        shard_headers=shard_headers,
        adapter_header=adapter_header,
        pairs=pairs,
        architecture=architecture,
        module_counts=dict(sorted(module_counts.items())),
    )


def check_expectations(plan: MergePlan, expect: Expectations) -> None:
    keys = list(plan.weight_map)
    checks: list[tuple[str, object, object]] = [
        ("architecture", expect.architecture, plan.architecture),
        ("shard_count", expect.shard_count, len(plan.shard_headers)),
        ("key_count", expect.key_count, len(keys)),
        ("total_size", expect.total_size, plan.total_size),
        ("adapter_tensor_count", expect.adapter_tensor_count, len(plan.adapter_header.tensors)),
        ("lora_pairs", expect.lora_pairs, len(plan.pairs)),
        ("scaling", expect.scaling, plan.lora.scaling),
        ("rank", expect.rank, plan.lora.rank),
    ]
    if expect.prefix_counts is not None:
        counts = prefix_counts(keys)
        for p, n in expect.prefix_counts.items():
            checks.append((f"prefix[{p}]", n, counts.get(p)))
    wrong = [f"{name}: 기대 {want!r} 실측 {got!r}" for name, want, got in checks if want is not None and want != got]
    _require(not wrong, "기대값 불일치 — " + "; ".join(wrong))


def plan_summary(plan: MergePlan) -> dict[str, Any]:
    targets_per_shard = collections.Counter(p.shard for p in plan.pairs.values())
    return {
        "architecture": plan.architecture,
        "shards": len(plan.shard_headers),
        "base_keys": len(plan.weight_map),
        "prefix_counts": prefix_counts(plan.weight_map),
        "total_size": plan.total_size,
        "adapter_tensors": len(plan.adapter_header.tensors),
        "lora_pairs": len(plan.pairs),
        "unconsumed_adapter_tensors": 0,
        "rank": plan.lora.rank,
        "lora_alpha": plan.lora.alpha,
        "scaling": plan.lora.scaling,
        "module_counts": plan.module_counts,
        "targets_per_shard": dict(sorted(targets_per_shard.items())),
        "untouched_tensors": len(plan.weight_map) - len(plan.pairs),
    }


# ---------------------------------------------------------------------------
# 병합
# ---------------------------------------------------------------------------


def merge_weight(w: Any, a: Any, b: Any, scaling: float) -> Any:
    import torch

    return (w.float() + scaling * (b.float() @ a.float())).to(torch.bfloat16)


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def merge_shards(plan: MergePlan, out_dir: Path, log: Callable[[str], None] = print) -> list[str]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    adapter_path = plan.adapter_dir / ADAPTER_WEIGHTS
    changed: list[str] = []
    for i, shard in enumerate(plan.shards, 1):
        t0 = time.monotonic()
        shard_pairs = plan.pairs_in_shard(shard)
        tensors: dict[str, Any] = {}
        with safe_open(str(plan.base_dir / shard), framework="pt") as fb, safe_open(
            str(adapter_path), framework="pt"
        ) as fa:
            for key in fb.keys():
                t = fb.get_tensor(key)
                pair = shard_pairs.get(key)
                if pair is not None:
                    t = merge_weight(t, fa.get_tensor(pair.a_key), fa.get_tensor(pair.b_key), plan.lora.scaling)
                    changed.append(key)
                tensors[key] = t
        out_path = out_dir / shard
        save_file(tensors, str(out_path), metadata={"format": "pt"})
        del tensors
        _fsync_file(out_path)
        log(f"[merge] {i}/{len(plan.shards)} {shard}: LoRA {len(shard_pairs)}개 적용, {time.monotonic() - t0:.1f}s")
    _require(sorted(changed) == sorted(plan.pairs), "병합 중 적용된 텐서 집합이 계획과 다르다")
    return sorted(changed)


def copy_aux_files(base_dir: Path, out_dir: Path) -> list[str]:
    missing = [n for n in AUX_FILES_REQUIRED if not (base_dir / n).is_file()]
    _require(not missing, f"base 에 필수 비가중치 파일이 없다: {missing}")
    copied: list[str] = []
    for name in (*AUX_FILES_REQUIRED, *AUX_FILES_OPTIONAL):
        src = base_dir / name
        if src.is_file():
            shutil.copyfile(src, out_dir / name)
            _fsync_file(out_dir / name)
            copied.append(name)
    return copied


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------


def _bytes_equal(fa: Any, fb: Any, a: TensorInfo, b: TensorInfo) -> bool:
    if a.nbytes != b.nbytes:
        return False
    fa.seek(a.start)
    fb.seek(b.start)
    remaining = a.nbytes
    while remaining:
        n = min(CHUNK_BYTES, remaining)
        if fa.read(n) != fb.read(n):
            return False
        remaining -= n
    return True


def classify_model_keys(
    model_shapes: Mapping[str, tuple[int, ...]],
    ckpt_shapes: Mapping[str, tuple[int, ...]],
    ignore_unexpected: Iterable[str] = (),
    ignore_missing: Iterable[str] = (),
    tied_keys: Iterable[str] = (),
) -> dict[str, Any]:
    ign_u = [re.compile(p) for p in ignore_unexpected]
    ign_m = [re.compile(p) for p in ignore_missing]
    tied = set(tied_keys)
    missing = sorted(k for k in model_shapes if k not in ckpt_shapes)
    missing_real = [k for k in missing if k not in tied and not any(p.search(k) for p in ign_m)]
    unexpected = sorted(k for k in ckpt_shapes if k not in model_shapes)
    ignored = [k for k in unexpected if any(p.search(k) for p in ign_u)]
    not_ignored = [k for k in unexpected if k not in set(ignored)]
    mismatches = [
        {"key": k, "checkpoint": list(ckpt_shapes[k]), "model": list(model_shapes[k])}
        for k in sorted(set(model_shapes) & set(ckpt_shapes))
        if tuple(ckpt_shapes[k]) != tuple(model_shapes[k])
    ]
    ignored_prefixes = collections.Counter(k.split(".", 1)[0] for k in ignored)
    return {
        "model_key_count": len(model_shapes),
        "checkpoint_key_count": len(ckpt_shapes),
        "missing": missing_real,
        "missing_tied_or_ignored": [k for k in missing if k not in set(missing_real)],
        "unexpected_ignored_count": len(ignored),
        "unexpected_ignored_by_prefix": dict(ignored_prefixes),
        "unexpected_ignored": ignored,
        "unexpected_not_ignored": not_ignored,
        "shape_mismatches": mismatches,
        "passed": not missing_real and not not_ignored and not mismatches,
    }


def transformers_key_check(output_dir: Path, ckpt_shapes: Mapping[str, tuple[int, ...]]) -> dict[str, Any]:
    import torch
    import transformers
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(str(output_dir))
    arch = (config.architectures or [None])[0]
    _require(arch is not None and hasattr(transformers, arch), f"transformers 에 아키텍처 {arch!r} 가 없다")
    cls = getattr(transformers, arch)
    with torch.device("meta"):
        model = cls._from_config(config)
    model_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    tie = bool(getattr(config.get_text_config(), "tie_word_embeddings", False)) or bool(
        getattr(config, "tie_word_embeddings", False)
    )
    tied_keys = list((getattr(model, "_tied_weights_keys", None) or {})) if tie else []
    result = classify_model_keys(
        model_shapes,
        ckpt_shapes,
        ignore_unexpected=getattr(model, "_keys_to_ignore_on_load_unexpected", None) or (),
        ignore_missing=getattr(model, "_keys_to_ignore_on_load_missing", None) or (),
        tied_keys=tied_keys,
    )
    result.update(
        {
            "architecture": arch,
            "device": "meta",
            "ignore_unexpected_patterns": list(getattr(model, "_keys_to_ignore_on_load_unexpected", None) or ()),
            "tie_word_embeddings": tie,
            "note": (
                "unexpected_ignored 는 모델 클래스의 _keys_to_ignore_on_load_unexpected 가 로딩 때 버리는 키다. "
                "mtp.* 는 transformers 추론에서 쓰지 않을 뿐 체크포인트에는 있어야 정상이다(MTP 추론기용)."
            ),
        }
    )
    del model
    return result


def verify_output(
    plan: MergePlan,
    output_dir: Path,
    *,
    expect: Expectations = Expectations(),
    transformers_check: bool = True,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    report: dict[str, Any] = {"output_dir": str(output_dir)}
    failures: list[str] = []
    t_all = time.monotonic()

    # (a) 헤더
    t0 = time.monotonic()
    a_fail: list[str] = []
    out_index = json.loads((output_dir / INDEX_NAME).read_text(encoding="utf-8"))
    if dict(out_index["weight_map"]) != plan.weight_map:
        a_fail.append("출력 index weight_map 이 base 와 다르다")
    out_headers: dict[str, Header] = {}
    for shard in plan.shards:
        path = output_dir / shard
        if not path.is_file():
            a_fail.append(f"출력 샤드가 없다: {shard}")
            continue
        oh = read_header(path)
        out_headers[shard] = oh
        bh = plan.shard_headers[shard]
        if set(oh.tensors) != set(bh.tensors):
            a_fail.append(f"{shard}: 키 집합이 다르다 (+{len(set(oh.tensors) - set(bh.tensors))} -{len(set(bh.tensors) - set(oh.tensors))})")
        for k in set(oh.tensors) & set(bh.tensors):
            o, b = oh.tensors[k], bh.tensors[k]
            if (o.dtype, o.shape) != (b.dtype, b.shape):
                a_fail.append(f"{k}: dtype/shape {o.dtype}{o.shape} != base {b.dtype}{b.shape}")
        if oh.metadata.get("format") != "pt":
            a_fail.append(f"{shard}: metadata format {oh.metadata!r}")
    extra_shards = sorted(
        p.name for p in output_dir.glob("*.safetensors") if p.name not in plan.shard_headers
    )
    if extra_shards:
        a_fail.append(f"base 에 없는 샤드가 출력에 있다: {extra_shards}")
    out_keys = [k for h in out_headers.values() for k in h.tensors]
    base_counts = prefix_counts(plan.weight_map)
    out_counts = prefix_counts(out_keys)
    if out_counts != base_counts:
        a_fail.append(f"접두사별 키 수 {out_counts} != base {base_counts}")
    if expect.key_count is not None and len(out_keys) != expect.key_count:
        a_fail.append(f"출력 키 {len(out_keys)} != 기대 {expect.key_count}")
    if expect.prefix_counts is not None:
        for p, n in expect.prefix_counts.items():
            if out_counts.get(p) != n:
                a_fail.append(f"출력 {p}* {out_counts.get(p)} != 기대 {n}")
    out_size = sum(t.nbytes for h in out_headers.values() for t in h.tensors.values())
    if out_size != plan.total_size:
        a_fail.append(f"출력 텐서 바이트 합 {out_size} != base total_size {plan.total_size}")
    report["a_headers"] = {
        "shards": len(out_headers),
        "keys": len(out_keys),
        "prefix_counts": out_counts,
        "tensor_bytes": out_size,
        "failures": a_fail,
        "passed": not a_fail,
        "seconds": round(time.monotonic() - t0, 2),
    }
    log(f"[verify a] 키 {len(out_keys)} 샤드 {len(out_headers)} {out_counts} → {'통과' if not a_fail else a_fail[:3]}")
    failures += [f"(a) {m}" for m in a_fail]

    # (b) 원시 바이트
    t0 = time.monotonic()
    differing: list[str] = []
    if not a_fail:
        for shard in plan.shards:
            bh, oh = plan.shard_headers[shard], out_headers[shard]
            with (plan.base_dir / shard).open("rb") as fb, (output_dir / shard).open("rb") as fo:
                for k in sorted(bh.tensors, key=lambda key: bh.tensors[key].start):
                    if not _bytes_equal(fb, fo, bh.tensors[k], oh.tensors[k]):
                        differing.append(k)
        target_set = set(plan.pairs)
        diff_set = set(differing)
        b_fail = []
        if diff_set != target_set:
            b_fail.append(
                f"달라진 텐서 집합이 LoRA 대상과 다르다: 대상인데 같음 {len(target_set - diff_set)} "
                f"{sorted(target_set - diff_set)[:3]}, 대상 아닌데 다름 {len(diff_set - target_set)} {sorted(diff_set - target_set)[:3]}"
            )
        report["b_bytes"] = {
            "changed": len(diff_set),
            "unchanged_identical": len(plan.weight_map) - len(diff_set),
            "lora_targets": len(target_set),
            "failures": b_fail,
            "passed": not b_fail,
            "seconds": round(time.monotonic() - t0, 2),
        }
        log(f"[verify b] 다름 {len(diff_set)} / 바이트 동일 {len(plan.weight_map) - len(diff_set)} → {'통과' if not b_fail else b_fail}")
        failures += [f"(b) {m}" for m in b_fail]
    else:
        report["b_bytes"] = {"skipped": "(a) 실패", "passed": False}

    # (c) 재계산 비트 일치 — merge_weight 와 따로 식을 적는다(병합 함수가 틀려도 잡히도록).
    t0 = time.monotonic()
    if not a_fail:
        mismatched: list[str] = []
        adapter_path = plan.adapter_dir / ADAPTER_WEIGHTS
        checked = 0
        with safe_open(str(adapter_path), framework="pt") as fa:
            for shard in plan.shards:
                shard_pairs = plan.pairs_in_shard(shard)
                if not shard_pairs:
                    continue
                with safe_open(str(plan.base_dir / shard), framework="pt") as fb, safe_open(
                    str(output_dir / shard), framework="pt"
                ) as fo:
                    for key, pair in sorted(shard_pairs.items()):
                        w = fb.get_tensor(key)
                        a = fa.get_tensor(pair.a_key)
                        b = fa.get_tensor(pair.b_key)
                        ref = (w.to(torch.float32) + plan.lora.scaling * (b.to(torch.float32) @ a.to(torch.float32))).to(
                            torch.bfloat16
                        )
                        out = fo.get_tensor(key)
                        same = out.dtype == torch.bfloat16 and out.shape == ref.shape and torch.equal(
                            out.contiguous().view(torch.uint8), ref.contiguous().view(torch.uint8)
                        )
                        if not same:
                            mismatched.append(key)
                        checked += 1
        c_fail = [f"재계산과 비트가 다른 텐서 {len(mismatched)}개: {mismatched[:3]}"] if mismatched else []
        if checked != len(plan.pairs):
            c_fail.append(f"재계산 {checked} != LoRA 대상 {len(plan.pairs)}")
        report["c_recompute"] = {
            "checked": checked,
            "bit_identical": checked - len(mismatched),
            "formula": FORMULA,
            "scaling": plan.lora.scaling,
            "torch_threads": torch.get_num_threads(),
            "failures": c_fail,
            "passed": not c_fail,
            "seconds": round(time.monotonic() - t0, 2),
        }
        log(f"[verify c] 재계산 {checked} 중 비트 일치 {checked - len(mismatched)} → {'통과' if not c_fail else c_fail}")
        failures += [f"(c) {m}" for m in c_fail]
    else:
        report["c_recompute"] = {"skipped": "(a) 실패", "passed": False}

    # (d) transformers meta 로딩 키 대조
    t0 = time.monotonic()
    if transformers_check:
        ckpt_shapes = {k: t.shape for h in out_headers.values() for k, t in h.tensors.items()}
        try:
            d = transformers_key_check(output_dir, ckpt_shapes)
        except Exception as exc:  # 보고서에 남기고 실패로 친다
            d = {"error": f"{type(exc).__name__}: {exc}", "passed": False}
        d_fail = [] if d.get("passed") else ["transformers 키 대조 실패"]
        if (
            d.get("passed")
            and expect.ignored_unexpected_prefix is not None
            and expect.prefix_counts is not None
        ):
            want = expect.prefix_counts.get(expect.ignored_unexpected_prefix)
            ignored = d.get("unexpected_ignored", [])
            if not all(k.startswith(expect.ignored_unexpected_prefix) for k in ignored) or len(ignored) != want:
                d_fail.append(f"무시된 unexpected 키가 {expect.ignored_unexpected_prefix}* {want}개가 아니다: {len(ignored)}")
        d["seconds"] = round(time.monotonic() - t0, 2)
        d["failures"] = d_fail
        d["passed"] = not d_fail
        report["d_transformers_meta"] = d
        log(
            f"[verify d] model 키 {d.get('model_key_count')} missing {len(d.get('missing', []))} "
            f"unexpected(무시·정상) {d.get('unexpected_ignored_count')} {d.get('unexpected_ignored_by_prefix')} "
            f"unexpected(기타) {len(d.get('unexpected_not_ignored', []))} → {'통과' if not d_fail else d_fail}"
        )
        failures += [f"(d) {m}" for m in d_fail]
    else:
        report["d_transformers_meta"] = {"skipped": "--skip-transformers-check", "passed": None}

    # 비가중치 파일은 base 와 바이트 동일해야 한다.
    aux_fail = [
        name
        for name in (*AUX_FILES_REQUIRED, *AUX_FILES_OPTIONAL)
        if (plan.base_dir / name).is_file()
        and (not (output_dir / name).is_file() or (output_dir / name).read_bytes() != (plan.base_dir / name).read_bytes())
    ]
    report["aux_files"] = {"mismatched_or_missing": aux_fail, "passed": not aux_fail}
    failures += [f"(aux) {n} 가 base 와 다르거나 없다" for n in aux_fail]

    report["failures"] = failures
    report["passed"] = not failures
    report["seconds"] = round(time.monotonic() - t_all, 2)
    report["verified_at"] = _utc_now()
    if failures:
        raise VerificationError("검증 실패: " + "; ".join(failures[:5]), report)
    return report


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            h.update(chunk)
    return h.hexdigest()


def hash_output_files(out_dir: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for p in sorted(out_dir.iterdir()):
        if p.name == MANIFEST_NAME or not p.is_file():
            continue
        files[p.name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return files


def tool_versions() -> dict[str, str]:
    import safetensors
    import torch

    versions = {"python": sys.version.split()[0], "torch": torch.__version__, "safetensors": safetensors.__version__}
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except ImportError:
        pass
    return versions


def peak_rss_mib() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def _write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _fsync_file(tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------


def run_merge(
    base_dir: Path,
    adapter_dir: Path,
    output_dir: Path,
    *,
    expect: Expectations = Expectations(),
    threads: int = MAX_THREADS,
    transformers_check: bool = True,
    keep_partial: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    import torch

    _require(1 <= threads <= MAX_THREADS, f"threads 는 1..{MAX_THREADS} 이어야 한다: {threads}")
    torch.set_num_threads(threads)
    output_dir = output_dir.absolute()
    _require(not output_dir.exists(), f"출력 경로가 이미 있다: {output_dir}")

    t_start = time.monotonic()
    started_at = _utc_now()
    plan = build_plan(base_dir, adapter_dir)
    check_expectations(plan, expect)
    summary = plan_summary(plan)
    log(f"[plan] {json.dumps(summary, ensure_ascii=False)}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output_dir.parent).free
    need = int(plan.total_size * 1.02) + 64 * 1024 * 1024
    _require(free >= need, f"디스크 여유 {free} < 필요 {need}")

    tmp_dir = output_dir.parent / f".{output_dir.name}.partial-{os.getpid()}"
    _require(not tmp_dir.exists(), f"임시 경로가 이미 있다: {tmp_dir}")
    tmp_dir.mkdir()
    try:
        t0 = time.monotonic()
        changed = merge_shards(plan, tmp_dir, log=log)
        copied = copy_aux_files(base_dir, tmp_dir)
        merge_seconds = time.monotonic() - t0
        log(f"[merge] 완료 {merge_seconds:.1f}s, 바뀐 텐서 {len(changed)}, 비가중치 {len(copied)}개 복사")

        verification = verify_output(
            plan, tmp_dir, expect=expect, transformers_check=transformers_check, log=log
        )

        t0 = time.monotonic()
        files = hash_output_files(tmp_dir)
        hash_seconds = time.monotonic() - t0
        manifest = {
            "created_at": started_at,
            "completed_at": _utc_now(),
            "output_dir": str(output_dir),
            "base": {"path": str(base_dir.absolute()), "architecture": plan.architecture},
            "adapter": {
                "path": str(adapter_dir.absolute()),
                "adapter_model_sha256": sha256_file(adapter_dir / ADAPTER_WEIGHTS),
                "adapter_config_sha256": sha256_file(adapter_dir / ADAPTER_CONFIG),
                "rank": plan.lora.rank,
                "lora_alpha": plan.lora.alpha,
            },
            "scaling": plan.lora.scaling,
            "merge_dtype_path": MERGE_DTYPE_PATH,
            "formula": FORMULA,
            "plan": summary,
            "changed_tensor_count": len(changed),
            "changed_tensors": changed,
            "aux_files_copied": copied,
            "aux_files_excluded": AUX_EXCLUDED,
            "files": files,
            "total_bytes": sum(f["bytes"] for f in files.values()),
            "tool_versions": tool_versions(),
            "runtime": {
                "torch_threads": threads,
                "nice": os.nice(0),
                "merge_seconds": round(merge_seconds, 1),
                "verify_seconds": verification["seconds"],
                "hash_seconds": round(hash_seconds, 1),
                "total_seconds": round(time.monotonic() - t_start, 1),
                "peak_rss_mib": peak_rss_mib(),
            },
            "verification": verification,
        }
        _write_json_atomic(tmp_dir / MANIFEST_NAME, manifest)
        _fsync_dir(tmp_dir)
        _require(not output_dir.exists(), f"출력 경로가 병합 중에 생겼다: {output_dir}")
        os.rename(tmp_dir, output_dir)
        _fsync_dir(output_dir.parent)
    except BaseException:
        if keep_partial:
            log(f"[fail] 임시 산출물을 남긴다: {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    log(f"[done] {output_dir} ({manifest['total_bytes'] / 1e9:.2f} GB, {manifest['runtime']['total_seconds']}s)")
    return manifest


def run_verify_only(
    base_dir: Path,
    adapter_dir: Path,
    output_dir: Path,
    *,
    expect: Expectations = Expectations(),
    threads: int = MAX_THREADS,
    transformers_check: bool = True,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    import torch

    _require(1 <= threads <= MAX_THREADS, f"threads 는 1..{MAX_THREADS} 이어야 한다: {threads}")
    torch.set_num_threads(threads)
    plan = build_plan(base_dir, adapter_dir)
    check_expectations(plan, expect)
    manifest_path = output_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    hash_mismatch: list[str] = []
    try:
        report = verify_output(plan, output_dir, expect=expect, transformers_check=transformers_check, log=log)
        error: VerificationError | None = None
    except VerificationError as exc:
        report, error = exc.report, exc
    if manifest.get("files"):
        current = hash_output_files(output_dir)
        hash_mismatch = sorted(n for n in set(current) | set(manifest["files"]) if current.get(n) != manifest["files"].get(n))
        report["manifest_sha256_mismatch"] = hash_mismatch
        if hash_mismatch:
            report["passed"] = False
            report.setdefault("failures", []).append(f"manifest sha256 불일치: {hash_mismatch[:5]}")
    report["peak_rss_mib"] = peak_rss_mib()
    if manifest:
        manifest.setdefault("verify_history", []).append(
            {"verified_at": report.get("verified_at"), "passed": report.get("passed"), "seconds": report.get("seconds")}
        )
        manifest["verification"] = report
        _write_json_atomic(manifest_path, manifest)
    if error is not None or hash_mismatch:
        raise VerificationError("검증 실패: " + "; ".join(report.get("failures", [])[:5]), report)
    return report


def run_dry(base_dir: Path, adapter_dir: Path, output_dir: Path, *, expect: Expectations) -> dict[str, Any]:
    plan = build_plan(base_dir, adapter_dir)
    check_expectations(plan, expect)
    missing_aux = [n for n in AUX_FILES_REQUIRED if not (base_dir / n).is_file()]
    _require(not missing_aux, f"base 에 필수 비가중치 파일이 없다: {missing_aux}")
    probe = output_dir.absolute()
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    summary = plan_summary(plan)
    summary.update(
        {
            "dry_run": True,
            "output_exists": output_dir.exists(),
            "disk_free_bytes": free,
            "disk_needed_bytes": int(plan.total_size * 1.02),
            "disk_ok": free >= int(plan.total_size * 1.02),
            "aux_files_present": [n for n in (*AUX_FILES_REQUIRED, *AUX_FILES_OPTIONAL) if (base_dir / n).is_file()],
        }
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="대응·shape·기대값만 검사하고 아무것도 쓰지 않는다")
    mode.add_argument("--verify", action="store_true", help="이미 있는 출력을 검증하고 manifest 를 갱신한다")
    parser.add_argument(
        "--expect",
        choices=sorted(EXPECTATION_PRESETS),
        default="qwen36-27b-v6.1",
        help="실측 기대값 묶음을 단언한다(기본 qwen36-27b-v6.1). 다른 모델이면 none",
    )
    parser.add_argument("--threads", type=int, default=MAX_THREADS, help=f"torch 스레드 수(1..{MAX_THREADS})")
    parser.add_argument("--skip-transformers-check", action="store_true", help="검증 (d)를 건너뛴다(합성 체크포인트용)")
    parser.add_argument("--keep-partial", action="store_true", help="실패 시 임시 디렉터리를 지우지 않는다")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    expect = EXPECTATION_PRESETS[args.expect]
    try:
        if args.dry_run:
            print(json.dumps(run_dry(args.base, args.adapter, args.output, expect=expect), ensure_ascii=False, indent=2))
        elif args.verify:
            report = run_verify_only(
                args.base,
                args.adapter,
                args.output,
                expect=expect,
                threads=args.threads,
                transformers_check=not args.skip_transformers_check,
            )
            print(json.dumps({"passed": report["passed"], "seconds": report["seconds"]}, ensure_ascii=False))
        else:
            manifest = run_merge(
                args.base,
                args.adapter,
                args.output,
                expect=expect,
                threads=args.threads,
                transformers_check=not args.skip_transformers_check,
                keep_partial=args.keep_partial,
            )
            print(json.dumps({"output": manifest["output_dir"], "runtime": manifest["runtime"]}, ensure_ascii=False))
    except VerificationError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        print(json.dumps(exc.report, ensure_ascii=False, indent=2)[:20000], file=sys.stderr)
        return 2
    except MergeError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
