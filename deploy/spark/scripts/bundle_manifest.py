#!/usr/bin/env python3
"""Spark 배포 번들의 매니페스트를 만들고 검증한다.

표준 라이브러리만 쓴다. 프로젝트 의존성이 없는 Spark 호스트의 python3(3.10+)에서
컨테이너를 띄우기 전에 그대로 돈다.

    create  모든 파일의 sha256·크기, 모델 출처(merge/fp8 manifest) 대조 결과, 코드 커밋,
            이미지 참조를 BUNDLE_MANIFEST.json에 기록한다. 대조가 어긋나면 쓰지 않는다.
    verify  매니페스트대로 파일이 있는지, 크기·sha256이 같은지, 배포에 필요한 자산 묶음이
            갖춰졌는지 확인한다. --quick 은 크기만 본다(같은 크기의 손상은 못 잡는다).

번들 배치는 컨테이너 경로와 1:1이다(deploy/spark/compose.yaml 볼륨):

    models/bge-m3/                                  embedding  -> agent-api /models/bge-m3
    models/bge-reranker-v2-m3/                      reranker   -> agent-api /models/bge-reranker-v2-m3
    output/indexes/{faiss.index,metadata.json}      rag_index  -> agent-api /output/indexes
    output/merged/qwen36_27b_shipcoat_v6_1_fp8/     model_fp8  -> vllm /models/shipcoat, agent-api 토크나이저
    output/merged/qwen36_27b_shipcoat_v6_1_bf16/    model_bf16 -> 롤백(선택)
    output/expert_reference/expert_arm_reference.json  mcp_pose_reference -> mcp ARM_REFERENCE_PATH
    data/eval/                                      eval_set   -> FP8 품질 게이트(선택)
    config/                                         배포 때 생성(롤백 설정). 매니페스트 대상이 아니다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "shipcoat-spark-bundle/1"
MANIFEST_NAME = "BUNDLE_MANIFEST.json"
SOURCES_NAME = "bundle_sources.json"
GENERATED_DIRS = ("config",)
_CHUNK = 8 * 1024 * 1024
_LARGE_FILE_BYTES = 256 * 1024 * 1024

FP8_MODEL_ROOT = "output/merged/qwen36_27b_shipcoat_v6_1_fp8"
BF16_MODEL_ROOT = "output/merged/qwen36_27b_shipcoat_v6_1_bf16"
_MODEL_REQUIRED = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
)

# group -> (bundle-relative root, files that must exist under it)
GROUPS: dict[str, tuple[str, tuple[str, ...]]] = {
    "model_fp8": (FP8_MODEL_ROOT, _MODEL_REQUIRED),
    "model_bf16": (BF16_MODEL_ROOT, _MODEL_REQUIRED),
    # search_index.load_embedder: SentenceTransformer(use_safetensors=True)
    "embedding": ("models/bge-m3", ("config.json", "modules.json", "tokenizer.json", "model.safetensors", "1_Pooling/config.json")),
    "reranker": ("models/bge-reranker-v2-m3", ("config.json", "tokenizer.json", "model.safetensors")),
    "rag_index": ("output/indexes", ("faiss.index", "metadata.json")),
    # 팔 레퍼런스는 **필수**다. 이 그룹을 실으면서 이 파일을 빼면 MCP는 뜨지만 모든
    # 시퀀스가 reference_data_missing이 된다 — 조용히 채점이 비는 것을 verify가 잡아야 한다.
    "mcp_pose_reference": ("output/expert_reference", ("expert_arm_reference.json",)),
    "eval_set": ("data/eval", ()),
}
MODEL_GROUPS = {"fp8": "model_fp8", "bf16": "model_bf16"}
DEPLOY_REQUIRED_GROUPS = ("embedding", "reranker", "rag_index")
# Manifests written by quantize_fp8_block.py / merge_lora_shards.py next to the weights.
SOURCE_MANIFESTS = {"model_fp8": "fp8_manifest.json", "model_bf16": "merge_manifest.json"}

_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class BundleError(Exception):
    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("; ".join(list(problems)[:5]))
        self.problems = list(problems)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def scan_bundle(bundle: Path) -> tuple[list[str], list[str]]:
    """Returns ``(files, problems)``: sorted relative POSIX paths of regular files, and symlinks found.

    The manifest itself and the deploy-time ``config/`` directory are skipped.
    """
    files: list[str] = []
    problems: list[str] = []
    for dirpath, dirnames, filenames in os.walk(bundle):
        base = Path(dirpath)
        if base == bundle:
            dirnames[:] = [name for name in dirnames if name not in GENERATED_DIRS]
        for name in list(dirnames):
            if (base / name).is_symlink():
                problems.append(f"symlinked directory: {(base / name).relative_to(bundle).as_posix()}")
                dirnames.remove(name)
        for name in filenames:
            path = base / name
            rel = path.relative_to(bundle).as_posix()
            if rel == MANIFEST_NAME:
                continue
            if path.is_symlink():
                problems.append(f"symlink: {rel}")
            elif not path.is_file():
                problems.append(f"not a regular file: {rel}")
            else:
                files.append(rel)
    return sorted(files), problems


def group_of(rel: str) -> str:
    for name, (root, _) in GROUPS.items():
        if rel.startswith(root + "/"):
            return name
    return "other"


def hash_files(bundle: Path, rels: Sequence[str], jobs: int) -> dict[str, dict[str, Any]]:
    def _one(rel: str) -> tuple[str, dict[str, Any]]:
        path = bundle / rel
        size = path.stat().st_size
        digest = sha256_file(path)
        if size >= _LARGE_FILE_BYTES:
            print(f"hashed {rel} ({size / 1e9:.2f} GB)", file=sys.stderr, flush=True)
        return rel, {"bytes": size, "sha256": digest}

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        return dict(pool.map(_one, rels))


def check_group_contents(bundle: Path, group: str, rels: Iterable[str]) -> list[str]:
    """Required files exist; for a model, every shard named by the safetensors index exists."""
    root, required = GROUPS[group]
    prefix = root + "/"
    present = {rel[len(prefix) :] for rel in rels if rel.startswith(prefix)}
    problems = [f"{group}: missing {root}/{name}" for name in required if name not in present]
    if group in SOURCE_MANIFESTS and "model.safetensors.index.json" in present:
        try:
            index = json.loads((bundle / root / "model.safetensors.index.json").read_text(encoding="utf-8"))
            shards = {str(shard) for shard in (index.get("weight_map") or {}).values()}
        except (OSError, ValueError, AttributeError) as exc:
            return [*problems, f"{group}: unreadable model.safetensors.index.json ({type(exc).__name__})"]
        if not shards:
            problems.append(f"{group}: model.safetensors.index.json has an empty weight_map")
        problems.extend(f"{group}: shard {root}/{shard} named by the index is missing" for shard in sorted(shards - present))
    return problems


def crosscheck_source_manifest(
    bundle: Path, group: str, digests: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any] | None:
    """Compares bundle hashes with the hashes the merge/FP8 tool recorded; ``None`` if that manifest is absent."""
    root, _ = GROUPS[group]
    rel = f"{root}/{SOURCE_MANIFESTS[group]}"
    if rel not in digests:
        return None
    data = json.loads((bundle / rel).read_text(encoding="utf-8"))
    verification = data.get("verification") if isinstance(data.get("verification"), dict) else {}
    if group == "model_fp8":
        expected = {str(k): str(v) for k, v in (data.get("files_sha256") or {}).items()}
        lineage: dict[str, Any] = {
            "created_utc": data.get("created_utc"),
            "input_merge_manifest_sha256": data.get("input_merge_manifest_sha256"),
            "quantized_tensors": data.get("quantized_tensors"),
            "total_tensors": data.get("total_tensors"),
        }
    else:
        expected = {str(k): str((v or {}).get("sha256")) for k, v in (data.get("files") or {}).items()}
        adapter = data.get("adapter") if isinstance(data.get("adapter"), dict) else {}
        base = data.get("base") if isinstance(data.get("base"), dict) else {}
        lineage = {
            "created_at": data.get("created_at"),
            "base_path": base.get("path"),
            "adapter_model_sha256": adapter.get("adapter_model_sha256"),
            "changed_tensor_count": data.get("changed_tensor_count"),
        }
    lineage["verification_passed"] = verification.get("passed")

    mismatched: list[str] = []
    missing: list[str] = []
    compared = 0
    for name, sha in sorted(expected.items()):
        entry = digests.get(f"{root}/{name}")
        if entry is None:
            missing.append(name)
            continue
        compared += 1
        if entry["sha256"] != sha:
            mismatched.append(name)
    return {
        "source_manifest": rel,
        "source_manifest_sha256": digests[rel]["sha256"],
        "compared": compared,
        "mismatched": mismatched,
        "missing": missing,
        **lineage,
    }


def parse_image_ref(value: str) -> tuple[str, dict[str, str | None]]:
    """``key=repo:tag[@sha256:...]`` -> ``(key, {"ref": "repo:tag", "digest": "sha256:..." | None})``."""
    key, sep, ref = value.partition("=")
    if not sep or not key or not ref:
        raise ValueError(f"expected KEY=IMAGE[@sha256:DIGEST], got {value!r}")
    name, at, digest = ref.partition("@")
    if at and not _DIGEST_RE.fullmatch(digest):
        raise ValueError(f"image digest must be sha256:<64 hex>, got {digest!r}")
    return key, {"ref": name, "digest": digest if at else None}


def build_manifest(
    bundle: Path,
    *,
    code_commit: str,
    images: Mapping[str, Mapping[str, str | None]],
    jobs: int = 4,
    created_utc: str | None = None,
) -> dict[str, Any]:
    if not _COMMIT_RE.fullmatch(code_commit):
        raise BundleError([f"code commit must be 7-40 lowercase hex, got {code_commit!r}"])
    rels, problems = scan_bundle(bundle)
    if problems:
        raise BundleError(problems)
    if not rels:
        raise BundleError(["bundle has no files"])

    digests = hash_files(bundle, rels, jobs)
    groups: dict[str, dict[str, Any]] = {}
    for rel, entry in digests.items():
        name = group_of(rel)
        info = groups.setdefault(name, {"root": GROUPS[name][0] if name in GROUPS else None, "files": 0, "bytes": 0})
        info["files"] += 1
        info["bytes"] += entry["bytes"]

    for name in groups:
        if name in GROUPS:
            problems.extend(check_group_contents(bundle, name, rels))

    provenance: dict[str, Any] = {}
    for name, manifest_name in SOURCE_MANIFESTS.items():
        if name not in groups:
            continue
        check = crosscheck_source_manifest(bundle, name, digests)
        if check is None:
            problems.append(f"{name}: {manifest_name} is missing; model lineage cannot be recorded")
            continue
        provenance[name] = check
        if check["compared"] == 0:
            problems.append(f"{name}: {manifest_name} lists no file hashes")
        problems.extend(f"{name}: {f} differs from {manifest_name}" for f in check["mismatched"])
        problems.extend(f"{name}: {f} listed in {manifest_name} is absent" for f in check["missing"])
    if problems:
        raise BundleError(problems)

    sources_path = bundle / SOURCES_NAME
    sources = json.loads(sources_path.read_text(encoding="utf-8")) if sources_path.is_file() else None
    return {
        "schema": SCHEMA,
        "created_utc": created_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": {"host_machine": platform.machine(), "python": platform.python_version()},
        "code_commit": code_commit,
        "images": {key: dict(value) for key, value in sorted(images.items())},
        "file_count": len(digests),
        "total_bytes": sum(entry["bytes"] for entry in digests.values()),
        "groups": dict(sorted(groups.items())),
        "provenance": provenance,
        "sources": sources,
        "files": [{"path": rel, **digests[rel]} for rel in rels],
    }


@dataclass
class VerifyReport:
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    files_checked: int = 0
    bytes_checked: int = 0
    mode: str = "full"

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "files_checked": self.files_checked,
            "bytes_checked": self.bytes_checked,
            "problems": self.problems,
            "warnings": self.warnings,
        }


def verify_bundle(
    bundle: Path,
    manifest: Mapping[str, Any],
    *,
    model: str = "fp8",
    quick: bool = False,
    jobs: int = 4,
    image_digests: Mapping[str, str] | None = None,
    code_commit: str | None = None,
) -> VerifyReport:
    report = VerifyReport(mode="quick" if quick else "full")
    if manifest.get("schema") != SCHEMA:
        report.problems.append(f"unknown manifest schema {manifest.get('schema')!r}")
        return report

    entries = {str(item["path"]): item for item in manifest.get("files", [])}
    if sum(int(item["bytes"]) for item in entries.values()) != manifest.get("total_bytes"):
        report.problems.append("manifest total_bytes does not equal the sum of its file sizes")

    to_hash: list[str] = []
    for rel, item in sorted(entries.items()):
        path = bundle / rel
        if path.is_symlink():
            report.problems.append(f"symlink where the manifest expects a file: {rel}")
        elif not path.is_file():
            report.problems.append(f"missing: {rel}")
        elif path.stat().st_size != int(item["bytes"]):
            report.problems.append(f"size differs: {rel} ({path.stat().st_size} != {item['bytes']})")
        else:
            to_hash.append(rel)
    if not quick:
        actual = hash_files(bundle, to_hash, jobs)
        report.problems.extend(f"sha256 differs: {rel}" for rel in to_hash if actual[rel]["sha256"] != entries[rel]["sha256"])
    report.files_checked = len(to_hash)
    report.bytes_checked = sum(int(entries[rel]["bytes"]) for rel in to_hash)

    present, scan_problems = scan_bundle(bundle)
    report.problems.extend(scan_problems)
    report.warnings.extend(f"not in manifest: {rel}" for rel in sorted(set(present) - set(entries)))

    if model not in MODEL_GROUPS:
        report.problems.append(f"unknown model {model!r}; expected one of {sorted(MODEL_GROUPS)}")
        return report
    groups = manifest.get("groups") or {}
    provenance = manifest.get("provenance") or {}
    for name in (*DEPLOY_REQUIRED_GROUPS, MODEL_GROUPS[model]):
        if name not in groups:
            report.problems.append(f"required asset group absent: {name}")
        else:
            report.problems.extend(check_group_contents(bundle, name, entries))
    if MODEL_GROUPS[model] in groups and MODEL_GROUPS[model] not in provenance:
        report.problems.append(f"{MODEL_GROUPS[model]}: no source manifest provenance recorded")
    for name, check in provenance.items():
        if check.get("mismatched") or check.get("missing"):
            report.problems.append(f"{name}: provenance cross-check recorded differences")

    recorded_images = manifest.get("images") or {}
    for key, digest in (image_digests or {}).items():
        recorded = recorded_images.get(key) or {}
        if not recorded.get("digest"):
            report.problems.append(f"image {key}: no digest recorded in the manifest")
        elif recorded["digest"] != digest:
            report.problems.append(f"image {key}: digest {digest} != recorded {recorded['digest']}")
    if code_commit is not None and manifest.get("code_commit") != code_commit:
        report.problems.append(f"code commit {code_commit} != recorded {manifest.get('code_commit')}")
    return report


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _default_jobs() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="hash the bundle and write BUNDLE_MANIFEST.json")
    create.add_argument("--bundle-dir", type=Path, required=True)
    create.add_argument("--code-commit", required=True, help="git rev-parse HEAD of the code baked into the images")
    create.add_argument("--image", action="append", default=[], metavar="KEY=IMAGE[@sha256:DIGEST]")
    create.add_argument("--jobs", type=int, default=_default_jobs())
    create.add_argument("--output", type=Path, default=None, help=f"default: <bundle-dir>/{MANIFEST_NAME}")

    verify = sub.add_parser("verify", help="check the bundle against its manifest")
    verify.add_argument("--bundle-dir", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, default=None, help=f"default: <bundle-dir>/{MANIFEST_NAME}")
    verify.add_argument("--model", choices=sorted(MODEL_GROUPS), default="fp8")
    verify.add_argument("--quick", action="store_true", help="sizes only; same-size corruption goes unnoticed")
    verify.add_argument("--jobs", type=int, default=_default_jobs())
    verify.add_argument("--image-digest", action="append", default=[], metavar="KEY=sha256:DIGEST")
    verify.add_argument("--code-commit", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    bundle = args.bundle_dir.resolve()
    if not bundle.is_dir():
        print(json.dumps({"ok": False, "problems": [f"bundle dir not found: {bundle}"]}))
        return 1

    if args.command == "create":
        try:
            images = dict(parse_image_ref(value) for value in args.image)
            manifest = build_manifest(bundle, code_commit=args.code_commit, images=images, jobs=args.jobs)
        except (BundleError, ValueError) as exc:
            problems = exc.problems if isinstance(exc, BundleError) else [str(exc)]
            print(json.dumps({"ok": False, "problems": problems}, ensure_ascii=False, indent=2))
            return 1
        output = args.output or bundle / MANIFEST_NAME
        _write_json_atomic(output, manifest)
        summary = {
            "ok": True,
            "manifest": str(output),
            "manifest_sha256": sha256_file(output),
            "file_count": manifest["file_count"],
            "total_bytes": manifest["total_bytes"],
            "groups": manifest["groups"],
            "provenance": {
                name: {k: check[k] for k in ("source_manifest_sha256", "compared", "mismatched", "missing")}
                for name, check in manifest["provenance"].items()
            },
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    manifest_path = args.manifest or bundle / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "problems": [f"cannot read manifest {manifest_path}: {type(exc).__name__}"]}))
        return 1
    image_digests: dict[str, str] = {}
    for value in args.image_digest:
        key, sep, digest = value.partition("=")
        if not sep or not _DIGEST_RE.fullmatch(digest):
            print(json.dumps({"ok": False, "problems": [f"expected KEY=sha256:<64 hex>, got {value!r}"]}))
            return 1
        image_digests[key] = digest
    report = verify_bundle(
        bundle,
        manifest,
        model=args.model,
        quick=args.quick,
        jobs=args.jobs,
        image_digests=image_digests,
        code_commit=args.code_commit,
    )
    payload = report.as_dict()
    payload["manifest_sha256"] = sha256_file(manifest_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
