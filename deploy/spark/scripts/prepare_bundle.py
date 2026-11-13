#!/usr/bin/env python3
"""Spark 배포 번들 디렉터리를 만든다 (표준 라이브러리만, A40 개발 컨테이너에서 실행).

rag_config_spark_vllm.yaml의 컨테이너 경로와 1:1인 배치로 자산을 모은다(배치는 bundle_manifest.py 문서 참고).

- 임베딩(KURE-v3)·재정렬(KURE-Reranker-nano): HF 캐시 main 스냅숏을 그대로 넣는다. 둘 다 model.safetensors가
  main에 있다. KURE-v3의 라이선스 고지 파일(LICENSE, LICENSE-OpenMDW-1.1, NOTICE)도 스냅숏에서 함께 온다.
- 스냅숏 파일은 blobs로 가는 심볼릭 링크다. 번들에는 실제 파일로 넣는다(심볼릭 링크는 매니페스트가 거부).
- --link: 같은 파일시스템이면 하드링크(모델 29GB를 복사하지 않음), 아니면 복사로 떨어진다.
- 선택 자산이 없으면 bundle_sources.json의 absent_optional에 적고 계속한다.

다음 단계: bundle_manifest.py create --bundle-dir <dir> --code-commit <sha> --image ...
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bundle_manifest as bm  # noqa: E402

DEFAULT_FP8_DIR = "/output/merged/qwen36_27b_shipcoat_v6_1_fp8"
DEFAULT_BF16_DIR = "/output/merged/qwen36_27b_shipcoat_v6_1_bf16"
DEFAULT_HUB_CACHE = "/hf-cache/hub"
DEFAULT_INDEX_DIR = "/output/indexes_20261113_kure3"  # build_index.py 산출물(2026-11-13, 25,483청크)
# mcp_servers/pose_estimation.py DEFAULT_POSE_REFERENCE; compose.yaml POSE_REFERENCE_PATH
DEFAULT_POSE_REFERENCE = "/output/expert_reference/expert_pose_reference.json"
# mcp_servers/pose_estimation.py DEFAULT_ARM_REFERENCE; compose.yaml ARM_REFERENCE_PATH.
# **채점이 실제로 읽는 파일이다** — 빠지면 모든 시퀀스가 reference_data_missing이다.
DEFAULT_ARM_REFERENCE = "/output/expert_reference/expert_arm_reference.json"
# mcp_servers/pose_estimation.py DEFAULT_AVATAR_ARM_REFERENCE — 2026-10 아바타로 옮겨 구운 기준. 이것도 필수다.
DEFAULT_AVATAR_ARM_REFERENCE = "/output/expert_reference/expert_arm_reference_avatar_2026-10.json"
DEFAULT_EVAL_SET = "/data/eval/eval_set_agentic.jsonl"
INDEX_FILES = ("faiss.index", "metadata.json")


class PrepareError(Exception):
    pass


@dataclass(frozen=True)
class CopyItem:
    group: str
    src: Path
    dst_rel: str


def hub_repo_dir(hub_cache: Path, repo_id: str) -> Path:
    return hub_cache / ("models--" + repo_id.replace("/", "--"))


def resolve_snapshot(repo_dir: Path, revision: str) -> Path:
    """HF cache layout: ``refs/<revision>`` holds the commit, ``snapshots/<commit>`` the files."""
    ref = repo_dir / "refs" / revision
    if not ref.is_file():
        raise PrepareError(f"{repo_dir.name}: no ref for revision {revision!r} ({ref})")
    commit = ref.read_text(encoding="utf-8").strip()
    snapshot = repo_dir / "snapshots" / commit
    if not snapshot.is_dir():
        raise PrepareError(f"{repo_dir.name}: snapshot {commit} for {revision!r} is missing")
    return snapshot


def _walk_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = Path(dirpath) / name
            if path.is_file():  # follows snapshot -> blob links; dangling links are skipped
                found.append(path)
    return sorted(found)


def plan_hf_model(
    group: str,
    hub_cache: Path,
    repo_id: str,
    *,
    exclude: Iterable[str] = (),
    overrides: dict[str, str] | None = None,
) -> tuple[list[CopyItem], dict[str, Any]]:
    """Main snapshot files minus ``exclude``, plus ``{file: revision}`` taken from other snapshots."""
    root = bm.GROUPS[group][0]
    repo = hub_repo_dir(hub_cache, repo_id)
    main = resolve_snapshot(repo, "main")
    excluded = set(exclude)
    items: dict[str, CopyItem] = {}
    for path in _walk_files(main):
        rel = path.relative_to(main).as_posix()
        if rel not in excluded:
            items[rel] = CopyItem(group, path, f"{root}/{rel}")
    source: dict[str, Any] = {
        "repo_id": repo_id,
        "main_snapshot": main.name,
        "excluded": sorted(name for name in excluded if (main / name).exists()),
        "overrides": {},
    }
    for rel, revision in (overrides or {}).items():
        snapshot = resolve_snapshot(repo, revision)
        path = snapshot / rel
        if not path.is_file():
            raise PrepareError(f"{repo_id}: {rel} not found in snapshot {snapshot.name} ({revision})")
        items[rel] = CopyItem(group, path, f"{root}/{rel}")
        source["overrides"][rel] = {"revision": revision, "snapshot": snapshot.name}
    return list(items.values()), source


def plan_dir(group: str, src_dir: Path, *, only: Sequence[str] | None = None) -> list[CopyItem]:
    root = bm.GROUPS[group][0]
    if not src_dir.is_dir():
        raise PrepareError(f"{group}: source directory not found: {src_dir}")
    names = list(only) if only is not None else [p.relative_to(src_dir).as_posix() for p in _walk_files(src_dir)]
    items = []
    for name in names:
        path = src_dir / name
        if not path.is_file():
            raise PrepareError(f"{group}: {path} not found")
        items.append(CopyItem(group, path, f"{root}/{name}"))
    return items




def plan_arm_reference(root: str, arm: Path, name: str = "expert_arm_reference.json") -> CopyItem:
    """팔 레퍼런스 복사 항목. **없으면 실패한다.**

    위 pose 파일과 달리 이것이 채점 경로의 **유일한 기준**이다. 빠진 채 배포하면
    MCP는 정상으로 뜨고(헬스체크도 통과한다) 모든 시퀀스가 `reference_data_missing`이
    된다 — 조용히 채점만 비는 실패다.

    `build_plan` 안에 인라인으로 두면 앞 단계(모델 선택·HF 스냅샷·인덱스)를 모두
    갖춰야 이 분기에 닿아 검사가 사실상 불가능하다. 그래서 떼어 둔다.
    """
    if not arm.is_file():
        raise PrepareError(
            f"arm_reference: {arm} not found — 이것이 없으면 모든 시퀀스가 "
            "reference_data_missing이다. trainer/scripts/build_arm_reference.py로 굽는다."
        )
    return CopyItem("mcp_pose_reference", arm, f"{root}/{name}")

def build_plan(args: argparse.Namespace) -> tuple[list[CopyItem], dict[str, Any], list[str]]:
    items: list[CopyItem] = []
    sources: dict[str, Any] = {}
    absent: list[str] = []

    if not args.skip_fp8:
        items += plan_dir("model_fp8", args.fp8_dir)
        sources["model_fp8"] = {"dir": str(args.fp8_dir)}
    if args.with_bf16:
        items += plan_dir("model_bf16", args.bf16_dir)
        sources["model_bf16"] = {"dir": str(args.bf16_dir)}
    if args.skip_fp8 and not args.with_bf16:
        raise PrepareError("no model selected: drop --skip-fp8 or add --with-bf16")

    embedding, sources["embedding"] = plan_hf_model("embedding", args.hf_hub_cache, args.embedding_repo)
    reranker, sources["reranker"] = plan_hf_model("reranker", args.hf_hub_cache, args.reranker_repo)
    items += embedding + reranker

    items += plan_dir("rag_index", args.index_dir, only=INDEX_FILES)
    sources["rag_index"] = {"dir": str(args.index_dir), "files": list(INDEX_FILES)}

    root = bm.GROUPS["mcp_pose_reference"][0]
    pose = args.pose_reference
    if pose.is_file():
        items.append(CopyItem("mcp_pose_reference", pose, f"{root}/expert_pose_reference.json"))
        sources["mcp_pose_reference"] = {"file": str(pose)}
    else:
        absent.append(f"mcp_pose_reference: {pose} not found (pose_estimation answers without a reference)")

    items.append(plan_arm_reference(root, args.arm_reference))
    sources["mcp_arm_reference"] = {"file": str(args.arm_reference)}
    items.append(plan_arm_reference(root, args.avatar_arm_reference, Path(DEFAULT_AVATAR_ARM_REFERENCE).name))
    sources["mcp_avatar_arm_reference"] = {"file": str(args.avatar_arm_reference)}

    if args.with_eval_set:
        if not args.eval_set.is_file():
            raise PrepareError(f"eval_set: {args.eval_set} not found")
        items.append(CopyItem("eval_set", args.eval_set, f"{bm.GROUPS['eval_set'][0]}/eval_set_agentic.jsonl"))
        sources["eval_set"] = {"file": str(args.eval_set)}
    else:
        absent.append("eval_set: not requested (--with-eval-set); the FP8 quality gate needs it")

    destinations = [item.dst_rel for item in items]
    duplicates = sorted({d for d in destinations if destinations.count(d) > 1})
    problems = [f"duplicate destination {d}" for d in duplicates]
    planned = set(destinations)
    for group in sorted({item.group for item in items}):
        root, required = bm.GROUPS[group]
        problems += [f"{group}: {root}/{name} would be missing" for name in required if f"{root}/{name}" not in planned]
    if problems:
        raise PrepareError("; ".join(problems))
    return items, sources, absent


def _summarize(items: Sequence[CopyItem]) -> dict[str, Any]:
    groups: dict[str, dict[str, int]] = {}
    for item in items:
        info = groups.setdefault(item.group, {"files": 0, "bytes": 0})
        info["files"] += 1
        info["bytes"] += item.src.stat().st_size
    return {"groups": groups, "total_bytes": sum(g["bytes"] for g in groups.values())}


def execute(bundle: Path, items: Sequence[CopyItem], *, link: bool) -> dict[str, int]:
    counts = {"linked": 0, "copied": 0}
    for item in items:
        dst = bundle / item.dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        real = item.src.resolve(strict=True)
        if link:
            try:
                os.link(real, dst)
            except OSError:
                pass  # other filesystem / mount point: copy instead
            else:
                counts["linked"] += 1
                continue
        shutil.copyfile(real, dst)
        counts["copied"] += 1
    return counts


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle-dir", type=Path, required=True, help="new or empty directory")
    parser.add_argument("--fp8-dir", type=Path, default=Path(DEFAULT_FP8_DIR))
    parser.add_argument("--skip-fp8", action="store_true")
    parser.add_argument("--with-bf16", action="store_true", help="include the BF16 rollback model (52GB)")
    parser.add_argument("--bf16-dir", type=Path, default=Path(DEFAULT_BF16_DIR))
    parser.add_argument("--hf-hub-cache", type=Path, default=Path(DEFAULT_HUB_CACHE))
    parser.add_argument("--embedding-repo", default="nlpai-lab/KURE-v3")
    parser.add_argument("--reranker-repo", default="nlpai-lab/KURE-Reranker-nano")
    parser.add_argument("--index-dir", type=Path, default=Path(DEFAULT_INDEX_DIR))
    parser.add_argument("--pose-reference", type=Path, default=Path(DEFAULT_POSE_REFERENCE))
    parser.add_argument("--arm-reference", type=Path, default=Path(DEFAULT_ARM_REFERENCE))
    parser.add_argument("--avatar-arm-reference", type=Path, default=Path(DEFAULT_AVATAR_ARM_REFERENCE))
    parser.add_argument("--with-eval-set", action="store_true")
    parser.add_argument("--eval-set", type=Path, default=Path(DEFAULT_EVAL_SET))
    parser.add_argument("--link", action="store_true", help="hardlink when on the same filesystem")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and sizes; write nothing")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        items, sources, absent = build_plan(args)
    except PrepareError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    summary = {"bundle_dir": str(args.bundle_dir), **_summarize(items), "absent_optional": absent}
    if args.dry_run:
        print(json.dumps({"ok": True, "dry_run": True, **summary}, ensure_ascii=False, indent=2))
        return 0

    bundle = args.bundle_dir
    if bundle.exists() and any(bundle.iterdir()):
        print(json.dumps({"ok": False, "error": f"bundle dir is not empty: {bundle}"}))
        return 1
    bundle.mkdir(parents=True, exist_ok=True)
    counts = execute(bundle, items, link=args.link)
    for generated in bm.GENERATED_DIRS:
        (bundle / generated).mkdir(exist_ok=True)
    record = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": sources,
        "absent_optional": absent,
        "link_requested": bool(args.link),
        **counts,
    }
    (bundle / bm.SOURCES_NAME).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, **summary, **counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
