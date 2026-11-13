"""Build and verify deterministic SHA-256 manifests for legacy snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ManifestEntry:
    size: int
    sha256: str


@dataclass(frozen=True)
class VerificationResult:
    missing: tuple[str, ...]
    extra: tuple[str, ...]
    mismatched: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not (self.missing or self.extra or self.mismatched)


def _digest(path: Path) -> ManifestEntry:
    if path.is_symlink():
        data = os.readlink(path).encode("utf-8", errors="surrogateescape")
        return ManifestEntry(size=len(data), sha256=hashlib.sha256(data).hexdigest())

    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            hasher.update(chunk)
    return ManifestEntry(size=size, sha256=hasher.hexdigest())


def build_manifest(root: Path) -> dict[str, ManifestEntry]:
    root = root.resolve()
    paths = sorted(
        (path for path in root.rglob("*") if path.is_file() or path.is_symlink()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    return {path.relative_to(root).as_posix(): _digest(path) for path in paths}


def write_manifest(target: Path, entries: Mapping[str, ManifestEntry]) -> None:
    payload = {
        "algorithm": "sha256",
        "entries": {name: asdict(entries[name]) for name in sorted(entries)},
        "version": 1,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_manifest(source: Path) -> dict[str, ManifestEntry]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or payload.get("algorithm") != "sha256":
        raise ValueError("unsupported manifest format")
    return {
        name: ManifestEntry(size=int(value["size"]), sha256=str(value["sha256"]))
        for name, value in payload["entries"].items()
    }


def verify_manifest(root: Path, expected: Mapping[str, ManifestEntry]) -> VerificationResult:
    actual = build_manifest(root)
    expected_names = set(expected)
    actual_names = set(actual)
    return VerificationResult(
        missing=tuple(sorted(expected_names - actual_names)),
        extra=tuple(sorted(actual_names - expected_names)),
        mismatched=tuple(
            sorted(name for name in expected_names & actual_names if expected[name] != actual[name])
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("root", type=Path)
    create.add_argument("manifest", type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("root", type=Path)
    verify.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)

    if args.command == "create":
        entries = build_manifest(args.root)
        write_manifest(args.manifest, entries)
        print(json.dumps({"ok": True, "files": len(entries)}))
        return 0

    result = verify_manifest(args.root, read_manifest(args.manifest))
    print(json.dumps(asdict(result) | {"ok": result.ok}, ensure_ascii=False))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
