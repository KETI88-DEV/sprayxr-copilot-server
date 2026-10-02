from __future__ import annotations

import hashlib
import json
from pathlib import Path

from deploy.spark.scripts.legacy_snapshot_manifest import (
    ManifestEntry,
    build_manifest,
    verify_manifest,
    write_manifest,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_build_manifest_is_relative_sorted_and_records_size_and_sha256(tmp_path: Path) -> None:
    (tmp_path / "z.txt").write_bytes(b"last")
    nested = tmp_path / "dir"
    nested.mkdir()
    (nested / "a.bin").write_bytes(b"first")

    manifest = build_manifest(tmp_path)

    assert list(manifest) == ["dir/a.bin", "z.txt"]
    assert manifest == {
        "dir/a.bin": ManifestEntry(size=5, sha256=_sha256(b"first")),
        "z.txt": ManifestEntry(size=4, sha256=_sha256(b"last")),
    }


def test_write_manifest_uses_a_stable_json_shape(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"
    write_manifest(
        target,
        {
            "z": ManifestEntry(size=2, sha256="b" * 64),
            "a": ManifestEntry(size=1, sha256="a" * 64),
        },
    )

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload == {
        "algorithm": "sha256",
        "entries": {
            "a": {"sha256": "a" * 64, "size": 1},
            "z": {"sha256": "b" * 64, "size": 2},
        },
        "version": 1,
    }


def test_verify_manifest_reports_missing_extra_and_changed_files(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_bytes(b"before")
    (tmp_path / "missing.txt").write_bytes(b"gone")
    expected = build_manifest(tmp_path)

    (tmp_path / "keep.txt").write_bytes(b"after!")
    (tmp_path / "missing.txt").unlink()
    (tmp_path / "extra.txt").write_bytes(b"extra")

    result = verify_manifest(tmp_path, expected)

    assert result.ok is False
    assert result.missing == ("missing.txt",)
    assert result.extra == ("extra.txt",)
    assert result.mismatched == ("keep.txt",)


def test_verify_manifest_accepts_an_exact_copy(tmp_path: Path) -> None:
    (tmp_path / "same.bin").write_bytes(b"same")
    expected = build_manifest(tmp_path)

    result = verify_manifest(tmp_path, expected)

    assert result.ok is True
    assert result.missing == ()
    assert result.extra == ()
    assert result.mismatched == ()
