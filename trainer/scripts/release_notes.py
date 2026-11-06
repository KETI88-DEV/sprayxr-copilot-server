"""Validate a release tag and extract its entry without importing the API runtime."""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


class ReleaseNotesError(ValueError):
    """A tag, package version, or changelog cannot identify one release."""


def version_from_tag(tag: str) -> str:
    if not re.fullmatch(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", tag):
        raise ReleaseNotesError("tag must use vMAJOR.MINOR.PATCH")
    return tag[1:]


def extract_release_notes(changelog: str, version: str) -> str:
    headings = list(re.finditer(r"^##[ \t]+(.+)$", changelog, re.MULTILINE))
    matches = [i for i, heading in enumerate(headings)
               if re.match(r"\[" + re.escape(version) + r"\](?:[ \t]|$)", heading[1])]
    if len(matches) != 1:
        raise ReleaseNotesError(f"expected exactly one changelog section for version {version}")
    index = matches[0]
    end = headings[index + 1].start() if index + 1 < len(headings) else len(changelog)
    notes = changelog[headings[index].end():end].strip()
    if not notes:
        raise ReleaseNotesError(f"empty changelog section for version {version}")
    return notes + "\n"


def build_release_notes(root: Path, tag: str) -> str:
    version = version_from_tag(tag)
    package = root / "trainer/demo_api/__init__.py"
    tree = ast.parse(package.read_text(encoding="utf-8"), filename=str(package))
    versions = [node.value.value for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)]
    if versions != [version]:
        raise ReleaseNotesError(f"tag version {version} must match the literal package version")
    return extract_release_notes((root / "CHANGELOG.md").read_text(encoding="utf-8"), version)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        notes = build_release_notes(Path(__file__).resolve().parents[2], args.tag)
    except (ReleaseNotesError, OSError, SyntaxError) as exc:
        parser.exit(1, f"Release validation failed: {exc}\n")
    args.output.write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    main()
