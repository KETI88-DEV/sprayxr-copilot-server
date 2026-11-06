"""Issue reconciliation survives a canceled pending push and preserves unchanged issues."""
import json
import subprocess
from pathlib import Path

import yaml


def test_issue_batch_reconciles_missed_push_and_preserves_manual_edits(tmp_path, monkeypatch):
    workflow = Path(__file__).resolve().parents[3] / ".github/workflows/issue-batches.yml"
    run = yaml.load(workflow.read_text(), Loader=yaml.BaseLoader)["jobs"]["sync"]["steps"][1]["run"]
    code = run.split("<<'PY'\n", 1)[1].split("\nPY", 1)[0]
    names = [".github/issue-batches/earlier.json", ".github/issue-batches/current.json"]
    for key, name in zip(("earlier", "current"), names):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"key": key, "title": key, "body": "Reported failure", "state": "open"}))
    issues = {}
    mutations = []

    def external(args, **kwargs):
        if args[:2] == ["git", "diff"]:
            return names[1]  # Earlier pending push was canceled; latest diff only contains current.
        if args[:2] == ["git", "ls-files"]:
            return "\n".join(names)
        assert args[:2] == ["gh", "api"]
        if "--paginate" in args:
            return json.dumps([list(issues.values())])
        if "--method" in args:
            method = args[args.index("--method") + 1]
            endpoint = args[args.index("--method") + 2]
            if "--input" in args:
                payload = json.loads(Path(args[args.index("--input") + 1]).read_text())
            else:
                payload = dict(arg.split("=", 1) for arg in args if "=" in arg)
            number = len(issues) + 1 if method == "POST" else int(endpoint.rsplit("/", 1)[1])
            previous = issues.get(number, {"number": number, "state": "open", "html_url": f"https://github.com/org/repo/issues/{number}", "updated_at": "now"})
            issues[number] = {**previous, **payload}
            mutations.append((method, number))
            return json.dumps(issues[number])
        return json.dumps(issues[int(args[-1].rsplit("/", 1)[1])])

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BEFORE_SHA", "1" * 40)
    monkeypatch.setenv("GH_REPO", "org/repo")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    monkeypatch.setattr(subprocess, "check_output", external)
    exec(compile(code, str(workflow), "exec"), {})
    assert {i["title"] for i in issues.values()} == {"earlier", "current"}
    first = issues[1]
    first["body"] += "\nHuman investigation note"
    first["state"] = "closed"
    before = len(mutations)
    exec(compile(code, str(workflow), "exec"), {})
    assert len(mutations) == before
    assert first["body"].endswith("Human investigation note") and first["state"] == "closed"
