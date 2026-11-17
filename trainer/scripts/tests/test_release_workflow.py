"""Execute the release workflow's real shell steps against local Git and a fake gh.

No GitHub, SSH, model, production database, or host Python is involved. Checkout
is represented by a fresh clone; each run step and GITHUB_ENV handoff is real.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
TAG = "v3.9.0"
VERSION_FILE = "trainer/demo_api/__init__.py"

FAKE_GH = r'''#!/usr/bin/env python3
import json,os,pathlib,subprocess,sys
path=pathlib.Path(os.environ['FAKE_GH_STATE']);state=json.loads(path.read_text())
args=sys.argv[1:];state.setdefault('calls',[]).append(args)
def save():path.write_text(json.dumps(state))
def fail(message):save();print(message,file=sys.stderr);sys.exit(75)
def git(*arguments):
    return subprocess.check_output(['git','--git-dir',os.environ['FAKE_ORIGIN'],*arguments],text=True).strip()
def release(tag):
    if tag not in state.get('releases',{}):fail('release not found')
    return state['releases'][tag]
if args[:1]==['api']:
    route=args[1]
    if state.get('fail_api_suffix') and route.endswith(state['fail_api_suffix']):
        del state['fail_api_suffix'];fail('injected temporary API failure')
    prefix='repos/'+os.environ['GH_REPO']+'/'
    if not route.startswith(prefix):fail('unsupported API repository')
    route=route[len(prefix):]
    if route.startswith('releases/tags/'):answer=release(route[len('releases/tags/'):])
    elif route=='releases/latest':answer=release(state.get('latest',''))
    elif route.startswith('git/ref/'):
        ref='refs/'+route[len('git/ref/'):];sha=git('rev-parse','--verify',ref)
        answer={'object':{'sha':sha,'type':git('cat-file','-t',sha)}}
    elif route.startswith('git/tags/'):
        sha=route[len('git/tags/'):];raw=git('cat-file','-p',sha);headers,message=raw.split('\n\n',1)
        values=dict(line.split(' ',1) for line in headers.splitlines())
        answer={'sha':sha,'tag':values['tag'],'object':{'sha':values['object'],'type':values['type']},'message':message}
    else:fail('unsupported API route: '+route)
elif args[:2]==['release','view']:
    answer=release(args[2])
elif args[:2] in (['release','create'],['release','edit']):
    tag=args[2]
    if args[1]=='create':
        if state.pop('fail_create_once',False):fail('injected temporary release create failure')
        if tag in state.get('releases',{}):fail('release already exists')
        git('rev-parse','--verify','refs/tags/'+tag)
        answer={'id':1,'tag_name':tag,'html_url':'https://example.invalid/releases/'+tag,
                'published_at':'2026-11-17T00:00:00Z','draft':False,'prerelease':False}
        state.setdefault('releases',{})[tag]=answer
    else:answer=release(tag)
    if '--latest=true' in args or '--latest' in args:state['latest']=tag
else:fail('unsupported gh command: '+repr(args))
save();print(json.dumps(answer))
'''


class ReleaseCase:
    def __init__(self, root: Path):
        self.root, self.setup = root, root / "setup"
        self.origin, self.bin = root / "origin.git", root / "bin"
        self.state_path, self.runs = root / "gh-state.json", 0
        self.setup.mkdir();self.bin.mkdir()
        self.git("init", "--bare", "-q", str(self.origin), cwd=root)
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "Release test")
        self.git("config", "user.email", "release-test@local.invalid")
        for relative in ("deploy/release/make_release.sh", "trainer/scripts/version_tag_check.py",
                         "trainer/scripts/release_notes.py"):
            self.write(relative, (ROOT / relative).read_text(encoding="utf-8"))
        self.write("trainer/__init__.py", "")
        self.write("trainer/scripts/__init__.py", "")
        self.write(VERSION_FILE, '__version__ = "3.9.0"\n')
        self.write("trainer/demo_api/service.py", "VALUE = 1\n")
        self.write("deploy/release/release_files.txt", VERSION_FILE + "\ntrainer/demo_api/service.py\n")
        self.write("deploy/release/README.release.md", "Deployment README\n")
        self.write("CHANGELOG.md", "# Changes\n\n## [3.9.0]\n\nRelease workflow test.\n")
        self.git("add", "--", ".")
        self.git("commit", "-q", "-m", "verified main source")
        self.source = self.git("rev-parse", "HEAD")
        empty = self.git("mktree", input="")
        self.previous = self.git("commit-tree", empty, "-m", "previous independent deployment")
        self.git("update-ref", "refs/heads/release", self.previous)
        self.git("remote", "add", "origin", str(self.origin))
        self.git("push", "-q", "origin", "main", "release")
        self.git("symbolic-ref", "HEAD", "refs/heads/main", bare=True)
        self.expected = self.expected_tree()
        self.state_path.write_text(json.dumps({"releases": {}, "calls": []}), encoding="utf-8")
        (self.bin / "gh").write_text(FAKE_GH, encoding="utf-8")
        (self.bin / "gh").chmod(0o755)

    def git(self, *args, cwd=None, bare=False, input=None, env=None):
        prefix = ["git", "--git-dir", str(self.origin)] if bare else ["git", "-C", str(cwd or self.setup)]
        result = subprocess.run([*prefix, *args], input=input, env=env, capture_output=True,
                                text=True, timeout=30, check=True)
        return result.stdout.strip()

    def write(self, relative, text):
        path = self.setup / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def expected_tree(self):
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            env = {**os.environ, "GIT_INDEX_FILE": str(Path(temporary) / "index")}
            self.git("read-tree", "--empty", env=env)
            for destination, original in [(VERSION_FILE, VERSION_FILE),
                    ("trainer/demo_api/service.py", "trainer/demo_api/service.py"),
                    ("README.md", "deploy/release/README.release.md")]:
                blob = self.git("rev-parse", f"{self.source}:{original}")
                self.git("update-index", "--add", "--cacheinfo", f"100644,{blob},{destination}", env=env)
            return self.git("write-tree", env=env)

    def configure(self, **values):
        state = self.state;state.update(values)
        self.state_path.write_text(json.dumps(state), encoding="utf-8")

    @property
    def state(self):
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def ref(self, path):
        return self.git("rev-parse", "--verify", path, bare=True)

    def run(self, **overrides):
        self.runs += 1
        run = self.root / f"run-{self.runs}"
        checkout = run / "checkout";run.mkdir()
        self.git("clone", "-q", "--no-hardlinks", str(self.origin), str(checkout), cwd=self.root)
        self.git("checkout", "-q", "--detach", self.source, cwd=checkout)
        environment = {**os.environ, "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
            "PYTHONPATH": str(checkout), "SOURCE_SHA": self.source, "RELEASE_TAG": TAG,
            "EXPECTED_RELEASE_TREE": self.expected, "MAKE_LATEST": "true", "GH_REPO": "local/sprayxr",
            "FAKE_GH_STATE": str(self.state_path), "FAKE_ORIGIN": str(self.origin),
            "RUNNER_TEMP": str(run), "GITHUB_ENV": str(run / "env"),
            "GITHUB_STEP_SUMMARY": str(run / "summary"), "GITHUB_RUN_ID": str(self.runs),
            "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_REF": "refs/heads/main",
            "GIT_ALLOW_PROTOCOL": "file",
            "GITHUB_SHA": self.source, "GITHUB_WORKSPACE": str(checkout), "GH_TOKEN": "fake-test-token",
            **overrides}
        workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8"))
        output = []
        for step in workflow["jobs"]["publish"]["steps"]:
            if "run" not in step:
                continue
            assert "if" not in step, "Keep resume guards in shell or model the exact Actions condition here"
            result = subprocess.run(["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", step["run"]],
                                    cwd=checkout, env=environment, capture_output=True, text=True, timeout=40)
            output.append(f"{step['name']}\n{result.stdout}\n{result.stderr}")
            if result.returncode:
                return SimpleNamespace(returncode=result.returncode, step=step["name"], output="\n".join(output))
            environment_file = run / "env"
            if environment_file.exists():
                for line in environment_file.read_text(encoding="utf-8").splitlines():
                    key, value = line.split("=", 1);environment[key] = value
        return SimpleNamespace(returncode=0, step="complete", output="\n".join(output))


@pytest.fixture
def case(tmp_path):
    return ReleaseCase(tmp_path)


def test_fresh_workflow_publishes_exact_independent_tree_and_annotated_tag(case):
    result = case.run()
    assert result.returncode == 0, result.output
    tag = case.ref("refs/tags/" + TAG)
    assert case.git("cat-file", "-t", tag, bare=True) == "tag"
    commit = case.ref(TAG + "^{commit}")
    assert case.ref("refs/heads/release") == commit
    assert case.ref(commit + "^{tree}") == case.expected
    assert case.git("show", "-s", "--format=%P", commit, bare=True) == case.previous
    assert set(case.git("ls-tree", "-r", "--name-only", commit, bare=True).splitlines()) == {
        VERSION_FILE, "trainer/demo_api/service.py", "README.md"}
    assert case.state["latest"] == TAG
    receipt = json.loads(case.git("notes", "--ref=refs/notes/releases/" + TAG, "show", commit, bare=True))
    assert receipt["source_main_commit"] == case.source
    assert receipt["release_tree"] == case.expected


def test_workflow_resumes_after_atomic_push_and_temporary_publication_failure(case):
    case.configure(fail_create_once=True)
    first = case.run()
    assert first.returncode != 0
    original_tag = case.ref("refs/tags/" + TAG)
    original_commit = case.ref("refs/heads/release")
    assert TAG not in case.state["releases"]
    second = case.run()
    assert second.returncode == 0, second.output
    assert case.ref("refs/tags/" + TAG) == original_tag
    assert case.ref("refs/heads/release") == original_commit
    assert TAG in case.state["releases"]


def test_workflow_resumes_after_published_release_receipt_failure_and_reuses_notes(case):
    case.configure(fail_api_suffix="releases/latest")
    first = case.run()
    assert first.returncode != 0
    original_tag = case.ref("refs/tags/" + TAG)
    assert TAG in case.state["releases"]
    second = case.run()
    assert second.returncode == 0, second.output
    assert case.ref("refs/tags/" + TAG) == original_tag
    original_notes = case.ref("refs/notes/releases/" + TAG)
    third = case.run()
    assert third.returncode == 0, third.output
    assert case.ref("refs/tags/" + TAG) == original_tag
    assert case.ref("refs/notes/releases/" + TAG) == original_notes


def test_wrong_prechecked_tree_is_rejected_before_remote_publication(case):
    result = case.run(EXPECTED_RELEASE_TREE="0" * 40)
    assert result.returncode != 0
    assert case.ref("refs/heads/release") == case.previous
    assert subprocess.run(["git", "--git-dir", str(case.origin), "rev-parse", "--verify", "refs/tags/" + TAG],
                          capture_output=True).returncode != 0
    assert not case.state["releases"]


def test_stale_main_source_is_rejected_before_remote_publication(case):
    case.write("new-main-note.txt", "main changed after operator verification\n")
    case.git("add", "--", "new-main-note.txt")
    case.git("commit", "-q", "-m", "concurrent main update")
    case.git("push", "-q", "origin", "main")
    result = case.run()
    assert result.returncode != 0
    assert case.ref("refs/heads/release") == case.previous
    assert subprocess.run(["git", "--git-dir", str(case.origin), "rev-parse", "--verify", "refs/tags/" + TAG],
                          capture_output=True).returncode != 0
    assert not case.state["releases"]


@pytest.mark.parametrize("mismatch", ["source", "tree"])
def test_resume_requires_the_original_source_and_prechecked_tree(case, mismatch):
    first = case.run()
    assert first.returncode == 0, first.output
    tag_before, branch_before = case.ref("refs/tags/" + TAG), case.ref("refs/heads/release")
    if mismatch == "source":
        # Documentation-only main changes keep the runtime tree identical, but
        # do not turn an old source attestation into an attestation for the new SHA.
        case.write("later-main-note.txt", "same runtime, different source commit\n")
        case.git("add", "--", "later-main-note.txt")
        case.git("commit", "-q", "-m", "later documentation source")
        case.git("push", "-q", "origin", "main")
        case.source = case.git("rev-parse", "HEAD")
        assert case.expected_tree() == case.expected
        result = case.run()
    else:
        result = case.run(EXPECTED_RELEASE_TREE="0" * 40)
    assert result.returncode != 0
    assert case.ref("refs/tags/" + TAG) == tag_before
    assert case.ref("refs/heads/release") == branch_before


@pytest.mark.parametrize("tamper", ["lightweight_tag", "wrong_tag_target", "release_branch"])
def test_resume_refuses_tampered_remote_refs_instead_of_rewriting_them(case, tamper):
    first = case.run()
    assert first.returncode == 0, first.output
    if tamper == "lightweight_tag":
        case.git("update-ref", "refs/tags/" + TAG, case.source, bare=True)
    elif tamper == "wrong_tag_target":
        payload = (f"object {case.source}\ntype commit\ntag {TAG}\n"
                   "tagger Release test <release-test@local.invalid> 0 +0000\n\nWrong target\n")
        wrong_tag = case.git("mktag", input=payload, bare=True)
        case.git("update-ref", "refs/tags/" + TAG, wrong_tag, bare=True)
    else:
        case.git("update-ref", "refs/heads/release", case.source, bare=True)
    tag_before, branch_before = case.ref("refs/tags/" + TAG), case.ref("refs/heads/release")
    release_before = case.state["releases"][TAG].copy()
    result = case.run()
    assert result.returncode != 0
    assert case.ref("refs/tags/" + TAG) == tag_before
    assert case.ref("refs/heads/release") == branch_before
    assert case.state["releases"][TAG] == release_before
