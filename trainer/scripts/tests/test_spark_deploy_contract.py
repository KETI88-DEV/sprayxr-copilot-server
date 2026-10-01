"""deploy/spark 정의가 rag_config_spark_vllm.yaml, 번들 배치, 코드와 어긋나지 않는지 단언한다.

compose 문법은 호스트의 ``docker compose -f deploy/spark/compose.yaml --env-file deploy/spark/.env.example config``로
검증한다. 여기서는 값의 정합성을 본다: vLLM argv는 실제 entrypoint.sh를 dry-run으로 돌려 얻는다.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import argparse
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import urlparse

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
SPARK = REPO / "deploy" / "spark"
COMPOSE = SPARK / "compose.yaml"
ENV_EXAMPLE = SPARK / ".env.example"
ENTRYPOINT = SPARK / "vllm" / "entrypoint.sh"
NGINX = SPARK / "web" / "nginx.conf"
AGENT_DOCKERFILE = SPARK / "images" / "agent.Dockerfile"
AGENT_LOCK = SPARK / "requirements" / "agent-api.lock.txt"
MCP_LOCK = SPARK / "requirements" / "mcp.lock.txt"
RAG_CONFIG = REPO / "trainer" / "configs" / "rag_config_spark_vllm.yaml"
BUNDLE = "/BUNDLE"

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:(:?[-?])([^}]*))?\}")
_FORBIDDEN_VLLM_FLAGS = ("--enable-auto-tool-choice", "--tool-call-parser", "--reasoning-parser")


def _load_script(name: str) -> ModuleType:
    module_name = f"spark_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, SPARK / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@functools.cache
def _compose() -> dict[str, Any]:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


@functools.cache
def _rag() -> dict[str, Any]:
    return yaml.safe_load(RAG_CONFIG.read_text(encoding="utf-8"))


@functools.cache
def _env_example() -> dict[str, str]:
    env: dict[str, str] = {}
    for raw in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        assert sep, f"malformed .env.example line: {raw!r}"
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        env[key.strip()] = value
    return env


def _interpolate(text: str, env: Mapping[str, str]) -> str:
    def _sub(match: re.Match[str]) -> str:
        name, op, arg = match.group(1), match.group(2), match.group(3)
        value = env.get(name)
        if op == ":-":
            return value if value else arg
        if op == "-":
            return value if value is not None else arg
        if op in (":?", "?"):
            assert value, f"compose requires {name}: {arg}"
            return value
        return value or ""

    return _VAR.sub(_sub, text)


def _env(**overrides: str) -> dict[str, str]:
    return {**_env_example(), "BUNDLE_DIR": BUNDLE, "RESULTS_DIR": "/RESULTS", **overrides}


def _service(name: str) -> dict[str, Any]:
    return _compose()["services"][name]


def _service_env(name: str, env: Mapping[str, str]) -> dict[str, str]:
    return {key: _interpolate(str(value), env) for key, value in _service(name)["environment"].items()}


def _volumes(name: str, env: Mapping[str, str]) -> list[tuple[str, str]]:
    pairs = []
    for spec in _service(name).get("volumes", []):
        parts = _interpolate(spec, env).split(":")
        pairs.append((parts[0], parts[1]))
    return pairs


def _run_entrypoint(**overrides: str) -> subprocess.CompletedProcess[str]:
    bash = shutil.which("bash")
    assert bash is not None
    env = {"PATH": "/usr/bin:/bin", **_service_env("vllm", _env(**overrides)), "VLLM_DRY_RUN": "1"}
    return subprocess.run([bash, str(ENTRYPOINT)], env=env, capture_output=True, text=True, check=False, timeout=30)


def _vllm_argv(**overrides: str) -> list[str]:
    proc = _run_entrypoint(**overrides)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.splitlines()


def _flag(argv: list[str], flag: str) -> str:
    assert argv.count(flag) == 1, f"{flag} appears {argv.count(flag)} times in {argv}"
    return argv[argv.index(flag) + 1]


def _command_port(name: str) -> int:
    command = _service(name)["command"]
    return int(command[command.index("--port") + 1])


def test_vllm_serves_the_name_and_window_the_agent_config_expects() -> None:
    argv = _vllm_argv()
    llm = _rag()["llm"]

    assert argv[:3] == ["vllm", "serve", "/models/shipcoat"]
    assert _flag(argv, "--served-model-name") == llm["served_model_name"]
    assert int(_flag(argv, "--max-model-len")) == int(llm["max_model_len"])
    assert "--enable-prefix-caching" in argv
    assert _flag(argv, "--generation-config") == "vllm"


def test_vllm_argv_never_carries_server_tool_or_reasoning_parsers() -> None:
    argv = _vllm_argv(ENABLE_MTP="1", LANGUAGE_MODEL_ONLY="1")
    assert not [arg for arg in argv if arg.startswith(_FORBIDDEN_VLLM_FLAGS)]
    assert not _rag()["llm"]["pass_native_tools"]

    for flag in (*_FORBIDDEN_VLLM_FLAGS, "--api-key"):
        proc = _run_entrypoint(VLLM_EXTRA_ARGS=f"{flag} value")
        assert proc.returncode == 2, (flag, proc.stdout)


def test_vllm_generation_override_matches_config_sampling() -> None:
    override = json.loads(_flag(_vllm_argv(), "--override-generation-config"))
    generation = _rag()["generation"]

    assert override == {key: generation[key] for key in ("temperature", "top_p", "top_k", "repetition_penalty")}


def test_mtp_is_off_by_default_and_adds_one_mtp_token_when_enabled() -> None:
    assert "--speculative-config" not in _vllm_argv()

    speculative = json.loads(_flag(_vllm_argv(ENABLE_MTP="1"), "--speculative-config"))
    assert speculative == {"method": "mtp", "num_speculative_tokens": 1}

    assert _run_entrypoint(ENABLE_MTP="yes").returncode == 2
    assert _run_entrypoint(ENABLE_MTP="1", MTP_NUM_SPECULATIVE_TOKENS="0").returncode == 2


def test_compose_defaults_agree_with_env_example() -> None:
    raw = COMPOSE.read_text(encoding="utf-8")
    example = _env_example()
    for name in ("SERVED_MODEL_NAME", "MAX_MODEL_LEN", "MODEL_SUBDIR", "MAX_NUM_SEQS", "GPU_MEMORY_UTILIZATION", "ENABLE_MTP"):
        defaults = set(re.findall(r"\$\{" + name + r":-([^}]*)\}", raw))
        assert defaults == {example[name]}, (name, defaults, example.get(name))


def test_agent_api_environment_contract() -> None:
    env = _service_env("agent-api", _env())
    config_path = Path(env["CONFIG_PATH"])

    assert env["DEMO_LIVE_MODE"] == "on"
    assert config_path == Path("/workspace/trainer/configs/rag_config_spark_vllm.yaml")
    assert (REPO / config_path.relative_to("/workspace")).is_file()
    assert env["LIVE_EAGER_INIT"] == "1"
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["TOOL_ADAPTER_NATIVE_TOOLS"] == "0"
    assert env["TOOL_ADAPTER_FORCE_SYNTHESIS"] == "1"
    assert env["ADAPTER_PATH"] == ""
    assert "--workers" not in _service("agent-api")["command"]
    assert _service("agent-api")["depends_on"]["vllm"]["condition"] == "service_healthy"
    assert _rag()["llm"]["api_key_env"] in env


def test_agent_api_reaches_vllm_and_mcp_by_service_name_and_port() -> None:
    base_url = urlparse(_rag()["llm"]["base_url"])
    mcp_url = urlparse(_service_env("agent-api", _env())["MCP_BASE_URL"])

    assert base_url.hostname == "vllm" and "vllm" in _compose()["services"]
    assert base_url.port == int(_flag(_vllm_argv(), "--port"))
    assert mcp_url.hostname == "mcp" and mcp_url.port == _command_port("mcp")
    assert any(f":{_command_port('agent-api')}" in port for port in _service("agent-api")["ports"])


def test_agent_config_paths_are_mounted_from_the_bundle_layout() -> None:
    bm = _load_script("bundle_manifest")
    rag = _rag()
    mounts = [(src, dst) for src, dst in _volumes("agent-api", _env()) if src.startswith(BUNDLE + "/")]

    def bundle_location(container_path: str) -> str:
        for src, dst in mounts:
            if container_path == dst or container_path.startswith(dst + "/"):
                return src[len(BUNDLE) + 1 :] + container_path[len(dst) :]
        raise AssertionError(f"{container_path} is not mounted into agent-api")

    assert bundle_location(rag["embedding"]["model_name"]) == bm.GROUPS["embedding"][0]
    assert bundle_location(rag["reranker"]["model_name"]) == bm.GROUPS["reranker"][0]
    assert bundle_location(rag["llm"]["tokenizer_path"]) == bm.GROUPS["model_fp8"][0]
    for key in ("faiss_index", "metadata_json"):
        location = bundle_location(rag["paths"][key])
        assert location.startswith(bm.GROUPS["rag_index"][0] + "/")
        assert location.rsplit("/", 1)[1] in bm.GROUPS["rag_index"][1]


def test_vllm_mounts_the_fp8_model_group_by_default() -> None:
    bm = _load_script("bundle_manifest")
    model_mounts = [src for src, dst in _volumes("vllm", _env()) if dst == "/models/shipcoat"]

    assert model_mounts == [f"{BUNDLE}/{bm.GROUPS['model_fp8'][0]}"]


def test_mcp_pose_reference_path_is_mounted_and_matches_the_code_default() -> None:
    from mcp_servers import pose_estimation

    env = _service_env("mcp", _env())
    path = env[pose_estimation.POSE_REFERENCE_ENV]
    assert path == pose_estimation.DEFAULT_POSE_REFERENCE
    assert any(path.startswith(dst + "/") for _src, dst in _volumes("mcp", _env()))


def test_mcp_arm_reference_path_is_mounted_and_matches_the_code_default() -> None:
    """팔 레퍼런스는 **채점 경로의 유일한 기준**이다.

    경로가 어긋나거나 마운트에 없으면 MCP는 정상으로 뜨고 **모든 시퀀스가
    `reference_data_missing`** 이 된다 — 서비스가 살아 있으니 헬스체크도 통과한다.
    조용히 채점만 비는 실패라 계약으로 묶는다.
    """
    from mcp_servers import pose_estimation

    env = _service_env("mcp", _env())
    path = env[pose_estimation.ARM_REFERENCE_ENV]
    assert path == pose_estimation.DEFAULT_ARM_REFERENCE
    assert any(path.startswith(dst + "/") for _src, dst in _volumes("mcp", _env()))


def test_the_bundle_requires_the_arm_reference_file() -> None:
    """번들에 이 파일이 없으면 `verify`가 실패해야 한다.

    `mcp_pose_reference` 그룹이 필수 파일 없이 선택이었을 때는, 디렉터리만 있고
    아티팩트가 없는 번들도 통과했다. 그러면 배포가 조용히 채점 없이 뜬다.

    깨뜨리는 법: `GROUPS["mcp_pose_reference"]`의 필수 파일 목록을 비우면 실패한다.
    """
    from mcp_servers import pose_estimation

    manifest = _load_script("bundle_manifest")
    root, required = manifest.GROUPS["mcp_pose_reference"]
    assert "expert_arm_reference.json" in required
    # 번들 상대 경로가 컨테이너 경로와 1:1이어야 마운트가 맞는다.
    assert pose_estimation.DEFAULT_ARM_REFERENCE.endswith(f"/{root}/expert_arm_reference.json".replace("//", "/")) or \
        pose_estimation.DEFAULT_ARM_REFERENCE == f"/{root}/expert_arm_reference.json"


def test_prepare_bundle_refuses_to_build_without_the_arm_reference(tmp_path: Path) -> None:
    """빠진 채로 번들을 만들 수 있으면 배포 직전까지 아무도 모른다.

    `prepare_bundle`의 기본 경로가 코드 상수와 같은지도 함께 본다 — 갈리면
    번들에는 있고 컨테이너는 다른 곳을 보게 된다.
    """
    from mcp_servers import pose_estimation

    prepare = _load_script("prepare_bundle")
    assert prepare.DEFAULT_ARM_REFERENCE == pose_estimation.DEFAULT_ARM_REFERENCE

    # 소스 문자열을 훑지 않고 **실제로 부른다.** 문자열 검사는 `raise`를
    # `absent.append`로 바꿔도 통과했다(돌연변이가 살아남았다).
    root = _load_script("bundle_manifest").GROUPS["mcp_pose_reference"][0]
    with pytest.raises(prepare.PrepareError) as raised:
        prepare.plan_arm_reference(root, tmp_path / "missing-arm.json")
    assert "arm_reference" in str(raised.value)
    assert "reference_data_missing" in str(raised.value), "왜 치명적인지가 메시지에 없다"

    present = tmp_path / "expert_arm_reference.json"
    present.write_text("{}", encoding="utf-8")
    item = prepare.plan_arm_reference(root, present)
    assert item.group == "mcp_pose_reference"
    # 번들 상대 경로가 컨테이너 경로와 1:1이어야 마운트가 맞는다.
    assert pose_estimation.DEFAULT_ARM_REFERENCE == f"/{item.dst_rel}"


def test_nginx_proxies_api_to_agent_api_without_buffering() -> None:
    text = NGINX.read_text(encoding="utf-8")
    block = re.search(r"location /api/ \{(.*?)\n    \}", text, flags=re.DOTALL)
    assert block is not None
    directives = dict(
        (m.group(1), m.group(2).strip()) for m in re.finditer(r"^\s*([a-z_]+)\s+([^;]*);", block.group(1), flags=re.MULTILINE)
    )
    upstream = re.search(r"set \$agent_api (http://[^;]+);", text)
    llm = _rag()["llm"]

    assert upstream is not None
    assert urlparse(upstream.group(1)).hostname == "agent-api"
    assert urlparse(upstream.group(1)).port == _command_port("agent-api")
    assert directives["proxy_pass"] == "$agent_api"
    assert directives["proxy_buffering"] == "off"
    assert directives["proxy_http_version"] == "1.1"
    read_timeout = int(directives["proxy_read_timeout"].rstrip("s"))
    assert read_timeout > max(llm["request_timeout_sec"], llm["tool_deadline_sec"])


def _lock(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        assert re.fullmatch(r"[a-z0-9][a-z0-9._-]*==[A-Za-z0-9.+!]+", line), f"not an exact lowercase pin: {raw!r}"
        name, version = line.split("==")
        assert name not in pins, f"duplicate pin {name}"
        pins[name] = version
    return pins


def test_lock_files_pin_exact_versions_without_gpu_training_or_forbidden_packages() -> None:
    agent = _lock(AGENT_LOCK)
    forbidden = {"torch", "triton", "bitsandbytes", "vllm", "unsloth", "flagembedding"}

    for pins in (agent, _lock(MCP_LOCK)):
        assert not forbidden & set(pins)
        assert not [name for name in pins if name.startswith("nvidia-")]
    assert {"faiss-cpu", "sentence-transformers", "transformers", "peft", "httpx", "fastapi", "jinja2", "pyyaml"} <= set(agent)


def test_mcp_lock_is_a_version_consistent_subset_of_the_agent_lock() -> None:
    agent = _lock(AGENT_LOCK)
    mcp = _lock(MCP_LOCK)

    assert {name: agent.get(name) for name in mcp} == mcp


def test_behavior_critical_lock_versions_match_the_evaluated_container() -> None:
    agent = _lock(AGENT_LOCK)
    checked = 0
    for name in ("fastapi", "pydantic", "httpx", "transformers", "tokenizers", "sentence-transformers", "faiss-cpu", "peft"):
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        checked += 1
        assert agent[name] == installed, f"{name}: lock {agent[name]} != installed {installed}"
    if checked == 0:
        pytest.skip("not running in the provisioned container")


def test_mcp_image_copy_list_is_enough_to_import_the_mcp_app(tmp_path: Path) -> None:
    text = AGENT_DOCKERFILE.read_text(encoding="utf-8")
    stage = text.split("FROM base AS mcp", 1)[1].split("\nFROM ", 1)[0]
    copies = re.findall(r"^COPY (?!--)(\S+) (\S+)$", stage, flags=re.MULTILINE)
    sources = [(src, dst) for src, dst in copies if not src.startswith("deploy/")]
    assert sources, "no code COPY lines found in the mcp stage"

    for src, dst in sources:
        origin = REPO / src
        target = tmp_path / dst
        if origin.is_dir():
            shutil.copytree(origin, target, ignore=shutil.ignore_patterns("test_*.py", "__pycache__", "tests"))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, target)
    proc = subprocess.run(
        [sys.executable, "-E", "-s", "-c", "import mcp_servers.main"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


# 이미지의 import 스모크(Dockerfile의 RUN python3 -c)는 x86_64 개발 서버에서 빌드하지 않으므로
# lock에 빠진 의존성을 CI가 아니라 Spark 실물 빌드에서야 발견한다. 2026-09-30에 python-multipart가
# 그렇게 빠졌다(/api/v1/sessions/complete 가 multipart/form-data 다). 여기서는 lock(+ agent-api는
# Dockerfile이 따로 설치하는 torch 휠 폐포)만 심볼릭 링크한 그림자 site-packages를 만들고 `-S`로
# 실제 site-packages를 끊어, 이미지에 설치될 배포만 있는 환경에서 같은 import를 돌린다.
# 차단은 "모듈이 보이지 않게" 해야 한다 — meta_path에서 예외를 던지면 transformers의
# find_spec("PIL") 같은 설치 여부 탐침이 함께 깨져 없는 실패를 만든다.
_SMOKE_IMPORT = re.compile(r'RUN python3 -c "((?:import|from) [^"]*)"')

# 스테이지 -> (lock 파일, torch 휠을 따로 설치하는가)
_IMAGE_STAGES = {
    "mcp": (MCP_LOCK, False),
    "agent-api": (AGENT_LOCK, True),
}


def _normalize_dist(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _torch_wheel_closure() -> set[str]:
    """agent-api Dockerfile이 lock과 별도로 설치하는 torch 휠이 끌고 오는 배포 이름."""
    closure: set[str] = set()
    pending = ["torch"]
    while pending:
        name = _normalize_dist(pending.pop())
        if name in closure:
            continue
        closure.add(name)
        try:
            requires = metadata.distribution(name).requires or []
        except metadata.PackageNotFoundError:
            continue
        for raw in requires:
            if ";" in raw and "extra" in raw.split(";", 1)[1]:
                continue
            dep = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", raw)
            if dep is not None:
                pending.append(dep.group(0))
    return closure


def _shadow_site_packages(allowed: set[str], root: Path) -> Path:
    """`allowed` 배포의 top-level 경로와 메타데이터만 링크한 sys.path 항목."""
    root.mkdir(parents=True, exist_ok=True)
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if not name or _normalize_dist(name) not in allowed:
            continue
        base = Path(str(dist.locate_file("")))
        tops = {str(f).split("/")[0] for f in (dist.files or [])}
        info = getattr(dist, "_path", None)
        if info is not None:
            tops.add(Path(str(info)).name)  # importlib.metadata.version()이 찾을 수 있게
        for top in tops:
            if top.startswith(".."):
                continue
            source = base / top
            target = root / top
            if source.exists() and not target.exists():
                target.symlink_to(source)
    return root


def _stage_smoke_statements(stage_name: str) -> str:
    text = AGENT_DOCKERFILE.read_text(encoding="utf-8")
    stage = text.split(f"FROM base AS {stage_name}", 1)[1].split("\nFROM ", 1)[0]
    found = _SMOKE_IMPORT.search(stage)
    assert found is not None, f"{stage_name} 스테이지에서 import 스모크를 찾지 못했다"
    return found.group(1)


def _allowed_dists(stage_name: str) -> set[str]:
    lock_path, installs_torch = _IMAGE_STAGES[stage_name]
    allowed = {_normalize_dist(name) for name in _lock(lock_path)}
    return allowed | _torch_wheel_closure() if installs_torch else allowed


def _run_image_smoke(stage_name: str, allowed: set[str], shadow_root: Path) -> subprocess.CompletedProcess[str]:
    shadow = _shadow_site_packages(allowed, shadow_root)
    return subprocess.run(
        [sys.executable, "-S", "-c", _stage_smoke_statements(stage_name)],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": f"{shadow}{os.pathsep}{REPO}"},
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )


@pytest.fixture(name="provisioned_container")
def _provisioned_container() -> None:
    """그림자 경로는 설치본에서 링크하므로 컨테이너 밖에서는 돌 수 없다.

    이 모듈의 다른 단언은 설치본과 무관하므로 모듈 전체를 건너뛰지는 않는다.
    """
    for name in ("fastapi", "torch"):
        try:
            metadata.distribution(name)
        except metadata.PackageNotFoundError:
            pytest.skip("not running in the provisioned container")


@pytest.mark.parametrize("stage_name", sorted(_IMAGE_STAGES))
def test_image_lock_covers_every_import_its_build_smoke_performs(
    stage_name: str, tmp_path: Path, provisioned_container: None
) -> None:
    proc = _run_image_smoke(stage_name, _allowed_dists(stage_name), tmp_path / "shadow")

    assert proc.returncode == 0, f"{stage_name}: {proc.stdout[-1500:]}\n{proc.stderr[-3000:]}"


@pytest.mark.parametrize(
    ("stage_name", "dropped"),
    [
        ("agent-api", "python-multipart"),
        ("agent-api", "faiss-cpu"),
        ("agent-api", "httpx"),
        ("mcp", "fastapi"),
        ("mcp", "pydantic"),
    ],
)
def test_image_lock_smoke_fails_when_a_pinned_dependency_is_missing(
    stage_name: str, dropped: str, tmp_path: Path, provisioned_container: None
) -> None:
    """위 단언이 실패할 수 있음을 보인다 — 그림자 경로가 실제로 차단한다는 대조군."""
    allowed = _allowed_dists(stage_name)
    assert dropped in allowed

    proc = _run_image_smoke(stage_name, allowed - {dropped}, tmp_path / "shadow")

    assert proc.returncode != 0, f"{stage_name}: {dropped} 없이도 import가 통과했다 — 차단되지 않는다"
