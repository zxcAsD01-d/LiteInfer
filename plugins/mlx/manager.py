"""Local MLX lifecycle plugin. JSON lines on stdout; never execute remote code."""
from contextlib import redirect_stdout
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import uuid
from urllib.parse import urlsplit

from model_config import ModelError, metadata, read_json

DEPENDENCIES = ["mlx==0.32.4", "mlx-lm==0.32.2", "transformers==5.23.0",
                "huggingface-hub==2.4.0", "modelscope==1.40.3"]
PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "*.jinja", "LICENSE*"]


def emit(stage: str, **fields) -> None:
    print(json.dumps({"stage": stage, **fields}, ensure_ascii=False), flush=True)


def apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def validate_source(request: dict) -> dict:
    source = request.get("source", "huggingface")
    repo = request.get("repository", "")
    if source not in ("huggingface", "modelscope", "local"):
        raise ModelError("invalid_source")
    if not isinstance(repo, str) or not repo.strip() or len(repo) > 1024:
        raise ModelError("repository_required")
    if source != "local" and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo):
        raise ModelError("invalid_repository")
    endpoint = request.get("endpoint") or None
    if endpoint:
        url = urlsplit(endpoint)
        local = url.hostname in ("localhost", "127.0.0.1", "::1")
        if not url.hostname or url.username or url.password or url.query or url.fragment or url.path not in ("", "/"):
            raise ModelError("invalid_endpoint")
        if url.scheme != "https" and not (url.scheme == "http" and local):
            raise ModelError("endpoint_requires_https")
    token_env = request.get("token_env")
    if token_env is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token_env):
        raise ModelError("invalid_token_env")
    revision = request.get("revision") or None
    if revision and (not isinstance(revision, str) or len(revision) > 256 or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in revision)):
        raise ModelError("invalid_revision")
    return {"source": source, "repository": repo, "endpoint": endpoint,
            "revision": revision, "token_env": token_env}


def download(request: dict, directory: Path, config_only: bool = False) -> Path:
    source = validate_source(request)
    if source["source"] == "local":
        directory = Path(source["repository"]).expanduser().resolve(strict=True)
        if not directory.is_dir():
            raise ModelError("local_model_must_be_directory")
        return directory
    patterns = ["config.json", "tokenizer_config.json", "chat_template.jinja"] if config_only else PATTERNS
    token = os.environ.get(source["token_env"]) if source["token_env"] else None
    if source["token_env"] and not token:
        raise ModelError("source_token_env_missing")
    with redirect_stdout(sys.stderr):
        if source["source"] == "huggingface":
            from huggingface_hub import snapshot_download
            result = snapshot_download(repo_id=source["repository"], revision=source["revision"],
                endpoint=source["endpoint"], token=token or False, local_dir=str(directory),
                allow_patterns=patterns, max_workers=4)
        else:
            from modelscope.hub.snapshot_download import snapshot_download
            result = snapshot_download(model_id=source["repository"], revision=source["revision"],
                endpoint=source["endpoint"], token=token, local_dir=str(directory),
                allow_patterns=patterns, max_workers=4)
    return Path(result)


def runtime_status(root: Path) -> dict:
    versions = {}
    for requirement in DEPENDENCIES:
        name = requirement.split("==")[0]
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    artifacts = []
    for path in sorted((root / "artifacts").glob("*/manifest.json")):
        record = read_json(path)
        artifacts.append({"artifact": path.parent.name, "alias": record["alias"], "metadata": record["metadata"]})
    return {"apple_silicon": apple_silicon(), "versions": versions,
            "ready": apple_silicon() and all(versions[r.split("==")[0]] == r.split("==")[1] for r in DEPENDENCIES),
            "artifacts": artifacts}


def install_runtime(root: Path) -> dict:
    if not apple_silicon() or sys.version_info < (3, 11):
        raise ModelError("requires_apple_silicon_python311")
    runtime = root / "runtime"
    emit("creating_environment")
    subprocess.run([sys.executable, "-m", "venv", str(runtime)], check=True, stdout=sys.stderr)
    emit("installing_dependencies")
    subprocess.run([str(runtime / "bin/python"), "-m", "pip", "install", "--disable-pip-version-check",
                    *DEPENDENCIES], check=True, stdout=sys.stderr)
    return {"installed": True}


def supported(directory: Path) -> dict:
    from mlx_lm.utils import _get_classes
    config = read_json(directory / "config.json")
    try:
        _get_classes(config)
    except (ImportError, ValueError) as error:
        raise ModelError("architecture_not_supported_by_mlx_lm") from error
    return metadata(directory, weights=False)


def precision_options(request: dict, info: dict) -> dict:
    precision = request.get("precision", "q4")
    if precision not in ("q4", "q8", "float16", "keep"):
        raise ModelError("invalid_precision")
    bits = 4 if precision == "q4" else 8
    group = request.get("group_size", 64)
    if type(group) is not int or group not in (32, 64, 128):
        raise ModelError("invalid_quantization_group")
    existing = info["quantization"]
    if existing and precision != "keep":
        if not isinstance(existing, dict) or precision not in ("q4", "q8") or existing.get("bits") != bits:
            raise ModelError("requantization_requires_unquantized_source")
    return {"quantize": not existing and precision in ("q4", "q8"), "q_bits": bits,
            "q_group_size": group, "dtype": "float16" if precision != "keep" and not existing else None}


def self_test(directory: Path) -> dict:
    import mlx.core as mx
    from mlx_lm import load
    with redirect_stdout(sys.stderr):
        model, tokenizer = load(str(directory), tokenizer_config={"trust_remote_code": False}, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ModelError("chat_template_required_for_chat_service")
    tokens = tokenizer.apply_chat_template([{"role": "user", "content": "Hello"}], tokenize=True, add_generation_prompt=True)
    logits = model(mx.array([tokens]))
    mx.eval(logits)
    if not mx.all(mx.isfinite(logits)).item() or logits.shape[-1] < len(tokenizer):
        raise ModelError("model_forward_or_vocabulary_check_failed")
    return metadata(directory)


def adapt_chat_end(directory: Path) -> None:
    """Preserve EOS and add ChatML's explicit message boundary for chat templates."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(directory), trust_remote_code=False)
    template = tokenizer.chat_template
    if not isinstance(template, str) or "<|im_end|>" not in template:
        return
    token = tokenizer.get_vocab().get("<|im_end|>")
    if token is None or tokenizer.encode("<|im_end|>", add_special_tokens=False) != [token]:
        return
    config = read_json(directory / "config.json")
    path = directory / "generation_config.json"
    generation = read_json(path) if path.exists() else {}
    existing = generation.get("eos_token_id", config.get("eos_token_id", []))
    ids = [existing] if type(existing) is int else list(existing or [])
    if token not in ids:
        ids.append(token)
        generation["eos_token_id"] = ids
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(generation, ensure_ascii=False))
        temporary.replace(path)


def prepare(root: Path, request: dict) -> dict:
    from mlx_lm.convert import convert
    artifact = uuid.uuid4().hex
    work = root / "artifacts" / artifact
    work.mkdir(parents=True)
    emit("downloading_config", artifact=artifact)
    source = download(request, work / "source", config_only=True)
    info = supported(source)
    options = precision_options(request, info)
    emit("downloading_weights", artifact=artifact, metadata=info)
    source = download(request, work / "source")
    metadata(source)
    output = work / "model"
    emit("converting", artifact=artifact)
    with redirect_stdout(sys.stderr):
        convert(str(source), str(output), trust_remote_code=False, **options)
    emit("validating", artifact=artifact)
    adapt_chat_end(output)
    info = self_test(output)
    alias = request.get("alias", "local-model")
    manifest = {"alias": alias, "metadata": info, "precision": request.get("precision", "q4")}
    temporary = work / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False))
    temporary.replace(work / "manifest.json")
    return {"artifact": artifact, "alias": alias, "metadata": info}


def artifact_path(root: Path, artifact: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", artifact):
        raise ModelError("invalid_artifact")
    directory = root / "artifacts" / artifact
    if not directory.resolve().is_relative_to(root.resolve()):
        raise ModelError("artifact_outside_directory")
    read_json(directory / "manifest.json")
    return directory / "model"


def main() -> None:
    request = json.load(sys.stdin)
    root = Path(request.pop("root")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    action = request["action"]
    if action == "status":
        result = runtime_status(root)
    elif action == "setup":
        result = install_runtime(root)
    elif action == "inspect":
        directory = download(request, root / "inspection" / uuid.uuid4().hex, config_only=True)
        result = supported(directory)
    elif action == "prepare":
        result = prepare(root, request)
    elif action == "serve":
        directory = artifact_path(root, request["artifact"])
        adapt_chat_end(directory)
        count, port = request["concurrency"], request["port"]
        args = [sys.executable, "-m", "mlx_lm", "server", "--model", str(directory),
                "--host", "127.0.0.1", "--port", str(port), "--decode-concurrency", str(count),
                "--prompt-concurrency", str(min(count, 8)), "--prefill-step-size", "128",
                "--prompt-cache-bytes", "268435456", "--temp", "0.7", "--top-k", "64", "--top-p", "0.9",
                "--log-level", "ERROR"]
        os.execv(sys.executable, args)
    else:
        raise ModelError("invalid_action")
    emit("completed", result=result)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Third-party exceptions can contain source tokens and private paths.
        code = str(error) if isinstance(error, ModelError) else type(error).__name__
        emit("failed", code=code)
        sys.exit(1)
