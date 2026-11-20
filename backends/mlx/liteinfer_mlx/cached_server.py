"""Version-pinned MLX-LM HTTP adapter with indexed, tenant-scoped SSD caching."""
from __future__ import annotations

import argparse
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import hmac
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import sys

import mlx.core as mx
from mlx_lm import server

from .cache_settings import CacheConfig
from .prompt_cache import TieredPromptCache

TENANT: ContextVar[str] = ContextVar("cache_tenant", default="")
SINGLE_OWNER: ContextVar[ScopedGenerator | None] = ContextVar("single_cache_owner", default=None)
ORIGINAL_STREAM_GENERATE = server.stream_generate


def model_fingerprint(paths: tuple[str | None, ...]) -> str:
    records: list[object] = ["prompt-v1", version("mlx"), version("mlx-lm"), paths]
    for value in paths:
        if value is None:
            continue
        directory = Path(value)
        if not directory.is_dir():
            raise ValueError("SSD prompt caching requires local model and adapter directories")
        for path in sorted(directory.rglob("*")):
            if path.suffix not in (".safetensors", ".json", ".jinja", ".model", ".tiktoken") or not path.is_file():
                continue
            info = path.stat()
            item: list[object] = [str(path.relative_to(directory)), info.st_size, info.st_mtime_ns, info.st_ctime_ns]
            if path.suffix in (".json", ".jinja"):
                item.append(hashlib.sha256(path.read_bytes()).hexdigest())
            records.append(item)
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


class ScopedName(str):
    tenant: str

    def __new__(cls, value: str, tenant: str) -> "ScopedName":
        result = super().__new__(cls, value)
        result.tenant = tenant
        return result


@dataclass
class ScopedDescription(server.ModelDescription):
    tenant: str = field(default_factory=TENANT.get)

    def __post_init__(self) -> None:
        self.model = ScopedName(self.model, self.tenant)


class ScopedProvider(server.ModelProvider):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.base_key: tuple[str | None, ...] | None = None
        self.fingerprints: dict[tuple[str | None, ...], str] = {}

    def load(self, model_path: str, adapter_path: str | None = None, draft_model_path: str | None = None):
        # Only cache identity changes with the tenant; weights remain resident.
        self.model_key = self.base_key
        try:
            result = super().load(str(model_path), adapter_path, draft_model_path)
        except Exception:
            self.base_key = self.model_key = None
            raise
        self.base_key = self.model_key
        if self.base_key not in self.fingerprints:
            self.fingerprints[self.base_key] = model_fingerprint(self.base_key)
        tenant = model_path.tenant if isinstance(model_path, ScopedName) else ""
        self.model_key = (self.fingerprints[self.base_key], tenant)
        return result


class ScopedGenerator(server.ResponseGenerator):
    def _tokenize(self, tokenizer, request, args):
        prompt, segments, kinds, initial, vision = super()._tokenize(tokenizer, request, args)
        self.current_prompt = prompt
        self.current_vision = vision
        # Saving the state before the final input token gives hybrid models a
        # reusable checkpoint even when the template has no natural segments.
        if vision is None and segments and len(segments[-1]) > 1:
            last, kind = segments.pop(), kinds.pop()
            segments.extend((last[:-1], last[-1:]))
            kinds.extend((kind, "assistant"))
        return prompt, segments, kinds, initial, vision

    def _serve_single(self, request, stream):
        context = SINGLE_OWNER.set(self)
        try:
            return super()._serve_single(request, stream)
        finally:
            SINGLE_OWNER.reset(context)


def checkpoint_stream(*args, **kwargs):
    owner = SINGLE_OWNER.get()
    if owner is not None and owner.current_vision is None and kwargs.get("draft_model") is None:
        callback = kwargs["prompt_progress_callback"]
        full = owner.current_prompt.copy()
        state = kwargs["prompt_cache"]
        model = owner.model_provider.model_key
        remaining = len(kwargs["prompt"])

        def progress(processed: int, total: int) -> None:
            if 0 < processed == total - 1:
                consumed = len(full) - remaining + processed
                owner.prompt_cache.insert_cache(model, full[:consumed], state, cache_type="user")
            callback(processed, total)

        kwargs["prompt_progress_callback"] = progress
    return ORIGINAL_STREAM_GENERATE(*args, **kwargs)


class CacheHandler(server.APIHandler):
    def handle_completion(self, request: server.CompletionRequest, stop_words: list[str]):
        salt = self.body.get("cache_salt", "")
        if not isinstance(salt, str) or len(salt) > 512:
            self._cache_response(400, {"error": "cache_salt must be a string of at most 512 characters"})
            return
        context = TENANT.set(hashlib.sha256(salt.encode()).hexdigest())
        try:
            return super().handle_completion(request, stop_words)
        finally:
            TENANT.reset(context)

    def _cache_response(self, status: int, body: dict[str, object]) -> None:
        encoded = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _cache_admin(self) -> bool:
        required = os.environ.get("LITEINFER_CACHE_ADMIN_TOKEN")
        supplied = self.headers.get("Authorization", "")
        if not required or not hmac.compare_digest(supplied.encode(), f"Bearer {required}".encode()):
            self._cache_response(403, {"error": "Cache administration requires the backend admin token"})
            return False
        return True

    def do_GET(self) -> None:
        if self.path != "/cache/stats":
            return super().do_GET()
        if self._cache_admin():
            self._cache_response(200, self.response_generator.prompt_cache.snapshot())

    def do_DELETE(self) -> None:
        if self.path != "/cache/stats":
            self._cache_response(404, {"error": "Unknown cache endpoint"})
        elif self._cache_admin():
            self.response_generator.prompt_cache.clear()
            self._cache_response(200, self.response_generator.prompt_cache.snapshot())


def install(config: CacheConfig, directory: Path | None) -> None:
    if version("mlx-lm") != "0.32.2":
        raise RuntimeError("Cached server adapter requires mlx-lm==0.32.2")
    server.ModelDescription, server.ModelProvider = ScopedDescription, ScopedProvider
    server.stream_generate = checkpoint_stream

    def run(host: str, port: int, provider: ScopedProvider) -> None:
        if mx.distributed.init().size() != 1:
            raise ValueError("The SSD cache adapter currently supports one local GPU process")
        cache = TieredPromptCache(config, directory)
        generator = ScopedGenerator(provider, cache)
        try:
            server._run_http_server(host, port, generator, handler_class=CacheHandler)
        finally:
            generator.stop_and_join()
            cache.close()

    server.run = run


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cache-config", type=Path)
    parser.add_argument("--cache-directory", type=Path)
    args, remaining = parser.parse_known_args()
    config = CacheConfig.from_file(args.cache_config) if args.cache_config else CacheConfig()
    install(config, args.cache_directory)
    sys.argv = [sys.argv[0], *remaining]
    signal.signal(signal.SIGTERM, lambda *_: signal.raise_signal(signal.SIGINT))
    server.main()


if __name__ == "__main__":
    main()
