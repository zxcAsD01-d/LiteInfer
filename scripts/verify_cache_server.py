#!/usr/bin/env python3
"""Isolated real-weight HTTP verification; never restarts an existing server."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from liteinfer_mlx.cache_settings import CacheConfig


def request(url: str, method: str = "GET", body: dict | None = None, token: str | None = None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = json.dumps(body).encode() if body is not None else None
    started = time.perf_counter()
    try:
        with urlopen(Request(url, data=payload, headers=headers, method=method), timeout=120) as response:
            return response.status, json.load(response), (time.perf_counter() - started) * 1000
    except HTTPError as error:
        return error.code, json.loads(error.read()), (time.perf_counter() - started) * 1000


@contextmanager
def running(model: Path, root: Path, port: int, token: str):
    config = CacheConfig(memory_max_entries=2, ssd_max_bytes=512 * 1024**2, min_tokens=1)
    settings = root / "settings.json"
    settings.write_text(json.dumps(config.as_dict()))
    env = {**os.environ, "LITEINFER_CACHE_ADMIN_TOKEN": token, "HF_HUB_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false"}
    command = [sys.executable, "-m", "liteinfer_mlx.cached_server", "--model", str(model),
               "--port", str(port), "--cache-config", str(settings),
               "--cache-directory", str(root / "ssd"), "--decode-concurrency", "3",
               "--prompt-concurrency", "3", "--log-level", "ERROR",
               "--chat-template-args", '{"enable_thinking":false}']
    with (root / "server.log").open("ab") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, env=env)
        try:
            wait_ready(process, root, port)
            yield f"http://127.0.0.1:{port}"
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def wait_ready(process, root: Path, port: int) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError((root / "server.log").read_text()[-6000:])
        try:
            if request(f"http://127.0.0.1:{port}/v1/models")[0] == 200:
                return
        except (URLError, TimeoutError):
            time.sleep(.1)
    raise TimeoutError("Cache HTTP test server did not become ready")


def completion(base: str, tenant: str, seeded: bool = False):
    body = {"model": "default_model", "cache_salt": tenant, "max_tokens": 8,
            "temperature": 0, "messages": [{"role": "user",
            "content": "Cache verification document. " * 64 + "Reply with one short sentence."}]}
    if seeded:
        body["seed"] = 7
    status, response, elapsed = request(base + "/v1/chat/completions", "POST", body)
    if status != 200:
        raise RuntimeError(f"Completion failed: {response}")
    return response, elapsed


def verify(model: Path, root: Path, port: int, token: str) -> dict[str, object]:
    with running(model, root, port, token) as base:
        assert request(base + "/cache/stats")[0] == 403
        first, cold_ms = completion(base, "tenant-a")
        warm, warm_ms = completion(base, "tenant-a")
        stats_a = request(base + "/cache/stats", token=token)[1]
        assert stats_a["hits"] >= 1, stats_a
        other, _ = completion(base, "tenant-b")
        stats_b = request(base + "/cache/stats", token=token)[1]
        assert stats_b["misses"] == stats_a["misses"] + 1
        restored, disk_ms = completion(base, "tenant-a")
        stats = request(base + "/cache/stats", token=token)[1]
        assert stats["ssd_hits"] > 0
        answers = [item["choices"][0]["message"] for item in (first, warm, other, restored)]
        assert all(answer == answers[0] for answer in answers)
        seeded, _ = completion(base, "seeded-tenant", seeded=True)
        seeded_warm, _ = completion(base, "seeded-tenant", seeded=True)
        assert seeded["choices"][0]["message"] == seeded_warm["choices"][0]["message"]
        seeded_stats = request(base + "/cache/stats", token=token)[1]
        assert seeded_stats["hits"] > stats["hits"]
    with running(model, root, port, token) as base:
        restarted, restart_ms = completion(base, "tenant-a")
        after_restart = request(base + "/cache/stats", token=token)[1]
        assert after_restart["ssd_hits"] > 0
        assert restarted["choices"][0]["message"] == answers[0]
        cleared = request(base + "/cache/stats", "DELETE", token=token)[1]
        assert cleared["entries"] == cleared["ssd"]["entries"] == 0
    return {"model": model.name, "requests": 7, "tenant_isolation": True, "seeded_reuse": True,
            "equivalent_answers": True, "ssd_restore": True, "restart_restore": True,
            "admin_authentication": True, "clear": True, "stats": stats,
            "after_restart": after_restart,
            "http_ms": {"cold": cold_ms, "warm": warm_ms, "ssd": disk_ms, "restart": restart_ms},
            "timing_scope": "one smoke request per path; includes HTTP and generation, not a latency benchmark"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="liteinfer-cache-http-") as directory:
        report = verify(args.model.resolve(), Path(directory), port, secrets.token_hex(32))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
