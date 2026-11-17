#!/usr/bin/env python3
"""Scan tracked and staged files before publication; do not print matched secrets."""
import re
import subprocess
from pathlib import Path

PATTERNS = {
    "personal home path": re.compile(rb"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
    "private tunnel": re.compile(rb"https://[a-z0-9-]+\.trycloudflare\.com"),
    "provider token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|sk-(?:local-|proj-)?[A-Za-z0-9_-]{24,})"),
    "LiteInfer key": re.compile(rb"li_[0-9a-f]{64}"),
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
BLOCKED = {"config.toml", "api-key.txt", "client-config.json", "publication.json", ".env"}


def main():
    files = subprocess.check_output(["git", "ls-files", "-z"]).split(b"\0")
    failures = []
    for item in files:
        if not item:
            continue
        path = Path(item.decode())
        if (path.name in BLOCKED and path != Path("examples/config.toml")) or path.suffix in (".log", ".sqlite3", ".safetensors", ".gguf", ".pt"):
            failures.append((str(path), "private file"))
        if not path.is_file():
            continue
        data = path.read_bytes()
        for name, pattern in PATTERNS.items():
            if pattern.search(data):
                failures.append((str(path), name))
    if failures:
        for path, reason in failures:
            print(f"BLOCKED {path}: {reason}")
        raise SystemExit(1)
    print("隐私扫描通过：未发现个人路径、私有隧道、实际密钥或模型权重。")


if __name__ == "__main__":
    main()
