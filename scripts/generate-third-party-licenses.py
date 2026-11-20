#!/usr/bin/env python3
"""Collect license and notice files for target-specific Rust dependencies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


def metadata(root: Path) -> dict[str, object]:
    result = subprocess.run(
        ["cargo", "metadata", "--offline", "--locked", "--filter-platform",
         "aarch64-apple-darwin", "--format-version", "1"],
        cwd=root, check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def license_files(directory: Path) -> list[Path]:
    prefixes = ("license", "licence", "copying", "notice", "copyright")
    return sorted(
        (path for path in directory.iterdir()
         if path.is_file() and path.name.lower().startswith(prefixes)),
        key=lambda path: path.name.lower(),
    )


def collect(packages: list[dict[str, object]]) -> dict[str, tuple[set[str], bytes]]:
    grouped: dict[str, tuple[set[str], bytes]] = {}
    for package in packages:
        if package["source"] is None or package["name"] == "liteinfer":
            continue
        manifest = Path(package["manifest_path"])
        files = license_files(manifest.parent)
        if not files:
            if package["name"] in {"objc2-core-foundation", "objc2-io-kit"}:
                files = [Path(__file__).resolve().parent.parent / "licenses/objc2-LICENSE.md"]
            else:
                raise RuntimeError(f"Missing license text for {package['name']} ({package['license']})")
        for path in files:
            body = path.read_bytes().replace(b"\r\n", b"\n").strip() + b"\n"
            digest = hashlib.sha256(body).hexdigest()
            names, _ = grouped.setdefault(digest, (set(), body))
            names.add(f"{package['name']} {package['version']} — {path.name}")
    return grouped


def render(groups: dict[str, tuple[set[str], bytes]]) -> bytes:
    chunks = [
        "# Rust 依赖的第三方许可与声明\n\n"
        "下列文本随 LiteInfer 1.0.0 macOS arm64 安装包中的 Rust 网关分发。\n"
        "相同文本合并列出来源 crate；其 SPDX 标识及版本以 Cargo.lock 为准。\n"
    ]
    for _, (packages, body) in sorted(groups.items()):
        title = ", ".join(sorted(packages))
        chunks.extend((f"\n## {title}\n\n", f"~~~~text\n{body.decode('utf-8', errors='replace')}\n~~~~\n"))
    return "".join(chunks).encode()


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: generate-third-party-licenses.py REPO_ROOT OUTPUT")
    root, output = Path(sys.argv[1]), Path(sys.argv[2])
    packages = metadata(root)["packages"]
    output.write_bytes(render(collect(packages)))


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.CalledProcessError, RuntimeError, KeyError,
            TypeError, json.JSONDecodeError) as error:
        print(f"license bundle generation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
