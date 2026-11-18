#!/usr/bin/env python3
"""使用 MLX-LM 将 HuggingFace 模型转换为 MLX 格式。"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return parsed


def validate_output(source: Path | None, output: Path) -> None:
    if source and (output == source or source in output.parents):
        raise ValueError("输出目录不能覆盖源目录或位于源目录内")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("输出目录必须不存在或为空目录")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="本地 HuggingFace 模型目录或 owner/model 仓库 ID")
    parser.add_argument("output", type=Path, help="新的 MLX 输出目录")
    parser.add_argument("--bits", type=positive_int, choices=(4, 8), default=4)
    parser.add_argument("--group-size", type=positive_int, choices=(32, 64, 128), default=64)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--python", default=sys.executable, help="安装 MLX-LM 的 Python 可执行文件")
    parser.add_argument("--no-quantize", action="store_true", help="保留浮点权重")
    args = parser.parse_args()
    source_argument = Path(args.source).expanduser()
    source = source_argument.resolve() if source_argument.is_dir() else None
    if source is None and (args.source.startswith(("./", "../", "~/", "/"))):
        parser.error("源目录不存在或不是目录")
    args.output = args.output.expanduser().resolve()
    try:
        validate_output(source, args.output)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [args.python, "-m", "mlx_lm", "convert", "--hf-path", str(source or args.source),
               "--mlx-path", str(args.output), "--dtype", args.dtype]
    if not args.no_quantize:
        command.extend(["--quantize", "--q-bits", str(args.bits),
                        "--q-group-size", str(args.group_size)])
    subprocess.run(command, check=True)
    print("转换完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
