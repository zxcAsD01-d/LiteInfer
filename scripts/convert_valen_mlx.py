#!/usr/bin/env python3
"""将 Valen 原生 Qwen3.5 检查点转换为 MLX 权重。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backends" / "mlx"))

from liteinfer_mlx.valen import convert_valen  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Valen 原生模型目录")
    parser.add_argument("output", type=Path, help="新建 MLX 输出目录")
    parser.add_argument("--bits", type=int, choices=(0, 4, 8), default=4,
                        help="主干精度：0=BF16，4/8=仿射量化；默认 4")
    args = parser.parse_args()
    try:
        result = convert_valen(args.source, args.output, args.bits)
    except (FileNotFoundError, ValueError, OSError) as error:
        parser.error(str(error))
    print(f"转换完成：{result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
