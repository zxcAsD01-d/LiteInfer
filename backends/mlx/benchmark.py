#!/usr/bin/env python3
"""Real-weight batch benchmark; exports aggregate numbers, never model paths or text."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_model, load_tokenizer

from liteinfer_mlx.config import BatchConfig, SamplingConfig
from liteinfer_mlx.engine import BatchEngine
from liteinfer_mlx.fused_layers import fuse_model


def measure(model: nn.Module, tokens: list[int], n: int, count: int, output: int) -> dict:
    config = BatchConfig(concurrency=n, queue_capacity=count, prefill_batch_size=n)
    engine = BatchEngine(model, "benchmark-model", config)
    try:
        # Warm the kernels before timing; keep the prefix to measure reuse too.
        engine.submit(tokens, tenant="benchmark", max_tokens=output)
        while engine.pending or engine.active:
            engine.step()
        before = engine.cache.snapshot()
        mx.reset_peak_memory()
        started = time.perf_counter()
        first: dict[int, float] = {}
        completed, generated, cached = 0, 0, 0
        for i in range(count):
            engine.submit(tokens + [10 + i % 8], tenant="benchmark", max_tokens=output,
                          sampling=SamplingConfig(seed=i))
        while engine.pending or engine.active:
            for event in engine.step():
                first.setdefault(event.id, time.perf_counter() - started)
                generated += 1
                if event.finish_reason:
                    completed += 1
                    cached += event.cached_tokens
        elapsed = time.perf_counter() - started
        if completed != count:
            raise RuntimeError("Batch did not complete all requests")
        return {"concurrency": n, "requests": count, "completed": completed,
                "seconds": elapsed, "completion_tokens": generated,
                "tokens_per_second": generated / elapsed, "cached_input_tokens": cached,
                "ttft_including_queue_ms_p95": sorted(first.values())[int((count-1)*.95)] * 1000,
                "peak_active": engine.peak_active, "peak_gpu_bytes": mx.get_peak_memory(),
                "warm_cache": before, "final_cache": engine.cache.snapshot()}
    finally:
        engine.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model-label", required=True, help="Public model name, never a local path")
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 3, 8])
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    parser.add_argument("--quantize-bits", type=int, choices=[4, 8])
    parser.add_argument("--fuse-llama", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.requests, args.max_tokens, *args.concurrencies) <= 0:
        parser.error("Request counts, concurrency and output budget must be positive")
    model, _ = load_model(Path(args.model))
    model.set_dtype(getattr(mx, args.dtype))
    if args.quantize_bits:
        nn.quantize(model, group_size=64, bits=args.quantize_bits)
    if args.fuse_llama:
        fuse_model(model)
    mx.eval(model.parameters())
    tokenizer = load_tokenizer(Path(args.tokenizer))
    tokens = tokenizer.encode("请简要介绍本地推理服务。" * 16)
    cases = [measure(model, tokens, n, args.requests, args.max_tokens) for n in args.concurrencies]
    report = {"label": "real-weight-mlx-components", "model": args.model_label,
        "dtype": args.dtype, "quantize_bits": args.quantize_bits,
        "fused_llama": args.fuse_llama, "sampling": asdict(SamplingConfig()),
        "mlx": "0.32.4", "mlx_lm": "0.32.2", "cases": cases,
        "note": "Warm prefix reuse, fixed output length, direct BatchEngine; excludes HTTP/tokenization/loading. No EOS stop configured; not a quality evaluation."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"model": args.model_label, "cases": [
        {key: row[key] for key in ("concurrency", "completed", "tokens_per_second", "cached_input_tokens")}
        for row in cases]}))


if __name__ == "__main__":
    main()
