"""Runnable batch example using the same public optimization components."""
import argparse
from dataclasses import asdict
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load_model, load_tokenizer

from .config import BatchConfig, SamplingConfig
from .engine import BatchEngine
from .fused_layers import fuse_model


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLX continuous batch inference example")
    parser.add_argument("--model", required=True, help="Local MLX model directory")
    parser.add_argument("--tokenizer", help="Separate local tokenizer directory, if needed")
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fuse-llama", action="store_true")
    parser.add_argument("--no-prefix-cache", action="store_true")
    parser.add_argument("--raw", action="store_true", help="Skip chat template explicitly")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    sampling = SamplingConfig(args.temperature, args.top_k, args.top_p, args.seed)
    config = BatchConfig(concurrency=args.concurrency, queue_capacity=len(args.prompt),
                         prefix_cache=not args.no_prefix_cache)
    if args.tokenizer:
        model, _ = load_model(Path(args.model))
        tokenizer = load_tokenizer(Path(args.tokenizer))
    else:
        model, tokenizer = load(args.model)
    if args.fuse_llama:
        fuse_model(model)
    stop_tokens = [[token] for token in tokenizer.eos_token_ids]
    engine = BatchEngine(model, "qwen3.5-0.8b-q4", config, stop_tokens, len(tokenizer))
    try:
        run(engine, tokenizer, args, sampling)
    finally:
        engine.close()


def run(engine: BatchEngine, tokenizer: TokenizerWrapper, args: argparse.Namespace, sampling: SamplingConfig) -> None:
    for prompt in args.prompt:
        if args.raw:
            tokens = tokenizer.encode(prompt)
        else:
            if not tokenizer.chat_template:
                raise ValueError("Model has no chat template; choose an SFT model or use --raw")
            tokens = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                   add_generation_prompt=True, tokenize=True)
        engine.submit(tokens, tenant="local-cli", max_tokens=args.max_tokens, sampling=sampling)
    started = time.perf_counter()
    count = 0
    while engine.active or engine.pending:
        for event in engine.step():
            count += 1
            print(json.dumps(asdict(event), ensure_ascii=False), flush=True)
    elapsed = time.perf_counter() - started
    print(json.dumps({"summary": {"completion_tokens": count, "seconds": elapsed,
        "tokens_per_second": count / elapsed, "peak_active": engine.peak_active,
        "cache": engine.cache.snapshot(), "peak_gpu_bytes": mx.get_peak_memory()}}), flush=True)


if __name__ == "__main__":
    main()
