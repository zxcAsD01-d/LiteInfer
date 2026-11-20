#!/usr/bin/env python3
"""Benchmark cache lookup and optional real-weight prefill/state recovery."""
import argparse
from collections import deque
from importlib.metadata import version
import json
from pathlib import Path
import platform
import statistics
import tempfile
import time

from liteinfer_mlx.cache_index import RadixIndex
from liteinfer_mlx.cache_settings import CacheConfig
from liteinfer_mlx.disk_cache import latency


def summarize(values: list[float]) -> dict[str, float | int]:
    return latency(deque(values))


def index_benchmark(trials: int) -> list[dict[str, object]]:
    results = []
    for size in (64, 1024, 8192):
        entries = [tuple([1] * 32 + [i] + [2] * 223) for i in range(size)]
        index = RadixIndex()
        for i, tokens in enumerate(entries):
            index.add("tenant", tokens, str(i), True)
        times: dict[str, list[float]] = {"scan": [], "radix": []}
        for step in range(trials + 5):
            query = entries[step % size][:-1] + (9,)
            for mode in (("scan", "radix") if step % 2 else ("radix", "scan")):
                started = time.perf_counter()
                if mode == "scan":
                    common = 0
                    for tokens in entries:
                        length = 0
                        for left, right in zip(tokens, query):
                            if left != right:
                                break
                            length += 1
                        common = max(common, length)
                else:
                    _, common = index.nearest("tenant", query)
                elapsed = (time.perf_counter() - started) * 1000
                if common != len(query) - 1:
                    raise RuntimeError("Index benchmark found an incorrect prefix")
                if step >= 5:
                    times[mode].append(elapsed)
        results.append({"entries": size, "prompt_tokens": 256,
                        "scan": summarize(times["scan"]), "radix": summarize(times["radix"]),
                        "speedup": statistics.median(times["scan"]) / statistics.median(times["radix"]),
                        "index": index.snapshot()})
    return results


def prefill(model, tokens: list[int]):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    for start in range(0, len(tokens), 128):
        model(mx.array([tokens[start:start + 128]]), cache=cache)
        mx.eval([c.state for c in cache])
    return cache


def state_benchmark(directory: Path, trials: int) -> dict[str, object]:
    import mlx.core as mx
    from mlx_lm import load
    from liteinfer_mlx.prompt_cache import TieredPromptCache
    model, tokenizer = load(str(directory))
    tokens = tokenizer.encode("Cache validation uses deterministic prefix state. " * 64)[:256]
    suffix = tokenizer.encode("Continue")[:1]
    config = CacheConfig(ssd_max_bytes=512 * 1024**2, min_tokens=1)
    with tempfile.TemporaryDirectory(prefix="liteinfer-cache-bench-") as root:
        cache = TieredPromptCache(config, Path(root))
        try:
            report = measure_states(model, tokens, suffix, cache, trials)
            report["model"] = directory.name
            return report
        finally:
            cache.close()


def measure_states(model, tokens, suffix, cache, trials: int) -> dict[str, object]:
    import mlx.core as mx
    reference = prefill(model, tokens)
    cache.insert_cache("model", tokens, reference)
    cache.disk.flush()
    expected = model(mx.array([suffix]), cache=reference)
    mx.eval(expected)
    durations: dict[str, list[float]] = {"recompute": [], "memory": [], "ssd": []}
    errors: list[float] = []
    for step in range(trials + 2):
        for mode in (("recompute", "memory", "ssd") if step % 2 else ("ssd", "memory", "recompute")):
            if mode == "ssd":
                cache.trim_to(n_sequences=0)
            elif mode == "memory":
                cache.fetch_nearest_cache("model", tokens + suffix)
            started = time.perf_counter()
            if mode == "recompute":
                restored = prefill(model, tokens)
            else:
                restored, rest = cache.fetch_nearest_cache("model", tokens + suffix)
                if restored is None or rest != suffix:
                    raise RuntimeError("Expected prefix was not restored")
            logits = model(mx.array([suffix]), cache=restored)
            mx.eval(logits)
            elapsed = (time.perf_counter() - started) * 1000
            error = mx.max(mx.abs(logits - expected)).item()
            if error != 0:
                raise RuntimeError(f"Restored logits differ by {error}")
            errors.append(error)
            if step >= 2:
                durations[mode].append(elapsed)
    return {"input_tokens": len(tokens),
            "suffix_tokens": len(suffix), "trials": trials, "mlx": version("mlx"),
            "mlx_lm": version("mlx-lm"), "max_abs_logit_error": max(errors),
            "timing_scope": "prefill or cache recovery plus one fixed continuation; excludes HTTP, tokenization and model load",
            "latency": {key: summarize(values) for key, values in durations.items()},
            "cache": cache.snapshot()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    if args.trials < 1:
        raise ValueError("trials must be positive")
    report = {"platform": platform.machine(), "index": index_benchmark(100)}
    if args.model:
        report["state"] = state_benchmark(args.model, args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
