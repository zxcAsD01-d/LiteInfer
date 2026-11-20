#!/usr/bin/env python3
"""Interleaved real-weight A/B benchmark for Qwen3.5 prefill and decode."""
import argparse
import gc
import json
import random
import statistics
import time
from pathlib import Path
from typing import NotRequired, TypedDict
from importlib.metadata import version

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_model

from liteinfer_mlx.config import BatchConfig, SamplingConfig
from liteinfer_mlx.engine import BatchEngine
from liteinfer_mlx.qwen35_fusion import fuse_qwen35
from liteinfer_mlx.qwen35_small_m import enable_qwen35_small_m


class Trial(TypedDict):
    prefill_tokens: int
    prefill_seconds: float
    prefill_tokens_per_second: float
    mean_time_to_first_token_seconds: float
    decode_tokens_per_second: float
    end_to_end_tokens_per_second: float
    seconds: float
    completion_tokens: int
    peak_memory_bytes: int
    token_sequences: list[list[int]]
    output_token_mismatches: NotRequired[int]


def make_prompts(count: int, length: int, vocab: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(vocab) for _ in range(length)] for _ in range(count)]


def run_trial(model: nn.Module, prompts: list[list[int]], output_tokens: int) -> Trial:
    batch_size = len(prompts)
    config = BatchConfig(
        concurrency=batch_size,
        queue_capacity=batch_size,
        prefill_batch_size=batch_size,
        prefill_step=2048,
        idle_prefill_step=2048,
        prefix_cache=False,
    )
    engine = BatchEngine(model, "qwen35-benchmark", config, stop_tokens=[])
    sampling = SamplingConfig(temperature=0.0, top_k=1, top_p=1.0, seed=0)
    for prompt in prompts:
        engine.submit(prompt, tenant="benchmark", max_tokens=output_tokens, sampling=sampling)

    mx.reset_peak_memory()
    generated = 0
    sequences: dict[int, list[int]] = {}
    first_token_times: dict[int, float] = {}
    started = time.perf_counter()
    try:
        with engine.batch.stats() as stats:
            while engine.active or engine.pending:
                events = engine.step()
                event_time = time.perf_counter() - started
                generated += len(events)
                for event in events:
                    first_token_times.setdefault(event.id, event_time)
                    sequences.setdefault(event.id, []).append(event.token)
        elapsed = time.perf_counter() - started
        expected = batch_size * output_tokens
        if generated != expected:
            raise RuntimeError(f"Expected {expected} generated tokens, got {generated}")
        ordered_sequences = [sequences[request_id] for request_id in sorted(sequences)]
        if any(len(sequence) != output_tokens for sequence in ordered_sequences):
            raise RuntimeError("A request stopped before its fixed output budget")
        expected_prefill = sum(len(prompt) - 1 for prompt in prompts)
        if stats.prompt_tokens != expected_prefill:
            raise RuntimeError(
                f"Expected {expected_prefill} prefill tokens, got {stats.prompt_tokens}"
            )
        return {
            "prefill_tokens": stats.prompt_tokens,
            "prefill_seconds": stats.prompt_time,
            "prefill_tokens_per_second": stats.prompt_tps,
            "mean_time_to_first_token_seconds": statistics.mean(first_token_times.values()),
            "decode_tokens_per_second": stats.generation_tps,
            "end_to_end_tokens_per_second": generated / elapsed,
            "seconds": elapsed,
            "completion_tokens": generated,
            "peak_memory_bytes": mx.get_peak_memory(),
            "token_sequences": ordered_sequences,
        }
    finally:
        engine.close()


def summarize(trials: list[Trial]) -> dict[str, object]:
    prefill_rates = [item["prefill_tokens_per_second"] for item in trials]
    decode_rates = [item["decode_tokens_per_second"] for item in trials]
    wall_rates = [item["end_to_end_tokens_per_second"] for item in trials]
    return {
        "prefill_median_tokens_per_second": statistics.median(prefill_rates),
        "prefill_min_tokens_per_second": min(prefill_rates),
        "prefill_max_tokens_per_second": max(prefill_rates),
        "prefill_median_seconds": statistics.median(
            item["prefill_seconds"] for item in trials
        ),
        "time_to_first_token_median_seconds": statistics.median(
            item["mean_time_to_first_token_seconds"] for item in trials
        ),
        "decode_median_tokens_per_second": statistics.median(decode_rates),
        "decode_min_tokens_per_second": min(decode_rates),
        "decode_max_tokens_per_second": max(decode_rates),
        "end_to_end_median_tokens_per_second": statistics.median(wall_rates),
        "trials": [{key: value for key, value in item.items() if key != "token_sequences"}
                   for item in trials],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--mode", choices=("compare", "baseline", "optimized"), default="compare")
    parser.add_argument("--optimization", choices=("small-m", "qwen35-fusion", "combined"),
                        default="small-m")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 6, 8])
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--generation-tokens", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--seed", type=int, default=713)
    parser.add_argument("--single-target", type=float, default=300.0)
    parser.add_argument("--batch8-target", type=float, default=1600.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batches) < 1:
        parser.error("Batches must be positive")
    if min(args.prompt_tokens, args.generation_tokens, args.trials) <= 0:
        parser.error("Request and token counts and trials must be positive")
    if args.warmups < 0:
        parser.error("Warmups cannot be negative")
    if args.mode == "compare" and 1 not in args.batches:
        parser.error("Comparison mode requires batch 1")
    if args.mode == "compare" and 8 not in args.batches:
        parser.error("Comparison mode requires batch 8")

    baseline, _ = load_model(args.model)
    models: dict[str, nn.Module] = {"baseline": baseline}
    wrappers = {"quantized_linear": 0, "quantized_embedding": 0}
    fusion: dict[str, int] | None = None
    if args.mode != "baseline":
        optimized = baseline if args.mode == "optimized" else load_model(args.model)[0]
        if args.optimization in ("qwen35-fusion", "combined"):
            fusion = fuse_qwen35(optimized)
        if args.optimization in ("small-m", "combined"):
            wrappers = enable_qwen35_small_m(optimized)
        models["optimized"] = optimized

    vocab = baseline.args.text_config["vocab_size"]
    cases: dict[str, dict[str, dict[str, object]]] = {}
    for batch_size in args.batches:
        prompts = make_prompts(batch_size, args.prompt_tokens, vocab, args.seed + batch_size)
        variants = list(models.items())
        for _, model in variants:
            for _ in range(args.warmups):
                run_trial(model, prompts[:batch_size], args.generation_tokens)
                mx.clear_cache()
                gc.collect()

        samples: dict[str, list[Trial]] = {name: [] for name, _ in variants}
        for trial_index in range(args.trials):
            trial_prompts = make_prompts(
                batch_size, args.prompt_tokens, vocab, args.seed + batch_size + trial_index + 1
            )
            order = variants if trial_index % 2 == 0 else list(reversed(variants))
            for name, model in order:
                samples[name].append(run_trial(model, trial_prompts, args.generation_tokens))
                mx.clear_cache()
                gc.collect()
            if "baseline" in samples and "optimized" in samples:
                baseline_trial = samples["baseline"][-1]
                optimized_trial = samples["optimized"][-1]
                mismatch_count = sum(
                    left != right
                    for baseline_sequence, optimized_sequence in zip(
                        baseline_trial["token_sequences"],
                        optimized_trial["token_sequences"],
                        strict=True,
                    )
                    for left, right in zip(baseline_sequence, optimized_sequence, strict=True)
                )
                optimized_trial["output_token_mismatches"] = mismatch_count
        cases[str(batch_size)] = {name: summarize(items) for name, items in samples.items()}

    result: dict[str, object] = {
        "date": time.strftime("%Y-%m-%d"),
        "hardware": mx.device_info(mx.gpu),
        "runtime": {"mlx": version("mlx"), "mlx_lm": version("mlx-lm")},
        "model": args.model_label,
        "optimization": args.optimization if args.mode != "baseline" else None,
        "method": {
            "prompt_tokens": args.prompt_tokens,
            "generation_tokens": args.generation_tokens,
            "concurrent_requests_per_trial": args.batches,
            "warmups_per_case": args.warmups,
            "timing_trials_per_case": args.trials,
            "throughput": "BatchEngine BatchStats decode tokens/s; wall-rate reported separately",
            "prefill_throughput": "BatchStats prompt_tokens / prompt_time; aggregate input tokens/s",
            "prefill_scope": "Last input token per request is counted in generation, not prefill",
            "time_to_first_token": "Submit-to-first-TokenEvent wall time; excludes model load, tokenization and HTTP",
            "prefix_cache": False,
        },
        "small_m_wrapped_layers": wrappers,
        "qwen35_fusion": fusion,
        "targets": {
            "single_request_tokens_per_second": args.single_target,
            "batch8_aggregate_tokens_per_second": args.batch8_target,
            "achieved": None,
        },
        "cases": cases,
    }
    if args.mode == "compare":
        one = cases["1"]["optimized"]["decode_median_tokens_per_second"]
        eight = cases["8"]["optimized"]["decode_median_tokens_per_second"]
        result["targets"]["achieved"] = one >= args.single_target and eight >= args.batch8_target
    else:
        result["mode"] = args.mode

    encoded = json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
