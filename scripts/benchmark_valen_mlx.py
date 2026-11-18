#!/usr/bin/env python3
"""Measure converted Valen MLX text backbone and decision-head kernels."""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from datetime import date
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.utils import load_model
from mlx_lm.models.cache import make_prompt_cache
from tokenizers import Tokenizer

from liteinfer_mlx.valen import ValenMixerHead
from liteinfer_mlx.valen_ops import replace_valen_depthwise_convs


def _measure(run, item_count: int, warmup: int, trials: int,
             throughput_name: str = "input_tokens_per_second") -> dict[str, float]:
    for _ in range(warmup):
        mx.eval(run())
    samples = []
    for _ in range(trials):
        start = time.perf_counter()
        mx.eval(run())
        samples.append(time.perf_counter() - start)
    median = statistics.median(samples)
    return {
        "median_ms": median * 1000,
        "p95_ms": float(np.percentile(samples, 95)) * 1000,
        throughput_name: item_count / median,
    }


def _tokens(model_dir: Path, length: int) -> mx.array:
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    text = ("Context: customer needs help with a delayed package. "
            "Instruction: score each candidate for relevance. "
            "Candidates: shipping, billing, account, other. Decision: ")
    ids = tokenizer.encode(text).ids
    if not ids:
        raise ValueError("Valen tokenizer 没有生成输入 token")
    repeated = (ids * ((length + len(ids) - 1) // len(ids)))[:length]
    return mx.array([repeated], dtype=mx.uint32)


def _forward(model, inputs: mx.array) -> mx.array:
    return model.model(inputs)


def _load_hidden(model_dir: Path, inputs: mx.array, warmup: int,
                 trials: int) -> tuple[object, dict[str, float], np.ndarray]:
    model, _ = load_model(model_dir, lazy=False)
    model.eval()
    run = lambda: _forward(model, inputs)
    baseline = _measure(run, inputs.shape[-1], warmup, trials)
    hidden = run()
    mx.eval(hidden)
    return model, baseline, np.asarray(hidden.astype(mx.float32), dtype=np.float32)


def _model_trial(model_dir: Path, inputs: mx.array, args: argparse.Namespace):
    model, prefill, hidden = _load_hidden(model_dir, inputs, args.warmup, args.trials)
    decode = _decode_measure(model, inputs, args.generation_tokens, args.warmup, args.trials)
    return model, {"prefill": prefill, "decode": decode}, hidden


def _baseline_trial(model_dir: Path, inputs: mx.array,
                    args: argparse.Namespace) -> tuple[dict[str, object], np.ndarray]:
    model, result, hidden = _model_trial(model_dir, inputs, args)
    del model
    gc.collect()
    mx.clear_cache()
    return result, hidden


def _quantized_trial(model_dir: Path, inputs: mx.array, args: argparse.Namespace,
                     reference: np.ndarray):
    model, result, hidden = _model_trial(model_dir, inputs, args)
    result.update(_hidden_error(reference, hidden))
    return model, result, hidden


def _hidden_error(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    difference = candidate - reference
    return {
        "hidden_rmse_vs_bf16": float(np.sqrt(np.mean(difference * difference))),
        "hidden_max_abs_error_vs_bf16": float(np.max(np.abs(difference))),
        "hidden_relative_l2_error_vs_bf16": float(
            np.linalg.norm(difference) / max(np.linalg.norm(reference), 1e-9)
        ),
    }


def _decode_once(model, inputs: mx.array, tokens: int) -> float:
    cache = make_prompt_cache(model)
    logits = model(inputs, cache=cache)
    mx.eval(logits)
    start = time.perf_counter()
    for index in range(tokens):
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_token)
        if index + 1 < tokens:
            logits = model(next_token[:, None], cache=cache)
            mx.eval(logits)
    return time.perf_counter() - start


def _decode_measure(model, inputs: mx.array, tokens: int, warmup: int,
                    trials: int) -> dict[str, float]:
    for _ in range(warmup):
        _decode_once(model, inputs, tokens)
    samples = [_decode_once(model, inputs, tokens) for _ in range(trials)]
    median = statistics.median(samples)
    return {
        "generated_tokens": tokens,
        "median_ms": median * 1000,
        "p95_ms": float(np.percentile(samples, 95)) * 1000,
        "decode_tokens_per_second": tokens / median,
    }


def _features(hidden: np.ndarray, candidates: int) -> mx.array:
    values = mx.array(hidden[0], dtype=mx.float32)
    if values.shape[0] < 3:
        raise ValueError("输入 token 数至少为 3")
    positions = np.linspace(1, values.shape[0] - 2, candidates, dtype=np.int32)
    selected = values[mx.array(positions)]
    context = mx.broadcast_to(values[0], selected.shape)
    instruction = mx.broadcast_to(values[1], selected.shape)
    decision = mx.broadcast_to(values[-1], selected.shape)
    return mx.stack((context, instruction, selected, decision), axis=1)


def _head_benchmark(model_dir: Path, hidden: np.ndarray, trials: int) -> dict[str, object]:
    metadata = json.loads((model_dir / "valen_conversion.json").read_text(encoding="utf-8"))
    config = metadata["head_config"]
    weights = mx.load(str(model_dir / "valen_head.safetensors"))
    head = ValenMixerHead(int(hidden.shape[-1]), config)
    head.load_weights(list(weights.items()), strict=True)
    head.eval()
    optimized = ValenMixerHead(int(hidden.shape[-1]), config, custom_layer_norm=True)
    optimized.load_weights(list(weights.items()), strict=True)
    optimized.eval()
    compiled_reference = head.compiled()
    compiled_optimized = optimized.compiled()
    results = {}
    for count in (4, 16, 64):
        features = _features(hidden, count)
        reference = head(features)
        custom = optimized(features)
        compiled_base = compiled_reference(features)
        compiled_custom = compiled_optimized(features)
        mx.eval(reference, custom, compiled_base, compiled_custom)
        max_error = float(mx.max(mx.abs(reference - custom)).item())
        if not mx.allclose(reference, custom, atol=1e-3, rtol=1e-3).item():
            raise RuntimeError(f"自定义 Mixer 算子数值偏差过大：{max_error}")
        compiled_error = float(mx.max(mx.abs(reference - compiled_base)).item())
        if not mx.allclose(reference, compiled_base, atol=1e-4, rtol=1e-4).item():
            raise RuntimeError(f"编译 Mixer 数值偏差过大：{compiled_error}")
        results[str(count)] = {
            "reference": _measure(lambda: head(features), count, 2, trials, "candidates_per_second"),
            "metal_layer_norm": _measure(lambda: optimized(features), count, 2, trials,
                                          "candidates_per_second"),
            "compiled_reference": _measure(lambda: compiled_reference(features), count, 2, trials,
                                           "candidates_per_second"),
            "compiled_metal_layer_norm": _measure(lambda: compiled_optimized(features), count, 2,
                                                  trials, "candidates_per_second"),
            "max_abs_error": max_error,
            "compiled_max_abs_error": compiled_error,
        }
    return results


def _head_agreement(head, reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    base_features = _features(reference, 64)
    candidate_features = _features(candidate, 64)
    base_scores = head(base_features).reshape(16, 4)
    candidate_scores = head(candidate_features).reshape(16, 4)
    base_probabilities = mx.softmax(base_scores, axis=-1)
    candidate_probabilities = mx.softmax(candidate_scores, axis=-1)
    mx.eval(base_probabilities, candidate_probabilities)
    return {
        "synthetic_top_choice_agreement": float(
            mx.mean(mx.argmax(base_scores, axis=-1) == mx.argmax(candidate_scores, axis=-1)).item()
        ),
        "synthetic_probability_mae": float(
            mx.mean(mx.abs(base_probabilities - candidate_probabilities)).item()
        ),
    }


def _synthetic_agreement(model_dir: Path, reference: np.ndarray, q4: np.ndarray,
                         q8: np.ndarray | None) -> dict[str, object]:
    metadata = json.loads((model_dir / "valen_conversion.json").read_text(encoding="utf-8"))
    head = ValenMixerHead(1024, metadata["head_config"])
    weights = mx.load(str(model_dir / "valen_head.safetensors"))
    head.load_weights(list(weights.items()), strict=True)
    head.eval()
    return {
        "q8": _head_agreement(head, reference, q8) if q8 is not None else None,
        "q4": _head_agreement(head, reference, q4),
        "note": "Scores from placeholder role spans, not compiled Valen decisions; not a quality metric.",
    }


def _custom_conv_benchmark(model, inputs: mx.array, args: argparse.Namespace):
    baseline = _forward(model, inputs)
    mx.eval(baseline)
    replaced = replace_valen_depthwise_convs(model)
    optimized = _forward(model, inputs)
    mx.eval(optimized)
    error = float(mx.max(mx.abs(baseline - optimized)).item())
    if not mx.allclose(baseline, optimized, atol=2e-2, rtol=2e-2).item():
        raise RuntimeError(f"自定义卷积算子改变 Qwen 隐层输出：max_abs_error={error}")
    prefill = _measure(lambda: _forward(model, inputs), args.prompt_tokens,
                       args.warmup, args.trials)
    decode = _decode_measure(model, inputs, args.generation_tokens, args.warmup, args.trials)
    return {"prefill": prefill, "decode": decode}, replaced, error


def _benchmark(args: argparse.Namespace) -> dict[str, object]:
    inputs = _tokens(args.q4_dir, args.prompt_tokens)
    bf16_result, bf16_hidden = _baseline_trial(args.bf16_dir, inputs, args)
    q8_result, q8_hidden = None, None
    if args.q8_dir is not None:
        q8_result, q8_hidden = _baseline_trial(args.q8_dir, inputs, args)
        q8_result.update(_hidden_error(bf16_hidden, q8_hidden))
    model, q4_result, q4_hidden = _quantized_trial(
        args.q4_dir, inputs, args, bf16_hidden
    )
    custom_result, replaced, custom_error = _custom_conv_benchmark(model, inputs, args)
    mixer_results = _head_benchmark(args.q4_dir, q4_hidden, args.trials)
    agreement = _synthetic_agreement(args.q4_dir, bf16_hidden, q4_hidden, q8_hidden)
    del model, q4_hidden, bf16_hidden, q8_hidden
    mx.clear_cache()
    return _report(args, bf16_result, q8_result, q4_result, custom_result,
                   replaced, custom_error, mixer_results, agreement)


def _report(args: argparse.Namespace, bf16: dict[str, object], q8: dict[str, object] | None,
            q4: dict[str, object], custom: dict[str, object], replaced: int,
            custom_error: float, mixer: dict[str, object], agreement: dict[str, object]) -> dict[str, object]:
    return {
        "date": date.today().isoformat(),
        "model": "Valen-Decision-1.5-0.8B",
        "source_revision": "e079395fe9b8e0763b384af403e636e91c1a1559",
        "hardware": {"chip": "Apple M1 Pro", "unified_memory_gib": 16},
        "runtime": {"mlx": version("mlx"), "mlx_lm": version("mlx-lm")},
        "method": {
            "prompt_tokens": args.prompt_tokens,
            "generation_tokens": args.generation_tokens,
            "prompt_source": "fixed English test prompt; token sequence repeated to requested length",
            "warmup_runs": args.warmup,
            "timing_trials": args.trials,
            "scope": "text-only backbone prefill/decode + Mixer synthetic role-feature microbench; not HTTP or multimodal",
        },
        "backbone": {
            "bf16": bf16,
            "q8_reference": q8,
            "q4_reference": q4,
            "q4_custom_depthwise_conv": custom,
            "depthwise_layers_replaced": replaced,
            "custom_conv_max_abs_error": custom_error,
        },
        "mixer_head": mixer,
        "synthetic_decision_head_agreement_vs_bf16": agreement,
        "note": "Synthetic role positions test operator cost and numerical equivalence only; no Valen task-quality claim.",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16-dir", type=Path, required=True)
    parser.add_argument("--q8-dir", type=Path)
    parser.add_argument("--q4-dir", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--generation-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/valen-mlx-m1pro.json"))
    args = parser.parse_args()
    if args.prompt_tokens < 8 or args.warmup < 1 or args.trials < 3:
        parser.error("prompt tokens 至少 8、warmup 至少 1、trials 至少 3")
    result = _benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
