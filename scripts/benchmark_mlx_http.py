#!/usr/bin/env python3
"""对 OpenAI 兼容 MLX HTTP 服务做无内容落盘的并发基准。"""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int((len(ordered) - 1) * ratio))] * 1000, 3)


def sse_usage(response, started: float) -> tuple[float | None, int]:
    first = None
    output_tokens = 0
    complete = False
    has_usage = False
    for raw in response:
        if not raw.startswith(b"data: "):
            continue
        first = first or time.perf_counter() - started
        data = raw[6:].strip()
        if data == b"[DONE]":
            complete = True
            continue
        event = json.loads(data)
        if "error" in event:
            raise ValueError("in-band error")
        usage = event.get("usage")
        if isinstance(usage, dict):
            has_usage = True
            output_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    if not complete or not has_usage:
        raise ValueError("incomplete stream")
    return first, output_tokens


def one(args: argparse.Namespace, number: int) -> dict[str, object]:
    prompt = args.prompt if args.prompt_mode == "same" else f"请求编号 {number}：{args.prompt}"
    body = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens, "stream": args.stream}
    if args.stream:
        body["stream_options"] = {"include_usage": True}
    headers = {"Content-Type": "application/json"}
    if args.key_env and os.environ.get(args.key_env):
        headers["Authorization"] = "Bearer " + os.environ[args.key_env]
    request = urllib.request.Request(args.url.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers=headers)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            if args.stream:
                ttft, tokens = sse_usage(response, started)
            else:
                payload = json.load(response)
                if "error" in payload:
                    raise ValueError("in-band error")
                usage = payload.get("usage", {})
                ttft = time.perf_counter() - started
                tokens = int(usage.get("completion_tokens", 0) or 0)
        return {"status": 200, "seconds": time.perf_counter() - started, "ttft": ttft, "tokens": tokens}
    except urllib.error.HTTPError as error:
        error.close()
        return {"status": error.code, "seconds": time.perf_counter() - started, "ttft": None, "tokens": 0}
    except (OSError, ValueError, KeyError, TypeError):
        return {"status": 0, "seconds": time.perf_counter() - started, "ttft": None, "tokens": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8787")
    parser.add_argument("--model", default="qwen3.5-0.8b-q4")
    parser.add_argument("--key-env", default="LITEINFER_API_KEY")
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 3, 6, 8, 10])
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt", default="请用一句话介绍本地推理服务。")
    parser.add_argument("--prompt-mode", choices=("same", "unique"), default="same")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.requests <= 0 or args.max_tokens <= 0 or any(value <= 0 for value in args.concurrencies):
        parser.error("requests、max-tokens 和并发数必须大于 0")
    if args.key_env and not os.environ.get(args.key_env):
        parser.error(f"请先设置 API Key 环境变量 {args.key_env}")
    reports = []
    for concurrency in args.concurrencies:
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(lambda number: one(args, number), range(args.requests)))
        elapsed = time.perf_counter() - started
        successful = [item for item in results if item["status"] == 200]
        latency = [float(item["seconds"]) for item in successful]
        ttft = [float(item["ttft"]) for item in successful if item["ttft"] is not None]
        tokens = sum(int(item["tokens"]) for item in successful)
        reports.append({
            "concurrency": concurrency,
            "requests": args.requests,
            "completed": len(successful),
            "error_count": len(results) - len(successful),
            "statuses": dict(Counter(str(item["status"]) for item in results)),
            "elapsed_seconds": round(elapsed, 3),
            "successful_requests_per_second": round(len(successful) / elapsed, 3),
            "output_tokens": tokens,
            "output_tokens_per_second": round(tokens / elapsed, 3),
            "latency_ms": {
                "mean": round(statistics.mean(latency) * 1000, 3) if latency else None,
                "p50": percentile(latency, 0.5),
                "p95": percentile(latency, 0.95),
            },
            "ttft_ms_p95": percentile(ttft, 0.95),
        })
    report = {"model": args.model, "stream": args.stream, "prompt_mode": args.prompt_mode,
              "max_tokens": args.max_tokens, "results": reports,
              "note": "只保存状态、延迟和后端 usage，不保存提示正文或模型输出。"}
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if all(report["error_count"] == 0 for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
