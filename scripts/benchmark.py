#!/usr/bin/env python3
"""Reproducible API load test. Reports contain numbers only, never credentials or prompts."""
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


def request(args, number):
    body = {"model": args.model, "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens, "stream": args.stream}
    if args.protocol == "openai":
        body["stream_options"] = {"include_usage": True}
    path = "/v1/messages" if args.protocol == "anthropic" else "/v1/chat/completions"
    raw = json.dumps(body).encode()
    headers = {"Authorization": "Bearer " + os.environ[args.key_env], "Content-Type": "application/json"}
    started = time.perf_counter()
    first = None
    try:
        with urllib.request.urlopen(urllib.request.Request(args.url.rstrip("/") + path, data=raw, headers=headers), timeout=args.timeout) as response:
            if args.stream:
                done = False
                usage = None
                tokens = 0
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    first = first or (time.perf_counter() - started)
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        done = True
                        continue
                    frame = json.loads(data)
                    if "error" in frame:
                        raise ValueError("in-band error")
                    if args.protocol == "anthropic":
                        done |= frame.get("type") == "message_stop"
                        if frame.get("type") == "message_delta":
                            usage = frame.get("usage")
                            tokens = usage.get("output_tokens", 0) if usage else 0
                    elif frame.get("usage") is not None:
                        usage = frame["usage"]
                        tokens = usage.get("completion_tokens", 0)
                if not done or usage is None:
                    raise ValueError("incomplete stream")
            else:
                result = json.load(response)
                usage = result["usage"]
                tokens = usage["output_tokens" if args.protocol == "anthropic" else "completion_tokens"]
            return {"status": 200, "seconds": time.perf_counter() - started, "ttft": first, "tokens": tokens}
    except urllib.error.HTTPError as error:
        error.close()
        return {"status": error.code, "seconds": time.perf_counter() - started, "ttft": None, "tokens": 0}
    except (OSError, ValueError, KeyError):
        return {"status": 0, "seconds": time.perf_counter() - started, "ttft": None, "tokens": 0}


def percentile(values, ratio):
    return round(sorted(values)[min(len(values) - 1, int((len(values) - 1) * ratio))] * 1000, 3) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8787")
    parser.add_argument("--key-env", default="LITEINFER_API_KEY")
    parser.add_argument("--model", default="qwen3.5-0.8b")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--protocol", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--prompt", default="你好，请用一句话介绍自己。")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--label", choices=["synthetic", "real-model"], required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.requests <= 0 or args.concurrency <= 0 or not os.environ.get(args.key_env):
        parser.error("requests/concurrency 必须大于 0，且密钥环境变量必须已设置")
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda number: request(args, number), range(args.requests)))
    elapsed = time.perf_counter() - started
    success = [r for r in results if r["status"] == 200]
    latencies = [r["seconds"] for r in success]
    ttfts = [r["ttft"] for r in success if r["ttft"] is not None]
    report = {"label": args.label, "protocol": args.protocol, "stream": args.stream,
              "requests": args.requests, "client_concurrency": args.concurrency,
              "statuses": dict(Counter(str(r["status"]) for r in results)),
              "elapsed_seconds": round(elapsed, 3), "successful_requests_per_second": round(len(success) / elapsed, 3),
              "latency_ms": {"mean": round(statistics.mean(latencies) * 1000, 3) if latencies else None,
                             "p50": percentile(latencies, .5), "p95": percentile(latencies, .95), "p99": percentile(latencies, .99)},
              "ttft_ms_p95": percentile(ttfts, .95), "backend_reported_output_tokens": sum(r["tokens"] for r in success),
              "note": "Synthetic tokens are fixtures, not model throughput. Real counts are reported by the configured backend."}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    if any(r["status"] not in (200, 429) for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
