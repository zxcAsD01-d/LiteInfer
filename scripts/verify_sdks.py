#!/usr/bin/env python3
"""Smoke-test official Python SDKs against a running service. Prints no responses or keys."""
import argparse
import importlib.metadata
import json
import os

from anthropic import Anthropic, DefaultHttpxClient as AnthropicHttpClient
from openai import OpenAI, DefaultHttpxClient as OpenAIHttpClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8787")
    parser.add_argument("--model", default="qwen3.5-0.8b-q4")
    args = parser.parse_args()
    key = os.environ["LITEINFER_API_KEY"]
    with OpenAI(base_url=args.url + "/v1", api_key=key, max_retries=0,
                http_client=OpenAIHttpClient(trust_env=False, timeout=180)) as client:
        response = client.chat.completions.create(model=args.model, messages=[{"role": "user", "content": "你好"}], max_tokens=16)
        assert response.model == args.model and response.usage.total_tokens > 0
        stream = client.chat.completions.create(model=args.model, messages=[{"role": "user", "content": "你好"}],
                                                max_tokens=16, stream=True, stream_options={"include_usage": True})
        frames = list(stream)
        assert frames[-1].usage.total_tokens > 0
    with Anthropic(base_url=args.url, api_key=key, max_retries=0,
                   http_client=AnthropicHttpClient(trust_env=False, timeout=180)) as client:
        response = client.messages.create(model=args.model, messages=[{"role": "user", "content": "你好"}], max_tokens=16)
        assert response.model == args.model and response.usage.input_tokens > 0
        with client.messages.stream(model=args.model, messages=[{"role": "user", "content": "你好"}], max_tokens=16) as stream:
            list(stream.text_stream)
            final = stream.get_final_message()
        assert final.model == args.model and final.usage.input_tokens > 0 and final.usage.output_tokens > 0
    print(json.dumps({"status": "passed", "cases": 4, "openai": importlib.metadata.version("openai"),
                      "anthropic": importlib.metadata.version("anthropic"), "scope": "Official SDK JSON and accumulated streaming message usage"}))


if __name__ == "__main__":
    main()
