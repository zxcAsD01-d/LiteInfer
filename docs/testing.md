# 测试与压测

所有压测报告只保存状态码、延迟、吞吐、token 总数和资源计数，不保存请求正文、Key 或模型输出。模拟后端的 token 是固定夹具，不能作为模型速度结论；真实模型报告注明后端配置和测试范围。

## 自动检查

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --all-targets --locked
cargo build --release --locked
python3 scripts/privacy_check.py
```

单元与集成测试覆盖：

- 配置、URL、并发和请求大小边界；
- Key 哈希、撤销、权限分离和数据库 0600；
- SSE 的 UTF-8 分块、CRLF、超限和异常结束；
- OpenAI JSON/SSE、Completions、工具调用和真实 usage；
- Anthropic JSON/SSE、文本块、工具参数增量、tool_result、错误和官方 SDK 版本头；
- 取消回收、队列满、3 活跃并发峰值、元数据脱敏和控制台 CSP。

## 可复现压测

先启动 `examples/mock_backend.rs`，再用合成配置启动 LiteInfer：

```sh
cargo run --example mock_backend
cargo run --release -- --config config.toml serve
LITEINFER_API_KEY='本地测试Key' python3 scripts/benchmark.py \
  --url http://127.0.0.1:8787 --model local-model --label synthetic \
  --protocol openai --stream --requests 1000 --concurrency 10 \
  --output benchmarks/synthetic-local.json
```

脚本支持 `openai` / `anthropic` 和 JSON / SSE。`--label synthetic` 只测 HTTP、队列、协议转换与元数据开销。将 `--label` 设为 `real-model` 时，必须说明实际后端与模型；不要把合成报告与模型 token/s 混合。

## 本轮结果

- `benchmarks/synthetic-matrix.json`：14,000 次合成请求，OpenAI/Anthropic JSON/SSE，客户端并发 3、10、64、256；3 活跃请求峰值，过载按预算返回 429，无 5xx，元数据无丢弃。
- `benchmarks/real-model-matrix.json`：TaiChu-80M-SFT q4 和 AL-0-SFT q4 各 60 次请求，双协议、JSON/SSE，10 客户端并发全部成功；只验证服务与传输，不评价模型回答质量。
- `benchmarks/sdk-verification.json`：OpenAI Python SDK 3.51.0 与 Anthropic Python SDK 2.8.0，合成及两个本地模型各完成普通/流式调用。
- `benchmarks/html-transport-verification.json`：私有本地素材经 Anthropic 流式工具参数和回执完成传输，两模型 usage 与分词器核对；素材内容不进入仓库。

真实模型压测应先确认 GPU 空闲、后端端口和模型缓存，再逐步增加并发。3 路并发是入口预算，不保证 GPU token/s 为单路的三倍；应同时观察后端 Metal 内存、KV 缓存、预填充和输出长度。

## 故障判定

HTTP 429 表示入口容量或排队预算已满，属于保护行为；不应自动无限重试。5xx、SSE 缺 usage、缺完成事件、非法工具 JSON、断连未释放槽位属于失败。报告中的 `active=0`、`queued=0` 和 `records_dropped=0` 是结束验收条件。
