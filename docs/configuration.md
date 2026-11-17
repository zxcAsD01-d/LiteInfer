# 配置与模型后端

复制 `examples/config.toml` 到仓库根目录 `config.toml`；本地配置已被 Git 忽略。配置未知字段会导致启动失败，便于发现拼写错误。通过 `liteinfer --config config.toml check` 检查。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| listen | 127.0.0.1:8787 | API 与控制台地址 |
| concurrency | 3 | 任意正整数 N，所有模型共享活跃上限 |
| queue_capacity | 24 | 任意非负整数，等待请求数；0 表示不排队 |
| ingress_capacity | 自动 | 默认 max(64, N + queue_capacity + 16)，限制鉴权/请求体阶段；可显式配置正整数 |
| stream_buffer_chunks | 8 | 每条流的背压缓冲块数，正整数 |
| metadata_queue_capacity | 4096 | 异步请求记录队列容量，正整数 |
| queue_timeout_seconds | 60 | 等待活跃槽位的时间 |
| request_timeout_seconds | 180 | 获得槽位后的后端总预算 |
| max_body_bytes | 8388608 | 请求体上限，超出返回 413 |
| max_response_bytes | 16777216 | JSON / SSE 整段响应预算 |
| database | .liteinfer/state.sqlite3 | Key 哈希和请求元数据 |
| allow_remote | false | 监听非回环地址必须显式开启 |

`max_response_bytes` 还约束工具参数；每个 SSE 事件单独最多 1 MiB。大型 HTML 应分块通过 SSE 传输。模型实际上下文与 KV 内存由后端配置，Rust 层不截断系统提示或聊天历史。

## 模型注册

每个 `[[models]]` 是一个公开模型 ID 与后端模型的映射，可以指向同一后端的不同模型，也可以指向不同端口。

```toml
[[models]]
id = "local-chat"
upstream_model = "your-model-id"
base_url = "http://127.0.0.1:8080/v1"
max_output_tokens = 1024
tools = true
vision = false
# api_key_env = "BACKEND_API_KEY"
```

`base_url` 包含 `/v1`，不包含具体接口路径。禁止 URL 内嵌凭据、查询参数和片段；不跟随后端重定向，不读取系统 HTTP 代理。后端地址来自管理员配置，客户端不能更改。`api_key_env` 只填写环境变量名称，具体密钥只从环境读取。

`tools` 与 `vision` 必须按模型及后端的真实能力设置。模型别名不会增加实际能力；响应模型名称恢复为公开 ID。当前 `/v1/models` 使用 OpenAI 格式。

## 后端示例

| 后端 | 通常的 Base URL | 备注 |
| --- | --- | --- |
| llama.cpp server | http://127.0.0.1:8080/v1 | GGUF、Metal，需启用流式 usage |
| MLX-LM server | http://127.0.0.1:8081/v1 | MLX 格式模型，Apple Silicon |
| Ollama | http://127.0.0.1:11434/v1 | upstream_model 填本地模型 tag |
| LM Studio | http://127.0.0.1:1234/v1 | 先启动其本地服务 |

这些是协议适配入口，不代表已测试每个软件版本、模型架构和工具模板。后端必须实现 OpenAI Chat Completions；JSON 需要 usage，SSE 需要 finish_reason、末尾 usage 帧及 `[DONE]`。不满足时显式报错。

示例启动方式（参数以本机安装版本的 `--help` 为准）：

```sh
# GGUF 模型由用户自行下载并确认授权
llama-server --model ./models/model.gguf --host 127.0.0.1 --port 8080 --parallel 3 --n-gpu-layers 99

# MLX 模型由用户自行准备；这是独立后端，不是 Rust 服务的 Python 依赖
python -m mlx_lm server --model ./models/mlx-model --host 127.0.0.1 --port 8081 --decode-concurrency 3 --prompt-concurrency 3 --prompt-cache-bytes 268435456
```

部分 MLX-LM 版本的 KV 量化会禁用批处理，应先测量再启用。不要把 KV 量化和高并发默认为同时加速。模型权重、模型卡与许可证不随 LiteInfer 分发。

## N 并发调优

并发和排队无固定产品上限，校验 `N + queue_capacity` 的整数溢出与 Tokio `Semaphore::MAX_PERMITS`。这是运行时可表示的容量，并非机器可承载的请求数。默认入口容量在运行时最大值处饱和，不会溢出；显式入口容量可以比调度容量低，以便较早限流。

先让后端 batch / parallel 与 N 对齐，再逐步增加 N 并测量总 token/s、首 token 延迟和 KV 内存。增加队列只增加等待容量；增加 SSE 缓冲会增加慢客户端占用。TCP 建连突发还受操作系统 listen backlog 和文件描述符预算影响，与请求调度的 N 是两个维度。可选 MLX 优化及直接调用方式见 [推理优化](mlx-optimization.md)。
