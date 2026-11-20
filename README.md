# LiteInfer 0.2.0

**为 macOS 本地模型提供轻量、可观测的 Rust API 服务。**

一个入口连接 MLX、llama.cpp 等 OpenAI 兼容推理后端，向客户端提供 OpenAI、Anthropic Messages，以及 Valen 决策 API。默认仅监听本机，默认 3 路、支持配置任意正整数 N 并发，带有有界队列、流式背压、API Key 管理和中文控制台。

LiteInfer 的服务核心使用 Rust；模型计算由独立推理后端执行。它不是新的模型训练框架，也不把协议兼容等同于模型拥有相应能力。

## 功能

- **多协议**：OpenAI Chat Completions / Completions、Anthropic Messages；JSON 和 SSE。Valen /v1/decisions 支持 JSON、文本、图片、视频及真实 token 用量。
- **通用模型注册**：公开别名映射实际模型和后端，不硬编码模型架构。
- **资源控制**：默认 3 活跃 + 24 等待，FIFO 准入、超时、取消和过载 429；并发与排队容量可自由配置。
- **流式管线**：共享连接池、UTF-8 安全解析、可配置背压通道（默认 8 块）、大小预算和断连回收。
- **可选 MLX 优化包**：编译稀疏采样、独立随机种子、租户隔离前缀 KV 缓存、连续批处理、自适应预填充和 Llama 投影融合；附可运行示例与实测报告。
- **网页模型管理**：HuggingFace、ModelScope、自定义镜像或本地目录，读取原生上下文，安装独立环境、下载、MLX 转换、自检、启动并即时注册 API；支持重启恢复。
- **工具调用**：转发 OpenAI function tools；转换 Anthropic tool_use / tool_result 和 JSON 参数增量。
- **真实统计**：使用后端返回的 token 用量；不以字符数估算，不伪造缺失用量。
- **中文控制台**：创建/撤销 Key，查看运行指标、最近请求和 token 用量。
- **隐私默认值**：Key 只保存 SHA-256；请求日志不保存正文、工具参数、密钥或上游错误原文。

## 快速开始

准备 Rust 1.85+，可从网页控制台安装模型，也可连接已有 OpenAI 兼容后端。Apple Silicon 推荐使用支持 Metal 的 MLX 或 llama.cpp 后端；Intel Mac 可连接适配其硬件的后端。

```sh
git clone https://github.com/zxcAsD01-d/LiteInfer.git
cd LiteInfer
cargo build --release --locked
```

也可以直接运行 `./target/release/liteinfer serve`：缺少配置时自动生成，终端显示当前进程管理密钥。打开控制台即可[完成模型安装与自动配置](docs/model-management.md)，无需手动填写后端配置。

网页安装流程默认使用 `mlx-community/Qwen3.5-0.8B-OptiQ-4bit`，注册 API 名称 `qwen3.5-0.8b`。首次启动不会自动下载权重；在控制台确认下载源后安装，权重需按其许可证自行下载。

MLX 后端使用 MLX-LM 原生推理与连续批处理。**Apple M1 Pro（16 GB）实测：单请求约 135 token/s，3 路批量约 230 token/s（实测 235），6 路批量约 276 token/s，8 路约 263 token/s。**批量数据是并行请求的聚合生成吞吐，不是单个请求速度，也不等同于 LiteInfer HTTP API 吞吐。模型为 `Qwen3.5-0.8B-OptiQ-4bit`，提示 128 token、生成 64 token；测试配置、软件版本与复现命令见[MLX 基准说明](docs/mlx-optimization.md)和[原始结果](benchmarks/qwen35-08b-mlx-m1pro.json)。

连接已有后端时，先复制 `examples/config.toml` 为 `config.toml`，编辑其中的 `models`：将 `upstream_model`、`base_url` 改为后端实际值，然后创建访问密钥：

```sh
./target/release/liteinfer check
./target/release/liteinfer key create --name local-client
```

保存命令返回的 `key`，它只显示一次。为控制台生成独立管理密钥，再启动：

```sh
export LITEINFER_ADMIN_KEY="$(openssl rand -hex 32)"
# 在自己的密码管理器保存该环境变量值，勿提交到仓库。
./target/release/liteinfer serve
```

打开 `http://127.0.0.1:8787/`，输入管理密钥即可管理 Key 和请求。API Key 与管理密钥分开。也可以设置 `LITEINFER_API_KEY`（至少24字符）作为环境Key，省略创建步骤。

## 自定义 N 并发与推理优化

```toml
concurrency = 16         # 任意正整数 N，默认 3
queue_capacity = 128     # 0 表示不排队；支持超过 4096
stream_buffer_chunks = 8
metadata_queue_capacity = 8192
# ingress_capacity = 256 # 可选：鉴权/请求体阶段容量，省略时自动计算
```

移除原来的 128 并发和 4096 排队上限；只保留整数溢出、运行时容量和正数校验。配置大 N 不会预分配 N 份模型权重，但实际请求仍消耗内存、文件描述符和后端槽位。网关和后端的并发应配套调整。

[MLX 推理优化与接入](docs/mlx-optimization.md)包含可安装源码、批处理 CLI、Python 接入示例、适用架构和基准脚本。Rust 网关保持独立，安装该包不会自动改动已运行的后端。

## 请求示例

把创建的Key放入本地环境变量 `LITEINFER_API_KEY`。示例模型名与 `examples/config.toml` 一致。

```sh
curl http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer $LITEINFER_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-0.8b","messages":[{"role":"user","content":"你好"}],"max_tokens":128}'
```

Anthropic 格式：

```sh
curl -N http://127.0.0.1:8787/v1/messages \
  -H "x-api-key: $LITEINFER_API_KEY" \
  -H 'anthropic-version: 2023-06-01' \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-0.8b","max_tokens":128,"messages":[{"role":"user","content":"你好"}],"stream":true}'
```

OpenAI Python SDK：

```python
import os
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key=os.environ["LITEINFER_API_KEY"])
response = client.chat.completions.create(
    model="qwen3.5-0.8b", messages=[{"role": "user", "content": "你好"}], max_tokens=128
)
print(response.choices[0].message.content)
```

Anthropic Python SDK（本地不需要 Anthropic 云端 Key）：

```python
import os
from anthropic import Anthropic

client = Anthropic(base_url="http://127.0.0.1:8787", api_key=os.environ["LITEINFER_API_KEY"])
with client.messages.stream(model="qwen3.5-0.8b", max_tokens=128,
                            messages=[{"role": "user", "content": "你好"}]) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
    print(stream.get_final_message().usage)
```

## Valen Decisions 示例

该路由使用官方 Decisions 请求结构，需传 model、input 和 questions；它输出分类/判断结果，不是聊天补全接口。图片和视频应以内嵌 base64 数据块提交。完整部署说明见 [Valen Decisions 部署说明](docs/valen-decisions.md)。

~~~sh
curl http://127.0.0.1:8787/v1/decisions \
  -H "Authorization: Bearer $LITEINFER_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"valen-decision-1.5-0.8b","input":"A package delivery is late.","questions":[{"name":"shipping","type":"predicate","instructions":"The request is about shipping."}]}'
~~~

## 兼容边界

支持标准 Messages 交互不代表实现全部 Anthropic 云端功能。`thinking`、PDF、服务端工具、beta、Responses API 暂不支持；`/v1/messages/count_tokens` 返回501。图片和工具需要模型本身具备能力并在配置启用。

流式输入用量在上游只提供末尾统计时，`message_start` 使用0占位，`message_delta` 返回实际输入和输出计数。后端必须支持流式usage和完成事件；缺失时返回错误。具体字段、停止原因和扩展限制见[兼容矩阵](docs/compatibility.md)。

## 文档与验证

- [文档导航](docs/README.md)：配置、后端、部署、架构、协议与迁移。
- [网页模型管理](docs/model-management.md)：纯前端配置校验、下载源、MLX 转换、自检、启动与恢复。
- [Valen MLX 实验](docs/valen-mlx.md)：原生权重转换、BF16/q8/q4 对照、自定义 Metal 算子及 M1 Pro 实测边界。
- [压测方法与结果](docs/testing.md)：区分模拟服务开销和真实模型推理；不把模拟 token 当成模型吞吐。
- [安全与隐私](SECURITY.md)：公开内容、密钥、日志与部署边界。
- [贡献指南](CONTRIBUTING.md)：检查命令和代码原则。

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --all-targets --locked
python3 scripts/privacy_check.py
```

项目权利人有权许可的原创软件采用 [Apache License 2.0](LICENSE)，允许商业使用，无须另行购买商用许可；分发时须遵守许可、署名、NOTICE 及修改标识等适用要求。授权范围及历史权利见 [LICENSING.md](LICENSING.md)，第三方例外见 [第三方声明](THIRD_PARTY_NOTICES.md)，商业使用与可选支持说明见 [COMMERCIAL.md](COMMERCIAL.md)。

模型权重和第三方后端分别遵守其许可证，仓库不包含模型、用户素材、私人配置或历史请求。
