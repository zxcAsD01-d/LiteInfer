# Valen Decisions 本地 API

LiteInfer 将 Valen Decisions 模型作为独立的决策 API 接入统一鉴权、限流、请求记录和模型别名管理。该接口返回类别概率、候选得分或图像框，不是聊天补全；服务端不会把决策响应伪装成 OpenAI 对话。

## 配置

Valen 本地后端应提供 `POST /v1/decisions`。将它作为模型注册到 `examples/config.toml`：

```toml
[[models]]
id = "valen-decision-1.5-0.8b-q8"
upstream_model = "valen-decision-1.5-0.8b-q8"
base_url = "http://127.0.0.1:18080/v1"
max_output_tokens = 1024
tools = false
vision = true
```

`upstream_model` 必须与本地 Valen 服务接受的模型 ID 相同；如果启动参数中模型名不同，请相应替换。LiteInfer 将 `/v1/decisions` 转发到后端 `{base_url}/decisions`，并要求响应含 `answers` 数组及 `usage.input_tokens`、`usage.output_tokens`。

## 请求

```sh
curl http://127.0.0.1:8787/v1/decisions \
  -H "Authorization: Bearer $LITEINFER_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"valen-decision-1.5-0.8b-q8","input":"A cat is on the sofa.","questions":[{"name":"cat","type":"predicate","instructions":"A cat is present."}]}'
```

`questions` 的内容和决策类型由 Valen 后端定义。图像请求可在 `input` 中传文本和内嵌 `input_image` 数据块；多模态字段应与所部署的 Valen 版本匹配。LiteInfer 保留上游答案和 usage，仅把返回的 `model` 改为客户端使用的别名。

目前该路由不支持流式输出；必需的 `input`、`questions` 或 usage 缺失时会返回错误。用普通 LiteInfer API Key 鉴权，管理密钥不可用作 API Key。

有关问题、答案和多模态格式的完整定义，请以所用本地 Valen 运行时版本为准。
