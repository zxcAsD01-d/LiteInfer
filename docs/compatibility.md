# 协议兼容矩阵

## OpenAI 格式

| 功能 | 状态 |
| --- | --- |
| GET /v1/models | 支持，公开模型 ID |
| POST /v1/chat/completions | 支持 JSON 与 SSE |
| POST /v1/completions | 支持单个字符串 prompt，JSON 与 SSE |
| messages / system / developer / assistant / tool | 支持 |
| function tools、tool_choice、tool_calls、tool 回传 | 转发给有能力的后端 |
| 图片内容 | 配置 vision=true 后转发 |
| max_tokens / max_completion_tokens | 同义归一化；冲突报错；超模型预算封顶 |
| temperature / top_p / seed / stop / response_format | 保留并转发，能力由后端决定 |
| stream_options.include_usage | 支持，未请求时过滤 usage 帧 |
| n | 仅支持 1 |
| Responses API、音频、嵌入、批量任务 | 未实现，404 |

未知 OpenAI 扩展字段转发到后端，不假定其已经支持。OpenAI SSE始终向后端请求 usage，确保监控有真实计数；客户端未请求时不展示该帧。模型名、采样参数与输出预算以规范化后的请求为准。正整数输出预算超出模型配置时封顶；0、负数、字符串和布尔值返回400。

## Anthropic 格式

接口 `POST /v1/messages`，使用 `x-api-key` 或 Bearer 鉴权。支持 `anthropic-version: 2023-06-01` 和官方 Python SDK 2.8.0 当前默认的 `2026-10-21`，也接受省略版本。其他显式版本拒绝。两者都限定为本文的 Messages 子集，不承诺新版本全部云端特性。

| 字段/能力 | 状态 |
| --- | --- |
| model / max_tokens / messages / system | 支持，max_tokens 必填正整数 |
| text / image(base64、HTTPS URL) | 支持，图片需模型声明 vision |
| tools(name,description,input_schema) | 映射到 OpenAI function tools |
| tool_use / tool_result 历史 | 支持，tool_result 支持字符串或 text 块 |
| tool_choice:auto/any/tool/none | 支持，默认关闭并行工具请求 |
| temperature / top_p / top_k / stop_sequences | 转发到后端 |
| stream | 完整事件序列与输入 JSON 增量 |
| stop_reason | stop→end_turn，length→max_tokens，tool_calls→tool_use |
| usage | 后端 prompt/completion 映射到 input/output |
| thinking、文档/PDF、服务端工具、MCP、beta 扩展 | 未实现，明确拒绝 |
| POST /v1/messages/count_tokens | 501，不估造模型分词数 |
| Anthropic 专用 prompt caching / citations / 搜索 / 文件 API | 未实现 |

支持主要 Messages 交互格式不等于实现 Anthropic 全部云端功能，也不保证每种 Agent 客户端都能使用。OpenAI 后端不能区分EOS与匹配停止字符串时，返回end_turn，stop_sequence为null。工具is_error标志不改变正文；工具执行结果仍由调用方提供。

### SSE 事件

`message_start → content_block_start → content_block_delta → content_block_stop → message_delta → message_stop`。文本使用 `text_delta`，工具参数使用 `input_json_delta`。网络分块不要求和token边界一致，中文UTF-8不会被拆成无效文本。并行工具增量先缓冲次要工具，在主工具结束后按完整块发送。

OpenAI 后端通常在流末尾才给输入用量；因此 message_start 使用0占位，最终 message_delta 返回实际 input_tokens / output_tokens。客户端应以末尾用量为准；若客户端强依赖开始帧的精确输入数，需要后端提供前置分词能力，此版本不声称支持。

后端缺少 usage、JSON 不完整、流提前断开或工具参数不是 JSON 对象时，发送协议对应 error。Anthropic 错误流不发送伪造的 message_stop。未发送响应头时，错误使用相应HTTP状态。

## 安全响应

后端错误文本不会直接透出到客户端，避免包含本机路径或凭据。API错误保留类型、可用状态和操作提示。管理接口与模型API使用不同密钥；管理Key不被当成推理Key接受。
