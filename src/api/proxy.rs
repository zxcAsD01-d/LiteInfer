use super::anthropic::{self, Translator};
use crate::{config::Model, error::ApiError, scheduler::Lease, sse, state::AppState, store};
use axum::{
    Json,
    body::Body,
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use serde_json::{Value, json};
use std::{convert::Infallible, sync::Arc, time::Duration};
use tokio::time::Instant;

pub fn prepare(
    state: &AppState,
    body: &mut Value,
    key: &str,
    completion: bool,
) -> Result<Model, ApiError> {
    let object = body
        .as_object_mut()
        .ok_or_else(|| ApiError::bad("请求必须是 JSON 对象"))?;
    let name = object
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::bad("model 必须是字符串"))?;
    let model = state.registry.get(name)?;
    if let Some(stream) = object.get("stream").filter(|v| !v.is_null()) {
        if !stream.is_boolean() {
            return Err(ApiError::bad("stream 必须是布尔值"));
        }
    }
    if let Some(options) = object.get("stream_options").filter(|v| !v.is_null()) {
        if !options.is_object()
            || options
                .get("include_usage")
                .is_some_and(|v| !v.is_boolean())
        {
            return Err(ApiError::bad("stream_options.include_usage 必须为布尔值"));
        }
    }
    if object
        .get("n")
        .filter(|v| !v.is_null())
        .is_some_and(|n| n.as_u64() != Some(1))
    {
        return Err(ApiError::bad("仅支持 n=1"));
    }
    if object
        .get("tools")
        .filter(|v| !v.is_null())
        .is_some_and(|v| v.as_array().is_none_or(|a| !a.is_empty()))
        && !model.tools
    {
        return Err(ApiError::bad("该模型未启用 tools 能力"));
    }
    if completion {
        if object
            .get("prompt")
            .is_none_or(|p| p.as_str().is_none_or(|s| s.is_empty()))
        {
            return Err(ApiError::bad("completions.prompt 必须是非空字符串"));
        }
    } else {
        validate_messages(object.get("messages"), &model)?;
    }
    if let Some(tools) = object.get_mut("tools").filter(|v| !v.is_null()) {
        for tool in tools
            .as_array_mut()
            .ok_or_else(|| ApiError::bad("tools 必须是数组"))?
        {
            if tool["type"] != "function" || !tool["function"]["name"].is_string() {
                return Err(ApiError::bad(
                    "仅支持具有 name/parameters 的 function tools",
                ));
            }
            if tool["function"].get("parameters").is_none() {
                tool["function"]["parameters"] = json!({"type":"object","properties":{}});
            } else if !tool["function"]["parameters"].is_object() {
                return Err(ApiError::bad("工具 parameters 必须为 JSON Schema 对象"));
            }
        }
    }
    normalize_budget(object, model.max_output_tokens)?;
    if let Some(parameter) = &model.cache_isolation_parameter {
        object.remove("cache_salt");
        object.remove("prompt_cache_key");
        object.insert(parameter.clone(), json!(store::hash(key)));
    }
    object.insert("model".into(), json!(model.upstream_model));
    Ok(model)
}

fn validate_messages(messages: Option<&Value>, model: &Model) -> Result<(), ApiError> {
    let messages = messages
        .and_then(Value::as_array)
        .filter(|m| !m.is_empty())
        .ok_or_else(|| ApiError::bad("messages 必须是非空数组"))?;
    for message in messages {
        if !message["role"]
            .as_str()
            .is_some_and(|s| matches!(s, "system" | "developer" | "user" | "assistant" | "tool"))
        {
            return Err(ApiError::bad("message.role 无效"));
        }
        let content = &message["content"];
        if !(content.is_string()
            || content.is_array()
            || (content.is_null() && message["tool_calls"].is_array()))
        {
            return Err(ApiError::bad("message.content 无效"));
        }
        if let Some(blocks) = content.as_array() {
            for block in blocks {
                match block["type"].as_str() {
                    Some("text") if block["text"].is_string() => (),
                    Some("image_url") if model.vision && block["image_url"]["url"].is_string() => {}
                    _ => return Err(ApiError::bad("不支持的内容块，或模型未启用 vision")),
                }
            }
        }
    }
    Ok(())
}

fn normalize_budget(
    object: &mut serde_json::Map<String, Value>,
    maximum: u32,
) -> Result<(), ApiError> {
    let a = object.get("max_tokens");
    let b = object.get("max_completion_tokens");
    if a.is_some() && b.is_some() && a != b {
        return Err(ApiError::bad("max_tokens 与 max_completion_tokens 冲突"));
    }
    let requested = match a.or(b) {
        None | Some(Value::Null) => 128,
        Some(value) => value
            .as_u64()
            .filter(|n| *n > 0)
            .ok_or_else(|| ApiError::bad("输出预算必须为正整数"))?,
    };
    object.remove("max_completion_tokens");
    object.insert("max_tokens".into(), json!(requested.min(maximum as u64)));
    Ok(())
}

pub async fn execute(
    state: Arc<AppState>,
    headers: HeaderMap,
    mut body: Value,
    anthropic: bool,
    completion: bool,
) -> Result<Response, ApiError> {
    let key = state.authenticate(&headers)?;
    let model = prepare(&state, &mut body, &key, completion)?;
    let stream = body["stream"].as_bool().unwrap_or(false);
    let include_usage = anthropic
        || body["stream_options"]["include_usage"]
            .as_bool()
            .unwrap_or(false);
    if stream {
        body["stream_options"] = json!({"include_usage":true});
    }
    let mut lease = Lease::acquire(
        state.clone(),
        key,
        model.id.clone(),
        if anthropic { "anthropic" } else { "openai" },
    )
    .await?;
    let deadline = Instant::now() + Duration::from_secs(state.config.request_timeout_seconds);
    let path = if completion {
        "completions"
    } else {
        "chat/completions"
    };
    let mut request = state
        .client
        .post(format!("{}/{path}", model.base_url.trim_end_matches('/')))
        .json(&body);
    if let Some(key) = state.upstream_keys.get(&model.id) {
        request = request.bearer_auth(key);
    } else if let Some(name) = &model.api_key_env {
        request = request
            .bearer_auth(std::env::var(name).map_err(|_| ApiError::bad("后端密钥环境变量未设置"))?);
    }
    let upstream = match tokio::time::timeout_at(deadline, request.send()).await {
        Ok(Ok(response)) => response,
        result => {
            let code = if result.is_err() {
                StatusCode::GATEWAY_TIMEOUT
            } else {
                StatusCode::BAD_GATEWAY
            };
            lease.finish(code.as_u16());
            return Err(ApiError::new(code, "无法连接推理后端或后端响应超时"));
        }
    };
    if !upstream.status().is_success() {
        let status = if upstream.status().is_client_error() {
            upstream.status()
        } else {
            StatusCode::BAD_GATEWAY
        };
        lease.finish(status.as_u16());
        return Err(ApiError::new(
            status,
            "推理后端拒绝请求，请检查模型能力和后端配置",
        ));
    }
    let request_id = lease.id.clone();
    let mut response = if stream {
        if !upstream
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|s| s.starts_with("text/event-stream"))
        {
            lease.finish(502);
            return Err(ApiError::new(StatusCode::BAD_GATEWAY, "后端未返回 SSE"));
        }
        let stream = upstream_events(
            upstream,
            lease,
            deadline,
            state.config.max_response_bytes,
            anthropic,
            include_usage,
        );
        let mut response = Response::new(Body::from_stream(buffered_stream(
            stream,
            deadline,
            state.config.stream_buffer_chunks,
        )));
        response.headers_mut().insert(
            "content-type",
            HeaderValue::from_static("text/event-stream"),
        );
        response
            .headers_mut()
            .insert("x-accel-buffering", HeaderValue::from_static("no"));
        response
    } else {
        let bytes = read_bounded(upstream, deadline, state.config.max_response_bytes)
            .await
            .inspect_err(|e| lease.finish(e.status.as_u16()))?;
        let mut payload: Value = serde_json::from_slice(&bytes).map_err(|_| {
            lease.finish(502);
            ApiError::new(StatusCode::BAD_GATEWAY, "后端返回无效 JSON")
        })?;
        validate_completion(&payload, completion).inspect_err(|_| lease.finish(502))?;
        anthropic::usage(&payload["usage"]).inspect_err(|_| lease.finish(502))?;
        lease.usage(&payload["usage"]);
        if anthropic {
            payload =
                anthropic::from_openai(&payload, &model.id).inspect_err(|_| lease.finish(502))?;
        } else {
            payload["model"] = json!(model.id);
        }
        lease.finish(200);
        Json(payload).into_response()
    };
    response.headers_mut().insert(
        "x-request-id",
        HeaderValue::from_str(&request_id).expect("generated id"),
    );
    response.headers_mut().insert(
        "cache-control",
        HeaderValue::from_static("no-store, no-transform"),
    );
    response
        .headers_mut()
        .insert("x-usage-source", HeaderValue::from_static("upstream"));
    Ok(response)
}

fn validate_completion(payload: &Value, completion: bool) -> Result<(), ApiError> {
    let choices = payload["choices"]
        .as_array()
        .filter(|c| c.len() == 1)
        .ok_or_else(|| ApiError::new(StatusCode::BAD_GATEWAY, "后端响应缺少唯一 choice"))?;
    if !choices[0]["finish_reason"].is_string()
        || (completion && !choices[0]["text"].is_string())
        || (!completion && !choices[0]["message"].is_object())
    {
        return Err(ApiError::new(
            StatusCode::BAD_GATEWAY,
            "后端 choice 格式无效",
        ));
    }
    Ok(())
}

pub(super) async fn read_bounded(
    response: reqwest::Response,
    deadline: Instant,
    limit: usize,
) -> Result<Vec<u8>, ApiError> {
    let mut stream = response.bytes_stream();
    let mut output = Vec::new();
    loop {
        let next = tokio::time::timeout_at(deadline, stream.next())
            .await
            .map_err(|_| ApiError::new(StatusCode::GATEWAY_TIMEOUT, "后端响应超时"))?;
        let Some(chunk) = next else {
            return Ok(output);
        };
        let bytes = chunk.map_err(|_| ApiError::new(StatusCode::BAD_GATEWAY, "后端响应中断"))?;
        if output.len() + bytes.len() > limit {
            return Err(ApiError::new(
                StatusCode::BAD_GATEWAY,
                "后端响应超过大小预算",
            ));
        }
        output.extend_from_slice(&bytes);
    }
}

struct StreamState {
    translator: Option<Translator>,
    finished: bool,
    usage: bool,
    model: String,
    include_usage: bool,
}

fn has_stream_output(choices: &[Value]) -> bool {
    let nonempty = |field: &Value| field.as_str().is_some_and(|text| !text.is_empty());
    choices.iter().any(|choice| {
        let delta = &choice["delta"];
        nonempty(&choice["text"])
            || ["content", "reasoning", "reasoning_content"]
                .iter()
                .any(|field| nonempty(&delta[field]))
            || delta["tool_calls"].as_array().is_some_and(|calls| {
                calls
                    .iter()
                    .any(|call| nonempty(&call["function"]["arguments"]))
            })
    })
}

impl StreamState {
    fn frame(
        &mut self,
        frame: sse::Frame,
        lease: &mut Lease,
        received: std::time::Instant,
    ) -> Result<Vec<Bytes>, ApiError> {
        if frame.data == "[DONE]" {
            if !self.finished || !self.usage {
                return Err(ApiError::new(
                    StatusCode::BAD_GATEWAY,
                    "后端流缺少 finish_reason 或 usage",
                ));
            }
            return match &mut self.translator {
                Some(translator) => translator.finish(),
                None => Ok(vec![Bytes::from_static(b"data: [DONE]\n\n")]),
            };
        }
        let mut payload: Value = serde_json::from_str(&frame.data)
            .map_err(|_| ApiError::new(StatusCode::BAD_GATEWAY, "后端 SSE JSON 无效"))?;
        let Some(choices) = payload["choices"].as_array() else {
            return Err(ApiError::new(
                StatusCode::BAD_GATEWAY,
                "后端 SSE 缺少 choices 数组",
            ));
        };
        if payload.get("error").is_some() {
            return Err(ApiError::new(StatusCode::BAD_GATEWAY, "后端流返回错误"));
        }
        if payload.get("usage").is_some_and(|u| !u.is_null()) {
            anthropic::usage(&payload["usage"])?;
            self.usage = true;
            lease.usage(&payload["usage"]);
        }
        if has_stream_output(choices) {
            lease.observe_output(received);
        }
        self.finished |= choices.iter().any(|c| c["finish_reason"].is_string());
        if let Some(translator) = &mut self.translator {
            return translator.push(&payload);
        }
        payload["model"] = json!(self.model);
        if !self.include_usage {
            payload
                .as_object_mut()
                .ok_or_else(|| ApiError::bad("后端 SSE 必须为对象"))?
                .remove("usage");
            if payload["choices"].as_array().is_some_and(Vec::is_empty) {
                return Ok(vec![]);
            }
        }
        Ok(vec![sse::data(&payload)])
    }
}

fn upstream_events(
    response: reqwest::Response,
    mut lease: Lease,
    deadline: Instant,
    limit: usize,
    anthropic: bool,
    include_usage: bool,
) -> impl Stream<Item = Result<Bytes, Infallible>> + Send + 'static {
    async_stream::stream! {
        let mut upstream = response.bytes_stream();
        let mut decoder = sse::Decoder::default();
        let mut state = StreamState {translator:anthropic.then(|| Translator::new(&lease.model)),finished:false,usage:false,model:lease.model.clone(),include_usage};
        let mut received = 0;
        let error = 'receive: loop {
            let chunk = match tokio::time::timeout_at(deadline, upstream.next()).await {
                Ok(Some(Ok(chunk))) => chunk,
                Ok(None) => break ApiError::new(StatusCode::BAD_GATEWAY, if decoder.is_clean() {"后端流未以完成事件结束"} else {"后端流在事件中途结束"}),
                Ok(Some(Err(_))) => break ApiError::new(StatusCode::BAD_GATEWAY,"后端流连接中断"),
                Err(_) => break ApiError::new(StatusCode::GATEWAY_TIMEOUT,"后端流超时"),
            };
            let arrival = std::time::Instant::now();
            received += chunk.len();
            if received > limit { break ApiError::new(StatusCode::BAD_GATEWAY,"后端流超过大小预算"); }
            let frames = match decoder.feed(&chunk, limit.min(1024*1024)) {
                Ok(frames) => frames, Err(_) => break ApiError::new(StatusCode::BAD_GATEWAY,"后端 SSE 格式或大小无效"),
            };
            for frame in frames {
                let done = frame.data == "[DONE]";
                match state.frame(frame,&mut lease,arrival) {
                    Ok(events) => { for event in events { yield Ok(event); } },
                    Err(error) => break 'receive error,
                }
                if done { lease.finish(200); return; }
            }
        };
        lease.finish(error.status.as_u16());
        let error = if anthropic { error.anthropic() } else { error };
        yield Ok(if anthropic { sse::event("error",&error.payload()) } else { sse::data(&error.payload()) });
        if !anthropic { yield Ok(Bytes::from_static(b"data: [DONE]\n\n")); }
    }
}

fn buffered_stream(
    stream: impl Stream<Item = Result<Bytes, Infallible>> + Send + 'static,
    deadline: Instant,
    capacity: usize,
) -> impl Stream<Item = Result<Bytes, Infallible>> + Send + 'static {
    let (sender, receiver) = tokio::sync::mpsc::channel(capacity);
    tokio::spawn(async move {
        futures_util::pin_mut!(stream);
        loop {
            let item = tokio::select! {
                _ = sender.closed() => break,
                item = stream.next() => match item {Some(item)=>item,None=>break},
            };
            match sender.try_send(item) {
                Ok(()) => (),
                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => break,
                Err(tokio::sync::mpsc::error::TrySendError::Full(item)) => {
                    if !matches!(
                        tokio::time::timeout_at(deadline, sender.send(item)).await,
                        Ok(Ok(()))
                    ) {
                        break;
                    }
                }
            }
        }
    });
    futures_util::stream::unfold(receiver, |mut receiver| async move {
        receiver.recv().await.map(|item| (item, receiver))
    })
}
