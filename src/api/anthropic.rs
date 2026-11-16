use crate::{error::ApiError, sse};
use bytes::Bytes;
use serde_json::{Value, json};
use std::collections::BTreeMap;

fn text(value: &Value) -> Result<String, ApiError> {
    if let Some(s) = value.as_str() {
        return Ok(s.into());
    }
    value
        .as_array()
        .ok_or_else(|| ApiError::bad("content 必须是文本或内容块数组"))?
        .iter()
        .map(|block| {
            if block["type"] != "text" {
                return Err(ApiError::bad("此位置仅支持 text 内容块"));
            }
            block["text"]
                .as_str()
                .map(String::from)
                .ok_or_else(|| ApiError::bad("text 必须是字符串"))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|parts| parts.join("\n"))
}

pub fn to_openai(body: &Value) -> Result<Value, ApiError> {
    for name in ["thinking", "output_config", "mcp_servers", "container"] {
        if body.get(name).is_some_and(|v| !v.is_null()) {
            return Err(ApiError::bad(format!("暂不支持 Anthropic {name}")));
        }
    }
    if body["max_tokens"].as_u64().is_none_or(|n| n == 0) {
        return Err(ApiError::bad("max_tokens 必须为正整数"));
    }
    let mut messages = Vec::new();
    if let Some(system) = body.get("system") {
        messages.push(json!({"role":"system","content":text(system)?}));
    }
    let input = body["messages"]
        .as_array()
        .filter(|m| !m.is_empty())
        .ok_or_else(|| ApiError::bad("messages 不能为空"))?;
    for message in input {
        convert_message(message, &mut messages)?;
    }
    let mut output = json!({"model":body["model"],"messages":messages,"max_tokens":body["max_tokens"],"stream":body.get("stream").unwrap_or(&Value::Bool(false))});
    for name in ["temperature", "top_p", "top_k"] {
        if let Some(value) = body.get(name) {
            output[name] = value.clone();
        }
    }
    if let Some(stop) = body.get("stop_sequences") {
        output["stop"] = stop.clone();
    }
    if let Some(tools) = body.get("tools") {
        let tools = tools
            .as_array()
            .ok_or_else(|| ApiError::bad("tools 必须是数组"))?;
        output["tools"] = tools.iter().map(|tool| {
            if !tool["name"].is_string() || !tool["input_schema"].is_object() || tool.get("type").is_some() {
                return Err(ApiError::bad("仅支持具有 name/input_schema 的客户端工具"));
            }
            Ok(json!({"type":"function","function":{"name":tool["name"],"description":tool.get("description").unwrap_or(&json!("")),"parameters":tool["input_schema"]}}))
        }).collect::<Result<Vec<_>, _>>()?.into();
        output["parallel_tool_calls"] = json!(false);
    }
    if let Some(choice) = body.get("tool_choice") {
        output["tool_choice"] = match choice["type"].as_str() {
            Some("auto") => json!("auto"),
            Some("any") => json!("required"),
            Some("none") => json!("none"),
            Some("tool") if choice["name"].is_string() => {
                json!({"type":"function","function":{"name":choice["name"]}})
            }
            _ => return Err(ApiError::bad("tool_choice 无效")),
        };
    }
    Ok(output)
}

fn convert_message(message: &Value, out: &mut Vec<Value>) -> Result<(), ApiError> {
    let role = message["role"]
        .as_str()
        .filter(|r| matches!(*r, "user" | "assistant"))
        .ok_or_else(|| ApiError::bad("Anthropic message.role 仅支持 user/assistant"))?;
    if message["content"].is_string() {
        out.push(json!({"role":role,"content":message["content"]}));
        return Ok(());
    }
    let blocks = message["content"]
        .as_array()
        .ok_or_else(|| ApiError::bad("content 无效"))?;
    let mut content = Vec::new();
    let mut calls = Vec::new();
    for block in blocks {
        match block["type"].as_str() {
            Some("text") if block["text"].is_string() => {
                content.push(json!({"type":"text","text":block["text"]}))
            }
            Some("image") if role == "user" => content.push(convert_image(block)?),
            Some("tool_use") if role == "assistant" => {
                if !block["id"].is_string()
                    || !block["name"].is_string()
                    || !block["input"].is_object()
                {
                    return Err(ApiError::bad("tool_use 需要 id/name/input 对象"));
                }
                calls.push(json!({"id":block["id"],"type":"function","function":{"name":block["name"],"arguments":block["input"].to_string()}}));
            }
            Some("tool_result") if role == "user" => {
                if !block["tool_use_id"].is_string() {
                    return Err(ApiError::bad("tool_result 缺少 tool_use_id"));
                }
                let result = text(block.get("content").unwrap_or(&json!("")))?;
                out.push(
                    json!({"role":"tool","tool_call_id":block["tool_use_id"],"content":result}),
                );
            }
            _ => return Err(ApiError::bad("不支持的 Anthropic 内容块或角色")),
        }
    }
    if !content.is_empty() || !calls.is_empty() {
        let mut converted = json!({"role":role,"content":if content.is_empty() { Value::Null } else { json!(content) }});
        if !calls.is_empty() {
            converted["tool_calls"] = json!(calls);
        }
        out.push(converted);
    }
    Ok(())
}

fn convert_image(block: &Value) -> Result<Value, ApiError> {
    let source = &block["source"];
    let url = match source["type"].as_str() {
        Some("base64") => {
            let media = source["media_type"]
                .as_str()
                .filter(|s| matches!(*s, "image/jpeg" | "image/png" | "image/gif" | "image/webp"))
                .ok_or_else(|| ApiError::bad("图片 media_type 无效"))?;
            let data = source["data"]
                .as_str()
                .ok_or_else(|| ApiError::bad("图片 data 无效"))?;
            format!("data:{media};base64,{data}")
        }
        Some("url") => {
            let url = source["url"]
                .as_str()
                .ok_or_else(|| ApiError::bad("图片 URL 无效"))?;
            if !url.starts_with("https://") {
                return Err(ApiError::bad("图片 URL 必须使用 HTTPS"));
            }
            url.into()
        }
        _ => return Err(ApiError::bad("不支持的图片来源")),
    };
    Ok(json!({"type":"image_url","image_url":{"url":url}}))
}

pub fn stop_reason(reason: &str) -> Result<&'static str, ApiError> {
    match reason {
        "tool_calls" | "function_call" => Ok("tool_use"),
        "length" => Ok("max_tokens"),
        "stop" => Ok("end_turn"),
        _ => Err(ApiError::new(
            axum::http::StatusCode::BAD_GATEWAY,
            "后端返回了无法映射的停止原因",
        )),
    }
}

pub fn usage(value: &Value) -> Result<Value, ApiError> {
    let input = value["prompt_tokens"].as_u64().ok_or_else(|| {
        ApiError::new(
            axum::http::StatusCode::BAD_GATEWAY,
            "后端未提供输入 token 用量",
        )
    })?;
    let output = value["completion_tokens"].as_u64().ok_or_else(|| {
        ApiError::new(
            axum::http::StatusCode::BAD_GATEWAY,
            "后端未提供输出 token 用量",
        )
    })?;
    Ok(json!({"input_tokens":input,"output_tokens":output}))
}

pub fn from_openai(body: &Value, model: &str) -> Result<Value, ApiError> {
    let choice = &body["choices"][0];
    let message = &choice["message"];
    let mut content = Vec::new();
    if let Some(text) = message["content"].as_str().filter(|s| !s.is_empty()) {
        content.push(json!({"type":"text","text":text}));
    }
    if let Some(calls) = message["tool_calls"].as_array() {
        for call in calls {
            let input = parse_arguments(call["function"]["arguments"].as_str().unwrap_or(""))?;
            content.push(json!({"type":"tool_use","id":call["id"],"name":call["function"]["name"],"input":input}));
        }
    }
    Ok(
        json!({"id":format!("msg_{}",uuid::Uuid::new_v4().simple()),"type":"message","role":"assistant","model":model,
        "content":content,"stop_reason":stop_reason(choice["finish_reason"].as_str().unwrap_or(""))?,"stop_sequence":null,"usage":usage(&body["usage"])?}),
    )
}

fn parse_arguments(raw: &str) -> Result<Value, ApiError> {
    serde_json::from_str::<Value>(raw)
        .ok()
        .filter(Value::is_object)
        .ok_or_else(|| {
            ApiError::new(
                axum::http::StatusCode::BAD_GATEWAY,
                "后端工具参数不是完整 JSON 对象",
            )
        })
}

#[derive(Default)]
struct Tool {
    id: String,
    name: String,
    arguments: String,
    emitted: usize,
}

pub struct Translator {
    model: String,
    id: String,
    started: bool,
    text_index: Option<usize>,
    text_closed: bool,
    next_index: usize,
    active_tool: Option<(u64, usize)>,
    tools: BTreeMap<u64, Tool>,
    finish_reason: Option<String>,
    usage: Option<Value>,
}

impl Translator {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.into(),
            id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
            started: false,
            text_index: None,
            text_closed: false,
            next_index: 0,
            active_tool: None,
            tools: BTreeMap::new(),
            finish_reason: None,
            usage: None,
        }
    }

    pub fn push(&mut self, frame: &Value) -> Result<Vec<Bytes>, ApiError> {
        let mut events = Vec::new();
        if !self.started {
            self.started = true;
            events.push(sse::event("message_start", &json!({"type":"message_start","message":{"id":self.id,"type":"message","role":"assistant","model":self.model,"content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}})));
        }
        if frame.get("usage").is_some_and(|v| !v.is_null()) {
            self.usage = Some(usage(&frame["usage"])?);
        }
        for choice in frame["choices"]
            .as_array()
            .ok_or_else(|| ApiError::bad("后端 SSE 缺少 choices"))?
        {
            if let Some(reason) = choice["finish_reason"].as_str() {
                self.finish_reason = Some(reason.into());
            }
            let delta = &choice["delta"];
            if let Some(text) = delta["content"].as_str().filter(|s| !s.is_empty()) {
                self.push_text(text, &mut events)?;
            }
            if let Some(calls) = delta["tool_calls"].as_array() {
                for call in calls {
                    self.push_tool(call, &mut events)?;
                }
            }
        }
        Ok(events)
    }

    fn push_text(&mut self, text: &str, events: &mut Vec<Bytes>) -> Result<(), ApiError> {
        if self.text_closed {
            return Err(ApiError::bad("后端在工具参数之后返回了交错文本"));
        }
        if self.text_index.is_none() {
            let index = self.next_index;
            self.next_index += 1;
            self.text_index = Some(index);
            events.push(sse::event("content_block_start", &json!({"type":"content_block_start","index":index,"content_block":{"type":"text","text":""}})));
        }
        let index = self.text_index.expect("text index initialized");
        events.push(sse::event("content_block_delta", &json!({"type":"content_block_delta","index":index,"delta":{"type":"text_delta","text":text}})));
        Ok(())
    }

    fn close_text(&mut self, events: &mut Vec<Bytes>) {
        if let Some(index) = self.text_index.filter(|_| !self.text_closed) {
            events.push(sse::event(
                "content_block_stop",
                &json!({"type":"content_block_stop","index":index}),
            ));
            self.text_closed = true;
        }
    }

    fn push_tool(&mut self, call: &Value, events: &mut Vec<Bytes>) -> Result<(), ApiError> {
        let key = call["index"]
            .as_u64()
            .ok_or_else(|| ApiError::bad("后端工具增量缺少 index"))?;
        let tool = self.tools.entry(key).or_default();
        if let Some(id) = call["id"].as_str() {
            tool.id.push_str(id);
        }
        if let Some(name) = call["function"]["name"].as_str() {
            tool.name.push_str(name);
        }
        if let Some(args) = call["function"]["arguments"].as_str() {
            tool.arguments.push_str(args);
        }
        if self.active_tool.is_none() && !tool.id.is_empty() && !tool.name.is_empty() {
            self.close_text(events);
            let index = self.next_index;
            self.next_index += 1;
            self.active_tool = Some((key, index));
            let tool = &self.tools[&key];
            events.push(sse::event("content_block_start", &json!({"type":"content_block_start","index":index,"content_block":{"type":"tool_use","id":tool.id,"name":tool.name,"input":{}}})));
        }
        if let Some((active, index)) = self.active_tool.filter(|(active, _)| *active == key) {
            let tool = self.tools.get_mut(&active).expect("inserted tool");
            let part = &tool.arguments[tool.emitted..];
            if !part.is_empty() {
                events.push(sse::event("content_block_delta", &json!({"type":"content_block_delta","index":index,"delta":{"type":"input_json_delta","partial_json":part}})));
                tool.emitted = tool.arguments.len();
            }
        }
        Ok(())
    }

    pub fn finish(&mut self) -> Result<Vec<Bytes>, ApiError> {
        let reason = stop_reason(
            self.finish_reason
                .as_deref()
                .ok_or_else(|| ApiError::bad("后端流缺少 finish_reason"))?,
        )?;
        let usage = self
            .usage
            .as_ref()
            .ok_or_else(|| ApiError::bad("后端流未提供 usage"))?
            .clone();
        let mut events = Vec::new();
        self.close_text(&mut events);
        if let Some((_, index)) = self.active_tool {
            events.push(sse::event(
                "content_block_stop",
                &json!({"type":"content_block_stop","index":index}),
            ));
        }
        for (key, tool) in &self.tools {
            parse_arguments(&tool.arguments)?;
            if tool.id.is_empty() || tool.name.is_empty() {
                return Err(ApiError::bad("后端工具标识不完整"));
            }
            if self.active_tool.is_some_and(|(active, _)| active == *key) {
                continue;
            }
            let index = self.next_index;
            self.next_index += 1;
            events.push(sse::event("content_block_start", &json!({"type":"content_block_start","index":index,"content_block":{"type":"tool_use","id":tool.id,"name":tool.name,"input":{}}})));
            events.push(sse::event("content_block_delta", &json!({"type":"content_block_delta","index":index,"delta":{"type":"input_json_delta","partial_json":tool.arguments}})));
            events.push(sse::event(
                "content_block_stop",
                &json!({"type":"content_block_stop","index":index}),
            ));
        }
        events.push(sse::event("message_delta", &json!({"type":"message_delta","delta":{"stop_reason":reason,"stop_sequence":null},"usage":usage})));
        events.push(sse::event("message_stop", &json!({"type":"message_stop"})));
        Ok(events)
    }
}
