use crate::{error::ApiError, scheduler::Lease, state::AppState};
use axum::{
    Json,
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};
use std::{sync::Arc, time::Duration};
use tokio::time::Instant;

pub async fn execute(
    state: Arc<AppState>,
    headers: HeaderMap,
    mut body: Value,
) -> Result<Response, ApiError> {
    let key = state.authenticate(&headers)?;
    let model = select_model(&state, &mut body)?;
    let mut lease = Lease::acquire(state.clone(), key, model.id.clone(), "decisions").await?;
    let deadline = Instant::now() + Duration::from_secs(state.config.request_timeout_seconds);
    let upstream = send_request(&state, &model, &body, deadline)
        .await
        .inspect_err(|error| lease.finish(error.status.as_u16()))?;
    let mut payload = read_response(upstream, deadline, state.config.max_response_bytes)
        .await
        .inspect_err(|error| lease.finish(error.status.as_u16()))?;
    record_usage(&payload, &mut lease).inspect_err(|error| lease.finish(error.status.as_u16()))?;
    payload["model"] = json!(model.id);
    lease.finish(StatusCode::OK.as_u16());
    let mut response = Json(payload).into_response();
    response.headers_mut().insert(
        "x-request-id",
        HeaderValue::from_str(&lease.id).map_err(|_| internal())?,
    );
    response.headers_mut().insert(
        "cache-control",
        HeaderValue::from_static("no-store, no-transform"),
    );
    Ok(response)
}

fn select_model(state: &AppState, body: &mut Value) -> Result<crate::config::Model, ApiError> {
    let object = body
        .as_object_mut()
        .ok_or_else(|| ApiError::bad("请求必须是 JSON 对象"))?;
    let name = object
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::bad("model 必须是字符串"))?;
    let model = state.registry.get(name)?;
    if !object.contains_key("input") || !object.get("questions").is_some_and(Value::is_array) {
        return Err(ApiError::bad("Decisions 请求需要 input 和 questions"));
    }
    if object.get("stream").and_then(Value::as_bool) == Some(true) {
        return Err(ApiError::bad("Decisions API 暂不支持流式响应"));
    }
    object.insert("model".into(), json!(model.upstream_model.clone()));
    Ok(model)
}

async fn send_request(
    state: &AppState,
    model: &crate::config::Model,
    body: &Value,
    deadline: Instant,
) -> Result<reqwest::Response, ApiError> {
    let mut request = state
        .client
        .post(format!(
            "{}/decisions",
            model.base_url.trim_end_matches('/')
        ))
        .json(body);
    if let Some(key) = state.upstream_keys.get(&model.id) {
        request = request.bearer_auth(key);
    } else if let Some(name) = &model.api_key_env {
        let key = std::env::var(name).map_err(|_| ApiError::bad("后端密钥环境变量未设置"))?;
        request = request.bearer_auth(key);
    }
    let result = tokio::time::timeout_at(deadline, request.send()).await;
    let upstream = match result {
        Ok(Ok(response)) => response,
        Ok(Err(_)) => return Err(gateway("无法连接 Decisions 推理后端")),
        Err(_) => {
            return Err(ApiError::new(
                StatusCode::GATEWAY_TIMEOUT,
                "推理后端响应超时",
            ));
        }
    };
    if upstream.status().is_success() {
        return Ok(upstream);
    }
    let status = if upstream.status().is_client_error() {
        upstream.status()
    } else {
        StatusCode::BAD_GATEWAY
    };
    Err(ApiError::new(
        status,
        if status.is_client_error() {
            "Valen 后端拒绝请求，请检查 Decisions 字段与媒体格式"
        } else {
            "Valen 推理后端返回错误"
        },
    ))
}

async fn read_response(
    response: reqwest::Response,
    deadline: Instant,
    limit: usize,
) -> Result<Value, ApiError> {
    let bytes = crate::api::proxy::read_bounded(response, deadline, limit).await?;
    serde_json::from_slice(&bytes).map_err(|_| gateway("推理后端返回无效 JSON"))
}

fn record_usage(payload: &Value, lease: &mut Lease) -> Result<(), ApiError> {
    let usage = payload
        .get("usage")
        .and_then(Value::as_object)
        .ok_or_else(|| gateway("Decisions 响应缺少 usage"))?;
    let input = usage.get("input_tokens").and_then(Value::as_u64);
    let output = usage.get("output_tokens").and_then(Value::as_u64);
    match (input, output) {
        (Some(input), Some(output)) => lease.decision_usage(input, output),
        _ => return Err(gateway("Decisions usage 格式无效")),
    }
    if !payload.get("answers").is_some_and(Value::is_array) {
        return Err(gateway("Decisions 响应缺少 answers"));
    }
    Ok(())
}

fn gateway(message: &'static str) -> ApiError {
    ApiError::new(StatusCode::BAD_GATEWAY, message)
}

fn internal() -> ApiError {
    ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "请求状态不可用")
}
