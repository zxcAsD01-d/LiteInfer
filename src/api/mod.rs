mod admin;
pub mod anthropic;
mod decisions;
pub mod models;
pub mod proxy;

use crate::{error::ApiError, state::AppState};
use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State, rejection::BytesRejection},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{delete, get, post},
};
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::sync::Arc;

pub fn router(state: Arc<AppState>) -> Router {
    let maximum = state.config.max_body_bytes;
    Router::new()
        .route("/",get(|| async {Html(include_str!("../../assets/console.html"))}))
        .route("/assets/console.js", get(||async {([("content-type","text/javascript; charset=utf-8")],include_str!("../../assets/console.js"))}))
        .route("/assets/console.css", get(||async {([("content-type","text/css; charset=utf-8")],include_str!("../../assets/console.css"))}))
        .route("/assets/model-config.js", get(||async {([("content-type","text/javascript; charset=utf-8")],include_str!("../../assets/model-config.js"))}))
        .route("/assets/models.js", get(||async {([("content-type","text/javascript; charset=utf-8")],include_str!("../../assets/models.js"))}))
        .route("/health",get(|| async {Json(json!({"status":"ok","service":"LiteInfer","version":env!("CARGO_PKG_VERSION")}))}))
        .route("/v1/models",get(models))
        .route("/v1/chat/completions",post(chat))
        .route("/v1/completions",post(completions))
        .route("/v1/decisions",post(decisions))
        .route("/v1/messages",post(messages))
        .route("/v1/messages/count_tokens",post(count_tokens))
        .route("/admin/metrics",get(admin::metrics))
        .route("/admin/requests",get(admin::requests))
        .route("/admin/keys",get(admin::keys).post(admin::create_key))
        .route("/admin/keys/{id}",delete(admin::revoke_key))
        .route("/admin/models", get(models::status).post(models::register))
        .route("/admin/mlx/runtime", get(models::runtime))
        .route("/admin/mlx/setup", post(models::setup))
        .route("/admin/mlx/inspect", post(models::inspect))
        .route("/admin/mlx/prepare", post(models::prepare))
        .route("/admin/mlx/launch", post(models::launch))
        .route("/admin/mlx/stop", post(models::stop))
        .fallback(|| async { ApiError::new(StatusCode::NOT_FOUND,"接口不存在") })
        .layer(DefaultBodyLimit::max(maximum))
        .layer(middleware::from_fn_with_state(state.clone(), response_headers))
        .with_state(state)
}

async fn response_headers(
    State(state): State<Arc<AppState>>,
    request: axum::extract::Request,
    next: Next,
) -> Response {
    let anthropic = request.uri().path().starts_with("/v1/messages");
    let permit = match state.ingress.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            state
                .metrics
                .rejected
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let error = ApiError::new(StatusCode::TOO_MANY_REQUESTS, "HTTP 请求连接已满");
            return if anthropic { error.anthropic() } else { error }.into_response();
        }
    };
    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    headers.insert(
        "x-content-type-options",
        "nosniff".parse().expect("constant"),
    );
    headers.insert("referrer-policy", "no-referrer".parse().expect("constant"));
    headers.insert("content-security-policy","default-src 'self'; img-src 'self' data:; script-src 'self'; style-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'".parse().expect("constant"));
    let (parts, body) = response.into_parts();
    let stream = body.into_data_stream().map(move |item| {
        let _ = &permit;
        item
    });
    Response::from_parts(parts, axum::body::Body::from_stream(stream))
}

pub(super) fn parse(bytes: Result<Bytes, BytesRejection>) -> Result<Value, ApiError> {
    let bytes =
        bytes.map_err(|error| ApiError::new(error.status(), "请求体无法读取或超过大小预算"))?;
    serde_json::from_slice(&bytes).map_err(|_| ApiError::bad("无效 JSON 请求体"))
}

async fn models(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    state.authenticate(&headers)?;
    crate::deployment::Deployment::reconcile(&state).await?;
    Ok(Json(
        json!({"object":"list","data":state.registry.list()?.iter().map(|m|json!({"id":m.id,"object":"model","created":0,"owned_by":"local","context_length":m.context_length})).collect::<Vec<_>>()}),
    ))
}

async fn chat(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Result<Response, ApiError> {
    proxy::execute(state, headers, parse(body)?, false, false).await
}

async fn completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Result<Response, ApiError> {
    proxy::execute(state, headers, parse(body)?, false, true).await
}

async fn decisions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Result<Response, ApiError> {
    decisions::execute(state, headers, parse(body)?).await
}

async fn messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Result<Response, ApiError> {
    async {
        state.authenticate(&headers)?;
        if headers.get("anthropic-beta").is_some() {
            return Err(ApiError::bad("暂不支持 Anthropic beta 扩展"));
        }
        if headers
            .get("anthropic-version")
            .is_some_and(|v| v != "2023-06-01" && v != "2026-10-21")
        {
            return Err(ApiError::bad(
                "支持的 anthropic-version: 2023-06-01 / 2026-10-21",
            ));
        }
        let body = anthropic::to_openai(&parse(body)?)?;
        proxy::execute(state, headers, body, true, false).await
    }
    .await
    .map_err(ApiError::anthropic)
}

async fn count_tokens(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let error = match state.authenticate(&headers) {
        Err(error) => error,
        Ok(_) => ApiError::new(
            StatusCode::NOT_IMPLEMENTED,
            "通用后端缺少统一的预分词接口；请使用实际响应 usage，不提供伪造计数",
        ),
    };
    error.anthropic().into_response()
}
