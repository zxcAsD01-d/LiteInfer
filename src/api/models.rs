use crate::{
    config::Model,
    deployment::{Deployment, Source},
    error::ApiError,
    state::AppState,
};
use axum::{
    Json,
    body::Bytes,
    extract::{State, rejection::BytesRejection},
    http::{HeaderMap, Method, StatusCode},
};
use serde_json::{Value, json};
use std::sync::Arc;

pub async fn cache(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    method: Method,
) -> Result<Json<Value>, ApiError> {
    state.admin(&headers)?;
    let alias = state
        .deployment
        .managed
        .try_lock()
        .map_err(|_| ApiError::new(StatusCode::CONFLICT, "受管模型正在启动或停止"))?
        .as_ref()
        .map(|process| process.alias.clone())
        .ok_or_else(|| ApiError::bad("当前没有运行中的受管 MLX 模型"))?;
    let model = state.registry.get(&alias)?;
    let url = cache_url(&model)?;
    let response = state
        .client
        .request(method, url)
        .bearer_auth(&state.deployment.cache_admin_token)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|_| ApiError::new(StatusCode::BAD_GATEWAY, "缓存管理后端不可用"))?;
    if !response.status().is_success() {
        return Err(ApiError::new(
            StatusCode::BAD_GATEWAY,
            "后端拒绝缓存管理请求",
        ));
    }
    Ok(Json(response.json().await.map_err(|_| {
        ApiError::new(StatusCode::BAD_GATEWAY, "缓存统计响应无效")
    })?))
}

fn cache_url(model: &Model) -> Result<reqwest::Url, ApiError> {
    let mut url =
        reqwest::Url::parse(&model.base_url).map_err(|_| ApiError::bad("后端地址无效"))?;
    if url.scheme() != "http" || url.host_str() != Some("127.0.0.1") {
        return Err(ApiError::bad("缓存管理仅支持受管本机后端"));
    }
    url.set_path("/cache/stats");
    url.set_query(None);
    Ok(url)
}

pub async fn status(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    state.admin(&headers)?;
    Deployment::reconcile(&state).await?;
    let mut result = state.deployment.snapshot()?;
    result["models"] = json!(state.registry.list()?);
    result["managed"] = match state.deployment.managed.try_lock() {
        Ok(process) => json!(process.as_ref().map(|p| &p.alias)),
        Err(_) => json!("正在启动或停止"),
    };
    Ok(Json(result))
}

pub async fn runtime(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    state.admin(&headers)?;
    Ok(Json(state.deployment.status().await?))
}

pub async fn setup(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    state.admin(&headers)?;
    Ok(Json(json!(Deployment::start_job(state, None)?)))
}

pub async fn inspect(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Result<Json<Value>, ApiError> {
    state.admin(&headers)?;
    let source: Source = serde_json::from_value(super::parse(body)?)
        .map_err(|_| ApiError::bad("模型源配置格式无效"))?;
    Ok(Json(state.deployment.inspect(source).await?))
}

pub async fn prepare(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Result<Json<Value>, ApiError> {
    state.admin(&headers)?;
    let source = serde_json::from_value(super::parse(body)?)
        .map_err(|_| ApiError::bad("模型源配置格式无效"))?;
    Ok(Json(json!(Deployment::start_job(state, Some(source))?)))
}

pub async fn launch(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Result<Json<Value>, ApiError> {
    state.admin(&headers)?;
    let request =
        serde_json::from_value(super::parse(body)?).map_err(|_| ApiError::bad("启动参数无效"))?;
    Ok(Json(Deployment::launch(state, request).await?))
}

pub async fn stop(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    state.admin(&headers)?;
    Ok(Json(Deployment::stop(state).await?))
}

pub async fn register(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Result<Json<Value>, ApiError> {
    state.admin(&headers)?;
    let model: Model = serde_json::from_value(super::parse(body)?)
        .map_err(|_| ApiError::bad("模型注册配置无效"))?;
    if let Some(name) = &model.api_key_env {
        std::env::var(name).map_err(|_| ApiError::bad("后端密钥环境变量未设置"))?;
    }
    model.validate().map_err(ApiError::bad)?;
    state.registry.install(model.clone(), true)?;
    Ok(Json(json!({"model":model,"status":"applied"})))
}
