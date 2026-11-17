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
    http::HeaderMap,
};
use serde_json::{Value, json};
use std::sync::Arc;

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
