use crate::{
    error::ApiError,
    state::AppState,
    store::{Store, hash},
};
use axum::{
    Json,
    body::Bytes,
    extract::{Path, State, rejection::BytesRejection},
    http::{HeaderMap, StatusCode},
};
use serde_json::{Value, json};
use std::sync::Arc;

async fn database<T: Send + 'static>(
    store: Store,
    work: impl FnOnce(Store) -> Result<T, rusqlite::Error> + Send + 'static,
) -> Result<T, ApiError> {
    tokio::task::spawn_blocking(move || work(store))
        .await
        .map_err(|_| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "数据库任务失败"))?
        .map_err(|_| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "数据库操作失败"))
}

pub async fn metrics(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    state.admin(&headers)?;
    Ok(Json(
        json!({"service":"LiteInfer","limits":{"concurrency":state.config.concurrency,"queue":state.config.queue_capacity},"metrics":state.metrics.snapshot()}),
    ))
}

pub async fn requests(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    state.admin(&headers)?;
    let recent = database(state.store.clone(), |store| store.recent()).await?;
    Ok(Json(json!({"data":recent})))
}

pub async fn keys(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    state.admin(&headers)?;
    let keys = database(state.store.clone(), |store| store.keys()).await?;
    Ok(Json(json!({"data":keys})))
}

pub async fn create_key(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Result<Json<Value>, ApiError> {
    state.admin(&headers)?;
    let body = super::parse(body)?;
    let name = body["name"]
        .as_str()
        .filter(|s| !s.trim().is_empty() && s.len() <= 128)
        .ok_or_else(|| ApiError::bad("name 需要 1..128 字节"))?
        .to_owned();
    let key = database(state.store.clone(), move |store| store.create_key(&name)).await?;
    state
        .keys
        .write()
        .map_err(|_| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "鉴权状态不可用"))?
        .insert(hash(&key.key), key.id.clone());
    Ok(Json(
        json!({"id":key.id,"key":key.key,"notice":"密钥仅显示一次"}),
    ))
}

pub async fn revoke_key(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    state.admin(&headers)?;
    let target = id.clone();
    let revoked = database(state.store.clone(), move |store| store.revoke_key(&target)).await?;
    state
        .keys
        .write()
        .map_err(|_| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "鉴权状态不可用"))?
        .retain(|_, value| value != &id);
    Ok(Json(json!({"id":id,"revoked":revoked})))
}
