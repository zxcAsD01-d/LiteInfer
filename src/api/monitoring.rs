use crate::{error::ApiError, state::AppState, store::timestamp};
use axum::{
    Json,
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
};
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;

#[derive(Default, Deserialize)]
pub struct Window {
    pub window: Option<String>,
}

pub async fn overview(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(query): Query<Window>,
) -> Result<Json<crate::store::analytics::Overview>, ApiError> {
    state.admin(&headers)?;
    let seconds = match query.window.as_deref().unwrap_or("1h") {
        "15m" => 900,
        "1h" => 3600,
        "24h" => 86400,
        _ => return Err(ApiError::bad("统计时段仅支持 15m / 1h / 24h")),
    };
    let store = state.store.clone();
    let result = tokio::task::spawn_blocking(move || store.overview(seconds, timestamp()))
        .await
        .map_err(|_| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "统计任务失败"))?
        .map_err(|_| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "读取运行统计失败"))?;
    Ok(Json(result))
}

pub async fn system(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    state.admin(&headers)?;
    let result = tokio::task::spawn_blocking(move || {
        state
            .telemetry
            .lock()
            .map(|mut monitor| monitor.snapshot(&state.config))
            .map_err(|_| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "系统采集状态不可用"))
    })
    .await
    .map_err(|_| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "系统采集任务失败"))??;
    Ok(Json(result))
}
