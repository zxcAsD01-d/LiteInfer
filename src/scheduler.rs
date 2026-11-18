use crate::{
    error::ApiError,
    state::AppState,
    store::{RequestRecord, timestamp},
};
use axum::http::StatusCode;
use std::{
    sync::{Arc, atomic::Ordering},
    time::{Duration, Instant},
};
use tokio::sync::OwnedSemaphorePermit;

struct Waiting {
    state: Arc<AppState>,
}
impl Drop for Waiting {
    fn drop(&mut self) {
        self.state.metrics.queued.fetch_sub(1, Ordering::Relaxed);
    }
}

pub struct Lease {
    state: Arc<AppState>,
    _slot: OwnedSemaphorePermit,
    _admission: OwnedSemaphorePermit,
    pub id: String,
    key_id: String,
    pub model: String,
    protocol: String,
    started: Instant,
    created: f64,
    status: u16,
    prompt: u64,
    completion: u64,
}

impl Lease {
    pub async fn acquire(
        state: Arc<AppState>,
        key: String,
        model: String,
        protocol: &str,
    ) -> Result<Self, ApiError> {
        let admission = state.admission.clone().try_acquire_owned().map_err(|_| {
            state.metrics.rejected.fetch_add(1, Ordering::Relaxed);
            ApiError::new(StatusCode::TOO_MANY_REQUESTS, "等待队列已满，请稍后重试")
        })?;
        let started = Instant::now();
        state.metrics.queued.fetch_add(1, Ordering::Relaxed);
        let waiting = Waiting {
            state: state.clone(),
        };
        let result = tokio::time::timeout(
            Duration::from_secs(state.config.queue_timeout_seconds),
            state.slots.clone().acquire_owned(),
        )
        .await;
        drop(waiting);
        let slot = result
            .map_err(|_| {
                state.metrics.rejected.fetch_add(1, Ordering::Relaxed);
                ApiError::new(StatusCode::TOO_MANY_REQUESTS, "排队超时，请稍后重试")
            })?
            .map_err(|_| ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "服务正在关闭"))?;
        let active = state.metrics.active.fetch_add(1, Ordering::Relaxed) + 1;
        state
            .metrics
            .peak_active
            .fetch_max(active, Ordering::Relaxed);
        Ok(Self {
            state,
            _slot: slot,
            _admission: admission,
            id: format!("req_{}", uuid::Uuid::new_v4().simple()),
            key_id: key,
            model,
            protocol: protocol.into(),
            started,
            created: timestamp(),
            status: 499,
            prompt: 0,
            completion: 0,
        })
    }

    pub fn usage(&mut self, usage: &serde_json::Value) {
        self.prompt = usage["prompt_tokens"].as_u64().unwrap_or(self.prompt);
        self.completion = usage["completion_tokens"]
            .as_u64()
            .unwrap_or(self.completion);
    }
    pub fn decision_usage(&mut self, input_tokens: u64, output_tokens: u64) {
        self.prompt = input_tokens;
        self.completion = output_tokens;
    }
    pub fn finish(&mut self, status: u16) {
        self.status = status;
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        let metrics = &self.state.metrics;
        metrics.active.fetch_sub(1, Ordering::Relaxed);
        match self.status {
            200..=299 => &metrics.completed,
            499 => &metrics.cancelled,
            _ => &metrics.failed,
        }
        .fetch_add(1, Ordering::Relaxed);
        metrics
            .prompt_tokens
            .fetch_add(self.prompt, Ordering::Relaxed);
        metrics
            .completion_tokens
            .fetch_add(self.completion, Ordering::Relaxed);
        let record = RequestRecord {
            id: self.id.clone(),
            key_id: self.key_id.clone(),
            model: self.model.clone(),
            protocol: self.protocol.clone(),
            status: self.status,
            prompt_tokens: self.prompt,
            completion_tokens: self.completion,
            duration_ms: self.started.elapsed().as_millis().min(u64::MAX as u128) as u64,
            created: self.created,
        };
        if self.state.records.try_send(record).is_err() {
            metrics.records_dropped.fetch_add(1, Ordering::Relaxed);
            tracing::error!("请求元数据队列已满或已关闭");
        }
    }
}
