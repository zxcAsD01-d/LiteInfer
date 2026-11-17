use super::{Deployment, Job, Source, internal};
use crate::{error::ApiError, state::AppState};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Launch {
    pub artifact: String,
    pub alias: String,
    pub port: u16,
    pub max_output_tokens: u32,
}

impl Deployment {
    pub(super) fn update(
        &self,
        id: &str,
        stage: &str,
        result: Option<Value>,
        error: Option<String>,
    ) -> Result<(), ApiError> {
        let mut jobs = self.jobs.lock().map_err(|_| internal())?;
        let job = jobs.iter_mut().find(|j| j.id == id).ok_or_else(internal)?;
        job.stage = stage.into();
        if let Some(result) = result {
            job.status = "completed".into();
            job.result = Some(result);
        }
        if let Some(error) = error {
            job.status = "failed".into();
            job.error = Some(error);
        }
        Ok(())
    }

    pub fn start_job(state: Arc<AppState>, source: Option<Source>) -> Result<Job, ApiError> {
        if let Some(source) = &source {
            source.validate()?;
        }
        let permit = state
            .deployment
            .permit
            .clone()
            .try_acquire_owned()
            .map_err(|_| ApiError::new(StatusCode::CONFLICT, "已有下载、转换或启动任务"))?;
        let action = if source.is_some() { "prepare" } else { "setup" };
        let job = Job {
            id: uuid::Uuid::new_v4().simple().to_string(),
            action: action.into(),
            status: "running".into(),
            stage: "queued".into(),
            result: None,
            error: None,
        };
        {
            let mut jobs = state.deployment.jobs.lock().map_err(|_| internal())?;
            if jobs.len() >= 32 {
                jobs.remove(0);
            }
            jobs.push(job.clone());
        }
        let id = job.id.clone();
        tokio::spawn(async move {
            let _permit = permit;
            let body = source.map_or_else(|| Ok(json!({})), serde_json::to_value);
            let result = match body {
                Ok(body) => state.deployment.plugin(action, body, Some(&id)).await,
                Err(_) => Err(internal()),
            };
            let update = match result {
                Ok(value) => state.deployment.update(&id, "completed", Some(value), None),
                Err(error) => state
                    .deployment
                    .update(&id, "failed", None, Some(error.message)),
            };
            if update.is_err() {
                tracing::error!("模型任务状态更新失败");
            }
        });
        Ok(job)
    }

    pub async fn stop(state: Arc<AppState>) -> Result<Value, ApiError> {
        let _permit = state
            .deployment
            .permit
            .clone()
            .try_acquire_owned()
            .map_err(|_| ApiError::new(StatusCode::CONFLICT, "模型管理器忙"))?;
        let mut managed = state.deployment.managed.lock().await;
        if let Some(process) = managed.as_mut() {
            super::lifecycle::terminate(&mut process.child).await?;
            state.registry.remove_runtime(&process.alias)?;
        }
        *managed = None;
        let plan = state.deployment.root.join("launch.json");
        match tokio::fs::remove_file(plan).await {
            Ok(()) => (),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
            Err(_) => return Err(internal()),
        }
        Ok(json!({"status":"stopped"}))
    }

    pub async fn restore(state: Arc<AppState>) -> Result<(), ApiError> {
        let path = state.deployment.root.join("launch.json");
        let bytes = match tokio::fs::read(path).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(_) => return Err(internal()),
        };
        let request = serde_json::from_slice(&bytes)
            .map_err(|_| ApiError::bad("已保存的 MLX 启动配置损坏"))?;
        Self::launch(state, request).await?;
        Ok(())
    }
}
