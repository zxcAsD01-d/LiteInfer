mod jobs;
mod lifecycle;
mod process;
pub use jobs::Launch;

use crate::{config::MlxConfig, error::ApiError};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};
use tokio::{
    process::Child,
    sync::{Mutex as AsyncMutex, Semaphore},
};

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub source: String,
    pub repository: String,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub token_env: Option<String>,
    #[serde(default = "default_alias")]
    pub alias: String,
    #[serde(default = "default_precision")]
    pub precision: String,
    #[serde(default = "default_group")]
    pub group_size: u32,
}
fn default_alias() -> String {
    "qwen3.5-0.8b".into()
}
fn default_precision() -> String {
    "q4".into()
}
fn default_group() -> u32 {
    64
}

impl Source {
    pub fn validate(&self) -> Result<(), ApiError> {
        if !matches!(self.source.as_str(), "huggingface" | "modelscope" | "local") {
            return Err(ApiError::bad(
                "下载源仅支持 huggingface / modelscope / local",
            ));
        }
        if self.repository.trim().is_empty() || self.repository.len() > 1024 {
            return Err(ApiError::bad("请填写仓库 ID 或本地目录"));
        }
        if self.source != "local"
            && (self.repository.split('/').count() != 2
                || self.repository.split('/').any(|s| {
                    s.is_empty()
                        || !s.starts_with(|c: char| c.is_ascii_alphanumeric())
                        || !s
                            .chars()
                            .all(|c| c.is_ascii_alphanumeric() || "_.-".contains(c))
                }))
        {
            return Err(ApiError::bad("仓库格式需要 owner/name"));
        }
        validate_alias(&self.alias)?;
        if !matches!(self.precision.as_str(), "q4" | "q8" | "float16" | "keep")
            || !matches!(self.group_size, 32 | 64 | 128)
        {
            return Err(ApiError::bad("精度或量化分组无效"));
        }
        if let Some(endpoint) = self.endpoint.as_ref().filter(|s| !s.is_empty()) {
            let url = reqwest::Url::parse(endpoint).map_err(|_| ApiError::bad("镜像地址无效"))?;
            let local = matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"));
            if url.host_str().is_none()
                || !url.username().is_empty()
                || url.password().is_some()
                || url.query().is_some()
                || url.fragment().is_some()
                || url.path() != "/"
                || !(url.scheme() == "https" || (url.scheme() == "http" && local))
            {
                return Err(ApiError::bad("镜像需为 HTTPS 根地址；仅本机允许 HTTP"));
            }
        }
        if self.token_env.as_ref().is_some_and(|s| {
            s.is_empty()
                || !s.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
                || !s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        }) {
            return Err(ApiError::bad("token_env 应为环境变量名"));
        }
        if self
            .revision
            .as_ref()
            .is_some_and(|s| s.len() > 256 || s.chars().any(char::is_control))
        {
            return Err(ApiError::bad("revision 无效"));
        }
        Ok(())
    }
}

pub fn validate_alias(alias: &str) -> Result<(), ApiError> {
    if alias.is_empty()
        || alias.len() > 128
        || !alias
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "_.-".contains(c))
    {
        return Err(ApiError::bad(
            "别名需为 1..128 个字母、数字、点、下划线或连字符",
        ));
    }
    Ok(())
}

#[derive(Clone, Serialize)]
pub struct Job {
    pub id: String,
    pub action: String,
    pub status: String,
    pub stage: String,
    pub result: Option<Value>,
    pub error: Option<String>,
}

pub struct Managed {
    pub alias: String,
    pub child: Child,
}

pub struct Deployment {
    pub config: MlxConfig,
    pub root: PathBuf,
    pub jobs: Mutex<Vec<Job>>,
    pub permit: Arc<Semaphore>,
    pub managed: AsyncMutex<Option<Managed>>,
}

pub fn internal() -> ApiError {
    ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "模型管理状态不可用")
}

impl Deployment {
    pub fn new(config: MlxConfig) -> Self {
        Self {
            root: PathBuf::from(&config.directory),
            config,
            jobs: Mutex::new(Vec::new()),
            permit: Arc::new(Semaphore::new(1)),
            managed: AsyncMutex::new(None),
        }
    }

    pub fn snapshot(&self) -> Result<Value, ApiError> {
        Ok(json!({"jobs": self.jobs.lock().map_err(|_| internal())?.clone()}))
    }

    pub async fn status(&self) -> Result<Value, ApiError> {
        self.plugin("status", json!({}), None).await
    }

    pub async fn inspect(&self, source: Source) -> Result<Value, ApiError> {
        source.validate()?;
        let _permit = self
            .permit
            .clone()
            .try_acquire_owned()
            .map_err(|_| ApiError::new(StatusCode::CONFLICT, "另一个模型任务正在执行"))?;
        self.plugin(
            "inspect",
            serde_json::to_value(source).map_err(|_| internal())?,
            None,
        )
        .await
    }

    pub async fn close(&self) -> Result<(), ApiError> {
        let mut managed = self.managed.lock().await;
        if let Some(process) = managed.as_mut() {
            lifecycle::terminate(&mut process.child).await?;
        }
        *managed = None;
        Ok(())
    }
}
