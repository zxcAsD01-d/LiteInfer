use serde::{Deserialize, Serialize};
use std::{collections::HashSet, net::SocketAddr, path::Path};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub listen: SocketAddr,
    pub concurrency: usize,
    pub queue_capacity: usize,
    pub ingress_capacity: Option<usize>,
    pub stream_buffer_chunks: usize,
    pub metadata_queue_capacity: usize,
    pub queue_timeout_seconds: u64,
    pub request_timeout_seconds: u64,
    pub max_body_bytes: usize,
    pub max_response_bytes: usize,
    pub database: String,
    pub allow_remote: bool,
    pub models: Vec<Model>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub id: String,
    pub upstream_model: String,
    pub base_url: String,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default = "default_output")]
    pub max_output_tokens: u32,
    #[serde(default)]
    pub tools: bool,
    #[serde(default)]
    pub vision: bool,
    #[serde(default)]
    pub cache_isolation_parameter: Option<String>,
}

fn default_output() -> u32 {
    1024
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen: "127.0.0.1:8787".parse().expect("constant address"),
            concurrency: 3,
            queue_capacity: 24,
            ingress_capacity: None,
            stream_buffer_chunks: 8,
            metadata_queue_capacity: 4096,
            queue_timeout_seconds: 60,
            request_timeout_seconds: 180,
            max_body_bytes: 8 * 1024 * 1024,
            max_response_bytes: 16 * 1024 * 1024,
            database: ".liteinfer/state.sqlite3".into(),
            allow_remote: false,
            models: vec![],
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("读取配置失败: {e}"))?;
        let config: Self = toml::from_str(&text).map_err(|e| format!("配置无效: {e}"))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.listen.ip().is_loopback() && !self.allow_remote {
            return Err("非本机监听需要显式设置 allow_remote=true".into());
        }
        self.validate_capacities()?;
        if self.queue_timeout_seconds == 0
            || self.request_timeout_seconds == 0
            || self.max_body_bytes == 0
            || self.max_response_bytes == 0
        {
            return Err("超时和大小预算必须大于 0".into());
        }
        if self.models.is_empty() {
            return Err("至少配置一个模型".into());
        }
        let mut ids = HashSet::new();
        for model in &self.models {
            model.validate()?;
            if !ids.insert(&model.id) {
                return Err("模型 id 重复".into());
            }
        }
        Ok(())
    }

    pub fn admission_capacity(&self) -> Result<usize, String> {
        self.concurrency
            .checked_add(self.queue_capacity)
            .filter(|n| *n <= tokio::sync::Semaphore::MAX_PERMITS)
            .ok_or_else(|| "concurrency + queue_capacity 超出运行时容量".into())
    }

    pub fn ingress_limit(&self) -> Result<usize, String> {
        Ok(self.ingress_capacity.unwrap_or(
            self.admission_capacity()?
                .saturating_add(16)
                .clamp(64, tokio::sync::Semaphore::MAX_PERMITS),
        ))
    }

    fn validate_capacities(&self) -> Result<(), String> {
        self.admission_capacity()?;
        for (name, value) in [
            ("concurrency", self.concurrency),
            ("ingress_capacity", self.ingress_limit()?),
            ("stream_buffer_chunks", self.stream_buffer_chunks),
            ("metadata_queue_capacity", self.metadata_queue_capacity),
        ] {
            if value == 0 || value > tokio::sync::Semaphore::MAX_PERMITS {
                return Err(format!("{name} 必须为正整数，且不超过运行时容量"));
            }
        }
        Ok(())
    }
}

impl Model {
    fn validate(&self) -> Result<(), String> {
        if self.id.trim().is_empty()
            || self.upstream_model.trim().is_empty()
            || self.max_output_tokens == 0
        {
            return Err("模型名称不能为空，输出预算必须大于 0".into());
        }
        let url = reqwest::Url::parse(&self.base_url).map_err(|_| "后端 URL 无效")?;
        if !matches!(url.scheme(), "http" | "https")
            || url.host_str().is_none()
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
        {
            return Err("后端 URL 必须为无凭据、查询参数和片段的 HTTP(S) 地址".into());
        }
        if let Some(name) = &self.api_key_env {
            if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                return Err("api_key_env 必须是环境变量名称".into());
            }
        }
        if let Some(name) = &self.cache_isolation_parameter {
            if !matches!(name.as_str(), "cache_salt" | "prompt_cache_key") {
                return Err("缓存隔离参数仅支持 cache_salt / prompt_cache_key".into());
            }
        }
        Ok(())
    }
}
