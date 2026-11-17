use super::{Deployment, internal};
use crate::error::ApiError;
use axum::http::StatusCode;
use serde_json::{Value, json};
use std::{path::PathBuf, process::Stdio, time::Duration};
use tokio::{
    fs,
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, Command},
    time::timeout,
};

async fn publish(path: &std::path::Path, source: &str) -> Result<(), ApiError> {
    let temporary = path.with_extension(format!("{}.tmp", uuid::Uuid::new_v4().simple()));
    fs::write(&temporary, source)
        .await
        .map_err(|_| internal())?;
    if fs::rename(&temporary, path).await.is_err() {
        if fs::remove_file(temporary).await.is_err() {
            tracing::error!("插件临时文件清理失败");
        }
        return Err(internal());
    }
    Ok(())
}

impl Deployment {
    async fn script(&self) -> Result<PathBuf, ApiError> {
        let directory = self.root.join("plugin");
        fs::create_dir_all(&directory)
            .await
            .map_err(|_| internal())?;
        publish(
            &directory.join("model_config.py"),
            include_str!("../../plugins/mlx/model_config.py"),
        )
        .await?;
        let path = directory.join("manager.py");
        publish(&path, include_str!("../../plugins/mlx/manager.py")).await?;
        Ok(path)
    }

    pub(super) async fn spawn_plugin(
        &self,
        action: &str,
        mut body: Value,
    ) -> Result<Child, ApiError> {
        let script = self.script().await?;
        let runtime = self.root.join("runtime/bin/python");
        let python = if action != "setup" && runtime.exists() {
            runtime
        } else {
            PathBuf::from(&self.config.python)
        };
        body["action"] = json!(action);
        body["root"] = json!(self.root.canonicalize().map_err(|_| internal())?);
        let mut child = Command::new(python)
            .arg("-u")
            .arg(script)
            .stdin(Stdio::piped())
            .stdout(if action == "serve" {
                Stdio::null()
            } else {
                Stdio::piped()
            })
            .stderr(Stdio::null())
            .env("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            .env("TOKENIZERS_PARALLELISM", "false")
            .kill_on_drop(true)
            .spawn()
            .map_err(|_| {
                ApiError::new(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Python 无法启动，请检查 mlx.python（需 Python 3.11+）",
                )
            })?;
        let mut stdin = child.stdin.take().ok_or_else(internal)?;
        stdin
            .write_all(&serde_json::to_vec(&body).map_err(|_| internal())?)
            .await
            .map_err(|_| internal())?;
        drop(stdin);
        Ok(child)
    }

    pub(super) async fn plugin(
        &self,
        action: &str,
        body: Value,
        job: Option<&str>,
    ) -> Result<Value, ApiError> {
        let mut child = self.spawn_plugin(action, body).await?;
        let stdout = child.stdout.take().ok_or_else(internal)?;
        let work = async {
            let mut lines = BufReader::new(stdout).lines();
            let mut result = None;
            while let Some(line) = lines.next_line().await.map_err(|_| internal())? {
                let event: Value = serde_json::from_str(&line)
                    .map_err(|_| ApiError::new(StatusCode::BAD_GATEWAY, "MLX 插件协议错误"))?;
                if let Some(stage) = event["stage"].as_str() {
                    if let Some(id) = job {
                        self.update(id, stage, None, None)?;
                    }
                    if stage == "completed" {
                        result = Some(event["result"].clone());
                    }
                    if stage == "failed" {
                        return Err(plugin_error(&event));
                    }
                }
            }
            if !child.wait().await.map_err(|_| internal())?.success() {
                return Err(plugin_error(&json!({})));
            }
            result.ok_or_else(|| plugin_error(&json!({})))
        };
        timeout(Duration::from_secs(self.config.task_timeout_seconds), work)
            .await
            .map_err(|_| ApiError::new(StatusCode::GATEWAY_TIMEOUT, "模型任务超时，已终止子进程"))?
    }
}

fn plugin_error(event: &Value) -> ApiError {
    let code = event["code"].as_str().unwrap_or("plugin_process_failed");
    let code = if code.len() <= 96 && code.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        code
    } else {
        "plugin_process_failed"
    };
    ApiError::new(
        StatusCode::BAD_GATEWAY,
        format!("MLX 插件失败：{code}；请检查下载源、依赖、架构和本机内存"),
    )
}
