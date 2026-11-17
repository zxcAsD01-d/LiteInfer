use super::{Deployment, Launch, Managed, internal, validate_alias};
use crate::{config::Model, error::ApiError, state::AppState};
use axum::http::StatusCode;
use serde_json::{Value, json};
use std::{sync::Arc, time::Duration};
use tokio::{
    process::Child,
    time::{sleep, timeout},
};

pub(super) async fn terminate(child: &mut Child) -> Result<(), ApiError> {
    if child.try_wait().map_err(|_| internal())?.is_none() {
        child.kill().await.map_err(|_| internal())?;
    }
    Ok(())
}

impl Deployment {
    async fn context(&self, request: &Launch) -> Result<Option<u64>, ApiError> {
        validate_alias(&request.alias)?;
        if request.port == 0 || request.max_output_tokens == 0 {
            return Err(ApiError::bad("端口或输出预算无效"));
        }
        if request.artifact.len() != 32
            || !request
                .artifact
                .chars()
                .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
        {
            return Err(ApiError::bad("模型产物 ID 无效"));
        }
        let path = self
            .root
            .join("artifacts")
            .join(&request.artifact)
            .join("manifest.json");
        let bytes = tokio::fs::read(path)
            .await
            .map_err(|_| ApiError::bad("模型产物不存在或尚未自检通过"))?;
        let manifest: Value =
            serde_json::from_slice(&bytes).map_err(|_| ApiError::bad("模型产物清单无效"))?;
        let context = manifest["metadata"]["context_length"].as_u64();
        if context.is_some_and(|n| u64::from(request.max_output_tokens) >= n) {
            return Err(ApiError::bad("输出预算必须小于模型上下文长度"));
        }
        Ok(context)
    }

    async fn ready(state: &AppState, child: &mut Child, base: &str) -> Result<String, ApiError> {
        let poll = async {
            loop {
                if child.try_wait().map_err(|_| internal())?.is_some() {
                    return Err(ApiError::new(StatusCode::BAD_GATEWAY, "MLX 后端启动失败"));
                }
                if let Ok(response) = state
                    .client
                    .get(format!("{base}/models"))
                    .timeout(Duration::from_secs(1))
                    .send()
                    .await
                {
                    if response.status().is_success() {
                        let value: Value = response
                            .json()
                            .await
                            .map_err(|_| ApiError::bad("后端模型列表无效"))?;
                        if let Some(id) = value["data"][0]["id"].as_str().filter(|s| !s.is_empty())
                        {
                            return Ok(id.to_owned());
                        }
                    }
                }
                sleep(Duration::from_millis(200)).await;
            }
        };
        timeout(Duration::from_secs(120), poll)
            .await
            .map_err(|_| ApiError::new(StatusCode::GATEWAY_TIMEOUT, "MLX 后端未能在两分钟内就绪"))?
    }

    async fn apply(state: &AppState, model: Model, request: &Launch) -> Result<(), ApiError> {
        let plan = state.deployment.root.join("launch.json");
        let temporary = state.deployment.root.join("launch.tmp");
        tokio::fs::write(
            &temporary,
            serde_json::to_vec(request).map_err(|_| internal())?,
        )
        .await
        .map_err(|_| internal())?;
        tokio::fs::rename(temporary, &plan)
            .await
            .map_err(|_| internal())?;
        if let Err(error) = state.registry.install(model, false) {
            tokio::fs::remove_file(plan).await.map_err(|_| internal())?;
            return Err(error);
        }
        Ok(())
    }

    pub async fn launch(state: Arc<AppState>, request: Launch) -> Result<Value, ApiError> {
        let context_length = state.deployment.context(&request).await?;
        Self::reconcile(&state).await?;
        if request.port == state.config.listen.port() {
            return Err(ApiError::bad("后端端口不能与网关相同"));
        }
        let _permit = state
            .deployment
            .permit
            .clone()
            .try_acquire_owned()
            .map_err(|_| ApiError::new(StatusCode::CONFLICT, "模型管理器忙"))?;
        let mut managed = state.deployment.managed.lock().await;
        if managed.is_some() {
            return Err(ApiError::bad("请先停止当前受管模型后再启动"));
        }
        if state.registry.list()?.iter().any(|m| m.id == request.alias) {
            return Err(ApiError::bad("别名已存在"));
        }
        let (child, model) = Self::backend(&state, &request, context_length).await?;
        Self::apply(&state, model, &request).await?;
        *managed = Some(Managed {
            alias: request.alias.clone(),
            child,
        });
        Ok(json!({"alias":request.alias,"status":"running","artifact":request.artifact}))
    }

    async fn backend(
        state: &AppState,
        request: &Launch,
        context_length: Option<u64>,
    ) -> Result<(Child, Model), ApiError> {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", request.port))
            .await
            .map_err(|_| ApiError::bad("后端端口已被占用"))?;
        drop(listener);
        let base_url = format!("http://127.0.0.1:{}/v1", request.port);
        let mut child = state
            .deployment
            .spawn_plugin(
                "serve",
                json!({"artifact": request.artifact,
            "port": request.port, "concurrency": state.config.concurrency}),
            )
            .await?;
        let upstream_model = Self::ready(state, &mut child, &base_url).await?;
        let model = Model {
            id: request.alias.clone(),
            upstream_model,
            base_url,
            context_length,
            max_output_tokens: request.max_output_tokens,
            api_key_env: None,
            tools: false,
            vision: false,
            cache_isolation_parameter: None,
        };
        Ok((child, model))
    }

    pub async fn reconcile(state: &AppState) -> Result<(), ApiError> {
        let Ok(mut managed) = state.deployment.managed.try_lock() else {
            return Ok(());
        };
        if let Some(process) = managed.as_mut() {
            if process.child.try_wait().map_err(|_| internal())?.is_some() {
                state.registry.remove_runtime(&process.alias)?;
                *managed = None;
            }
        }
        Ok(())
    }
}
