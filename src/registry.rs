use crate::{
    config::{Config, Model},
    error::ApiError,
};
use axum::http::StatusCode;
use std::{collections::HashSet, fs, io::Write, path::PathBuf, sync::RwLock};

struct Models {
    live: Vec<Model>,
    saved: Vec<Model>,
}

pub struct Registry {
    models: RwLock<Models>,
    configured: HashSet<String>,
    path: PathBuf,
}

fn unavailable() -> ApiError {
    ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "模型注册表不可用")
}

impl Registry {
    pub fn new(config: &Config) -> Result<Self, String> {
        let path = PathBuf::from(&config.database).with_extension("models.json");
        let saved: Vec<Model> = match fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|_| "模型注册表损坏")?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(_) => return Err("无法读取模型注册表".into()),
        };
        let configured: HashSet<String> = config.models.iter().map(|m| m.id.clone()).collect();
        let mut ids = configured.clone();
        for model in &saved {
            model.validate()?;
            if !ids.insert(model.id.clone()) {
                return Err("已保存模型与配置中的模型 ID 冲突".into());
            }
        }
        let mut live = config.models.clone();
        live.extend(saved.clone());
        Ok(Self {
            models: RwLock::new(Models { live, saved }),
            configured,
            path,
        })
    }

    pub fn list(&self) -> Result<Vec<Model>, ApiError> {
        Ok(self.models.read().map_err(|_| unavailable())?.live.clone())
    }

    pub fn get(&self, id: &str) -> Result<Model, ApiError> {
        self.models
            .read()
            .map_err(|_| unavailable())?
            .live
            .iter()
            .find(|m| m.id == id)
            .cloned()
            .ok_or_else(|| ApiError::new(StatusCode::NOT_FOUND, "模型不存在，请查询 /v1/models"))
    }

    pub fn install(&self, model: Model, persist: bool) -> Result<(), ApiError> {
        model.validate().map_err(ApiError::bad)?;
        if self.configured.contains(&model.id) {
            return Err(ApiError::bad("此 ID 已由启动配置占用，请使用其他别名"));
        }
        let mut data = self.models.write().map_err(|_| unavailable())?;
        if data.live.iter().any(|m| m.id == model.id) {
            return Err(ApiError::bad("模型 ID 已存在，请先选择其他别名"));
        }
        if persist {
            let mut saved = data.saved.clone();
            saved.push(model.clone());
            self.save(&saved)?;
            data.saved = saved;
        }
        data.live.push(model);
        Ok(())
    }

    pub fn remove_runtime(&self, id: &str) -> Result<(), ApiError> {
        let mut data = self.models.write().map_err(|_| unavailable())?;
        if self.configured.contains(id) || data.saved.iter().any(|m| m.id == id) {
            return Err(ApiError::bad("只能停止由 MLX 管理器启动的模型"));
        }
        data.live.retain(|m| m.id != id);
        Ok(())
    }

    fn save(&self, models: &[Model]) -> Result<(), ApiError> {
        let bytes = serde_json::to_vec_pretty(models).map_err(|_| unavailable())?;
        let temporary = self
            .path
            .with_extension(format!("{}.tmp", uuid::Uuid::new_v4().simple()));
        let mut options = fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&temporary).map_err(|_| unavailable())?;
        let result = file
            .write_all(&bytes)
            .and_then(|_| file.sync_all())
            .and_then(|_| fs::rename(&temporary, &self.path));
        if result.is_err() {
            if fs::remove_file(&temporary).is_err() {
                tracing::error!("模型注册表临时文件清理失败");
            }
            return Err(unavailable());
        }
        Ok(())
    }
}
