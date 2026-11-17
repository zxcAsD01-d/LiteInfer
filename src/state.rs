use crate::{
    config::Config,
    error::ApiError,
    store::{self, RequestRecord, Store},
};
use axum::http::{HeaderMap, StatusCode};
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    sync::{
        Arc, RwLock,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio::sync::{Semaphore, mpsc};

#[derive(Default)]
pub struct Metrics {
    pub active: AtomicUsize,
    pub queued: AtomicUsize,
    pub peak_active: AtomicUsize,
    pub completed: AtomicU64,
    pub failed: AtomicU64,
    pub cancelled: AtomicU64,
    pub rejected: AtomicU64,
    pub prompt_tokens: AtomicU64,
    pub completion_tokens: AtomicU64,
    pub records_dropped: AtomicU64,
}

impl Metrics {
    pub fn snapshot(&self) -> Value {
        json!({"active":self.active.load(Ordering::Relaxed),"queued":self.queued.load(Ordering::Relaxed),
            "peak_active":self.peak_active.load(Ordering::Relaxed),"completed":self.completed.load(Ordering::Relaxed),
            "failed":self.failed.load(Ordering::Relaxed),"cancelled":self.cancelled.load(Ordering::Relaxed),
            "rejected":self.rejected.load(Ordering::Relaxed),"prompt_tokens":self.prompt_tokens.load(Ordering::Relaxed),
            "completion_tokens":self.completion_tokens.load(Ordering::Relaxed),
            "records_dropped":self.records_dropped.load(Ordering::Relaxed)})
    }
}

pub struct AppState {
    pub config: Config,
    pub client: reqwest::Client,
    pub store: Store,
    pub keys: RwLock<HashMap<String, String>>,
    pub admin_hash: Option<String>,
    pub slots: Arc<Semaphore>,
    pub admission: Arc<Semaphore>,
    pub ingress: Arc<Semaphore>,
    pub metrics: Arc<Metrics>,
    pub records: mpsc::Sender<RequestRecord>,
    pub upstream_keys: HashMap<String, String>,
    pub registry: crate::registry::Registry,
    pub deployment: crate::deployment::Deployment,
}

impl AppState {
    pub fn new(
        config: Config,
        static_key: Option<String>,
        admin_key: Option<String>,
    ) -> Result<Arc<Self>, String> {
        config.validate()?;
        let registry = crate::registry::Registry::new(&config)?;
        let deployment = crate::deployment::Deployment::new(config.mlx.clone());
        let store =
            Store::open(std::path::Path::new(&config.database)).map_err(|_| "数据库初始化失败")?;
        let mut keys = store.active_keys().map_err(|_| "读取 Key 失败")?;
        if let Some(key) = static_key {
            if key.len() < 24 {
                return Err("LITEINFER_API_KEY 至少 24 字符".into());
            }
            keys.insert(store::hash(&key), "environment".into());
        }
        if admin_key.as_ref().is_some_and(|k| k.len() < 24) {
            return Err("LITEINFER_ADMIN_KEY 至少 24 字符".into());
        }
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(5))
            .pool_idle_timeout(Duration::from_secs(60))
            .pool_max_idle_per_host(config.concurrency)
            .tcp_nodelay(true)
            .build()
            .map_err(|_| "HTTP 客户端初始化失败")?;
        let mut upstream_keys = HashMap::new();
        for model in &config.models {
            if let Some(name) = &model.api_key_env {
                let key = std::env::var(name).map_err(|_| "后端密钥环境变量未设置")?;
                upstream_keys.insert(model.id.clone(), key);
            }
        }
        let metrics = Arc::new(Metrics::default());
        let records = record_writer(
            store.clone(),
            metrics.clone(),
            config.metadata_queue_capacity,
        )?;
        Ok(Arc::new(Self {
            slots: Arc::new(Semaphore::new(config.concurrency)),
            admission: Arc::new(Semaphore::new(config.admission_capacity()?)),
            ingress: Arc::new(Semaphore::new(config.ingress_limit()?)),
            config,
            client,
            store,
            keys: RwLock::new(keys),
            admin_hash: admin_key.map(|k| store::hash(&k)),
            metrics,
            records,
            upstream_keys,
            registry,
            deployment,
        }))
    }

    pub fn authenticate(&self, headers: &HeaderMap) -> Result<String, ApiError> {
        let bearer = headers
            .get("authorization")
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.strip_prefix("Bearer "));
        let secret = bearer
            .or_else(|| headers.get("x-api-key").and_then(|h| h.to_str().ok()))
            .filter(|s| s.len() <= 512)
            .ok_or_else(|| ApiError::new(StatusCode::UNAUTHORIZED, "需要有效 API Key"))?;
        self.keys
            .read()
            .map_err(|_| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "鉴权状态不可用"))?
            .get(&store::hash(secret))
            .cloned()
            .ok_or_else(|| ApiError::new(StatusCode::UNAUTHORIZED, "API Key 无效或已撤销"))
    }

    pub fn admin(&self, headers: &HeaderMap) -> Result<(), ApiError> {
        let supplied = headers
            .get("x-admin-key")
            .and_then(|h| h.to_str().ok())
            .filter(|s| s.len() <= 512);
        if self
            .admin_hash
            .as_ref()
            .zip(supplied)
            .is_some_and(|(expected, actual)| *expected == store::hash(actual))
        {
            Ok(())
        } else {
            Err(ApiError::new(StatusCode::UNAUTHORIZED, "需要有效管理密钥"))
        }
    }
}

fn record_writer(
    store: Store,
    metrics: Arc<Metrics>,
    capacity: usize,
) -> Result<mpsc::Sender<RequestRecord>, String> {
    let (sender, mut receiver) = mpsc::channel::<RequestRecord>(capacity);
    let mut db = store
        .writer_connection()
        .map_err(|_| "元数据连接初始化失败")?;
    std::thread::Builder::new()
        .name("metadata-writer".into())
        .spawn(move || {
            let mut count = 0;
            while let Some(record) = receiver.blocking_recv() {
                let mut batch = vec![record];
                while batch.len() < 32 {
                    match receiver.try_recv() {
                        Ok(record) => batch.push(record),
                        Err(_) => break,
                    }
                }
                if Store::record_batch(&mut db, &batch).is_err() {
                    metrics
                        .records_dropped
                        .fetch_add(batch.len() as u64, Ordering::Relaxed);
                    tracing::error!("请求元数据写入失败");
                }
                let old = count;
                count += batch.len();
                if old / 100 != count / 100 && store.prune().is_err() {
                    tracing::error!("请求记录清理失败");
                }
            }
        })
        .map_err(|_| "元数据线程启动失败")?;
    Ok(sender)
}
