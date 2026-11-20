use crate::{
    config::Config,
    store::{REQUEST_RETENTION, timestamp},
};
use serde_json::{Value, json};
use std::{
    path::Path,
    time::{Duration, Instant},
};
use sysinfo::{Disks, System};

#[cfg(target_os = "macos")]
mod macos;

pub struct Telemetry {
    started: Instant,
    started_at: f64,
    sampled: Option<Instant>,
    sampled_at: f64,
    disk_sampled: Option<Instant>,
    cpu_ready: bool,
    system: System,
    disks: Disks,
    hardware: Value,
    hardware_sampled: Option<Instant>,
}

impl Default for Telemetry {
    fn default() -> Self {
        Self {
            started: Instant::now(),
            started_at: timestamp(),
            sampled: None,
            sampled_at: 0.0,
            disk_sampled: None,
            cpu_ready: false,
            system: System::new(),
            disks: Disks::new(),
            hardware: Value::Null,
            hardware_sampled: None,
        }
    }
}

impl Telemetry {
    fn sample(&mut self) {
        if self
            .hardware_sampled
            .is_none_or(|at| at.elapsed() >= Duration::from_secs(10))
        {
            #[cfg(target_os = "macos")]
            {
                self.hardware = macos::sample();
            }
            self.hardware_sampled = Some(Instant::now());
        }
        if self
            .sampled
            .is_none_or(|at| at.elapsed() >= Duration::from_secs(2))
        {
            self.cpu_ready = self.sampled.is_some();
            self.system.refresh_cpu_usage();
            self.system.refresh_memory();
            self.sampled = Some(Instant::now());
            self.sampled_at = timestamp();
        }
        if self
            .disk_sampled
            .is_none_or(|at| at.elapsed() >= Duration::from_secs(30))
        {
            self.disks.refresh(true);
            self.disk_sampled = Some(Instant::now());
        }
    }

    fn storage(&self, directory: &str) -> Value {
        let path = Path::new(directory);
        let absolute = if path.is_absolute() {
            path.to_path_buf()
        } else {
            match std::env::current_dir() {
                Ok(root) => root.join(path),
                Err(_) => return Value::Null,
            }
        };
        let resolved = absolute.ancestors().find_map(|p| p.canonicalize().ok());
        let disk = resolved.as_ref().and_then(|path| {
            self.disks
                .list()
                .iter()
                .filter(|disk| path.starts_with(disk.mount_point()))
                .max_by_key(|disk| disk.mount_point().components().count())
        });
        disk.map(|disk| {
            json!({"total_bytes":disk.total_space(),
            "available_bytes":disk.available_space(),
            "used_bytes":disk.total_space().saturating_sub(disk.available_space())})
        })
        .unwrap_or(Value::Null)
    }

    pub fn snapshot(&mut self, config: &Config) -> Value {
        self.sample();
        json!({"version":env!("CARGO_PKG_VERSION"), "pid":std::process::id(),
            "os":System::name(), "os_version":System::os_version(),
            "architecture":std::env::consts::ARCH, "started_at":self.started_at,
            "uptime_seconds":self.started.elapsed().as_secs(), "sampled_at":self.sampled_at,
            "cpu":{"usage_percent":self.cpu_ready.then(|| self.system.global_cpu_usage()),
                "cores":self.system.cpus().len(),
                "brand":self.system.cpus().first().map(|cpu| cpu.brand())},
            "memory":{"total_bytes":self.system.total_memory(),
                "used_bytes":self.system.used_memory(), "available_bytes":self.system.available_memory()},
            "disk":self.storage(&config.mlx.directory),
            "hardware":self.hardware,
            "limits":{"concurrency":config.concurrency, "queue_capacity":config.queue_capacity,
                "ingress_capacity":config.ingress_limit().ok(),
                "queue_timeout_seconds":config.queue_timeout_seconds,
                "request_timeout_seconds":config.request_timeout_seconds,
                "max_body_bytes":config.max_body_bytes, "max_response_bytes":config.max_response_bytes,
                "stream_buffer_chunks":config.stream_buffer_chunks,
                "metadata_queue_capacity":config.metadata_queue_capacity},
            "cache_config":config.mlx.cache, "listen":config.listen.to_string(),
            "request_retention":REQUEST_RETENTION})
    }
}
