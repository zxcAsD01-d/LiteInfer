use liteinfer::{config::Config, scheduler::Lease, state::AppState};
use serde_json::json;
use std::{
    sync::{Arc, atomic::Ordering},
    time::{Duration, Instant},
};
use tokio::sync::Barrier;

fn state(concurrency: usize, queue_capacity: usize) -> (Arc<AppState>, tempfile::TempDir) {
    let folder = tempfile::tempdir().unwrap();
    let mut config: Config = toml::from_str(include_str!("../examples/config.toml")).unwrap();
    config.concurrency = concurrency;
    config.queue_capacity = queue_capacity;
    config.database = folder.path().join("state.sqlite3").to_str().unwrap().into();
    (AppState::new(config, None, None).unwrap(), folder)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn n_way_capacity_is_reached_and_released_with_zero_queue() {
    let mut rows = Vec::new();
    for n in [1, 2, 3, 8, 32, 256, 1024] {
        let (app, _folder) = state(n, 0);
        let started = Instant::now();
        for _ in 0..4 {
            let barrier = Arc::new(Barrier::new(n + 1));
            let mut tasks = Vec::new();
            for _ in 0..n {
                let (app, barrier) = (app.clone(), barrier.clone());
                tasks.push(tokio::spawn(async move {
                    let mut lease = Lease::acquire(app, "fixture".into(), "model".into(), "openai")
                        .await
                        .unwrap();
                    barrier.wait().await;
                    lease.finish(200);
                }));
            }
            tokio::time::timeout(Duration::from_secs(10), barrier.wait())
                .await
                .unwrap();
            for task in tasks {
                task.await.unwrap();
            }
        }
        assert_eq!(app.metrics.peak_active.load(Ordering::Relaxed), n);
        assert_eq!(app.metrics.active.load(Ordering::Relaxed), 0);
        assert_eq!(app.slots.available_permits(), n);
        assert_eq!(app.admission.available_permits(), n);
        rows.push(json!({"concurrency":n,"requests":4*n,"peak_active":n,
            "elapsed_ms":started.elapsed().as_secs_f64()*1000.0,"metrics":app.metrics.snapshot()}));
    }
    if let Ok(path) = std::env::var("LITEINFER_SCHEDULER_REPORT") {
        std::fs::write(path, serde_json::to_string_pretty(&json!({"label":"synthetic-scheduler",
            "note":"Barrier-held leases measure scheduler capacity, not HTTP or model token throughput.",
            "cases":rows})).unwrap()+"\n").unwrap();
    }
}

#[tokio::test]
async fn fifo_queue_cancellation_and_overload_preserve_permits() {
    let (app, _folder) = state(1, 2);
    let lease = Lease::acquire(app.clone(), "a".into(), "model".into(), "openai")
        .await
        .unwrap();
    let mut tasks = Vec::new();
    for key in ["b", "c"] {
        let app = app.clone();
        tasks.push(tokio::spawn(async move {
            Lease::acquire(app, key.into(), "model".into(), "openai")
                .await
                .unwrap()
        }));
        tokio::task::yield_now().await;
    }
    assert_eq!(app.metrics.queued.load(Ordering::Relaxed), 2);
    assert!(
        Lease::acquire(app.clone(), "d".into(), "model".into(), "openai")
            .await
            .is_err()
    );
    tasks.pop().unwrap().abort();
    tokio::task::yield_now().await;
    drop(lease);
    drop(tasks.pop().unwrap().await.unwrap());
    assert_eq!(app.metrics.active.load(Ordering::Relaxed), 0);
    assert_eq!(app.metrics.queued.load(Ordering::Relaxed), 0);
    assert_eq!(app.admission.available_permits(), 3);
}
