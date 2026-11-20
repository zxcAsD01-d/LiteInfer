use super::{REQUEST_RETENTION, RequestRecord, Store};
use rusqlite::params;
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Default, Serialize)]
pub struct Summary {
    pub requests: u64,
    pub succeeded: u64,
    pub failed: u64,
    pub cancelled: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub avg_duration_ms: Option<f64>,
    pub p50_duration_ms: Option<u64>,
    pub p95_duration_ms: Option<u64>,
    pub success_rate: Option<f64>,
    pub inference: Inference,
}

#[derive(Default, Serialize)]
pub struct Inference {
    pub output_tokens: u64,
    pub speed_samples: u64,
    pub avg_output_tokens_per_second: Option<f64>,
    pub p95_first_output_ms: Option<f64>,
    pub cache_samples: u64,
    pub cache_hits: u64,
    pub cached_prompt_tokens: u64,
    pub cache_hit_rate: Option<f64>,
    pub prompt_token_reuse_rate: Option<f64>,
}

#[derive(Serialize)]
pub struct Bucket {
    pub start: f64,
    pub requests: u64,
    pub errors: u64,
    pub avg_duration_ms: Option<f64>,
    pub output_tokens_per_second: f64,
    pub avg_output_tokens_per_second: Option<f64>,
}

#[derive(Serialize)]
pub struct ModelStats {
    pub model: String,
    pub summary: Summary,
}

#[derive(Serialize)]
pub struct Overview {
    pub start: f64,
    pub end: f64,
    pub window_seconds: u64,
    pub bucket_seconds: u64,
    pub summary: Summary,
    pub series: Vec<Bucket>,
    pub models: Vec<ModelStats>,
    pub retained_records: u64,
    pub oldest_record_at: Option<f64>,
    pub retention_limit: usize,
}

fn summarize(records: &[&RequestRecord]) -> Summary {
    let mut result = Summary::default();
    let mut durations = Vec::with_capacity(records.len());
    for record in records {
        result.requests += 1;
        if (200..300).contains(&record.status) {
            result.succeeded += 1;
        } else if record.status == 499 {
            result.cancelled += 1;
        } else {
            result.failed += 1;
        }
        result.prompt_tokens += record.prompt_tokens;
        result.completion_tokens += record.completion_tokens;
        durations.push(record.duration_ms);
    }
    if !durations.is_empty() {
        durations.sort_unstable();
        let count = durations.len();
        result.avg_duration_ms =
            Some(durations.iter().map(|n| *n as f64).sum::<f64>() / count as f64);
        result.p50_duration_ms = Some(durations[(count * 50).div_ceil(100) - 1]);
        result.p95_duration_ms = Some(durations[(count * 95).div_ceil(100) - 1]);
        result.success_rate = Some(result.succeeded as f64 / result.requests as f64);
    }
    result.inference = inference(records);
    result
}

fn inference(records: &[&RequestRecord]) -> Inference {
    let mut result = Inference::default();
    let (mut speed_sum, mut cache_input) = (0.0, 0u64);
    let mut first_output = Vec::new();
    for row in records.iter().filter(|r| (200..300).contains(&r.status)) {
        result.output_tokens += row.completion_tokens;
        if let Some(speed) = row.output_tokens_per_second {
            speed_sum += speed;
            result.speed_samples += 1;
        }
        if let Some(ms) = row.first_output_ms {
            first_output.push(ms);
        }
        if let Some(tokens) = row.cached_prompt_tokens {
            result.cache_samples += 1;
            result.cache_hits += u64::from(tokens > 0);
            result.cached_prompt_tokens += tokens;
            cache_input += row.prompt_tokens;
        }
    }
    first_output.sort_unstable_by(f64::total_cmp);
    result.p95_first_output_ms = first_output
        .len()
        .checked_sub(1)
        .map(|_| first_output[(first_output.len() * 95).div_ceil(100) - 1]);
    result.avg_output_tokens_per_second = ratio(speed_sum, result.speed_samples);
    result.cache_hit_rate = ratio(result.cache_hits as f64, result.cache_samples);
    result.prompt_token_reuse_rate = ratio(result.cached_prompt_tokens as f64, cache_input);
    result
}

fn ratio(numerator: f64, denominator: u64) -> Option<f64> {
    (denominator > 0).then(|| numerator / denominator as f64)
}

fn series(records: &[RequestRecord], start: f64, step: u64) -> Vec<Bucket> {
    let mut buckets: Vec<Vec<&RequestRecord>> = (0..30).map(|_| Vec::new()).collect();
    for record in records {
        let index = ((record.finished_at.unwrap_or(record.created) - start) / step as f64) as usize;
        if let Some(bucket) = buckets.get_mut(index) {
            bucket.push(record);
        }
    }
    buckets
        .iter()
        .enumerate()
        .map(|(i, rows)| {
            let summary = summarize(rows);
            Bucket {
                start: start + (i as u64 * step) as f64,
                requests: summary.requests,
                errors: summary.failed,
                avg_duration_ms: summary.avg_duration_ms,
                output_tokens_per_second: summary.inference.output_tokens as f64 / step as f64,
                avg_output_tokens_per_second: summary.inference.avg_output_tokens_per_second,
            }
        })
        .collect()
}

impl Store {
    pub fn overview(&self, seconds: u64, end: f64) -> Result<Overview, rusqlite::Error> {
        let start = end - seconds as f64;
        let mut db = self.connect()?;
        let snapshot = db.transaction()?;
        let mut statement = snapshot.prepare(
            "SELECT * FROM requests
            WHERE COALESCE(finished_at, created)>=?1 AND COALESCE(finished_at, created)<?2",
        )?;
        let records = statement
            .query_map(params![start, end], read_record)?
            .collect::<Result<Vec<_>, _>>()?;
        let summary = summarize(&records.iter().collect::<Vec<_>>());
        let mut groups: BTreeMap<String, Vec<&RequestRecord>> = BTreeMap::new();
        for record in &records {
            groups.entry(record.model.clone()).or_default().push(record);
        }
        let models = groups
            .into_iter()
            .map(|(model, rows)| ModelStats {
                model,
                summary: summarize(&rows),
            })
            .collect();
        let (retained_records, oldest_record_at) =
            snapshot.query_row("SELECT COUNT(*),MIN(created) FROM requests", [], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })?;
        Ok(Overview {
            start,
            end,
            window_seconds: seconds,
            bucket_seconds: seconds / 30,
            summary,
            series: series(&records, start, seconds / 30),
            models,
            retained_records,
            oldest_record_at,
            retention_limit: REQUEST_RETENTION,
        })
    }
}

pub(super) fn read_record(row: &rusqlite::Row<'_>) -> Result<RequestRecord, rusqlite::Error> {
    Ok(RequestRecord {
        id: row.get(0)?,
        key_id: row.get(1)?,
        model: row.get(2)?,
        protocol: row.get(3)?,
        status: row.get(4)?,
        prompt_tokens: row.get(5)?,
        completion_tokens: row.get(6)?,
        duration_ms: row.get(7)?,
        created: row.get(8)?,
        finished_at: row.get("finished_at")?,
        cached_prompt_tokens: row.get("cached_prompt_tokens")?,
        first_output_ms: row.get("first_output_ms")?,
        output_tokens_per_second: row.get("output_tokens_per_second")?,
    })
}
