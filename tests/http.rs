use axum::{
    Json, Router,
    body::Body,
    http::HeaderValue,
    response::{IntoResponse, Response},
    routing::post,
};
use liteinfer::{
    api,
    config::{Config, Model},
    state::AppState,
};
use reqwest::{Client, StatusCode};
use serde_json::{Value, json};
use std::{
    convert::Infallible,
    sync::{Arc, atomic::Ordering},
    time::Duration,
};

const KEY: &str = "test-key-with-sufficient-entropy-example";
const ADMIN: &str = "test-admin-with-sufficient-entropy-example";

struct Harness {
    url: String,
    client: Client,
    state: Arc<AppState>,
    _folder: tempfile::TempDir,
    tasks: Vec<tokio::task::JoinHandle<()>>,
}
impl Drop for Harness {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
    }
}

async fn start(router: Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (url, task)
}

impl Harness {
    async fn new() -> Self {
        let folder = tempfile::tempdir().unwrap();
        let (backend, task) = start(
            Router::new()
                .route("/v1/chat/completions", post(mock))
                .route("/v1/completions", post(mock)),
        )
        .await;
        let config = Config {
            database: folder.path().join("state.sqlite3").to_str().unwrap().into(),
            queue_capacity: 2,
            request_timeout_seconds: 2,
            max_body_bytes: 4096,
            models: vec![Model {
                id: "alias".into(),
                upstream_model: "actual-model".into(),
                base_url: format!("{backend}/v1"),
                api_key_env: None,
                max_output_tokens: 64,
                tools: true,
                vision: false,
                cache_isolation_parameter: None,
            }],
            ..Config::default()
        };
        let state = AppState::new(config, Some(KEY.into()), Some(ADMIN.into())).unwrap();
        let (url, api) = start(api::router(state.clone())).await;
        Self {
            url,
            client: Client::new(),
            state,
            _folder: folder,
            tasks: vec![task, api],
        }
    }
    fn post(&self, path: &str, body: Value) -> reqwest::RequestBuilder {
        self.client
            .post(format!("{}{path}", self.url))
            .bearer_auth(KEY)
            .json(&body)
    }
    async fn idle(&self) {
        for _ in 0..100 {
            if self.state.metrics.active.load(Ordering::Relaxed) == 0
                && self.state.metrics.queued.load(Ordering::Relaxed) == 0
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("scheduler did not release slots");
    }
}

fn body(mode: &str, stream: bool) -> Value {
    json!({"model":"alias","messages":[{"role":"user","content":mode}],"stream":stream,"max_tokens":12,"stream_options":{"include_usage":true}})
}

async fn mock(Json(body): Json<Value>) -> Response {
    assert_eq!(body["model"], "actual-model");
    let prompt = body["messages"].to_string();
    if prompt.contains("backend-error") {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            "private backend error /private/path secret",
        )
            .into_response();
    }
    if prompt.contains("slow") {
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    let tool = prompt.contains("tool");
    let hang = prompt.contains("hang");
    let missing = prompt.contains("missing-usage");
    let usage = json!({"prompt_tokens":13,"completion_tokens":4,"total_tokens":17});
    let message = if tool {
        json!({"role":"assistant","content":null,"tool_calls":[{"id":"call_test","type":"function","function":{"name":"write_file","arguments":"{\"content\":\"中文\"}"}}]})
    } else {
        json!({"role":"assistant","content":"你好"})
    };
    let finish = if tool { "tool_calls" } else { "stop" };
    if !body["stream"].as_bool().unwrap_or(false) {
        return Json(json!({"id":"upstream","object":"chat.completion","created":0,"model":"actual-model","choices":[{"index":0,"message":message,"text":"你好","finish_reason":finish}],"usage":usage})).into_response();
    }
    let output = async_stream::stream! {
        let deltas=if tool{vec![json!({"tool_calls":[{"index":0,"id":"call_test","type":"function","function":{"name":"write_file","arguments":"{\"content\":"}}]}),json!({"tool_calls":[{"index":0,"function":{"arguments":"\"中文\"}"}}]})]}else{vec![json!({"role":"assistant","content":"你"}),json!({"content":"好"})]};
        for delta in deltas{yield Ok::<_,Infallible>(liteinfer::sse::data(&json!({"model":"actual-model","object":"chat.completion.chunk","id":"upstream","created":0,"choices":[{"index":0,"delta":delta,"finish_reason":null}]})));}
        if hang{tokio::time::sleep(Duration::from_secs(30)).await;}
        yield Ok(liteinfer::sse::data(&json!({"model":"actual-model","choices":[{"index":0,"delta":{},"finish_reason":finish}]})));
        if !missing{yield Ok(liteinfer::sse::data(&json!({"model":"actual-model","choices":[],"usage":usage})));}
        yield Ok(bytes::Bytes::from_static(b"data: [DONE]\n\n"));
    };
    let mut response = Response::new(Body::from_stream(output));
    response.headers_mut().insert(
        "content-type",
        HeaderValue::from_static("text/event-stream"),
    );
    response
}

#[tokio::test]
async fn openai_json_stream_and_completions_roundtrip() {
    let app = Harness::new().await;
    let json: Value = app
        .post("/v1/chat/completions", body("hello", false))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(json["model"], "alias");
    assert_eq!(json["usage"]["total_tokens"], 17);
    let response = app
        .post("/v1/chat/completions", body("hello", true))
        .send()
        .await
        .unwrap();
    assert_eq!(response.headers()["x-usage-source"], "upstream");
    let stream = response.text().await.unwrap();
    assert!(stream.contains("[DONE]"));
    assert!(stream.contains("\"model\":\"alias\""));
    assert!(stream.contains("\"total_tokens\":17"));
    assert!(!stream.contains("actual-model"));
    let response = app
        .post(
            "/v1/completions",
            json!({"model":"alias","prompt":"hello","max_tokens":8}),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    app.idle().await;
}

#[tokio::test]
async fn anthropic_json_stream_and_tool_use_roundtrip() {
    let app = Harness::new().await;
    for tool in [false, true] {
        for stream in [false, true] {
            let body = json!({"model":"alias","max_tokens":10,"messages":[{"role":"user","content":if tool{"tool"}else{"hello"}}],"stream":stream});
            let response = app
                .client
                .post(format!("{}/v1/messages", app.url))
                .header("x-api-key", KEY)
                .header("anthropic-version", "2023-06-01")
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 200);
            let text = response.text().await.unwrap();
            if stream {
                assert!(text.contains("event: message_stop"));
                assert!(text.contains("\"input_tokens\":13"));
                assert!(text.contains(if tool {
                    "input_json_delta"
                } else {
                    "text_delta"
                }));
            } else {
                let json: Value = serde_json::from_str(&text).unwrap();
                assert_eq!(json["usage"]["output_tokens"], 4);
                assert_eq!(
                    json["stop_reason"],
                    if tool { "tool_use" } else { "end_turn" }
                );
            }
        }
    }
    app.idle().await;
}

#[tokio::test]
async fn errors_have_protocol_shapes_and_do_not_expose_upstream_details() {
    let app = Harness::new().await;
    let response = app
        .client
        .post(format!("{}/v1/messages", app.url))
        .json(&json!({}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 401);
    assert_eq!(response.json::<Value>().await.unwrap()["type"], "error");
    let response = app
        .post("/v1/chat/completions", body("backend-error", false))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let text = response.text().await.unwrap();
    assert!(!text.contains("private"));
    let mut request = body("hello", false);
    request["model"] = json!("missing");
    assert_eq!(
        app.post("/v1/chat/completions", request)
            .send()
            .await
            .unwrap()
            .status(),
        404
    );
    let mut request = body("hello", false);
    request["max_tokens"] = json!(0);
    assert_eq!(
        app.post("/v1/chat/completions", request)
            .send()
            .await
            .unwrap()
            .status(),
        400
    );
    let response = app
        .client
        .post(format!("{}/v1/messages", app.url))
        .header("x-api-key", KEY)
        .body("broken")
        .send()
        .await
        .unwrap();
    assert_eq!(response.json::<Value>().await.unwrap()["type"], "error");
}

#[tokio::test]
async fn missing_usage_is_an_inband_error_and_never_fake_success() {
    let app = Harness::new().await;
    let response = app
        .post("/v1/chat/completions", body("missing-usage", true))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(response.contains("\"error\""));
    app.idle().await;
    assert_eq!(app.state.metrics.failed.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn overload_is_bounded_at_three_active_and_two_waiting() {
    let app = Harness::new().await;
    let mut tasks = Vec::new();
    for _ in 0..20 {
        let request = app.post("/v1/chat/completions", body("slow", false));
        tasks.push(tokio::spawn(async move {
            request.send().await.unwrap().status()
        }));
    }
    let mut rejected = 0;
    let mut accepted = 0;
    for task in tasks {
        match task.await.unwrap() {
            StatusCode::OK => accepted += 1,
            StatusCode::TOO_MANY_REQUESTS => rejected += 1,
            code => panic!("unexpected {code}"),
        }
    }
    assert!(rejected > 0 && accepted >= 3);
    app.idle().await;
    assert_eq!(app.state.metrics.peak_active.load(Ordering::Relaxed), 3);
}

#[tokio::test]
async fn cancelled_stream_releases_slot_and_upstream() {
    let app = Harness::new().await;
    let mut response = app
        .post("/v1/chat/completions", body("hang", true))
        .send()
        .await
        .unwrap();
    assert!(response.chunk().await.unwrap().is_some());
    drop(response);
    app.idle().await;
    assert_eq!(app.state.metrics.cancelled.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn admin_keys_are_created_revoked_and_hidden_from_api_keys() {
    let app = Harness::new().await;
    assert_eq!(
        app.client
            .get(format!("{}/admin/metrics", app.url))
            .bearer_auth(KEY)
            .send()
            .await
            .unwrap()
            .status(),
        401
    );
    let response = app
        .client
        .post(format!("{}/admin/keys", app.url))
        .header("x-admin-key", ADMIN)
        .json(&json!({"name":"test"}))
        .send()
        .await
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();
    let key = response["key"].as_str().unwrap();
    let id = response["id"].as_str().unwrap();
    assert_eq!(
        app.client
            .get(format!("{}/v1/models", app.url))
            .bearer_auth(key)
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    let listing = app
        .client
        .get(format!("{}/admin/keys", app.url))
        .header("x-admin-key", ADMIN)
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(!listing.contains(key));
    assert_eq!(
        app.client
            .delete(format!("{}/admin/keys/{id}", app.url))
            .header("x-admin-key", ADMIN)
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    assert_eq!(
        app.client
            .get(format!("{}/v1/models", app.url))
            .bearer_auth(key)
            .send()
            .await
            .unwrap()
            .status(),
        401
    );
}

#[tokio::test]
async fn large_body_is_rejected_and_no_count_estimate_is_fabricated() {
    let app = Harness::new().await;
    assert_eq!(
        app.post("/v1/chat/completions", body(&"a".repeat(5000), false))
            .send()
            .await
            .unwrap()
            .status(),
        413
    );
    let response = app
        .post("/v1/messages/count_tokens", json!({}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 501);
    assert_eq!(response.json::<Value>().await.unwrap()["type"], "error");
}

#[tokio::test]
async fn latest_sdk_version_and_stream_timeout_are_handled() {
    let app = Harness::new().await;
    let request =
        json!({"model":"alias","max_tokens":8,"messages":[{"role":"user","content":"hello"}]});
    let response = app
        .client
        .post(format!("{}/v1/messages", app.url))
        .header("x-api-key", KEY)
        .header("anthropic-version", "2026-10-21")
        .json(&request)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let response = app
        .post("/v1/chat/completions", body("hang", true))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(response.contains("超时"));
    app.idle().await;
    assert_eq!(app.state.metrics.failed.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn null_options_and_zero_argument_tools_are_compatible() {
    let app = Harness::new().await;
    let mut request = body("hello", false);
    for key in ["stream", "tools", "stream_options", "n"] {
        request[key] = Value::Null;
    }
    assert_eq!(
        app.post("/v1/chat/completions", request)
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    let mut request = body("hello", false);
    request["tools"] = json!([{"type":"function","function":{"name":"clock"}}]);
    assert_eq!(
        app.post("/v1/chat/completions", request)
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    let mut request = body("hello", true);
    request["stream_options"] = json!({"include_usage":"true"});
    assert_eq!(
        app.post("/v1/chat/completions", request)
            .send()
            .await
            .unwrap()
            .status(),
        400
    );
}

#[tokio::test]
async fn metadata_is_persisted_without_request_content_and_assets_use_strict_csp() {
    let app = Harness::new().await;
    app.post("/v1/chat/completions", body("a-private-prompt", false))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    app.idle().await;
    let mut rows = vec![];
    for _ in 0..100 {
        rows = app.state.store.recent().unwrap();
        if !rows.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].completion_tokens, 4);
    let serialized = serde_json::to_string(&rows).unwrap();
    assert!(!serialized.contains("a-private-prompt"));
    for asset in ["/", "/assets/console.js", "/assets/console.css"] {
        let response = app
            .client
            .get(format!("{}{asset}", app.url))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        assert!(
            !response.headers()["content-security-policy"]
                .to_str()
                .unwrap()
                .contains("unsafe-inline")
        );
    }
}
