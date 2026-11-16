//! Synthetic transport benchmark only. Never use this backend for inference.
use axum::{
    Json, Router,
    body::Body,
    http::HeaderValue,
    response::{IntoResponse, Response},
    routing::post,
};
use serde_json::{Value, json};
use std::{convert::Infallible, time::Duration};

async fn complete(Json(body): Json<Value>) -> Response {
    let delay = std::env::var("MOCK_DELAY_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(5);
    tokio::time::sleep(Duration::from_millis(delay)).await;
    let completion = body.get("prompt").is_some();
    let base = json!({"id":"mock-synthetic","object":if completion{"text_completion"}else{"chat.completion"},"created":0,"model":"mock"});
    let usage = json!({"prompt_tokens":10,"completion_tokens":8,"total_tokens":18});
    if !body["stream"].as_bool().unwrap_or(false) {
        let mut output = base;
        output["choices"] = json!([{"index":0,"message":{"role":"assistant","content":"mock response"},"text":"mock response","finish_reason":"stop"}]);
        output["usage"] = usage;
        return Json(output).into_response();
    }
    let stream = async_stream::stream! {
        for text in ["mock ","response"]{
            let mut frame=base.clone();frame["object"]=json!("chat.completion.chunk");
            frame["choices"]=json!([{"index":0,"delta":{"content":text},"text":text,"finish_reason":null}]);
            yield Ok::<_,Infallible>(liteinfer::sse::data(&frame));
        }
        let mut end=base;end["choices"]=json!([{"index":0,"delta":{},"finish_reason":"stop"}]);yield Ok(liteinfer::sse::data(&end));
        let usage=json!({"id":"mock-synthetic","model":"mock","object":"chat.completion.chunk","created":0,"choices":[],"usage":usage});
        yield Ok(liteinfer::sse::data(&usage));yield Ok(bytes::Bytes::from_static(b"data: [DONE]\n\n"));
    };
    let mut response = Response::new(Body::from_stream(stream));
    response.headers_mut().insert(
        "content-type",
        HeaderValue::from_static("text/event-stream"),
    );
    response
}

#[tokio::main(worker_threads = 2)]
async fn main() {
    let app = Router::new()
        .route("/v1/chat/completions", post(complete))
        .route("/v1/completions", post(complete));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:18080")
        .await
        .expect("benchmark port");
    println!("SYNTHETIC benchmark backend on 127.0.0.1:18080 (not a model)");
    axum::serve(listener, app).await.expect("mock server");
}
