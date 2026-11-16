use axum::{
    Json,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};

#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub kind: &'static str,
    pub message: String,
    pub anthropic: bool,
}

impl ApiError {
    pub fn new(status: StatusCode, message: impl Into<String>) -> Self {
        let kind = match status.as_u16() {
            401 => "authentication_error",
            403 => "permission_error",
            404 => "not_found_error",
            429 => "rate_limit_error",
            500..=599 => "api_error",
            _ => "invalid_request_error",
        };
        Self {
            status,
            kind,
            message: message.into(),
            anthropic: false,
        }
    }
    pub fn bad(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, message)
    }
    pub fn anthropic(mut self) -> Self {
        self.anthropic = true;
        self
    }
    pub fn payload(&self) -> Value {
        if self.anthropic {
            json!({"type":"error","error":{"type":self.kind,"message":self.message}})
        } else {
            json!({"error":{"type":self.kind,"message":self.message,"param":null,"code":self.status.as_u16().to_string()}})
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut headers = HeaderMap::new();
        if self.status == StatusCode::TOO_MANY_REQUESTS {
            headers.insert("retry-after", "2".parse().expect("constant header"));
        }
        (self.status, headers, Json(self.payload())).into_response()
    }
}
