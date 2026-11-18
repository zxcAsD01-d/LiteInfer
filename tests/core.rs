use liteinfer::{
    api::anthropic::{self, Translator},
    config::Config,
    deployment::Source,
    sse::{Decoder, Frame},
    store::Store,
};
use serde_json::{Value, json};

fn events(bytes: Vec<bytes::Bytes>) -> Vec<Value> {
    bytes
        .into_iter()
        .map(|b| {
            let s = String::from_utf8(b.to_vec()).unwrap();
            serde_json::from_str(s.lines().find_map(|l| l.strip_prefix("data: ")).unwrap()).unwrap()
        })
        .collect()
}

#[test]
fn configuration_rejects_unsafe_or_ambiguous_values() {
    let source = include_str!("../examples/config.toml");
    let config: Config = toml::from_str(source).unwrap();
    config.validate().unwrap();
    let mut bad = config.clone();
    bad.concurrency = 0;
    assert!(bad.validate().is_err());
    let mut bad = config.clone();
    bad.models.push(bad.models[0].clone());
    assert!(bad.validate().is_err());
    let mut bad = config.clone();
    bad.listen = "0.0.0.0:8787".parse().unwrap();
    assert!(bad.validate().is_err());
    let mut bad = config.clone();
    bad.models[0].base_url = "https://user:password@example.org/v1".into();
    assert!(bad.validate().is_err());
    let mut bad = config;
    bad.models[0].base_url = "file:///tmp/test".into();
    assert!(bad.validate().is_err());
}

#[test]
fn model_manager_defaults_to_qwen35_alias_and_q4() {
    let source: Source = serde_json::from_value(json!({
        "source": "huggingface",
        "repository": "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"
    }))
    .unwrap();
    assert_eq!(source.alias, "qwen3.5-0.8b");
    assert_eq!(source.precision, "q4");
    source.validate().unwrap();
}

#[test]
fn capacities_accept_large_n_and_reject_overflow_without_panicking() {
    let mut config: Config = toml::from_str(include_str!("../examples/config.toml")).unwrap();
    config.concurrency = 1024;
    config.queue_capacity = 10000;
    config.validate().unwrap();
    assert_eq!(config.admission_capacity().unwrap(), 11024);
    assert_eq!(config.ingress_limit().unwrap(), 11040);
    config.queue_capacity = 0;
    config.ingress_capacity = Some(4096);
    config.validate().unwrap();
    assert_eq!(config.ingress_limit().unwrap(), 4096);
    for (active, queue) in [(usize::MAX, 1), (1, tokio::sync::Semaphore::MAX_PERMITS)] {
        config.concurrency = active;
        config.queue_capacity = queue;
        assert!(config.validate().is_err());
    }
    config.concurrency = 1;
    config.queue_capacity = 0;
    config.ingress_capacity = Some(0);
    assert!(config.validate().is_err());
    config.ingress_capacity = None;
    config.stream_buffer_chunks = 0;
    assert!(config.validate().is_err());
    config.stream_buffer_chunks = 1;
    config.metadata_queue_capacity = usize::MAX;
    assert!(config.validate().is_err());
}

#[test]
fn keys_are_hashed_revocable_and_not_retrievable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("state.sqlite3");
    let store = Store::open(&path).unwrap();
    let key = store.create_key("example").unwrap();
    assert_eq!(key.key.len(), 67);
    assert!(!store.active_keys().unwrap().contains_key(&key.key));
    assert_eq!(store.active_keys().unwrap().values().next(), Some(&key.id));
    assert!(
        !serde_json::to_string(&store.keys().unwrap())
            .unwrap()
            .contains(&key.key)
    );
    assert!(store.revoke_key(&key.id).unwrap());
    assert!(store.active_keys().unwrap().is_empty());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
}

#[test]
fn sse_handles_every_utf8_split_and_crlf() {
    let raw = "event: message\r\ndata: {\"text\":\"中文🙂\"}\r\n\r\ndata: [DONE]\n\n".as_bytes();
    for split in 0..=raw.len() {
        let mut decoder = Decoder::default();
        let mut frames = decoder.feed(&raw[..split], 1024).unwrap();
        frames.extend(decoder.feed(&raw[split..], 1024).unwrap());
        assert_eq!(
            frames,
            vec![
                Frame {
                    event: Some("message".into()),
                    data: "{\"text\":\"中文🙂\"}".into()
                },
                Frame {
                    event: None,
                    data: "[DONE]".into()
                }
            ]
        );
        assert!(decoder.is_clean());
    }
}

#[test]
fn sse_rejects_invalid_utf8_and_unbounded_events() {
    assert!(Decoder::default().feed(&[255, 10], 100).is_err());
    assert!(Decoder::default().feed(b"data: 12345678", 8).is_err());
    let mut decoder = Decoder::default();
    assert!(decoder.feed(b"data: unfinished", 100).unwrap().is_empty());
    assert!(!decoder.is_clean());
}

#[test]
fn anthropic_tool_history_and_choice_convert() {
    let value=anthropic::to_openai(&json!({"model":"local","max_tokens":40,"system":[{"type":"text","text":"help"}],
        "messages":[{"role":"assistant","content":[{"type":"tool_use","id":"call_a","name":"write","input":{"content":"中文"}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"call_a","content":[{"type":"text","text":"ok"}]}]}],
        "tools":[{"name":"write","input_schema":{"type":"object"}}],"tool_choice":{"type":"tool","name":"write"}})).unwrap();
    assert_eq!(value["messages"][0]["role"], "system");
    assert_eq!(value["messages"][2]["tool_call_id"], "call_a");
    assert_eq!(value["tool_choice"]["function"]["name"], "write");
    assert_eq!(value["parallel_tool_calls"], false);
    assert_eq!(
        serde_json::from_str::<Value>(
            value["messages"][1]["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap()
        )
        .unwrap()["content"],
        "中文"
    );
}

#[test]
fn unsupported_anthropic_capabilities_are_explicit() {
    for extra in [
        json!({"thinking":{"type":"enabled","budget_tokens":10}}),
        json!({"messages":[{"role":"user","content":[{"type":"document"}]}]}),
    ] {
        let mut value =
            json!({"model":"local","max_tokens":10,"messages":[{"role":"user","content":"hello"}]});
        value
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        assert!(anthropic::to_openai(&value).is_err());
    }
}

#[test]
fn translated_json_preserves_actual_usage_and_tool_input() {
    let response=anthropic::from_openai(&json!({"choices":[{"message":{"content":null,"tool_calls":[{"id":"c","function":{"name":"write","arguments":"{\"x\":1}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":14,"completion_tokens":8}}),"alias").unwrap();
    assert_eq!(
        response["usage"],
        json!({"input_tokens":14,"output_tokens":8})
    );
    assert_eq!(response["content"][0]["input"], json!({"x":1}));
    assert_eq!(response["model"], "alias");
    assert_eq!(response["stop_reason"], "tool_use");
    assert!(anthropic::from_openai(&json!({}), "alias").is_err());
}

#[test]
fn anthropic_text_stream_has_balanced_blocks_and_final_usage() {
    let mut t = Translator::new("alias");
    let mut output = Vec::new();
    for delta in [json!({"content":"中"}), json!({"content":"文"})] {
        output.extend(events(
            t.push(&json!({"choices":[{"delta":delta,"finish_reason":null}]}))
                .unwrap(),
        ));
    }
    output.extend(events(t.push(&json!({"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":2}})).unwrap()));
    output.extend(events(t.finish().unwrap()));
    let types: Vec<_> = output.iter().map(|e| e["type"].as_str().unwrap()).collect();
    assert_eq!(
        types,
        vec![
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop"
        ]
    );
    assert_eq!(
        output[5]["usage"],
        json!({"input_tokens":20,"output_tokens":2})
    );
}

#[test]
fn parallel_tool_deltas_are_serialized_into_valid_blocks() {
    let mut t = Translator::new("alias");
    let mut output = Vec::new();
    let chunks = [
        json!({"index":0,"id":"call_a","function":{"name":"write","arguments":"{\"a\":"}}),
        json!({"index":1,"id":"call_b","function":{"name":"write","arguments":"{\"b\":2}"}}),
        json!({"index":0,"function":{"arguments":"1}"}}),
    ];
    for call in chunks {
        output.extend(events(
            t.push(&json!({"choices":[{"delta":{"tool_calls":[call]}}]}))
                .unwrap(),
        ));
    }
    t.push(&json!({"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":4,"completion_tokens":8}})).unwrap();
    output.extend(events(t.finish().unwrap()));
    let starts: Vec<_> = output
        .iter()
        .filter(|v| v["type"] == "content_block_start")
        .collect();
    assert_eq!(starts.len(), 2);
    assert_eq!(starts[0]["index"], 0);
    assert_eq!(starts[1]["index"], 1);
    assert_eq!(
        output
            .iter()
            .filter(|v| v["type"] == "content_block_stop")
            .count(),
        2
    );
}

#[test]
fn incomplete_tools_and_missing_usage_never_report_success() {
    let mut t = Translator::new("alias");
    t.push(&json!({"choices":[{"delta":{},"finish_reason":"stop"}]}))
        .unwrap();
    assert!(t.finish().is_err());
    let mut t = Translator::new("alias");
    t.push(&json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"write","arguments":"{"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1}})).unwrap();
    assert!(t.finish().is_err());
}
