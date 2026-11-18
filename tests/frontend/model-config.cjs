const assert = require("node:assert/strict");
const fs = require("node:fs");
const { sourceConfig, summarize } = require("../../assets/model-config.js");
const valid = { source: "huggingface", repository: "jiaheng0815/feng-30m", alias: "gpt-6.5astra", precision: "q4", group_size: 64 };
assert.equal(sourceConfig(valid).repository, valid.repository);
for (const source of ["huggingface", "modelscope", "local"]) assert.equal(sourceConfig({ ...valid, source }).source, source);
for (const endpoint of ["https://hf.example", "http://127.0.0.1:8000"]) assert.equal(sourceConfig({ ...valid, endpoint }).endpoint, endpoint);
for (const change of [{ repository: "../model" }, { alias: "<script>" }, { endpoint: "https://name:secret@example.org" },
  { endpoint: "http://example.org" }, { token_env: "secret-value" }, { token_env: "1TOKEN" },
  { repository: ".owner/model" }, { revision: "bad\nrevision" }, { group_size: 0 }, { precision: "q2" }]) {
  assert.throws(() => sourceConfig({ ...valid, ...change }));
}
const model = { model_type: "qwen3", hidden_size: 448, num_hidden_layers: 11, vocab_size: 16384, max_position_embeddings: 32768 };
assert.equal(summarize(model).context_length, 32768);
assert.equal(summarize({ model_type: "qwen3", text_config: model }).context_length, 32768);
assert.equal(summarize(model).recommended_output_tokens, 1024);
assert.throws(() => summarize({ ...model, max_position_embeddings: 0 }));
const consoleHtml = fs.readFileSync(require.resolve("../../assets/console.html"), "utf8");
assert.match(consoleHtml, /value="mlx-community\/Qwen3.5-0.8B-OptiQ-4bit"/);
assert.match(consoleHtml, /id="model-alias" value="qwen3.5-0.8b"/);
console.log("前端配置校验通过：下载源、镜像、别名、32K 配置，无网络请求。");
