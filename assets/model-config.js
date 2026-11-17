(function (root) {
  "use strict";
  function sourceConfig(input) {
    const value = { source: input.source, repository: input.repository.trim(), alias: input.alias.trim(),
      precision: input.precision, group_size: Number(input.group_size) };
    if (!["huggingface", "modelscope", "local"].includes(value.source)) throw new Error("请选择下载源");
    if (!value.repository || value.repository.length > 1024) throw new Error("仓库或目录长度需为 1..1024");
    if (value.source !== "local" && !/^[A-Za-z0-9][A-Za-z0-9_.-]*\/[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(value.repository)) {
      throw new Error("仓库格式需要 owner/name");
    }
    if (!/^[A-Za-z0-9_.-]{1,128}$/.test(value.alias)) throw new Error("模型别名格式无效");
    if (!["q4", "q8", "float16", "keep"].includes(value.precision) || ![32, 64, 128].includes(value.group_size)) {
      throw new Error("精度或量化分组无效");
    }
    for (const name of ["endpoint", "revision", "token_env"]) {
      const text = (input[name] || "").trim();
      if (text) value[name] = text;
    }
    if (value.endpoint) {
      const url = new URL(value.endpoint);
      const local = ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname);
      if (url.username || url.password || url.search || url.hash || url.pathname !== "/" ||
        !(url.protocol === "https:" || (local && url.protocol === "http:"))) throw new Error("镜像需为 HTTPS 根地址，本机可用 HTTP");
    }
    if (value.token_env && !/^[A-Za-z_][A-Za-z0-9_]*$/.test(value.token_env)) throw new Error("填写令牌环境变量名，不要填写实际令牌");
    if (value.revision && (value.revision.length > 256 || /[\x00-\x1f\x7f-\x9f]/.test(value.revision))) throw new Error("revision 无效");
    return value;
  }
  function summarize(config, tokenizer = {}) {
    const text = config.text_config || config;
    if (!config.model_type || !Number.isSafeInteger(text.vocab_size) || text.vocab_size <= 0 ||
      !Number.isSafeInteger(text.hidden_size) || text.hidden_size <= 0 ||
      !Number.isSafeInteger(text.num_hidden_layers) || text.num_hidden_layers <= 0) throw new Error("模型配置缺少有效架构/词表/层数");
    const context = text.max_position_embeddings ?? text.n_positions ?? null;
    if (context !== null && (!Number.isSafeInteger(context) || context <= 0)) throw new Error("上下文长度无效");
    return { model_type: config.model_type, context_length: context, vocab_size: text.vocab_size,
      layers: text.num_hidden_layers, chat_template: Boolean(tokenizer.chat_template),
      recommended_output_tokens: context ? Math.min(1024, Math.max(1, context - 1)) : 1024 };
  }
  const api = { sourceConfig, summarize };
  if (typeof module !== "undefined") module.exports = api;
  else root.LiteInferConfig = Object.freeze(api);
})(typeof globalThis === "undefined" ? this : globalThis);
