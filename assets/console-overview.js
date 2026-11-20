(function (root) {
  "use strict";
  const U = root.LiteInferUI, Charts = root.LiteInferCharts;
  const rate = (value) => value == null ? "—" : U.number(value, value > 0 && value < .1 ? 3 : 1) + " tok/s";
  function performance(report) {
    const p = report.summary.inference;
    U.text("output-speed", rate(p.avg_output_tokens_per_second));
    U.text("speed-note", p.speed_samples ? U.number(p.speed_samples) + " 个流式样本" : "暂无可计速的流式样本");
    U.text("output-throughput", rate(p.output_tokens / report.window_seconds));
    U.text("throughput-note", U.compact(p.output_tokens) + " 输出 token · 含空闲时间");
    U.text("first-output", U.duration(p.p95_first_output_ms));
    U.text("request-cache-hit", U.percent(p.cache_hit_rate));
    U.text("request-cache-note", p.cache_samples ?
      U.number(p.cache_hits) + " / " + U.number(p.cache_samples) + " 请求 · 复用 " + U.percent(p.prompt_token_reuse_rate) : "后端暂无缓存用量");
  }
  function overview(report) {
    const s = report.summary;
    U.text("stat-requests", U.number(s.requests));
    U.text("stat-success", U.percent(s.success_rate));
    U.text("stat-latency", U.duration(s.p95_duration_ms));
    U.text("stat-tokens", U.compact(s.prompt_tokens + s.completion_tokens));
    U.text("stat-requests-note", `平均 ${U.number(s.requests / (report.window_seconds / 60), 1)} 个请求 / 分钟`);
    U.text("stat-success-note", `${U.number(s.succeeded)} 次成功 · ${U.number(s.failed)} 次失败 · ${U.number(s.cancelled)} 次取消`);
    U.text("stat-latency-note", `P50 ${U.duration(s.p50_duration_ms)} · 含排队与生成`);
    U.text("stat-tokens-note", `输入 ${U.compact(s.prompt_tokens)} / 输出 ${U.compact(s.completion_tokens)}`);
    U.text("chart-interval", `每 ${report.bucket_seconds >= 60 ? U.number(report.bucket_seconds / 60) + " 分钟" : report.bucket_seconds + " 秒"}一个时间桶`);
    Charts.sparkline(U.el("spark-requests"), report.series.map((p) => p.requests));
    Charts.sparkline(U.el("spark-latency"), report.series.map((p) => p.avg_duration_ms || 0), true);
    performance(report);
    Charts.traffic(report);
  }
  function resource(id, used, total, note) {
    U.text(`${id}-value`, total ? U.percent(used / total) : "暂无采集");
    U.el(`${id}-progress`).value = total ? Math.min(100, used / total * 100) : 0;
    U.text(`${id}-note`, note);
  }
  function system(info) {
    U.text("host-label", `${info.os || "本地节点"} · ${info.architecture}`);
    U.text("cpu-value", info.cpu.usage_percent == null ? "采集中" : `${U.number(info.cpu.usage_percent, 1)}%`);
    U.el("cpu-progress").value = info.cpu.usage_percent || 0;
    U.text("cpu-note", `${info.cpu.cores} 核心 · ${info.cpu.brand || "本地 CPU"}`);
    resource("memory", info.memory.used_bytes, info.memory.total_bytes,
      `${U.bytes(info.memory.used_bytes)} / ${U.bytes(info.memory.total_bytes)}`);
    resource("disk", info.disk?.used_bytes, info.disk?.total_bytes,
      info.disk ? `${U.bytes(info.disk.available_bytes)} 可用 / ${U.bytes(info.disk.total_bytes)}` : "磁盘数据不可用");
    U.text("uptime-value", U.uptime(info.uptime_seconds));
    U.text("app-version", `v${info.version}`);
    hardware(info.hardware);
    settings(info);
  }
  function hardware(data) {
    const levels = { normal: ["正常", "normal"], warning: ["警告", "warning"], critical: ["严重", "critical"] };
    const [label, tone] = levels[data?.memory_pressure] || ["不可用", "muted"];
    U.text("hardware-sampled", data?.sampled_at ? "采样于 " + U.clock(data.sampled_at) : "当前平台无设备读数");
    U.text("memory-pressure", label);
    U.el("memory-pressure").dataset.level = tone;
    U.text("device-temperature", data?.temperature ? U.number(data.temperature.celsius, 1) + " °C" : "不可用");
    U.text("temperature-note", data?.temperature?.sensor || "没有可读温度传感器");
    const watts = data?.battery_discharge_watts;
    U.text("device-power", watts == null ? "不可用" : U.number(watts, 1) + " W");
    U.text("power-note", watts != null ? "电池电压 × 放电电流" : data?.battery_present ? "当前没有放电读数" : "未检测到电池");
  }
  function settings(info) {
    const l = info.limits, c = info.cache_config;
    U.el("settings-limits").replaceChildren(U.dataList([["监听地址", info.listen], ["推理并发上限", U.number(l.concurrency)],
      ["等待队列容量", U.number(l.queue_capacity)], ["HTTP 连接容量", U.number(l.ingress_capacity)],
      ["排队超时", `${l.queue_timeout_seconds} 秒`], ["请求超时", `${l.request_timeout_seconds} 秒`],
      ["请求 / 响应预算", `${U.bytes(l.max_body_bytes)} / ${U.bytes(l.max_response_bytes)}`],
      ["元数据保留上限", `${U.number(info.request_retention)} 条`], ["网关进程 PID", info.pid]]));
    U.el("settings-cache").replaceChildren(U.dataList([["内存条目 / 容量", `${U.number(c.memory_max_entries)} / ${U.bytes(c.memory_max_bytes)}`],
      ["SSD 缓存", c.ssd_max_bytes ? "已启用" : "关闭"], ["SSD 条目 / 容量", `${U.number(c.ssd_max_entries)} / ${U.bytes(c.ssd_max_bytes)}`],
      ["SSD 有效期", `${c.ttl_seconds} 秒`], ["最小准入 token", U.number(c.min_tokens)],
      ["写入队列条目", c.write_queue_entries], ["写入队列容量", U.bytes(c.write_queue_bytes)],
      ["GPU / 显存", "当前未采集"]]));
  }
  function modelIdentity(model, index) {
    const box = U.node("div", "model-cell");
    const avatar = U.node("span", `model-avatar ${index % 2 ? "blue" : ""}`, model.id.slice(0, 1).toUpperCase());
    const name = U.node("div"); name.append(U.node("strong", "", model.id),
      U.node("div", "table-secondary", model.context_length ? `${U.number(model.context_length)} 上下文` : model.upstream_model));
    box.append(avatar, name); return box;
  }
  function models(result, report) {
    const target = U.el("overview-models"); target.replaceChildren();
    U.text("nav-model-count", result.models.length); U.text("overview-model-count", result.models.length);
    if (!result.models.length) { U.emptyRow(target, 5, "尚未注册模型，前往模型管理开始部署"); return; }
    result.models.forEach((model, index) => {
      const summary = report?.models.find((m) => m.model === model.id)?.summary;
      const row = U.node("tr"), name = U.node("td"), status = U.node("td");
      name.append(modelIdentity(model, index));
      status.append(U.badge(result.managed === model.id ? "受管运行" : "已注册", result.managed === model.id ? "" : "neutral"));
      row.append(name, status, U.node("td", "", U.number(report ? summary?.requests ?? 0 : null)),
        U.node("td", "", U.percent(summary?.success_rate)), U.node("td", "", U.duration(summary?.avg_duration_ms)));
      target.append(row);
    });
  }
  function operation(title, description, tone = "", name = "check") {
    const row = U.node("div", `operation ${tone}`), symbol = U.node("span", "operation-symbol"), content = U.node("div");
    symbol.append(U.icon(name)); content.append(U.node("strong", "", title), U.node("p", "", description));
    row.append(symbol, content); return row;
  }
  function operations(state) {
    const target = U.el("operations"), metrics = state.metrics?.metrics;
    if (!metrics) return;
    const limits = state.metrics.limits;
    U.text("capacity-active", "活跃 " + metrics.active + " / " + limits.concurrency);
    U.text("capacity-queued", "排队 " + metrics.queued + " / " + limits.queue);
    U.text("capacity-peak", "进程峰值 " + metrics.peak_active);
    const rows = [];
    if (metrics.queued) rows.push(operation(`${metrics.queued} 个请求正在排队`, `当前 ${metrics.active} 个活跃请求，可结合并发上限评估负载。`, "warning", "clock"));
    else rows.push(operation("网关已连接", `${metrics.active} 个活跃请求，队列空闲；进程累计完成 ${U.number(metrics.completed)} 次请求。`));
    if (metrics.records_dropped) rows.push(operation("请求记录发生丢弃", `累计 ${U.number(metrics.records_dropped)} 条未写入，历史统计可能不完整。`, "error", "info"));
    else if (state.overview?.summary.failed) rows.push(operation("所选时段存在失败请求", `${state.overview.summary.failed} 次失败，可在请求日志中按结果筛选。`, "warning", "info"));
    else rows.push(operation("元数据记录正常", "未报告记录丢弃。保留最近 10,000 条元数据，不记录请求正文。", "", "list"));
    const rejected = metrics.rejected;
    if (rejected) rows.push(operation("存在请求拒绝", `进程累计 ${U.number(rejected)} 次，可能来自连接容量、等待队列或排队超时。`, "warning", "info"));
    else rows.push(operation(state.system?.cache_config.ssd_max_bytes ? "SSD 缓存已配置" : "SSD 缓存未开启",
      state.system?.cache_config.ssd_max_bytes ? "受管 MLX 服务可使用二级缓存，命中效率可在缓存分析中查看。" : "可在 config.toml 配置 SSD 容量，重启后扩大可复用前缀工作集。", "", "cache"));
    target.replaceChildren(...rows);
  }
  function inventory(result) {
    const target = U.el("model-inventory"); target.replaceChildren();
    if (!result.models.length) { target.append(U.node("div", "inline-empty", "尚未注册模型，使用下方部署流程添加第一个模型。")); return; }
    result.models.forEach((model, index) => {
      const card = U.node("article", "model-card"), top = U.node("div", "model-card-top");
      top.append(modelIdentity(model, index), U.badge(result.managed === model.id ? "受管运行" : "已注册", result.managed === model.id ? "" : "neutral"));
      const details = U.node("div", "model-card-details");
      for (const [label, value] of [["上下文长度", model.context_length ? U.number(model.context_length) : "未提供"],
        ["输出预算", U.number(model.max_output_tokens)], ["缓存隔离", model.cache_isolation_parameter ? "按 Key" : "后端管理"]]) {
        const item = U.node("div"); item.append(U.node("span", "", label), U.node("strong", "", value)); details.append(item);
      }
      const address = U.node("p", "mono", model.base_url);
      card.append(top, details, address); target.append(card);
    });
  }
  root.LiteInferOverview = Object.freeze({ overview, system, models, operations, inventory });
})(window);
