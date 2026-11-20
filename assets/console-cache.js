(function (root) {
  "use strict";
  const U = root.LiteInferUI;
  const latencyValue = (value) => value == null ? "—" : U.number(value, 2) + " ms";
  function stat(label, value, note, name, color) {
    const card = U.node("article", "stat-card"), top = U.node("div", "stat-top"), symbol = U.node("span", `stat-icon ${color}`);
    symbol.append(U.icon(name)); top.append(U.node("span", "", label), symbol);
    card.append(top, U.node("div", "stat-value", value), U.node("div", "stat-bottom", note)); return card;
  }
  function stats(cache) {
    U.el("cache-stats").replaceChildren(
      stat("请求命中率", U.percent(cache?.requests ? cache.hit_rate : null), cache ? U.number(cache.hits) + " 命中 / " + U.number(cache.requests) + " 次查询" : "暂无缓存样本", "check", "mint"),
      stat("Token 复用率", U.percent(cache?.input_tokens ? cache.token_hit_rate : null), cache ? "复用 " + U.compact(cache.saved_tokens) + " 个输入 tokens" : "暂无输入 token 样本", "tokens", "mint"),
      stat("查找延迟 P95", latencyValue(cache?.lookup_latency?.samples ? cache.lookup_latency.p95_ms : null), "包含状态复制或 SSD 恢复", "clock", "blue"),
      stat("SSD 命中", cache ? U.number(cache.ssd_hits) : "—", cache?.ssd?.enabled === false ? "SSD 缓存未启用" : "SSD 状态恢复次数", "cache", "violet"));
  }
  function tier(title, count, size, maximum, hits, disabled = false) {
    const block = U.node("div", "tier-block"), header = U.node("div", "tier-heading"), heading = U.node("div");
    heading.append(U.icon(title === "内存缓存" ? "server" : "cache"), U.node("span", "", title));
    header.append(heading, U.badge(disabled ? "关闭" : "已启用", disabled ? "neutral" : ""));
    const detail = U.node("div", "tier-stats"); detail.append(U.node("span", "", `${U.number(count)} 个条目`), U.node("span", "", `${U.bytes(size)} / ${U.bytes(maximum)}`));
    const progress = U.node("progress"); progress.max = 100; progress.value = maximum ? size / maximum * 100 : 0; progress.setAttribute("aria-label", `${title}容量使用率`);
    block.append(header, detail, progress, U.node("p", "", disabled ? "在配置中设置容量并重启受管服务后生效" : `累计命中 ${U.number(hits)} 次`)); return block;
  }
  function indices(cache) {
    const ram = cache.index, disk = cache.ssd.index;
    const sections = [U.node("h3", "", "内存前缀索引"), U.dataList([["索引条目 / 作用域", `${ram.entries} / ${ram.scopes}`],
      ["节点 / 压缩边 tokens", `${U.number(ram.nodes)} / ${U.number(ram.edge_tokens)}`],
      ["平均访问节点", U.number(ram.mean_nodes_visited, 2)],
      ["查找 P50 / P95", `${U.number(ram.lookup_p50_us, 2)} / ${U.number(ram.lookup_p95_us, 2)} μs`]])];
    if (disk) sections.push(U.node("h3", "", "SSD 常驻索引"), U.dataList([["索引条目 / 作用域", `${disk.entries} / ${disk.scopes}`],
      ["平均访问节点", U.number(disk.mean_nodes_visited, 2)],
      ["查找 P50 / P95", `${U.number(disk.lookup_p50_us, 2)} / ${U.number(disk.lookup_p95_us, 2)} μs`]]));
    U.el("cache-index").replaceChildren(...sections);
  }
  function latency(cache) {
    const table = U.node("table"), head = U.node("thead"), titles = U.node("tr");
    for (const title of ["路径", "样本", "P50", "P95"]) titles.append(U.node("th", "", title));
    head.append(titles); const body = U.node("tbody");
    for (const [name, data] of [["总体查找", cache.lookup_latency], ["SSD 读取", cache.ssd.read_latency], ["SSD 写入", cache.ssd.write_latency]]) {
      const row = U.node("tr");
      for (const value of [name, U.number(data?.samples), latencyValue(data?.samples ? data.p50_ms : null), latencyValue(data?.samples ? data.p95_ms : null)]) row.append(U.node("td", "", value));
      body.append(row);
    }
    table.append(head, body); U.el("cache-latency").replaceChildren(table);
  }
  function render(cache) {
    stats(cache); U.text("cache-state", "当前受管 MLX 后端 · 进程累计数据"); U.el("cache-clear").disabled = false;
    const disk = cache.ssd, config = cache.settings;
    U.el("cache-tiers").replaceChildren(tier("内存缓存", cache.entries, cache.bytes, config.memory_max_bytes, cache.memory_hits, !config.memory_max_entries || !config.memory_max_bytes),
      tier("SSD 缓存", disk.entries || 0, disk.bytes || 0, config.ssd_max_bytes, disk.hits || 0, !config.ssd_max_bytes));
    indices(cache); latency(cache);
    const notes = (cache.analysis.length ? cache.analysis : ["当前未发现缓存异常。命中率受输入重复度、租户分布与容量影响。"])
      .map((note) => U.node("p", "cache-note", note));
    const counters = U.dataList([["精确 / 前缀命中", U.number(cache.exact_hits) + " / " + U.number(cache.prefix_hits)],
      ["SSD 查询命中率", U.percent(disk.hits + disk.misses > 0 ? disk.hit_rate : null)],
      ["内存 / SSD 淘汰", U.number(cache.evictions) + " / " + U.number(disk.evictions || 0)],
      ["SSD 过期", U.number(disk.expirations || 0)], ["读 / 写错误", `${disk.read_errors || 0} / ${disk.write_errors || 0}`],
      ["队列丢弃 / 准入跳过", `${disk.queue_drops || 0} / ${disk.admission_skips || 0}`],
      ["待写入条目 / 字节", `${disk.pending_writes || 0} / ${U.bytes(disk.pending_bytes || 0)}`]]);
    U.el("cache-analysis").replaceChildren(...notes, counters);
  }
  function unavailable(message) {
    stats(null); U.text("cache-state", message); U.el("cache-clear").disabled = true;
    for (const id of ["cache-tiers", "cache-index", "cache-latency", "cache-analysis"]) {
      U.el(id).replaceChildren(U.node("div", "inline-empty", id === "cache-tiers" ? message : "暂无可用数据"));
    }
  }
  root.LiteInferCache = Object.freeze({ render, unavailable });
})(window);
