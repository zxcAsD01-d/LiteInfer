(function (root) {
  "use strict";
  const number = (value, digits = 0) => value == null || !Number.isFinite(Number(value)) ? "—" :
    Number(value).toLocaleString("zh-CN", { maximumFractionDigits: digits, minimumFractionDigits: digits });
  const compact = (value) => value == null ? "—" : value >= 1e6 ? `${number(value / 1e6, 2)}M` :
    value >= 1e4 ? `${number(value / 1e3, 1)}k` : number(value);
  const percent = (value, digits = 1) => value == null ? "—" : `${number(value * 100, digits)}%`;
  const duration = (value) => value == null ? "—" : value >= 1000 ? `${number(value / 1000, 2)} s` : `${number(value)} ms`;
  function bytes(value) {
    if (value == null) return "—";
    const units = ["B", "KiB", "MiB", "GiB", "TiB"];
    const index = value > 0 ? Math.min(4, Math.floor(Math.log(value) / Math.log(1024))) : 0;
    return `${number(value / 1024 ** index, index ? 1 : 0)} ${units[index]}`;
  }
  function uptime(seconds) {
    if (seconds == null) return "—";
    if (seconds >= 86400) return `${Math.floor(seconds / 86400)} 天 ${Math.floor(seconds % 86400 / 3600)} 小时`;
    if (seconds >= 3600) return `${Math.floor(seconds / 3600)} 小时 ${Math.floor(seconds % 3600 / 60)} 分钟`;
    return `${Math.floor(seconds / 60)} 分钟 ${seconds % 60} 秒`;
  }
  const clock = (value) => value == null ? "—" : new Date(value * 1000).toLocaleTimeString("zh-CN", { hour12: false });
  const datetime = (value) => value == null ? "—" : new Date(value * 1000).toLocaleString("zh-CN", { hour12: false });
  const el = (id) => document.getElementById(id);
  function node(tag, className, text) {
    const value = document.createElement(tag);
    if (className) value.className = className;
    if (text != null) value.textContent = String(text);
    return value;
  }
  function svg(tag, attrs = {}) {
    const value = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [name, content] of Object.entries(attrs)) value.setAttribute(name, String(content));
    return value;
  }
  function icon(name) {
    const value = svg("svg", { class: "icon", "aria-hidden": "true" });
    value.append(svg("use", { href: `#i-${name}` }));
    return value;
  }
  function text(id, value) { el(id).textContent = value == null ? "—" : String(value); }
  function badge(label, tone = "") { return node("span", `pill ${tone}`, label); }
  function emptyRow(target, columns, label) {
    const row = node("tr");
    const cell = node("td", "empty-cell", label);
    cell.colSpan = columns; row.append(cell); target.replaceChildren(row);
  }
  function dataList(entries) {
    const list = node("dl", "data-list");
    for (const [label, value] of entries) {
      const row = node("div", "data-row");
      row.append(node("dt", "", label), node("dd", "", value ?? "—")); list.append(row);
    }
    return list;
  }
  function requestState(status) {
    if (status >= 200 && status < 300) return { name: "成功", tone: "", filter: "success" };
    if (status === 499) return { name: "已取消", tone: "warning", filter: "cancelled" };
    return { name: "失败", tone: "error", filter: "error" };
  }
  function filterRequests(rows, filters) {
    const search = (filters.search || "").trim().toLowerCase();
    return rows.filter((row) => (!search || `${row.id} ${row.model}`.toLowerCase().includes(search)) &&
      (!filters.model || row.model === filters.model) &&
      (!filters.protocol || row.protocol === filters.protocol) &&
      (!filters.status || requestState(row.status).filter === filters.status));
  }
  function csv(rows) {
    return "\ufeff" + rows.map((row) => row.map((value) => {
      let text = String(value ?? "");
      if (/^[=+@\-\t\r]/.test(text)) text = "'" + text;
      return `"${text.replace(/"/g, '""')}"`;
    }).join(",")).join("\r\n");
  }
  function download(content, filename, type) {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const link = node("a"); link.href = url; link.download = filename;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const api = { number, compact, percent, duration, bytes, uptime, clock, datetime, el, node, svg, icon, text,
    badge, emptyRow, dataList, requestState, filterRequests, csv, download };
  if (typeof module !== "undefined") module.exports = api;
  else root.LiteInferUI = Object.freeze(api);
})(typeof globalThis === "undefined" ? this : globalThis);
