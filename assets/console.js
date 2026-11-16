"use strict";
const element = (id) => document.getElementById(id);
let refreshing = false;
let connected = false;

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "x-admin-key": element("admin").value, "Content-Type": "application/json" },
    signal: AbortSignal.timeout(10000),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error?.message || "请求失败");
  return result;
}

function cell(row, value) {
  const td = document.createElement("td");
  td.textContent = String(value ?? "");
  row.append(td);
  return td;
}

function renderMetrics(stats) {
  element("metrics").replaceChildren();
  const labels = {
    active: "活跃请求", queued: "排队", completed: "完成", failed: "失败",
    cancelled: "取消", prompt_tokens: "输入 Tokens", completion_tokens: "输出 Tokens",
    records_dropped: "记录丢弃",
  };
  for (const [key, label] of Object.entries(labels)) {
    const box = document.createElement("div");
    const number = document.createElement("span");
    number.className = "number";
    number.textContent = stats.metrics[key];
    const caption = document.createElement("span");
    caption.className = "label";
    caption.textContent = label;
    box.append(number, caption);
    element("metrics").append(box);
  }
}

function renderKeys(keys) {
  element("keys").replaceChildren();
  for (const key of keys) {
    const row = document.createElement("tr");
    cell(row, key.name);
    cell(row, key.id);
    cell(row, key.revoked ? "已撤销" : "有效");
    const action = cell(row, "");
    if (!key.revoked) {
      const button = document.createElement("button");
      button.textContent = "撤销";
      button.onclick = () => revokeKey(key.id);
      action.append(button);
    }
    element("keys").append(row);
  }
}

function renderRequests(requests) {
  element("requests").replaceChildren();
  for (const request of requests) {
    const row = document.createElement("tr");
    for (const field of ["model", "protocol", "status", "prompt_tokens", "completion_tokens", "duration_ms"]) {
      cell(row, request[field]);
    }
    element("requests").append(row);
  }
}

async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    const [stats, keys, requests] = await Promise.all([
      api("/admin/metrics"), api("/admin/keys"), api("/admin/requests"),
    ]);
    renderMetrics(stats);
    renderKeys(keys.data);
    renderRequests(requests.data);
    element("status").textContent = `已连接 · 最大并发 ${stats.limits.concurrency} · 队列 ${stats.limits.queue}`;
    connected = true;
  } catch (error) {
    connected = false;
    element("status").textContent = error.message;
  } finally {
    refreshing = false;
  }
}

async function createKey() {
  try {
    const result = await api("/admin/keys", {
      method: "POST", body: JSON.stringify({ name: element("name").value }),
    });
    element("secret").textContent = `请立即保存，此密钥仅显示一次：\n${result.key}`;
    element("name").value = "";
    await refresh();
  } catch (error) { element("status").textContent = error.message; }
}

async function revokeKey(id) {
  try {
    await api(`/admin/keys/${encodeURIComponent(id)}`, { method: "DELETE" });
    element("secret").textContent = "";
    await refresh();
  } catch (error) { element("status").textContent = error.message; }
}

element("refresh").onclick = refresh;
element("create").onclick = createKey;
element("admin").oninput = () => { connected = false; };
setInterval(() => { if (connected && !document.hidden) refresh(); }, 5000);
