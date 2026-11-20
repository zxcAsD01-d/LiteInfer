(function (root) {
  "use strict";
  const U = root.LiteInferUI;
  let records = [], currentPage = 1, filtered = [];
  function requestRow(record) {
    const row = U.node("tr"), id = U.node("td", "mono", record.id.length > 19 ? record.id.slice(0, 16) + "…" : record.id);
    id.title = record.id;
    const model = U.node("td"); model.append(U.node("strong", "", record.model), U.node("div", "table-secondary", record.protocol));
    const status = U.node("td"), state = U.requestState(record.status); status.append(U.badge(`${record.status} ${state.name}`, state.tone));
    row.append(id, model, status, U.node("td", "", `${U.number(record.prompt_tokens)} / ${U.number(record.completion_tokens)}`),
      U.node("td", "", U.duration(record.duration_ms)), U.node("td", "", U.clock(record.created)));
    row.lastChild.title = U.datetime(record.created); return row;
  }
  function renderTable(target, rows) {
    if (!rows.length) { U.emptyRow(target, 6, "暂无符合条件的请求"); return; }
    target.replaceChildren(...rows.map(requestRow));
  }
  function options(id, values, placeholder) {
    const target = U.el(id), selected = target.value;
    const first = U.node("option", "", placeholder); first.value = "";
    target.replaceChildren(first);
    [...new Set(values)].sort().forEach((value) => { const option = U.node("option", "", value); option.value = value; target.append(option); });
    target.value = values.includes(selected) ? selected : "";
  }
  function applyFilters(resetPage = true) {
    if (resetPage) currentPage = 1;
    filtered = U.filterRequests(records, { search: U.el("request-search").value, model: U.el("request-model").value,
      status: U.el("request-status").value, protocol: U.el("request-protocol").value });
    const pages = Math.max(1, Math.ceil(filtered.length / 10)); currentPage = Math.min(currentPage, pages);
    renderTable(U.el("requests"), filtered.slice((currentPage - 1) * 10, currentPage * 10));
    U.text("requests-count", `匹配 ${filtered.length} 条 / 最近 ${records.length} 条请求`);
    U.text("requests-page", `${currentPage} / ${pages}`);
    U.el("requests-prev").disabled = currentPage === 1; U.el("requests-next").disabled = currentPage === pages;
  }
  function update(rows) {
    records = rows; renderTable(U.el("overview-requests"), rows.slice(0, 5));
    options("request-model", rows.map((r) => r.model), "全部模型");
    options("request-protocol", rows.map((r) => r.protocol), "全部协议"); applyFilters(false);
  }
  function exportCsv() {
    const rows = [["请求 ID", "模型", "协议", "状态", "输入 Tokens", "输出 Tokens", "耗时 ms", "请求时间"],
      ...filtered.map((r) => [r.id, r.model, r.protocol, r.status, r.prompt_tokens, r.completion_tokens, r.duration_ms, U.datetime(r.created)])];
    U.download(U.csv(rows), "liteinfer-requests.csv", "text/csv;charset=utf-8");
  }
  function init() {
    for (const id of ["request-search", "request-model", "request-status", "request-protocol"]) U.el(id).addEventListener("input", () => applyFilters());
    U.el("requests-prev").onclick = () => { currentPage--; applyFilters(false); };
    U.el("requests-next").onclick = () => { currentPage++; applyFilters(false); };
    U.el("requests-export").onclick = exportCsv; applyFilters();
  }
  root.LiteInferRecords = Object.freeze({ init, update });
})(window);
