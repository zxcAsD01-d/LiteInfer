(function (root) {
  "use strict";
  const U = root.LiteInferUI, O = root.LiteInferOverview;
  const state = { connected: false, refreshing: false, live: true, window: "1h", view: "overview",
    metrics: null, overview: null, system: null, models: null, cache: null };
  let adminKey = "", session = 0, toastTimer, confirmResolve, refreshPending = false, clearing = false;
  const controllers = new Set();
  const pages = { overview: "运行总览", models: "模型管理", cache: "缓存分析",
    requests: "请求日志", keys: "访问密钥", settings: "系统设置" };

  async function api(path, options = {}) {
    const { timeout = 10000, key = adminKey, ...request } = options;
    if (!key) throw new Error("请先连接管理服务");
    const controller = new AbortController(), timer = setTimeout(() => controller.abort(), timeout);
    controllers.add(controller);
    try {
      const response = await fetch(path, { ...request, cache: "no-store", credentials: "omit",
        headers: { "x-admin-key": key, "Content-Type": "application/json" }, signal: controller.signal });
      let result;
      try { result = await response.json(); }
      catch { throw new Error(`管理接口返回了无效响应（HTTP ${response.status}）`); }
      if (!response.ok) throw Object.assign(new Error(result.error?.message || result.error || `请求失败（${response.status}）`), { status: response.status });
      return result;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("请求超时或连接已取消，请重试");
      if (error instanceof TypeError) throw new Error("网络连接失败，请检查服务是否运行");
      throw error;
    } finally { clearTimeout(timer); controllers.delete(controller); }
  }
  function toast(message, error = false) {
    clearTimeout(toastTimer); U.text("toast", message); U.el("toast").classList.toggle("error", error); U.el("toast").hidden = false;
    toastTimer = setTimeout(() => { U.el("toast").hidden = true; }, 4000);
  }
  async function copy(value) {
    try { await navigator.clipboard.writeText(value); toast("已复制到剪贴板"); }
    catch { toast("复制失败，请选择文本手动复制", true); }
  }
  function confirmAction({ title, description, accept = "确认" }) {
    if (confirmResolve) return Promise.resolve(false);
    U.text("confirm-title", title); U.text("confirm-description", description); U.text("confirm-accept", accept);
    const dialog = U.el("confirm-dialog"); dialog.returnValue = "";
    const result = new Promise((resolve) => { confirmResolve = resolve; }); dialog.showModal(); return result;
  }
  function requireConnection() {
    if (state.connected) return true;
    U.el("connect-dialog").showModal(); return false;
  }
  function connectionStatus(stale = false) {
    const badge = U.el("connection-badge"); badge.className = `connection-badge ${stale ? "stale" : "healthy"}`;
    badge.replaceChildren(U.node("span", `status-dot ${stale ? "warning" : "healthy"}`), U.node("span", "", stale ? "数据待更新" : "服务已连接"));
    U.el("node-dot").className = `status-dot ${stale ? "warning" : "healthy"}`;
    U.text("node-state", stale ? "部分数据未同步" : "本地节点已连接");
  }
  async function connect(event) {
    event.preventDefault(); const button = U.el("connect-submit"), candidate = U.el("admin").value.trim();
    button.disabled = true; U.text("connect-error", "");
    try {
      await api("/admin/metrics", { key: candidate });
      controllers.forEach((controller) => controller.abort()); session++; adminKey = candidate; state.connected = true;
      U.el("admin").value = ""; U.el("connect-dialog").close(); U.el("connection-notice").hidden = true;
      U.text("account-label", "管理员"); connectionStatus(); toast("管理服务已连接"); await refresh();
    } catch (error) { U.text("connect-error", error.message); }
    finally { button.disabled = false; }
  }
  function disconnect() {
    adminKey = ""; session++; state.connected = false;
    controllers.forEach((controller) => controller.abort());
    U.text("secret", ""); U.el("admin").value = ""; root.location.reload();
  }
  function renderSnapshots() {
    if (state.overview && state.view === "overview") O.overview(state.overview);
    if (state.system) O.system(state.system);
    if (state.models) { O.models(state.models, state.overview); O.inventory(state.models); root.LiteInferModels.render(state.models); }
    O.operations(state);
  }
  async function refreshCache(generation) {
    if (clearing) return;
    const managed = state.models?.managed;
    if (!managed || managed === "正在启动或停止") {
      state.cache = null; root.LiteInferCache.unavailable(managed ? "受管模型正在启动或停止" : "未启动受管 MLX 模型，缓存数据暂不可用"); return;
    }
    try {
      const cache = await api("/admin/mlx/cache");
      if (generation !== session || clearing) return;
      state.cache = cache; root.LiteInferCache.render(cache);
    } catch (error) {
      if (generation !== session || clearing) return;
      if (error.status === 401) { disconnect(); return; }
      if (state.cache) {
        U.text("cache-state", "统计未更新，显示上次快照：" + error.message);
        U.el("cache-clear").disabled = true;
      } else root.LiteInferCache.unavailable("缓存统计暂不可用：" + error.message);
    }
  }
  async function refresh(queue = true) {
    if (!state.connected) return;
    if (state.refreshing) { refreshPending ||= queue; return; }
    const requestedWindow = state.window;
    state.refreshing = true; const generation = session; U.el("refresh").classList.add("refreshing");
    const sources = [["metrics", "/admin/metrics", "进程指标"], ["overview", "/admin/overview?window=" + requestedWindow, "请求统计"],
      ["system", "/admin/system", "系统资源"], ["models", "/admin/models", "模型状态"],
      ["requests", "/admin/requests", "请求日志"]];
    if (state.view === "keys") sources.push(["keys", "/admin/keys", "密钥列表"]);
    try {
      const results = await Promise.allSettled(sources.map(([, path]) => api(path)));
      if (generation !== session) return;
      if (results.some((result) => result.status === "rejected" && result.reason.status === 401)) {
        disconnect(); return;
      }
      const errors = [];
      results.forEach((result, i) => {
        const [name, , label] = sources[i];
        if (result.status === "rejected") {
          errors.push(`${label}：${result.reason.message}`); return;
        }
        if (name === "requests") root.LiteInferRecords.update(result.value.data);
        else if (name === "keys") root.LiteInferKeys.render(result.value.data);
        else if (name !== "overview" || requestedWindow === state.window) state[name] = result.value;
      });
      renderSnapshots(); connectionStatus(errors.length > 0);
      U.el("status").hidden = !errors.length; U.text("status", errors.length ? `部分数据未更新，保留上次快照。${errors.join("；")}` : "");
      if (results.some((result) => result.status === "fulfilled")) U.text("updated-at", U.clock(Date.now() / 1000) + " 同步");
      if (state.view === "cache") await refreshCache(generation);
    } catch (error) {
      connectionStatus(true);
      U.el("status").hidden = false;
      U.text("status", "更新失败，已保留快照。" + error.message);
    } finally {
      state.refreshing = false;
      U.el("refresh").classList.remove("refreshing");
      if (refreshPending) { refreshPending = false; void refresh(); }
    }
  }
  function setNavigation(open) {
    U.el("sidebar").classList.toggle("open", open); U.el("nav-scrim").hidden = !open;
    U.el("menu-toggle").setAttribute("aria-expanded", String(open));
  }
  function navigate() {
    const selected = root.location.hash.slice(1);
    if (selected === "main") return;
    const view = Object.hasOwn(pages, selected) ? selected : "overview";
    state.view = view; const title = pages[view];
    document.querySelectorAll(".view").forEach((section) => { section.hidden = section.id !== `view-${view}`; });
    document.querySelectorAll("[data-view]").forEach((link) => {
      link.classList.toggle("active", link.dataset.view === view);
      if (link.dataset.view === view) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current");
    });
    U.text("page-title", title); U.text("breadcrumb-current", title);
    document.title = `${title} · LiteInfer`; U.el("time-window").closest("label").hidden = view !== "overview";
    U.el("export-report").hidden = view !== "overview"; setNavigation(false);
    if (view === "overview" && state.overview) O.overview(state.overview);
    if (view === "cache" || view === "keys") void refresh();
    root.scrollTo({ top: 0 });
  }
  async function clearCache() {
    if (clearing || !requireConnection()) return;
    if (!await confirmAction({ title: "清理全部推理缓存？", description: "将删除当前受管模型的内存与 SSD 缓存，后续请求可能需要重新预填充。正在运行的推理不受影响，完成后仍可能写入新缓存。", accept: "清理缓存" })) return;
    clearing = true;
    U.el("cache-clear").disabled = true;
    try { state.cache = await api("/admin/mlx/cache", { method: "DELETE" }); root.LiteInferCache.render(state.cache); toast("缓存已清理，累计计数保留"); }
    catch (error) { toast(error.message, true); }
    finally { clearing = false; void refresh(); }
  }
  function exportReport() {
    if (!requireConnection()) return;
    if (!state.overview || state.overview.window_seconds !== { "15m": 900, "1h": 3600, "24h": 86400 }[state.window]) {
      toast("所选时段的统计尚未更新", true); return;
    }
    const report = { exported_at: new Date().toISOString(), window: state.window,
      overview: state.overview, metrics: state.metrics, system: state.system, cache: state.cache };
    U.download(JSON.stringify(report, null, 2), "liteinfer-runtime-report.json", "application/json"); toast("运行报告已导出");
  }
  function initDialogs() {
    U.el("connect-form").onsubmit = connect;
    U.el("connect-dialog").addEventListener("close", () => { U.el("admin").value = ""; U.text("connect-error", ""); });
    for (const id of ["connect-open", "notice-connect"]) U.el(id).onclick = () => U.el("connect-dialog").showModal();
    document.querySelectorAll("[data-close]").forEach((button) => { button.onclick = () => U.el(button.dataset.close).close(); });
    U.el("confirm-form").onsubmit = (event) => { event.preventDefault(); U.el("confirm-dialog").close("yes"); };
    U.el("confirm-cancel").onclick = () => U.el("confirm-dialog").close();
    U.el("confirm-dialog").addEventListener("close", () => {
      const resolve = confirmResolve; confirmResolve = null; if (resolve) resolve(U.el("confirm-dialog").returnValue === "yes");
    });
  }
  function initActions() {
    U.el("chart-mode").onchange = () => root.LiteInferCharts.traffic(state.overview);
    U.el("refresh").onclick = () => { if (requireConnection()) refresh(); };
    U.el("time-window").onchange = () => {
      state.window = U.el("time-window").value;
      if (state.connected) refresh();
    };
    U.el("live-toggle").onclick = () => {
      state.live = !state.live; U.el("live-toggle").setAttribute("aria-pressed", String(state.live));
      U.el("live-toggle").replaceChildren(U.node("span", "live-indicator"), document.createTextNode(state.live ? "每 5 秒刷新" : "自动刷新已暂停"));
      if (state.live) refresh();
    };
    U.el("disconnect").onclick = disconnect; U.el("cache-clear").onclick = clearCache; U.el("export-report").onclick = exportReport;
    U.el("menu-toggle").onclick = () => setNavigation(!U.el("sidebar").classList.contains("open"));
    U.el("nav-scrim").onclick = () => setNavigation(false);
    U.el("quick-search").onclick = () => { root.location.hash = "requests"; requestAnimationFrame(() => U.el("request-search").focus()); };
    document.addEventListener("keydown", (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "k") { event.preventDefault(); U.el("quick-search").click(); }
      if (event.key === "Escape") setNavigation(false);
    });
    document.querySelectorAll("[data-copy]").forEach((button) => { button.onclick = () => copy(U.el(button.dataset.copy).textContent); });
  }
  root.LiteInferConsole = Object.freeze({ state, api, refresh, toast, copy, confirm: confirmAction, requireConnection });
  initDialogs(); initActions(); root.LiteInferRecords.init(); root.LiteInferKeys.init(); root.LiteInferCache.unavailable("等待连接受管 MLX 服务");
  U.text("openai-url", `${root.location.origin}/v1`); U.text("anthropic-url", `${root.location.origin}/v1/messages`);
  root.addEventListener("hashchange", navigate); navigate();
  let resizeTimer; root.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { if (state.view === "overview" && state.overview) root.LiteInferCharts.traffic(state.overview); }, 150); });
  document.addEventListener("visibilitychange", () => { if (!document.hidden && state.live) refresh(); });
  setInterval(() => { if (state.connected && state.live && !document.hidden) refresh(false); }, 5000);
})(window);
