(function (root) {
  "use strict";
  const U = root.LiteInferUI;
  let busy = false;
  const actions = ["model-runtime", "model-setup", "model-inspect", "model-prepare", "model-launch", "model-stop"];
  const C = () => root.LiteInferConsole;
  function sourceForm() {
    const values = {};
    for (const name of ["source", "repository", "endpoint", "revision", "token_env", "alias", "precision", "group_size"]) values[name] = U.el(`model-${name}`).value;
    return root.LiteInferConfig.sourceConfig(values);
  }
  function message(value, error = false) { U.text("model-status", value); U.el("model-status").classList.toggle("error", error); }
  async function action(work, local = false) {
    if (busy || (!local && !C().requireConnection())) return;
    busy = true; actions.forEach((id) => { U.el(id).disabled = true; });
    U.el("view-models").setAttribute("aria-busy", "true");
    try { await work(); }
    catch (error) { message(error.message, true); C().toast(error.message, true); }
    finally {
      busy = false; actions.forEach((id) => { U.el(id).disabled = false; });
      U.el("view-models").setAttribute("aria-busy", "false");
    }
  }
  function info(value) {
    U.el("model-info").hidden = false; U.text("model-info", JSON.stringify(value, null, 2));
    if (value.recommended_output_tokens) U.el("model-output").value = value.recommended_output_tokens;
  }
  function addArtifact(item) {
    const select = U.el("model-artifact");
    if (Array.from(select.options).some((option) => option.value === item.artifact)) return;
    const option = U.node("option", "", `${item.alias} · ${item.metadata?.context_length || "未知"} 上下文 · ${item.artifact.slice(0, 8)}`);
    option.value = item.artifact; option.dataset.alias = item.alias; option.dataset.metadata = JSON.stringify(item.metadata || {});
    select.append(option); if (!select.value) select.value = item.artifact;
  }
  function render(result) {
    const target = U.el("model-jobs"); target.replaceChildren();
    const names = { prepare: "准备模型", setup: "安装 MLX 环境" }, states = { running: "执行中", completed: "已完成", failed: "失败" };
    for (const job of [...result.jobs].reverse()) {
      const row = U.node("div", "job-row"), detail = U.node("div");
      detail.append(U.node("strong", "", names[job.action] || job.action), U.node("p", "", `${job.stage}${job.error ? ` · ${job.error}` : ""}`));
      row.append(detail, U.badge(states[job.status] || job.status, job.status === "failed" ? "error" : job.status === "running" ? "warning" : "")); target.append(row);
      if (job.result?.artifact) addArtifact(job.result);
    }
    if (!result.jobs.length) target.append(U.node("div", "inline-empty", "暂无部署任务"));
    U.text("model-managed", result.managed ? `受管：${result.managed}` : "未启动受管模型");
    U.el("model-managed").className = `pill ${result.managed ? "" : "neutral"}`;
  }
  async function runtime() {
    message("正在检查本机运行环境…");
    const result = await C().api("/admin/mlx/runtime", { timeout: 30000 });
    info(result); result.artifacts.forEach(addArtifact);
    message(result.ready ? "MLX 运行环境已就绪" : "环境尚未就绪，可安装独立 MLX 环境"); await C().refresh();
  }
  async function prepare() {
    await C().api("/admin/mlx/prepare", { method: "POST", body: JSON.stringify(sourceForm()) });
    message("已开始下载、转换和自检，部署任务会自动更新。"); await C().refresh();
  }
  async function launch() {
    const alias = U.el("model-alias").value.trim(), artifact = U.el("model-artifact").value;
    const port = Number(U.el("model-port").value), output = Number(U.el("model-output").value);
    if (!/^[A-Za-z0-9_.-]{1,128}$/.test(alias)) throw new Error("模型别名格式无效");
    if (!artifact) throw new Error("请先选择通过自检的模型产物");
    if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error("端口需为 1–65535 的整数");
    if (!Number.isSafeInteger(output) || output < 1 || output > 4294967295) throw new Error("输出预算需为有效正整数");
    message("正在加载模型，后端就绪后自动注册到 API，请勿重复提交…");
    await C().api("/admin/mlx/launch", { method: "POST", timeout: 180000,
      body: JSON.stringify({ artifact, alias, port, max_output_tokens: output }) });
    message(`模型 ${alias} 已启动并应用`); C().toast("模型已启动"); await C().refresh();
  }
  async function stop() {
    if (!await C().confirm({ title: "停止受管模型？", description: "将关闭当前受管推理进程，正在执行的请求可能中断。停止后可从已验证产物重新启动。", accept: "停止模型" })) return;
    await C().api("/admin/mlx/stop", { method: "POST" }); message("受管模型已停止"); await C().refresh();
  }
  function init() {
    U.el("model-runtime").onclick = () => action(runtime);
    U.el("model-setup").onclick = () => action(async () => {
      await C().api("/admin/mlx/setup", { method: "POST" }); message("已开始安装独立 MLX 环境，可在部署任务中查看进度。"); await C().refresh();
    });
    U.el("model-preview").onclick = () => action(async () => { info(sourceForm()); message("表单校验通过；预览在浏览器内生成。"); }, true);
    U.el("model-inspect").onclick = () => action(async () => {
      message("正在读取配置并检查模型架构…");
      info(await C().api("/admin/mlx/inspect", { method: "POST", body: JSON.stringify(sourceForm()), timeout: 120000 })); message("模型配置校验通过");
    });
    U.el("model-prepare").onclick = () => action(prepare); U.el("model-launch").onclick = () => action(launch); U.el("model-stop").onclick = () => action(stop);
    U.el("model-artifact").onchange = () => {
      const option = U.el("model-artifact").selectedOptions[0];
      if (option.dataset.alias) U.el("model-alias").value = option.dataset.alias;
      if (option.dataset.metadata) info(JSON.parse(option.dataset.metadata));
    };
    U.el("model-file").onchange = () => action(async () => {
      const file = U.el("model-file").files[0]; if (!file) return;
      if (file.size > 1024 * 1024) throw new Error("config.json 超过 1 MiB");
      info(root.LiteInferConfig.summarize(JSON.parse(await file.text()))); message("本地配置已在浏览器内校验，未上传文件。");
    }, true);
  }
  root.LiteInferModels = Object.freeze({ render }); init();
})(window);
