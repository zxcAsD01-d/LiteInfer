"use strict";
const modelEl = (id) => document.getElementById(id);
let modelBusy = false;
let modelPolling = false;

function sourceForm() {
  const values = {};
  for (const name of ["source", "repository", "endpoint", "revision", "token_env", "alias", "precision", "group_size"]) {
    values[name] = modelEl(`model-${name}`).value;
  }
  return LiteInferConfig.sourceConfig(values);
}

function modelMessage(value) { modelEl("model-status").textContent = value; }

async function modelAction(work) {
  if (modelBusy) return;
  modelBusy = true;
  try { await work(); } catch (error) { modelMessage(error.message); }
  finally { modelBusy = false; }
}

function showModelInfo(info) {
  modelEl("model-info").textContent = JSON.stringify(info, null, 2);
  if (info.recommended_output_tokens) modelEl("model-output").value = info.recommended_output_tokens;
}

async function refreshModels() {
  const result = await api("/admin/models");
  modelEl("model-jobs").replaceChildren();
  for (const job of result.jobs) {
    const row = document.createElement("div");
    row.textContent = `${job.action} · ${job.status} · ${job.stage}${job.error ? ` · ${job.error}` : ""}`;
    modelEl("model-jobs").append(row);
    if (job.result?.artifact) addArtifact(job.result);
  }
  modelEl("model-live").textContent = result.models.map((m) => `${m.id} · ${m.base_url}`).join("\n") || "尚未注册模型";
  modelEl("model-managed").textContent = result.managed ? `受管模型：${result.managed}` : "未启动受管模型";
  modelPolling = result.jobs.some((job) => job.status === "running");
}

function addArtifact(item) {
  const select = modelEl("model-artifact");
  if (Array.from(select.options).some((option) => option.value === item.artifact)) return;
  const option = document.createElement("option");
  option.value = item.artifact;
  option.textContent = `${item.alias} · ${item.metadata?.context_length || "未知"} 上下文 · ${item.artifact.slice(0, 8)}`;
  select.append(option);
  select.value = item.artifact;
  option.dataset.alias = item.alias;
  option.dataset.metadata = JSON.stringify(item.metadata || {});
}

modelEl("model-runtime").onclick = () => modelAction(async () => {
  modelMessage("正在检查本机运行环境…");
  const info = await api("/admin/mlx/runtime", { timeout: 30000 });
  showModelInfo(info);
  info.artifacts.forEach(addArtifact);
  modelMessage(info.ready ? "MLX 运行环境已就绪" : "环境尚未就绪，可点击安装运行环境");
  await refreshModels();
});
modelEl("model-setup").onclick = () => modelAction(async () => {
  await api("/admin/mlx/setup", { method: "POST" });
  modelMessage("已开始安装独立 MLX 环境，可在下方查看阶段");
  await refreshModels();
});
modelEl("model-preview").onclick = () => modelAction(async () => {
  showModelInfo(sourceForm());
  modelMessage("表单校验通过；此预览完全在浏览器内生成");
});
modelEl("model-inspect").onclick = () => modelAction(async () => {
  const source = sourceForm();
  modelMessage("正在读取模型配置并检查架构…");
  showModelInfo(await api("/admin/mlx/inspect", { method: "POST", body: JSON.stringify(source), timeout: 120000 }));
  modelMessage("模型配置校验通过，原生上下文已读取");
});
modelEl("model-prepare").onclick = () => modelAction(async () => {
  await api("/admin/mlx/prepare", { method: "POST", body: JSON.stringify(sourceForm()) });
  modelMessage("已开始下载、转换和自检；成功后在产物列表选择并启动");
  await refreshModels();
});
modelEl("model-launch").onclick = () => modelAction(async () => {
  const alias = modelEl("model-alias").value.trim();
  if (!/^[A-Za-z0-9_.-]{1,128}$/.test(alias)) throw new Error("模型别名格式无效");
  const artifact = modelEl("model-artifact").value;
  if (!artifact) throw new Error("请先选择通过自检的模型产物");
  modelMessage("正在加载模型，后端就绪后自动应用到 API…");
  await api("/admin/mlx/launch", { method: "POST", timeout: 180000, body: JSON.stringify({ artifact, alias,
    port: Number(modelEl("model-port").value), max_output_tokens: Number(modelEl("model-output").value) }) });
  modelMessage(`模型 ${alias} 已启动并应用`);
  await refreshModels();
});
modelEl("model-artifact").onchange = () => {
  const option = modelEl("model-artifact").selectedOptions[0];
  if (option.dataset.alias) modelEl("model-alias").value = option.dataset.alias;
  if (option.dataset.metadata) showModelInfo(JSON.parse(option.dataset.metadata));
};
modelEl("model-stop").onclick = () => modelAction(async () => {
  await api("/admin/mlx/stop", { method: "POST" });
  modelMessage("受管模型已停止");
  await refreshModels();
});
modelEl("model-file").onchange = () => modelAction(async () => {
  const file = modelEl("model-file").files[0];
  if (!file) return;
  if (file.size > 1024 * 1024) throw new Error("config.json 超过 1MiB");
  showModelInfo(LiteInferConfig.summarize(JSON.parse(await file.text())));
  modelMessage("本地 config.json 已在浏览器内校验，未上传文件");
});
setInterval(() => {
  if (modelPolling && !document.hidden) refreshModels().catch((error) => { modelPolling = false; modelMessage(error.message); });
}, 2000);
