(function (root) {
  "use strict";
  const U = root.LiteInferUI;
  function render(keys) {
    const target = U.el("keys"); target.replaceChildren();
    U.text("keys-summary", `${keys.filter((key) => !key.revoked).length} 个有效密钥 · ${keys.filter((key) => key.revoked).length} 个已撤销`);
    if (!keys.length) { U.emptyRow(target, 5, "尚未创建访问密钥"); return; }
    for (const key of keys) {
      const row = U.node("tr"), name = U.node("td"), status = U.node("td"), action = U.node("td");
      name.append(U.node("strong", "", key.name)); status.append(U.badge(key.revoked ? "已撤销" : "有效", key.revoked ? "neutral" : ""));
      if (!key.revoked) {
        const button = U.node("button", "text-button danger-text", "撤销");
        button.onclick = () => revoke(key); action.append(button);
      } else action.textContent = "—";
      row.append(name, U.node("td", "mono", key.id), status, U.node("td", "", U.datetime(key.created)), action); target.append(row);
    }
  }
  async function revoke(key) {
    const C = root.LiteInferConsole;
    if (!await C.confirm({ title: `撤销「${key.name}」？`, description: "该密钥将立即失效，使用它的应用无法再发起新请求。此操作不可恢复。", accept: "确认撤销" })) return;
    try {
      await C.api(`/admin/keys/${encodeURIComponent(key.id)}`, { method: "DELETE" });
      U.el("secret").textContent = ""; U.el("secret-panel").hidden = true;
      C.toast("访问密钥已撤销"); await C.refresh();
    } catch (error) { C.toast(error.message, true); }
  }
  async function create(event) {
    event.preventDefault(); const C = root.LiteInferConsole, button = U.el("create");
    const name = U.el("name").value.trim(); U.text("key-error", "");
    if (!name || new TextEncoder().encode(name).length > 128) { U.text("key-error", "名称需要 1–128 字节"); return; }
    button.disabled = true;
    try {
      const result = await C.api("/admin/keys", { method: "POST", body: JSON.stringify({ name }) });
      U.text("secret", result.key); U.el("secret-panel").hidden = false; U.el("key-dialog").close();
      U.el("name").value = ""; C.toast("密钥已创建，请立即复制保存"); await C.refresh();
    } catch (error) { U.text("key-error", error.message); }
    finally { button.disabled = false; }
  }
  function init() {
    U.el("key-create-open").onclick = () => { if (root.LiteInferConsole.requireConnection()) U.el("key-dialog").showModal(); };
    U.el("key-form").onsubmit = create;
    U.el("secret-close").onclick = () => { U.text("secret", ""); U.el("secret-panel").hidden = true; };
    U.el("secret-copy").onclick = () => root.LiteInferConsole.copy(U.el("secret").textContent);
    render([]);
  }
  root.LiteInferKeys = Object.freeze({ init, render });
})(window);
