# macOS 安装

安装包适用于 Apple Silicon（arm64），最低 macOS 14。安装包包含网关、内嵌控制台和示例配置。部署本地 MLX 模型另需 Python 3.11+ 和模型文件，可在控制台的模型管理中安装运行环境；连接已有推理后端不需要 Python。

1. 从 GitHub Release 下载同名的 `.pkg` 与 `.pkg.sha256`，在下载目录运行 `shasum -a 256 -c LiteInfer-1.0.0-macOS-arm64.pkg.sha256`。
2. 打开安装包，按 macOS Installer 指引安装。此版本未经过 Developer ID 签名或 Apple 公证，可能被 Gatekeeper 拦截。需要签名部署时，可从发布标签自行构建。
3. 在工作目录以普通用户启动。默认配置及状态保存在此目录，终端显示管理密钥。

```bash
mkdir -p "$HOME/Library/Application Support/LiteInfer"
cd "$HOME/Library/Application Support/LiteInfer"
/usr/local/bin/liteinfer serve
```

在浏览器打开 `http://127.0.0.1:8787`，输入终端显示的管理密钥。

升级前停止旧网关并备份原工作目录中的 `config.toml` 和 SQLite 数据库。安装器仅替换网关、文档和示例。从原工作目录启动新版本，数据库会自动添加监控字段；回退到 0.3.0 时请使用升级前的数据库备份。

在 Apple Silicon Mac 上运行 `bash scripts/package-macos.sh` 可重建未签名包，产物位于 `target/packages/`。
