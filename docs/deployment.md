# macOS 部署

## 前台运行

先启动模型后端，再在项目工作目录执行 `liteinfer --config config.toml serve`。配置中的相对数据库路径以进程工作目录为基准。不要在多个进程同时运行同一个服务配置；每个实例拥有独立的内存Key快照和调度上限。

Rust服务可在macOS与Linux构建；Metal/MLX由后端平台决定。`.github/workflow-templates/` 提供 macOS/Linux CI 和双架构 macOS 发布模板，当前尚未启用或执行。维护者使用具有 workflow 权限的凭据，将模板复制到 `.github/workflows/` 并提交后即可启用。发布模板会在推送版本标签时生成 Apple Silicon 和 Intel 压缩包。二进制暂未进行 Apple 签名或公证，企业分发需要自己的签名流程。

## launchd

可将 `examples/dev.liteinfer.plist` 复制到自己的 `~/Library/LaunchAgents/`，把 `/opt/liteinfer` 改成实际安装目录。此目录示例不是安装要求。使用 `ProgramArguments` 数组，不通过shell解释配置路径。

```sh
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/dev.liteinfer.plist
launchctl kickstart -k gui/$(id -u)/dev.liteinfer
# 停止自动服务
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/dev.liteinfer.plist
```

示例仅启用API Key数据库鉴权，不在plist写入管理Key。需要控制台时，推荐通过当前会话环境或自己维护的受保护服务配置注入管理Key。不要把实际凭据放入公开plist；launchd不自动继承交互shell环境，必须确认使用的环境变量已经注入。

模型后端与Rust服务分别管理生命周期。先停Rust入口，等待在途请求完成后再停止后端；SIGINT/SIGTERM触发优雅关闭，在途任务受配置总超时约束。启动时不会自动下载模型、运行模型提供的代码或打开公网隧道。

## 局域网与公网

默认只监听 `127.0.0.1`。需要局域网时显式设置 `listen = "0.0.0.0:8787"` 和 `allow_remote = true`，并配置主机防火墙、HTTPS反向代理。API不提供匿名推理，即使没有配置任何Key也只会返回401。

TLS在受控反向代理终止。流式连接应关闭代理缓冲，并设置足够的读超时；不要自动重试已经发送内容的工具调用。管理接口应由代理限制为本机或受信网络。内置CORS不放开跨域访问，控制台与API同源。

Cloudflare等隧道属于外部部署选项，项目不内置私人隧道URL或凭据。API服务不承诺临时URL永久可用。

## 更新与回滚

停止服务、备份本地配置和数据库、构建新版本、执行检查再启动。模型权重在后端管理。发布前不要把数据库、配置、日志或模型目录打进压缩包。`cargo build --release --locked` 使用锁文件固定依赖。
