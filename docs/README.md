# 文档导航

| 文档 | 内容 |
| --- | --- |
| [快速开始](../README.md) | 构建、启动、两种协议、控制台 |
| [macOS 安装与升级](install-macos.md) | Apple Silicon 安装包、校验和迁移 |
| [运行控制台](console-design.md) | 页面设计、推理统计与设备指标口径 |
| [内存与 SSD 分层缓存](ssd-cache.md) | 缓存配置、索引、延迟与隐私边界 |
| [配置与后端](configuration.md) | 参数、模型注册、MLX / llama.cpp / Ollama / LM Studio |
| [协议兼容](compatibility.md) | 字段、工具、流式事件、已知边界 |
| [架构与调度](architecture.md) | Rust 模块、并发、取消、缓存与 usage |
| [macOS 部署](deployment.md) | 前台、launchd、局域网与反向代理 |
| [测试与压测](testing.md) | 回归测试、复现方法、性能口径 |
| [迁移说明](migration.md) | 从专用 Python 服务迁移到通用服务 |
| [隐私与安全](../SECURITY.md) | 脱敏、密钥、日志和漏洞反馈 |

文档与代码一同版本管理。兼容矩阵描述当前版本；未实现的功能不会标记为支持。

- [MLX 推理优化与接入](mlx-optimization.md)：可选优化包、连续批处理、缓存、稀疏采样和适用范围。
- [网页模型管理与自动配置](model-management.md)：HuggingFace / ModelScope 镜像、安装运行环境、MLX 转换、模型校验和运行时应用。
- [Valen Decisions 本地 API](valen-decisions.md)：决策模型注册、多模态请求和 usage 处理。
- [Valen MLX 转换与基准](valen-mlx.md)：原生检查点转换、q8/q4 速度与误差、手写 Metal 算子验证。
- [Qwen3.5 small-M 实验与性能分析](qwen35-performance-analysis.md)：混合 4/8-bit 内核、交错 A/B、预填充与解码口径、数值风险及带宽模型。
