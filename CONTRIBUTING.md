# 贡献指南

提交前说明具体问题、预期行为和验证方法。改动只处理相关问题；函数保持单一职责，复杂逻辑增加流程图或模块说明。生产代码不吞错误，不把模拟后端作为真实推理性能证明。

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --all-targets --locked
cargo build --release --locked
python3 scripts/privacy_check.py
```

协议与调度改动需要覆盖成功、失败和取消路径；修复UTF-8/SSE时增加跨字节边界用例。用户可见能力变化同步更新中文文档与兼容矩阵。PR应说明具体行为变化与实际完成的测试，未运行的检查要明确标注。

禁止提交实际Key、个人主目录路径、请求数据库、私有隧道地址、用户素材和模型权重。使用通用相对路径与占位环境变量。

此项目的本地管理HTTP接口会即时刷新Key快照；离线CLI修改数据库后需重启服务。请不要让多个独立服务进程共同管理同一Key数据库。

修改 `backends/mlx` 时，在 Apple Silicon 的独立 Python 环境安装该包并执行 `python -m unittest discover -s backends/mlx/tests -v`。涉及私有 MLX-LM 批处理契约或投影融合时，保持依赖锁定并验证 full/incremental logits、缓存分叉、取消和真实权重示例；不要把这类检查混入不具备 Metal 的 Linux Rust 作业。
