# 第三方来源与许可

以下条目明确根软件许可的例外，不替代上游完整条款或全部分发义务。

- Rust 依赖：版本见 Cargo.toml、Cargo.lock，依各 crate 的原始许可和署名使用；根 LICENSE 不替代这些许可。
- macOS 安装包附有目标平台 Rust 依赖许可清单及 Rust 标准库版权声明。
- 控制台图标源自 Lucide，许可见 `assets/LUCIDE-LICENSE.txt`（安装包内为 `LUCIDE-LICENSE.txt`）。
- MLX、MLX-LM 及可选推理后端：由用户单独安装，适用其各自条款，版本见 backends/mlx/pyproject.toml。
- Qwen 等模型权重：按具体模型页及权利人声明自行获取；本仓库不含模型、用户素材、私人配置或历史请求。
- 历史主项目 MIT 授权保持有效，见 [原文](licenses/legacy/MIT.txt) 和 [版本边界](LICENSING.md)。

## Qwen3.5 实验内核

`backends/mlx/liteinfer_mlx/qwen35_small_m.py` 的 4-bit small-M Metal 内核
改编自 MLX-LM 的 `fast_qmm.py`，8-bit 内核及 Python 接入为本项目新增。
`qwen35_fusion.py` 的注意力、DeltaNet 与 MLP 计算改编自 MLX-LM 0.31.3，
本项目增加仅推理使用的混合位宽投影分组。上述改编代码保留上游 MIT 许可，
不由根 Apache-2.0 许可替代；完整声明见
[随包分发的 MIT 来源与条款](backends/mlx/liteinfer_mlx/QMM_MIT_NOTICE.txt)。
