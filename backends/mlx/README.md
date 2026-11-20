# LiteInfer MLX 可选组件

这是实际执行 MLX 张量推理的可选 Python 包；Rust 网关不依赖它。包含连续批处理引擎、编译稀疏采样、前缀 KV 缓存、自适应预填充、停止串过滤和可选 Llama 投影融合。

Apple Silicon、Python 3.11+，固定验证版本 MLX 0.32.4 / MLX-LM 0.32.2。安装、可运行示例、服务接入和限制见 [中文文档](../../docs/mlx-optimization.md)。

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install ./backends/mlx
liteinfer-mlx-batch --model ./models/mlx-chat \
  --prompt '你好' --prompt '介绍一下自己' --concurrency 3 --max-tokens 128
python -m unittest discover -s backends/mlx/tests -v
```

`--prompt` 可重复。CLI 使用模型聊天模板，输出逐 token JSON 事件和统计；模型文件与分词器分开时可加 `--tokenizer`。缺聊天模板时明确报错，仅在确实需要续写时用 `--raw`。

这不是独立 HTTP 服务，也不是 MLX-LM server 的自动补丁。应在自建后端的单一 GPU 工作线程内调用 `BatchEngine`，由 HTTP 层处理鉴权、SSE、超时与取消。`BatchEngine` 实例绑定一个模型，多模型可各建实例；各实例并发之和仍共享同一 GPU 预算。

仓库根目录还提供两个可复用脚本：`scripts/convert_mlx.py` 用于标准 MLX-LM q4/q8 转换，`scripts/benchmark_mlx_http.py` 用于 OpenAI 兼容 HTTP 的并发、p95、TTFT 和后端 token/s 测试。

## Qwen3.5 small-M 实验

`liteinfer-mlx-batch --qwen35-small-m` 显式开启 M=6–8 的混合 4/8-bit Metal
矩阵内核，同时关闭不支持递归状态的前缀缓存；单发和不支持的形状使用原生算子。
该选项默认关闭，不能与 `--fuse-llama` 同时使用。它不会自动接入 Rust 网关
或 MLX-LM HTTP server。

M1 Pro 的 5 轮交错 A/B 中，small-M 单发 151.3、8 发合计 442.8 解码 token/s；
8 发相对原生 301.3 提升约 46.9%。单发走原生回退，不将短测差异计为优化。
batch 6/8 的 greedy 输出仍有差异，尚未通过质量验收，也未达到 300/1600 目标。
原始数据与限制见 [性能分析](../../docs/qwen35-performance-analysis.md)。

在仓库根目录、安装本包的独立 Apple Silicon 环境中复现：

```sh
python backends/mlx/benchmark_qwen35.py \
  --model ./models/Qwen3.5-0.8B-OptiQ-4bit \
  --model-label mlx-community/Qwen3.5-0.8B-OptiQ-4bit \
  --optimization small-m --mode compare --batches 1 6 8 \
  --prompt-tokens 128 --generation-tokens 64 --warmups 2 --trials 5 \
  --output benchmarks/qwen35-smallm-reproduction.json

python -m unittest discover -s backends/mlx/tests -v
python -m pytest backends/mlx/tests/test_qwen35_fusion.py -q
```

运行测试前需在同一环境安装 `pytest`，因为测试发现时也会导入投影融合检查。
基准使用固定种子的任意 token 输入，仅用于性能和数值诊断，不能替代文本任务
准确率评测。当前脚本也支持非默认的投影融合诊断，已有失败结果保留在报告中。
