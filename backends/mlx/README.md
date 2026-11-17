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
