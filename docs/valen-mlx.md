# Valen 的 MLX 转换与性能实验

本实验从 Valen-Team/Valen-Decision-1.5-0.8B 原生 BF16 检查点直接映射 Qwen3.5 文本骨干权重，不经过 ONNX 反量化。原始检查点保留不动。转换器不执行模型仓库中的 Python 代码；决策 Mixer 以 FP32 sidecar 保存，视觉塔和 Grounding 权重以源精度保留在 valen_unconverted_components.safetensors。

**当前产物只支持 Qwen3.5 文本骨干前向，不是可直接接入 Decisions API 的完整 Valen MLX 服务。**MLX 多模态前处理、Valen 的 block attention mask、视频/图片路径以及完整决策请求编译尚未移植。本文的 token/s 是本机直接调用 MLX 模型的结果，不包含 HTTP、分词器服务、排队，也不代表 Valen 任务准确率。

## 转换

使用 Apple Silicon、Python 3.11+ 和仓库锁定的 MLX 版本：

~~~sh
python3 -m venv .venv
source .venv/bin/activate
pip install ./backends/mlx
hf download Valen-Team/Valen-Decision-1.5-0.8B \
  --revision e079395fe9b8e0763b384af403e636e91c1a1559 \
  --local-dir .liteinfer/valen-native

PYTHONPATH=backends/mlx python scripts/convert_valen_mlx.py \
  .liteinfer/valen-native .liteinfer/valen-mlx-bf16 --bits 0
PYTHONPATH=backends/mlx python scripts/convert_valen_mlx.py \
  .liteinfer/valen-native .liteinfer/valen-mlx-q8 --bits 8
PYTHONPATH=backends/mlx python scripts/convert_valen_mlx.py \
  .liteinfer/valen-native .liteinfer/valen-mlx-q4 --bits 4
~~~

转换器会校验源配置、保留原目录，并拒绝覆盖非空输出目录。MLX 主干权重按原生层名映射，检查点里的 FP32 Mixer 单独转换成 MLX 参数；没有被转换的视觉与定位组件会保存在 sidecar，供后续移植使用。

## M1 Pro 基准

测试日期为 2026-11-19。MacBook Pro M1 Pro、统一内存 16 GB、MLX 0.32.4、MLX-LM 0.32.2；固定 128 输入 token、32 输出 token，2 次预热、10 次计时。推理为单请求直接模型调用，生成计时不包含预填充。原始数据见 [benchmarks/valen-mlx-m1pro.json](../benchmarks/valen-mlx-m1pro.json)。

| 骨干精度 | 预填充吞吐 | 单请求解码吞吐 | 相对 BF16 的隐藏状态相对 L2 误差 |
| --- | ---: | ---: | ---: |
| BF16 | 1,827 输入 token/s | 73.6 输出 token/s | 基线 |
| q8，group 64 | 1,696 输入 token/s | 113.1 输出 token/s | 4.16% |
| q4，group 64 | 1,553 输入 token/s | 140.0 输出 token/s | 38.13% |

隐藏状态误差来自一个固定输入样例，只能比较量化偏差，不能替代验证集准确率。q4 解码最快，但误差显著更大；在没有决策集评测前，不能据此认定 q4 适合 Valen 决策任务。q8 解码约为 BF16 的 1.54 倍，质量风险较低，可作为后续验证候选。用占位角色区间对 Mixer 打分，q8 与 BF16 的前选项一致率为 93.75%，q4 为 75%；这些区间不是 Valen 编译器生成的真实特征，不能作为模型质量结论。

## 手写算子验证

liteinfer_mlx.valen_ops.depthwise_conv1d 为 Qwen3.5 线性注意力的深度卷积实现了直接 Metal kernel。q4 主干上替换 18 层后，最大绝对误差为 0；最终预填充为 1,576 token/s、解码为 139.5 token/s，原生 MLX 对照为 1,553 / 140.0 token/s。预填充略快、解码略慢，结果没有显示稳定收益，因此保留为可选实验路径，不自动启用。

Valen Mixer 另有 FP32 Metal LayerNorm 算子，并和标准 MLX 计算图、编译图对照。所有候选数上的数值最大误差均低于 8e-7；手写 LayerNorm 与标准实现、mx.compile 图的速度随候选数变化，没有稳定收益，均未设为默认路径。Mixer 单次约需 0.6–1.6 ms，而 32 token 主干解码约需 229–435 ms。

复跑完整基准：

~~~sh
PYTHONPATH=backends/mlx python scripts/benchmark_valen_mlx.py \
  --bf16-dir .liteinfer/valen-mlx-bf16 \
  --q8-dir .liteinfer/valen-mlx-q8 \
  --q4-dir .liteinfer/valen-mlx-q4 \
  --prompt-tokens 128 --generation-tokens 32 \
  --warmup 2 --trials 10 \
  --output benchmarks/valen-mlx-m1pro.json
~~~

算子正确性测试：

~~~sh
PYTHONPATH=backends/mlx python -m unittest discover \
  -s backends/mlx/tests -p 'test_valen.py' -v
~~~

后续接入 Decisions API 前，需要实现并验证原生 Valen 请求编译和 block mask，再用官方决策样例比较 BF16、q8、q4 的答案一致率、概率误差以及多模态输出。
