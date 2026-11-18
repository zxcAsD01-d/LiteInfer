"""Experimental Metal operators for Valen's Qwen 3.5 inference path."""
from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn


@lru_cache(maxsize=1)
def _depthwise_kernel():
    source = """
        uint index = thread_position_in_grid.y * 256 + thread_position_in_grid.x;
        uint total = B * S * C;
        if (index < total) {
          uint channel = index % C;
          uint step = (index / C) % S;
          uint batch = index / (C * S);
          auto row = x + (batch * (S + K - 1) + step) * C + channel;
          float value = 0.0f;
          for (uint tap = 0; tap < K; ++tap) {
            value += static_cast<float>(row[tap * C]) *
                     static_cast<float>(weight[channel * K + tap]);
          }
          out[index] = static_cast<T>(value);
        }
    """
    return mx.fast.metal_kernel(
        name="valen_depthwise_conv1d",
        input_names=["x", "weight"],
        output_names=["out"],
        source=source,
    )


def depthwise_conv1d(inputs: mx.array, weight: mx.array) -> mx.array:
    """Run a valid, channels-last depthwise Conv1d with MLX's Metal kernel."""
    if inputs.ndim != 3 or weight.ndim != 3 or weight.shape[-1] != 1:
        raise ValueError("需要 [batch, time, channels] 输入和 [channels, kernel, 1] 权重")
    batch, input_steps, channels = inputs.shape
    out_channels, kernel, _ = weight.shape
    if channels != out_channels or kernel <= 0 or input_steps < kernel:
        raise ValueError("深度卷积通道或时序长度无效")
    if inputs.dtype != weight.dtype:
        raise ValueError("深度卷积输入与权重精度必须相同")
    if not mx.metal.is_available():
        layer = nn.Conv1d(channels, channels, kernel, groups=channels, bias=False)
        layer.weight = weight
        return layer(inputs)
    steps = input_steps - kernel + 1
    output = _depthwise_kernel()(
        inputs=[inputs, weight],
        template=[("B", batch), ("S", steps), ("C", channels), ("K", kernel),
                  ("T", inputs.dtype)],
        grid=(256, (batch * steps * channels + 255) // 256, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, steps, channels)],
        output_dtypes=[inputs.dtype],
    )[0]
    return output


class MLXDepthwiseConv1d(nn.Module):
    """Drop-in replacement for the linear-attention depthwise convolution."""

    def __init__(self, weight: mx.array) -> None:
        super().__init__()
        if weight.ndim != 3 or weight.shape[-1] != 1:
            raise ValueError("卷积权重必须为 [channels, kernel, 1]")
        self.weight = weight

    def __call__(self, inputs: mx.array) -> mx.array:
        return depthwise_conv1d(inputs, self.weight)


def replace_valen_depthwise_convs(model: nn.Module) -> int:
    """Replace Qwen3.5 linear-attention convolutions in a loaded MLX model."""
    text_model = getattr(getattr(model, "language_model", None), "model", None)
    if text_model is None or not hasattr(text_model, "layers"):
        raise ValueError("模型不是 MLX-LM Qwen3.5 文本骨干")
    replaced = 0
    for layer in text_model.layers:
        if getattr(layer, "is_linear", False):
            layer.linear_attn.conv1d = MLXDepthwiseConv1d(layer.linear_attn.conv1d.weight)
            replaced += 1
    if replaced == 0:
        raise ValueError("模型没有可替换的线性注意力层")
    return replaced
