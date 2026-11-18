"""Valen Qwen 3.5 text-backbone conversion and MLX decision head."""
from __future__ import annotations

import json
from functools import lru_cache
from importlib.metadata import version
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_5 import Model, ModelArgs
from mlx_lm.utils import quantize_model, save_config, save_model
from mlx.utils import tree_flatten


class MixerBlock(nn.Module):
    def __init__(self, width: int, token_hidden: int, channel_hidden: int,
                 custom_layer_norm: bool = False) -> None:
        super().__init__()
        layer_norm = _MetalLayerNorm if custom_layer_norm else nn.LayerNorm
        self.token_norm = layer_norm(width, eps=1e-5)
        self.token_mlp = nn.Sequential(
            nn.Linear(4, token_hidden), nn.GELU(approx="none"), nn.Linear(token_hidden, 4)
        )
        self.channel_norm = layer_norm(width, eps=1e-5)
        self.channel_mlp = nn.Sequential(
            nn.Linear(width, channel_hidden),
            nn.GELU(approx="none"),
            nn.Linear(channel_hidden, width),
        )

    def __call__(self, inputs: mx.array) -> mx.array:
        mixed = self.token_mlp(self.token_norm(inputs).transpose(0, 2, 1)).transpose(0, 2, 1)
        inputs = inputs + mixed
        return inputs + self.channel_mlp(self.channel_norm(inputs))


class ValenMixerHead(nn.Module):
    """MLX implementation of Valen's four-role, candidate-wise Mixer."""

    def __init__(self, hidden_size: int, config: dict[str, int], custom_layer_norm: bool = False) -> None:
        super().__init__()
        width = config["head_width"]
        layer_norm = _MetalLayerNorm if custom_layer_norm else nn.LayerNorm
        self.input_norm = layer_norm(hidden_size, eps=1e-5)
        self.projection = nn.Linear(hidden_size, width)
        self.roles = mx.zeros((1, 4, width), dtype=mx.float32)
        blocks = [
            MixerBlock(width, config["head_token_hidden_dim"],
                       config["head_channel_hidden_dim"], custom_layer_norm)
            for _ in range(config["head_layers"])
        ]
        self.blocks = nn.Sequential(*blocks)
        self.output = nn.Sequential(layer_norm(width, eps=1e-5), nn.Linear(width, 1))

    def __call__(self, features: mx.array) -> mx.array:
        if features.ndim != 3 or features.shape[1] != 4:
            raise ValueError("Valen Mixer 输入必须是 [候选数, 4, hidden_size]")
        values = self.projection(self.input_norm(features.astype(mx.float32))) + self.roles
        values = self.blocks(values)
        return self.output(values[:, 2, :]).squeeze(-1)

    def compiled(self):
        """Return an MLX-compiled graph for repeated decision-head calls."""
        return mx.compile(lambda features: self(features))


class _MetalLayerNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = mx.ones((width,), dtype=mx.float32)
        self.bias = mx.zeros((width,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, values: mx.array) -> mx.array:
        return metal_layer_norm(values, self.weight, self.bias, self.eps)


@lru_cache(maxsize=16)
def _layer_norm_kernel(width: int, eps: float):
    source = f"""
        uint row = thread_position_in_grid.y;
        uint lane = thread_position_in_threadgroup.x;
        auto xrow = x + row * D;
        float sum = 0.0f;
        for (uint i = lane; i < D; i += 32) {{ sum += xrow[i]; }}
        sum = simd_sum(sum);
        float mean = sum / D;
        float square_sum = 0.0f;
        for (uint i = lane; i < D; i += 32) {{
          float centered = xrow[i] - mean;
          square_sum += centered * centered;
        }}
        float variance = simd_sum(square_sum) / D;
        float scale = rsqrt(variance + {eps:.12g}f);
        auto yrow = out + row * D;
        for (uint i = lane; i < D; i += 32) {{
          yrow[i] = (xrow[i] - mean) * scale * weight[i] + bias[i];
        }}
    """
    return mx.fast.metal_kernel(
        name="valen_layer_norm_f32",
        input_names=["x", "weight", "bias"],
        output_names=["out"],
        source=source,
    )


def metal_layer_norm(values: mx.array, weight: mx.array, bias: mx.array, eps: float) -> mx.array:
    """Single-kernel FP32 LayerNorm for Valen's inference-only Mixer head."""
    if values.ndim < 1 or values.shape[-1] != weight.size or bias.size != weight.size:
        raise ValueError("LayerNorm 权重与输入形状不匹配")
    if values.dtype != mx.float32 or weight.dtype != mx.float32 or bias.dtype != mx.float32:
        raise ValueError("Valen Metal LayerNorm 仅接受 FP32")
    if not mx.metal.is_available():
        raise RuntimeError("Valen Metal LayerNorm 需要 Apple Metal GPU")
    width = values.shape[-1]
    rows = values.size // width
    kernel = _layer_norm_kernel(width, float(eps))
    result = kernel(
        inputs=[values.reshape(rows, width), weight, bias],
        template=[("D", width)],
        grid=(32, rows, 1), threadgroup=(32, 1, 1),
        output_shapes=[(rows, width)], output_dtypes=[mx.float32],
    )[0]
    return result.reshape(values.shape)


def _read_config(source: Path) -> dict:
    config_path = source / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"缺少模型配置：{config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "valen_qwen" or "text_config" not in config:
        raise ValueError("源模型不是受支持的 Valen Qwen 检查点")
    if not (source / "model.safetensors").is_file():
        raise FileNotFoundError(f"缺少原生权重：{source / 'model.safetensors'}")
    return config


def _mlx_head_name(name: str) -> str:
    if name.startswith("blocks."):
        parts = name.split(".")
        if len(parts) < 4 or not parts[1].isdigit():
            raise ValueError(f"无效的 Mixer 权重名称：{name}")
        module_name = parts[2]
        module_path = f"blocks.layers.{parts[1]}.{module_name}"
        if module_name in ("token_mlp", "channel_mlp"):
            module_path += ".layers"
        return module_path + "." + ".".join(parts[3:])
    if name.startswith("output."):
        return "output.layers." + name[len("output."):]
    return name


def _decision_weights(raw: dict[str, mx.array]) -> tuple[dict[str, mx.array], dict[str, mx.array]]:
    head, sidecar = {}, {}
    for key, value in raw.items():
        if key.startswith("head."):
            head[_mlx_head_name(key[len("head."):])] = value
        elif key.startswith("backbone.visual."):
            sidecar["vision." + key[len("backbone.visual."):]] = value
        elif key.startswith("grounding."):
            sidecar[key] = value
    if not head:
        raise ValueError("原生检查点中没有 Valen 决策头权重")
    return head, sidecar


def _make_head(config: dict[str, object], raw: dict[str, mx.array]) -> ValenMixerHead:
    head_config = config.get("valen_head", {})
    if not isinstance(head_config, dict):
        raise ValueError("Valen Mixer 配置无效或缺失")
    required = ("head_width", "head_layers", "head_token_hidden_dim", "head_channel_hidden_dim")
    if any(type(head_config.get(key)) is not int or head_config[key] <= 0 for key in required):
        raise ValueError("Valen Mixer 配置无效或缺失")
    head = ValenMixerHead(config["text_config"]["hidden_size"], head_config)
    head_weights, _ = _decision_weights(raw)
    head.load_weights(list(head_weights.items()), strict=True)
    head.eval()
    return head


def load_valen_weights(source: Path) -> tuple[Model, ValenMixerHead, dict[str, object], dict[str, mx.array]]:
    """Load official BF16 backbone and FP32 Mixer weights without executing remote code."""
    source = source.expanduser().resolve()
    config = _read_config(source)
    model = Model(ModelArgs.from_dict({
        "model_type": "qwen3_5", "text_config": config["text_config"]
    }))
    raw = mx.load(str(source / "model.safetensors"))
    backbone = {
        "model.language_model" + key[len("backbone.language_model"):]: value
        for key, value in raw.items()
        if key.startswith("backbone.language_model.")
    }
    if not backbone:
        raise ValueError("原生检查点中没有 Qwen3.5 文本骨干权重")
    model.load_weights(list(model.sanitize(backbone).items()), strict=True)
    model.eval()
    head = _make_head(config, raw)
    _, sidecar = _decision_weights(raw)
    del raw, backbone
    return model, head, config, sidecar


def _mlx_config(source_config: dict, bits: int) -> dict:
    config = dict(source_config)
    for key in ("auto_map", "valen_head", "valen_grounding", "valen_media_kwargs",
                "valen_full_attention_mask", "valen_decision_version", "qwen_execution",
                "valen_max_length", "valen_backbone_autocast_dtype", "vision_config",
                "image_token_id", "video_token_id", "vision_start_token_id",
                "vision_end_token_id"):
        config.pop(key, None)
    config["model_type"] = "qwen3_5"
    config["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    config["dtype"] = "bfloat16"
    config["eos_token_id"] = source_config["text_config"].get("eos_token_id")
    if bits:
        config["quantization"] = {"group_size": 64, "bits": bits, "mode": "affine"}
        config["quantization_config"] = config["quantization"]
    return config


def _save_conversion(source: Path, output: Path, model: Model, head: ValenMixerHead,
                     sidecar: dict[str, mx.array], config: dict[str, object],
                     mlx_config: dict[str, object], bits: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    save_model(output, model, donate_model=False)
    save_config(mlx_config, output / "config.json")
    mx.save_safetensors(str(output / "valen_head.safetensors"), dict(tree_flatten(head.parameters())),
                        metadata={"format": "mlx", "head": "valen-mixer-fp32"})
    if sidecar:
        mx.save_safetensors(str(output / "valen_unconverted_components.safetensors"), sidecar,
                            metadata={"format": "source-dtype", "status": "retained-not-MLX-runtime"})
    _copy_source_files(source, output)
    _write_manifest(output, config, bits)


def _copy_source_files(source: Path, output: Path) -> None:
    names = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json")
    for name in names:
        path = source / name
        if path.is_file():
            shutil.copy2(path, output / name)
    shutil.copy2(source / "config.json", output / "valen_source_config.json")


def _write_manifest(output: Path, config: dict[str, object], bits: int) -> None:
    manifest = {
        "source_model": "Valen-Team/Valen-Decision-1.5-0.8B",
        "source_model_commit": "e079395fe9b8e0763b384af403e636e91c1a1559",
        "backbone": "Qwen3.5 text path converted; vision tower retained in original dtype",
        "backbone_dtype": "bfloat16" if bits == 0 else f"affine-{bits}bit-g64",
        "decision_head": "valen_head.safetensors (FP32)",
        "head_config": config["valen_head"],
        "grounding_head": "retained in original dtype; not yet MLX-executable",
        "unconverted_sidecar": "valen_unconverted_components.safetensors",
        "mlx": version("mlx"),
        "mlx_lm": version("mlx-lm"),
    }
    (output / "valen_conversion.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def convert_valen(source: Path, output: Path, bits: int = 0) -> Path:
    """Convert Valen text weights to MLX; store the FP32 Mixer as a sidecar."""
    if bits not in (0, 4, 8):
        raise ValueError("bits 只支持 0（BF16）、4 或 8")
    source, output = source.expanduser().resolve(), output.expanduser().resolve()
    if output == source or source in output.parents:
        raise ValueError("输出目录不能覆盖源权重或位于源目录内")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("输出目录必须不存在或为空目录")
    config = _read_config(source)
    model, head, _, sidecar = load_valen_weights(source)
    mlx_config = _mlx_config(config, bits)
    if bits:
        model, mlx_config = quantize_model(model, mlx_config, 64, bits)
    _save_conversion(source, output, model, head, sidecar, config, mlx_config, bits)
    return output
