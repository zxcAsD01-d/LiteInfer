"""Inference-only projection packing for the dense MLX Qwen3.5 text model."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from collections.abc import Hashable
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models import gated_delta
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.qwen3_5 import GatedDeltaNet, Model
from mlx_lm.models.qwen3_next import Qwen3NextAttention, Qwen3NextMLP

Linear = nn.Linear | nn.QuantizedLinear | nn.QQLinear


def _projection_key(layer: Linear) -> Hashable:
    return (type(layer), layer.weight.shape[1:], getattr(layer, "group_size", None),
            getattr(layer, "bits", None), getattr(layer, "mode", None),
            layer.get("bias") is not None, layer.get("biases") is not None)


def pack_linears(layers: list[Linear]) -> Linear:
    """Concatenate output rows without dequantization or changing groups."""
    if not layers or type(layers[0]) not in (nn.Linear, nn.QuantizedLinear, nn.QQLinear):
        raise TypeError("Projection packing requires supported dense Linear layers")
    first = layers[0]
    quantized = isinstance(first, (nn.QuantizedLinear, nn.QQLinear))
    for layer in layers:
        if type(layer) is not type(first) or layer.weight.shape[1:] != first.weight.shape[1:]:
            raise ValueError("Projection types and input shapes must match")
        if layer.training or (layer.get("bias") is None) != (first.get("bias") is None):
            raise ValueError("Projection packing requires eval mode and matching biases")
        if quantized and (layer.group_size, layer.bits, layer.mode) != (first.group_size, first.bits, first.mode):
            raise ValueError("Quantization settings must match")
    packed = type(first).__new__(type(first))
    nn.Module.__init__(packed)
    packed.weight = mx.concatenate([layer.weight for layer in layers], axis=0)
    if quantized:
        packed.group_size, packed.bits, packed.mode = first.group_size, first.bits, first.mode
        packed.scales = mx.concatenate([layer.scales for layer in layers], axis=0)
        biases = [layer.get("biases") for layer in layers]
        if any(b is None for b in biases) and not all(b is None for b in biases):
            raise ValueError("Quantization biases must match")
        if biases[0] is not None:
            packed.biases = mx.concatenate(biases, axis=0)
        if isinstance(first, nn.QQLinear):
            packed._quantized = True
    if first.get("bias") is not None:
        packed.bias = mx.concatenate([layer.bias for layer in layers], axis=0)
    return packed.eval()


class GroupedLinear(nn.Module):
    """Pack compatible projections together and restore their original row order."""

    def __init__(self, layers: list[Linear]):
        super().__init__()
        if not layers:
            raise ValueError("At least one projection is required")
        grouped: dict[Hashable, list[tuple[int, Linear]]] = {}
        for index, layer in enumerate(layers):
            grouped.setdefault(_projection_key(layer), []).append((index, layer))
        self.packed: list[Linear] = []
        self.members: list[list[tuple[int, int]]] = []
        for compatible in grouped.values():
            members = [layer for _, layer in compatible]
            self.packed.append(pack_linears(members))
            self.members.append([(index, layer.weight.shape[0]) for index, layer in compatible])
        self.output_count = len(layers)

    def __call__(self, values: mx.array) -> mx.array:
        if len(self.packed) == 1:
            return self.packed[0](values)
        outputs: list[mx.array | None] = [None] * self.output_count
        for packed, members in zip(self.packed, self.members, strict=True):
            boundaries = []
            total = 0
            for _, width in members[:-1]:
                total += width
                boundaries.append(total)
            chunks = mx.split(packed(values), boundaries, axis=-1)
            for (index, _), chunk in zip(members, chunks, strict=True):
                outputs[index] = chunk
        if any(output is None for output in outputs):
            raise RuntimeError("A packed projection did not produce every output")
        return mx.concatenate(outputs, axis=-1)


class PackedMLP(nn.Module):
    def __init__(self, previous: Qwen3NextMLP):
        super().__init__()
        self.gate_up = GroupedLinear([previous.gate_proj, previous.up_proj])
        self.down_proj = previous.down_proj

    def __call__(self, values: mx.array) -> mx.array:
        gate, up = mx.split(self.gate_up(values), 2, axis=-1)
        return self.down_proj(swiglu(gate, up))


class PackedAttention(nn.Module):
    def __init__(self, previous: Qwen3NextAttention):
        super().__init__()
        self.qkv = GroupedLinear([previous.q_proj, previous.k_proj, previous.v_proj])
        for name in ("num_attention_heads", "num_key_value_heads", "head_dim", "scale",
                     "o_proj", "q_norm", "k_norm", "rope"):
            setattr(self, name, getattr(previous, name))

    def __call__(self, values: mx.array, mask=None, cache=None) -> mx.array:
        batch, length, _ = values.shape
        q_end = self.num_attention_heads * self.head_dim * 2
        k_end = q_end + self.num_key_value_heads * self.head_dim
        query, key, value = mx.split(self.qkv(values), [q_end, k_end], axis=-1)
        query, gate = mx.split(query.reshape(batch, length, self.num_attention_heads, -1), 2, axis=-1)
        gate = gate.reshape(batch, length, -1)
        query = self.q_norm(query).transpose(0, 2, 1, 3)
        key = self.k_norm(key.reshape(batch, length, self.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        value = value.reshape(batch, length, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        offset = cache.offset if cache is not None else 0
        query, key = self.rope(query, offset=offset), self.rope(key, offset=offset)
        if cache is not None:
            key, value = cache.update_and_fetch(key, value)
        output = scaled_dot_product_attention(query, key, value, cache=cache, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output * mx.sigmoid(gate))


class PackedDeltaNet(nn.Module):
    def __init__(self, previous: GatedDeltaNet):
        super().__init__()
        self.input_proj = GroupedLinear([previous.in_proj_qkv, previous.in_proj_z,
                                        previous.in_proj_b, previous.in_proj_a])
        for name in ("num_v_heads", "num_k_heads", "head_k_dim", "head_v_dim", "key_dim",
                     "value_dim", "conv_dim", "conv_kernel_size", "conv1d",
                     "A_log", "dt_bias", "norm", "out_proj"):
            setattr(self, name, getattr(previous, name))

    def _convolve(self, qkv: mx.array, mask, cache) -> mx.array:
        batch, length, _ = qkv.shape
        previous = cache[0] if cache is not None and cache[0] is not None else mx.zeros(
            (batch, self.conv_kernel_size - 1, self.conv_dim), dtype=qkv.dtype)
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        inputs = mx.concatenate([previous, qkv], axis=1)
        if cache is not None:
            keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, length)
                positions = (ends[:, None] + mx.arange(keep))[..., None]
                cache[0] = mx.take_along_axis(inputs, positions, axis=1)
            else:
                cache[0] = mx.contiguous(inputs[:, -keep:, :])
        return nn.silu(self.conv1d(inputs))

    def __call__(self, inputs: mx.array, mask=None, cache=None) -> mx.array:
        batch, length, _ = inputs.shape
        qkv_end, z_end = self.conv_dim, self.conv_dim + self.value_dim
        qkv, z, b, a = mx.split(self.input_proj(inputs), [qkv_end, z_end, z_end + self.num_v_heads], axis=-1)
        z = z.reshape(batch, length, self.num_v_heads, self.head_v_dim)
        conv = self._convolve(qkv, mask, cache)
        q, k, v = [part.reshape(batch, length, heads, dim) for part, heads, dim in zip(
            mx.split(conv, [self.key_dim, 2 * self.key_dim], axis=-1),
            [self.num_k_heads, self.num_k_heads, self.num_v_heads],
            [self.head_k_dim, self.head_k_dim, self.head_v_dim], strict=True)]
        inverse = k.shape[-1] ** -0.5
        if hasattr(gated_delta, "normalize_qk"):
            q, k = gated_delta.normalize_qk(q, k, inv_scale=inverse, eps=1e-6)
        else:
            q = (inverse ** 2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inverse * mx.fast.rms_norm(k, None, 1e-6)
        state = cache[1] if cache is not None else None
        output, state = gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias,
                                          state, mask, use_kernel=True)
        if cache is not None:
            cache[1] = state
            cache.advance(length)
        output = self.norm(output, z).reshape(batch, length, -1)
        return self.out_proj(output)


def fuse_qwen35(lm: Model) -> dict[str, int]:
    """Fuse a loaded eval model in memory; save canonical weights beforehand."""
    if type(lm) is not Model or lm.training:
        raise TypeError("Fusion requires an eval-mode MLX Qwen3.5 model")
    replacements = []
    calls_removed = 0
    for layer in lm.layers:
        attention = layer.linear_attn if layer.is_linear else layer.self_attn
        expected = GatedDeltaNet if layer.is_linear else Qwen3NextAttention
        if type(attention) is not expected or type(layer.mlp) is not Qwen3NextMLP:
            raise TypeError("Fusion requires unmodified dense Qwen3.5 layers")
        if getattr(attention, "sharding_group", None) is not None:
            raise ValueError("Distributed sharding is not supported by projection packing")
        inputs = ([attention.in_proj_qkv, attention.in_proj_z, attention.in_proj_b, attention.in_proj_a]
                  if layer.is_linear else [attention.q_proj, attention.k_proj, attention.v_proj])
        calls_removed += len(inputs) - len({_projection_key(projection) for projection in inputs})
        mlp_projections = (layer.mlp.gate_proj, layer.mlp.up_proj)
        calls_removed += len(mlp_projections) - len({_projection_key(projection)
                                                       for projection in mlp_projections})
        packed = PackedDeltaNet(attention) if layer.is_linear else PackedAttention(attention)
        replacements.append((packed, PackedMLP(layer.mlp)))
    for layer, (attention, mlp) in zip(lm.layers, replacements, strict=True):
        if layer.is_linear:
            layer.linear_attn = attention
        else:
            layer.self_attn = attention
        layer.mlp = mlp
    lm.eval()
    mx.eval(lm.parameters())
    return {"layers": len(replacements), "matrix_calls_removed_per_forward": calls_removed}
