"""Fuse projections without changing attention, quantization groups or weights."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.cache import KVCache, BatchKVCache
from mlx_lm.models.llama import Attention, MLP, Model

Linear = nn.Linear | nn.QuantizedLinear


def merge_linears(layers: list[Linear]) -> Linear:
    if not layers or type(layers[0]) not in (nn.Linear, nn.QuantizedLinear):
        raise TypeError("Only plain Linear / QuantizedLinear projections are supported")
    first = layers[0]
    for layer in layers:
        if type(layer) is not type(first) or layer.weight.shape[1:] != first.weight.shape[1:]:
            raise ValueError("Projection types and input shapes must match")
        if ("bias" in layer) != ("bias" in first):
            raise ValueError("Projection bias configuration must match")
        if isinstance(first, nn.QuantizedLinear):
            if (layer.group_size, layer.bits, layer.mode) != (first.group_size, first.bits, first.mode):
                raise ValueError("Quantization settings must match")
            if first.mode != "affine":
                raise ValueError("Fusion currently supports affine quantization only")
    outputs = sum(layer.weight.shape[0] for layer in layers)
    if isinstance(first, nn.QuantizedLinear):
        merged = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
        nn.Module.__init__(merged)
        merged.group_size, merged.bits, merged.mode = first.group_size, first.bits, first.mode
        merged.scales = mx.concatenate([layer.scales for layer in layers])
        merged.biases = mx.concatenate([layer.biases for layer in layers])
    else:
        merged = nn.Linear(first.weight.shape[1], outputs, bias=False)
    merged.weight = mx.concatenate([layer.weight for layer in layers])
    if "bias" in first:
        merged.bias = mx.concatenate([layer.bias for layer in layers])
    return merged


class FusedAttention(nn.Module):
    def __init__(self, previous: Attention) -> None:
        super().__init__()
        self.qkv_proj = merge_linears([previous.q_proj, previous.k_proj, previous.v_proj])
        self.o_proj, self.rope = previous.o_proj, previous.rope
        self.n_heads, self.n_kv_heads = previous.n_heads, previous.n_kv_heads
        self.head_dim, self.scale = previous.head_dim, previous.scale

    def __call__(self, x: mx.array, mask: mx.array | None = None,
                 cache: KVCache | BatchKVCache | None = None) -> mx.array:
        batch, length, _ = x.shape
        q_end = self.n_heads * self.head_dim
        k_end = q_end + self.n_kv_heads * self.head_dim
        queries, keys, values = mx.split(self.qkv_proj(x), [q_end, k_end], axis=-1)
        queries = queries.reshape(batch, length, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(batch, length, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(batch, length, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        offset = cache.offset if cache is not None else 0
        queries, keys = self.rope(queries, offset=offset), self.rope(keys, offset=offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        result = scaled_dot_product_attention(queries, keys, values, cache=cache,
                                               scale=self.scale, mask=mask)
        return self.o_proj(result.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class FusedMLP(nn.Module):
    def __init__(self, previous: MLP) -> None:
        super().__init__()
        self.gate_up = merge_linears([previous.gate_proj, previous.up_proj])
        self.down_proj = previous.down_proj

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.gate_up(x), 2, axis=-1)
        return self.down_proj(swiglu(gate, up))


def fuse_model(model: Model) -> None:
    if type(model) is not Model or any(type(layer.self_attn) is not Attention
                                      or type(layer.mlp) is not MLP for layer in model.layers):
        raise TypeError("Fusion requires the unmodified mlx-lm Llama implementation")
    # Build all replacements before mutating the model, so validation is atomic.
    replacements = [(FusedAttention(layer.self_attn), FusedMLP(layer.mlp)) for layer in model.layers]
    for layer, (attention, mlp) in zip(model.layers, replacements):
        layer.self_attn, layer.mlp = attention, mlp
    mx.eval(model.parameters())
