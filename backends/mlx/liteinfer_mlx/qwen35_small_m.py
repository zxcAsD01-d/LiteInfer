"""Small-batch affine QMM kernels for the mixed-bit Qwen3.5 checkpoint.

The 4-bit tile follows the MLX-LM small-M Metal kernel under its MIT license.
The 8-bit tile is an equivalent extension for the tied vocabulary projection.
"""
from __future__ import annotations

from math import prod

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path

_THREADS = 256
_MIN_ROWS = 6
_MAX_ROWS = 8
_MIN_COLUMNS = 1024
_MIN_REDUCTION = 512

_Q4_SOURCE = r"""
    const int K = KD, N = ND, M = MD;
    const int KPS = KD / 8;
    uint tid = thread_position_in_threadgroup.x;
    uint tgid = threadgroup_position_in_grid.x;
    uint sg = tid >> 5;
    uint lane = tid & 31;
    int n0 = (int)tgid * 8;

    threadgroup bfloat16_t tile[8 * 512];
    threadgroup float partial[8 * 64];
    simdgroup_matrix<float, 8, 8> accum = simdgroup_matrix<float, 8, 8>(0);
    threadgroup bfloat16_t* bt = tile + sg * 512;
    int kbeg = (int)sg * KPS;

    for (int kk = 0; kk < KPS; kk += 64) {
        int ka = kbeg + kk;
        int j = (int)(lane & 7);
        int kq = (int)(lane >> 3);
        int n = n0 + j;
        if (n < N) {
            int group = ka >> 6;
            float scale = (float)sc[(size_t)n * (K / 64) + group];
            float bias = (float)bi[(size_t)n * (K / 64) + group];
            const device uint* wr = w + (size_t)n * (K / 8) + (ka >> 3) + kq * 2;
            uint p0 = wr[0], p1 = wr[1];
            for (int t = 0; t < 8; ++t)
                bt[(kq * 16 + t) * 8 + j] = (bfloat16_t)((float)((p0 >> (4 * t)) & 15u) * scale + bias);
            for (int t = 0; t < 8; ++t)
                bt[(kq * 16 + 8 + t) * 8 + j] = (bfloat16_t)((float)((p1 >> (4 * t)) & 15u) * scale + bias);
        } else {
            for (int t = 0; t < 16; ++t) bt[(kq * 16 + t) * 8 + j] = (bfloat16_t)0;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<bfloat16_t, 8, 8> A, B;
        for (int kt = 0; kt < 8; ++kt) {
            simdgroup_load(A, x + ka + kt * 8, K);
            simdgroup_load(B, bt + kt * 64, 8);
            simdgroup_multiply_accumulate(accum, A, B, accum);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(accum, partial + sg * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = (int)tid; i < 64; i += 256) {
        int m = i >> 3, j = i & 7, n = n0 + j;
        if (m < M && n < N) {
            float sum = 0.0f;
            for (int q = 0; q < 8; ++q) sum += partial[q * 64 + i];
            out[(size_t)m * N + n] = (bfloat16_t)sum;
        }
    }
"""

_Q8_SOURCE = r"""
    const int K = KD, N = ND, M = MD;
    const int KPS = KD / 8;
    uint tid = thread_position_in_threadgroup.x;
    uint tgid = threadgroup_position_in_grid.x;
    uint sg = tid >> 5;
    uint lane = tid & 31;
    int n0 = (int)tgid * 8;

    threadgroup bfloat16_t tile[8 * 512];
    threadgroup float partial[8 * 64];
    simdgroup_matrix<float, 8, 8> accum = simdgroup_matrix<float, 8, 8>(0);
    threadgroup bfloat16_t* bt = tile + sg * 512;
    int kbeg = (int)sg * KPS;

    for (int kk = 0; kk < KPS; kk += 64) {
        int ka = kbeg + kk;
        int j = (int)(lane & 7);
        int kq = (int)(lane >> 3);
        int n = n0 + j;
        if (n < N) {
            int group = ka >> 6;
            float scale = (float)sc[(size_t)n * (K / 64) + group];
            float bias = (float)bi[(size_t)n * (K / 64) + group];
            const device uint* wr = w + (size_t)n * (K / 4) + (ka >> 2) + kq * 4;
            uint p0 = wr[0], p1 = wr[1], p2 = wr[2], p3 = wr[3];
            for (int t = 0; t < 4; ++t) bt[(kq * 16 + t) * 8 + j] = (bfloat16_t)((float)((p0 >> (8 * t)) & 255u) * scale + bias);
            for (int t = 0; t < 4; ++t) bt[(kq * 16 + 4 + t) * 8 + j] = (bfloat16_t)((float)((p1 >> (8 * t)) & 255u) * scale + bias);
            for (int t = 0; t < 4; ++t) bt[(kq * 16 + 8 + t) * 8 + j] = (bfloat16_t)((float)((p2 >> (8 * t)) & 255u) * scale + bias);
            for (int t = 0; t < 4; ++t) bt[(kq * 16 + 12 + t) * 8 + j] = (bfloat16_t)((float)((p3 >> (8 * t)) & 255u) * scale + bias);
        } else {
            for (int t = 0; t < 16; ++t) bt[(kq * 16 + t) * 8 + j] = (bfloat16_t)0;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<bfloat16_t, 8, 8> A, B;
        for (int kt = 0; kt < 8; ++kt) {
            simdgroup_load(A, x + ka + kt * 8, K);
            simdgroup_load(B, bt + kt * 64, 8);
            simdgroup_multiply_accumulate(accum, A, B, accum);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(accum, partial + sg * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = (int)tid; i < 64; i += 256) {
        int m = i >> 3, j = i & 7, n = n0 + j;
        if (m < M && n < N) {
            float sum = 0.0f;
            for (int q = 0; q < 8; ++q) sum += partial[q * 64 + i];
            out[(size_t)m * N + n] = (bfloat16_t)sum;
        }
    }
"""

_Q4_KERNEL = mx.fast.metal_kernel(
    name="liteinfer_qwen35_q4_small_m",
    input_names=["x", "w", "sc", "bi"],
    output_names=["out"],
    source=_Q4_SOURCE,
)
_Q8_KERNEL = mx.fast.metal_kernel(
    name="liteinfer_qwen35_q8_small_m",
    input_names=["x", "w", "sc", "bi"],
    output_names=["out"],
    source=_Q8_SOURCE,
)
def _eligible(layer: nn.Module, values: mx.array) -> bool:
    if mx.default_device() != mx.gpu or values.dtype != mx.bfloat16:
        return False
    if layer.mode != "affine" or layer.group_size != 64 or layer.get("biases") is None:
        return False
    rows = prod(values.shape[:-1])
    columns = layer.weight.shape[0]
    reduction = values.shape[-1]
    return (
        _MIN_ROWS <= rows <= _MAX_ROWS
        and (layer.bits == 4 or rows == _MAX_ROWS)
        and columns >= _MIN_COLUMNS
        and reduction >= _MIN_REDUCTION
        and reduction % 512 == 0
        and layer.bits in (4, 8)
    )


def quantized_matmul(layer: nn.Module, values: mx.array) -> mx.array:
    """Run a supported M=6..8 affine QMM, otherwise use the MLX reference."""
    if not _eligible(layer, values):
        return mx.quantized_matmul(
            values,
            layer.weight,
            scales=layer.scales,
            biases=layer.get("biases"),
            transpose=True,
            group_size=layer.group_size,
            bits=layer.bits,
            mode=layer.mode,
        )

    rows, reduction = prod(values.shape[:-1]), values.shape[-1]
    columns = layer.weight.shape[0]
    flat = values.reshape(rows, reduction)
    if rows < _MAX_ROWS:
        flat = mx.concatenate(
            [flat, mx.zeros((_MAX_ROWS - rows, reduction), dtype=flat.dtype)], axis=0
        )
    kernel = _Q4_KERNEL if layer.bits == 4 else _Q8_KERNEL
    (output,) = kernel(
        inputs=[flat, layer.weight, layer.scales, layer.biases],
        template=[("KD", reduction), ("ND", columns), ("MD", rows)],
        output_shapes=[(rows, columns)],
        output_dtypes=[mx.bfloat16],
        grid=(((columns + 7) // 8) * _THREADS, 1, 1),
        threadgroup=(_THREADS, 1, 1),
    )
    return output.reshape(*values.shape[:-1], columns)


class SmallMQuantizedLinear(nn.QuantizedLinear):
    def __call__(self, values: mx.array) -> mx.array:
        output = quantized_matmul(self, values)
        if self.get("bias") is not None:
            output = output + self.bias
        return output


class SmallMQuantizedEmbedding(nn.QuantizedEmbedding):
    def as_linear(self, values: mx.array) -> mx.array:
        return quantized_matmul(self, values)


def _copy_module(source: nn.Module, target_type: type[nn.Module]) -> nn.Module:
    target = target_type.__new__(target_type)
    nn.Module.__init__(target)
    for name, value in source.items():
        setattr(target, name, value)
    for name in ("group_size", "bits", "mode", "num_embeddings", "dims"):
        if hasattr(source, name):
            setattr(target, name, getattr(source, name))
    target.eval()
    return target


def enable_qwen35_small_m(model: nn.Module) -> dict[str, int]:
    """Wrap Qwen3.5 quantized layers with shape-gated 4/8-bit small-M kernels."""
    from mlx_lm.models.qwen3_5 import Model

    if type(model) is not Model or model.training:
        raise TypeError("Small-M kernels require an eval-mode MLX Qwen3.5 model")
    counts = {"quantized_linear": 0, "quantized_embedding": 0}

    def replace(_path, module):
        if type(module) is nn.QuantizedLinear:
            counts["quantized_linear"] += 1
            return _copy_module(module, SmallMQuantizedLinear)
        if type(module) is nn.QuantizedEmbedding:
            counts["quantized_embedding"] += 1
            return _copy_module(module, SmallMQuantizedEmbedding)
        return module

    model.update_modules(
        tree_map_with_path(replace, model.leaf_modules(), is_leaf=nn.Module.is_module)
    )
    model.eval()
    mx.eval(model.parameters())
    return counts
