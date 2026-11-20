"""Projection packing checks against the unmodified MLX Qwen3.5 graph."""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_5 import Model, ModelArgs

from liteinfer_mlx.qwen35_fusion import fuse_qwen35, pack_linears


MODES = [("affine", False), ("mxfp4", False), ("nvfp4", False),
         ("mxfp8", False), ("nvfp4", True), ("mxfp8", True)]


@pytest.mark.parametrize("mode,activations", MODES)
def test_packed_projection_preserves_quantized_rows(mode, activations):
    mx.random.seed(17)
    layers = [nn.Linear(64, width, bias=False).eval() for width in (32, 64)]
    layers = [layer.to_quantized(mode=mode, quantize_input=activations).eval() for layer in layers]
    values = mx.random.normal((2, 5, 64)).astype(mx.bfloat16)
    expected = mx.concatenate([layer(values) for layer in layers], axis=-1)
    packed = pack_linears(layers)
    actual = packed(values)
    mx.eval(expected, actual)
    np.testing.assert_allclose(np.asarray(actual.astype(mx.float32)),
                               np.asarray(expected.astype(mx.float32)), rtol=0.02, atol=0.02)
    assert packed.weight.size == sum(layer.weight.size for layer in layers)
    assert packed.scales.size == sum(layer.scales.size for layer in layers)


def tiny_model():
    args = ModelArgs.from_dict({"model_type": "qwen3_5", "text_config": {
        "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        "linear_num_value_heads": 2, "linear_num_key_heads": 2,
        "linear_key_head_dim": 64, "linear_value_head_dim": 64,
        "vocab_size": 128, "max_position_embeddings": 1024,
        "tie_word_embeddings": True, "full_attention_interval": 4}})
    model = Model(args)
    model.set_dtype(mx.bfloat16)
    model.eval()
    return model


def hidden_with_prefix(model, tokens):
    cache = model.make_cache()
    first = model.language_model.model(tokens[:, :4], cache=cache)
    mx.eval(first, [entry.state for entry in cache])
    last = model.language_model.model(tokens[:, 4:], cache=cache)
    mx.eval(last, [entry.state for entry in cache])
    return mx.concatenate([first, last], axis=1)


@pytest.mark.parametrize("mode", [None, "affine", "mxfp4", "nvfp4", "mxfp8"])
def test_fused_qwen35_preserves_prefix_continuation(mode):
    mx.random.seed(8)
    reference, fused = tiny_model(), tiny_model()
    fused.load_weights(tree_flatten(reference.parameters()), strict=True)
    if mode is not None:
        nn.quantize(reference, mode=mode)
        nn.quantize(fused, mode=mode)
        reference.eval()
        fused.eval()
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])
    expected = hidden_with_prefix(reference, tokens)
    info = fuse_qwen35(fused)
    actual = hidden_with_prefix(fused, tokens)
    np.testing.assert_allclose(np.asarray(actual.astype(mx.float32)),
                               np.asarray(expected.astype(mx.float32)), rtol=0.03, atol=0.05)
    assert info["layers"] == 4
    assert info["matrix_calls_removed_per_forward"] == 15
    with pytest.raises(TypeError, match="unmodified"):
        fuse_qwen35(fused)


def test_packing_rejects_training_and_mismatched_layers():
    with pytest.raises(ValueError, match="eval"):
        pack_linears([nn.Linear(64, 32)])
    with pytest.raises(ValueError, match="shapes"):
        pack_linears([nn.Linear(64, 32).eval(), nn.Linear(32, 32).eval()])
