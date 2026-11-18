import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from liteinfer_mlx.valen import ValenMixerHead, metal_layer_norm
from liteinfer_mlx.valen_ops import depthwise_conv1d


class ValenOperatorTests(unittest.TestCase):
    def test_depthwise_conv_matches_mlx_reference(self):
        for dtype in (mx.float16, mx.bfloat16, mx.float32):
            inputs = mx.random.normal((2, 19, 24)).astype(dtype)
            reference = nn.Conv1d(24, 24, 4, groups=24, bias=False)
            reference.weight = mx.random.normal((24, 4, 1)).astype(dtype)
            expected = reference(inputs)
            actual = depthwise_conv1d(inputs, reference.weight)
            mx.eval(expected, actual)
            self.assertTrue(mx.allclose(actual, expected, atol=2e-3, rtol=2e-3).item())

    @unittest.skipUnless(mx.metal.is_available(), "Metal kernel requires Apple Silicon")
    def test_metal_layer_norm_matches_mlx_reference(self):
        for rows, width in ((7, 256), (1, 1024), (19, 1024)):
            values = mx.random.normal((rows, width)).astype(mx.float32)
            reference = nn.LayerNorm(width, eps=1e-5)
            expected = reference(values)
            actual = metal_layer_norm(values, reference.weight, reference.bias, reference.eps)
            mx.eval(expected, actual)
            self.assertTrue(mx.allclose(actual, expected, atol=2e-5, rtol=2e-5).item())

    @unittest.skipUnless(mx.metal.is_available(), "Mixer Metal path requires Apple Silicon")
    def test_custom_mixer_head_matches_mlx_graph(self):
        config = {
            "head_width": 32,
            "head_layers": 2,
            "head_token_hidden_dim": 8,
            "head_channel_hidden_dim": 64,
        }
        reference = ValenMixerHead(48, config)
        optimized = ValenMixerHead(48, config, custom_layer_norm=True)
        optimized.load_weights(tree_flatten(reference.parameters()), strict=True)
        features = mx.random.normal((11, 4, 48)).astype(mx.float32)
        expected = reference(features)
        actual = optimized(features)
        mx.eval(expected, actual)
        self.assertEqual(actual.shape, (11,))
        self.assertTrue(mx.allclose(actual, expected, atol=1e-4, rtol=1e-4).item())


if __name__ == "__main__":
    unittest.main()
