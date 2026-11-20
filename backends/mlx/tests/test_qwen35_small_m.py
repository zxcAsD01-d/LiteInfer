"""Numerical checks for the shape-gated Qwen3.5 Metal kernels."""
import unittest

import mlx.core as mx
import mlx.nn as nn

from liteinfer_mlx.qwen35_small_m import quantized_matmul


class SmallMKernelTests(unittest.TestCase):
    def test_affine_q4_and_q8_match_mlx_for_eight_rows(self):
        mx.random.seed(29)
        values = mx.random.normal((8, 512)).astype(mx.bfloat16)
        for bits in (4, 8):
            with self.subTest(bits=bits):
                layer = nn.Linear(512, 1024, bias=False).to_quantized(
                    group_size=64, bits=bits
                ).eval()
                expected = mx.quantized_matmul(
                    values,
                    layer.weight,
                    scales=layer.scales,
                    biases=layer.biases,
                    transpose=True,
                    group_size=64,
                    bits=bits,
                    mode="affine",
                )
                actual = quantized_matmul(layer, values)
                mx.eval(expected, actual)
                error = mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32)))
                self.assertLessEqual(float(error.item()), 0.2)

    def test_unsupported_row_count_falls_back_to_mlx(self):
        mx.random.seed(31)
        values = mx.random.normal((1, 512)).astype(mx.bfloat16)
        layer = nn.Linear(512, 1024, bias=False).to_quantized(
            group_size=64, bits=8
        ).eval()
        expected = layer(values)
        actual = quantized_matmul(layer, values)
        mx.eval(expected, actual)
        self.assertTrue(mx.all(expected == actual).item())

    def test_eligible_bfloat16_qmm_matches_reference_logits(self):
        mx.random.seed(37)
        values = mx.random.normal((8, 1024)).astype(mx.bfloat16)
        layer = nn.Linear(1024, 2048, bias=False).to_quantized(
            group_size=64, bits=4
        ).eval()
        expected = layer(values)
        actual = quantized_matmul(layer, values)
        mx.eval(expected, actual)
        error = mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32)))
        self.assertLessEqual(float(error.item()), 0.2)


if __name__ == "__main__":
    unittest.main()
