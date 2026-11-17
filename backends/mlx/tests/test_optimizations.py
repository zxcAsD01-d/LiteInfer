import copy
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.models.llama import Model, ModelArgs

from liteinfer_mlx.config import BatchConfig, SamplingConfig
from liteinfer_mlx.engine import BatchEngine
from liteinfer_mlx.fused_layers import fuse_model, merge_linears
from liteinfer_mlx.prefix_cache import CacheScope, PrefixCache
from liteinfer_mlx.sampling import candidate_logits, make_sampler, vocabulary_processor
from liteinfer_mlx.stop_filter import StopFilter


def model() -> Model:
    mx.random.seed(7)
    result = Model(ModelArgs(model_type="llama", hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        rms_norm_eps=1e-5, vocab_size=128))
    mx.eval(result.parameters())
    return result


def cache(length: int) -> list[KVCache]:
    layer = KVCache()
    values = mx.arange(length * 8).reshape(1, 2, length, 4).astype(mx.float32)
    layer.update_and_fetch(values, values)
    mx.eval(layer.state)
    return [layer]


class SamplingTests(unittest.TestCase):
    def test_invalid_sampling(self):
        for opts in ({"temperature":float("nan")}, {"top_k":0}, {"top_p":0}, {"seed":-1}):
            with self.assertRaises(ValueError):
                SamplingConfig(**opts)

    def test_sparse_distribution_matches_reference(self):
        logits = mx.array([[0., 3., 1., 5., 2., 4.]])
        values, indices = candidate_logits(logits, .7, 3, 1.)
        reference = mx.softmax(mx.where(logits >= 3, logits / .7, -float("inf")), axis=-1)
        actual = mx.zeros_like(reference)
        actual[0, indices[0]] = mx.softmax(values, axis=-1)[0]
        self.assertTrue(mx.allclose(actual, reference, atol=1e-6).item())

    def test_seed_reproducible_and_requests_independent(self):
        config = SamplingConfig(seed=42, top_k=3)
        first, other, reference = [make_sampler(config) for _ in range(3)]
        logits = mx.array([1., 2., 3., 4., 5.])
        one, two = [], []
        for _ in range(20):
            one.append(first(logits).item())
            other(logits).item()
            other(logits).item()
            two.append(reference(logits).item())
        self.assertEqual(one, two)
        self.assertTrue(set(one) <= {2, 3, 4})
        self.assertEqual(make_sampler(SamplingConfig(temperature=0))(logits).item(), 4)

    def test_top_p_and_padded_vocabulary(self):
        sample = make_sampler(SamplingConfig(top_k=4, top_p=.01))
        self.assertEqual(sample(mx.array([0., 1., 2., 100.])).item(), 3)
        masked = vocabulary_processor(3)(mx.array([]), mx.array([1., 2., 3., 100.]))
        self.assertTrue(mx.isneginf(masked[-1]).item())
        with self.assertRaises(ValueError):
            vocabulary_processor(5)(mx.array([]), masked)


class CacheTests(unittest.TestCase):
    def test_scope_and_branch_mutation_are_isolated(self):
        store = PrefixCache()
        scope = CacheScope("a:b", "c")
        original = cache(4)
        store.save(scope, [1, 2, 3, 4], original)
        hit, suffix = store.fetch(scope, [1, 2, 3, 7])
        self.assertEqual(suffix, [7])
        self.assertEqual(hit[0].offset, 3)
        hit[0].keys[..., :1, :] = -999
        again, _ = store.fetch(scope, [1, 2, 3, 8])
        self.assertNotEqual(again[0].keys[0, 0, 0, 0].item(), -999)
        miss, _ = store.fetch(CacheScope("a", "b:c"), [1, 2, 3, 4])
        self.assertIsNone(miss)

    def test_lru_budget_deduplication_and_clear(self):
        store = PrefixCache(max_bytes=1024, max_entries=2)
        a, b, c = [CacheScope("m", name) for name in "abc"]
        store.save(a, [1, 2], cache(2))
        self.assertFalse(store.save(a, [1], cache(1)))
        store.save(a, [1, 2, 3], cache(3))
        self.assertEqual(len(store.entries), 1)
        store.save(b, [1, 2], cache(2))
        store.fetch(a, [1, 2, 4])
        store.save(c, [1, 2], cache(2))
        self.assertIsNone(store.fetch(b, [1, 2])[0])
        self.assertLessEqual(store.nbytes, 1024)
        store.clear(a)
        self.assertEqual(store.nbytes, sum(entry.nbytes for entry in store.entries.values()))
        store.clear()
        self.assertEqual(store.nbytes, 0)

    def test_disabled_oversized_and_unsupported_cache(self):
        scope = CacheScope("m", "t")
        self.assertFalse(PrefixCache(max_bytes=1).save(scope, [1], cache(1)))
        self.assertFalse(PrefixCache(max_entries=0).save(scope, [1], cache(1)))
        with self.assertRaises(TypeError):
            PrefixCache().save(scope, [1], [object()])
        with self.assertRaises(ValueError):
            PrefixCache().save(scope, [1, 2], cache(1))


class FusionTests(unittest.TestCase):
    def test_full_and_incremental_logits_float_and_quantized(self):
        for quantized in (False, True):
            original = model()
            if quantized:
                nn.quantize(original, group_size=32, bits=4)
            fused = copy.deepcopy(original)
            fuse_model(fused)
            tokens = mx.array([[1, 2, 3, 4]])
            self.assertTrue(mx.allclose(original(tokens), fused(tokens), atol=1e-4, rtol=1e-4).item())
            left, right = make_prompt_cache(original), make_prompt_cache(fused)
            for index in range(4):
                token = tokens[:, index:index+1]
                self.assertTrue(mx.allclose(original(token, cache=left), fused(token, cache=right),
                                            atol=1e-4, rtol=1e-4).item())

    def test_rejects_mismatched_projection_and_repeated_fusion(self):
        with self.assertRaises(ValueError):
            merge_linears([nn.Linear(4, 4), nn.Linear(8, 4)])
        with self.assertRaises(ValueError):
            merge_linears([nn.Linear(4, 4, bias=True), nn.Linear(4, 4, bias=False)])
        value = model()
        fuse_model(value)
        with self.assertRaises(TypeError):
            fuse_model(value)


class EngineTests(unittest.TestCase):
    def test_continuous_batches_reuse_prefix_and_complete(self):
        engine = BatchEngine(model(), "fixture", BatchConfig(concurrency=3, prefill_step=2, idle_prefill_step=4))
        try:
            ids = [engine.submit([1, 2, 3, 4], tenant="a", max_tokens=4,
                                 sampling=SamplingConfig(seed=i)) for i in range(6)]
            finished = {}
            for _ in range(100):
                for event in engine.step():
                    if event.finish_reason:
                        finished[event.id] = event
                if not engine.active and not engine.pending:
                    break
            self.assertEqual(set(finished), set(ids))
            self.assertEqual(engine.peak_active, 3)
            self.assertGreater(engine.cache.hits, 0)
            self.assertTrue(all(event.completion_tokens == 4 for event in finished.values()))
            self.assertTrue(all(event.prompt_tokens == 4 for event in finished.values()))
        finally:
            engine.close()

    def test_queue_overload_cancellation_and_close(self):
        engine = BatchEngine(model(), "fixture", BatchConfig(concurrency=1, queue_capacity=1))
        try:
            a = engine.submit([1, 2], tenant="a")
            b = engine.submit([2, 3], tenant="a")
            with self.assertRaises(BufferError):
                engine.submit([3, 4], tenant="a")
            self.assertTrue(engine.cancel(b))
            engine.step()
            self.assertTrue(engine.cancel(a))
            self.assertFalse(engine.active or engine.pending)
            self.assertFalse(engine.cancel(a))
            with self.assertRaises(ValueError):
                engine.batch.set_prefill_step(0)
        finally:
            engine.close()
        with self.assertRaises(RuntimeError):
            engine.submit([1], tenant="a")

    def test_cache_reuse_preserves_greedy_tokens(self):
        engine = BatchEngine(model(), "fixture")
        try:
            results = []
            for _ in range(2):
                engine.submit([1, 2, 3, 4], tenant="a", max_tokens=5,
                              sampling=SamplingConfig(temperature=0))
                output = []
                while engine.active or engine.pending:
                    output.extend(event.token for event in engine.step())
                results.append(output)
            self.assertEqual(results[0], results[1])
            self.assertGreater(engine.cache.saved_tokens, 0)
        finally:
            engine.close()


class StopTests(unittest.TestCase):
    def test_unicode_overlap_and_every_split(self):
        for split in range(8):
            stop = StopFilter(("完成了", "完成"))
            text = "你好完成了更多"
            result = stop.push(text[:split]) + stop.push(text[split:]) + stop.push("", final=True)
            self.assertEqual(result, "你好")
        stop = StopFilter(("abc",))
        self.assertEqual(stop.push("a"), "")
        self.assertEqual(stop.push("", final=True), "a")
        with self.assertRaises(ValueError):
            StopFilter(("",))


if __name__ == "__main__":
    unittest.main()
