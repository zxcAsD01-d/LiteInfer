from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache

from liteinfer_mlx.cache_settings import CacheConfig
from liteinfer_mlx.prompt_cache import TieredPromptCache


def state(length=4, hybrid=False):
    kv = KVCache()
    values = mx.arange(length * 8).reshape(1, 2, length, 4).astype(mx.float32)
    kv.update_and_fetch(values, values)
    layers = [kv]
    if hybrid:
        recurrent = ArraysCache(2)
        recurrent[0], recurrent[1] = mx.ones((1, 2, 4)), mx.ones((1, 2, 2, 2))
        layers.append(recurrent)
    mx.eval([layer.state for layer in layers])
    return layers


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = CacheConfig(memory_max_entries=1, memory_max_bytes=1024**2,
                                  ssd_max_bytes=1024**2, min_tokens=1)

    def store(self, **changes):
        cache = TieredPromptCache(replace(self.config, **changes), Path(self.temporary.name))
        self.addCleanup(cache.close)
        return cache

    def test_spill_reload_restart_and_branch_mutation(self):
        cache = self.store()
        cache.insert_cache(("model", "tenant"), [1, 2, 3, 4], state())
        cache.disk.flush()
        cache.insert_cache(("model", "tenant"), [9, 8, 7, 6], state())
        hit, rest = cache.fetch_nearest_cache(("model", "tenant"), [1, 2, 3, 5])
        self.assertEqual(rest, [5])
        self.assertEqual(hit[0].offset, 3)
        hit[0].keys[..., 0, :] = -999
        again, _ = cache.fetch_nearest_cache(("model", "tenant"), [1, 2, 3, 5])
        self.assertNotEqual(again[0].keys[0, 0, 0, 0].item(), -999)
        self.assertEqual(cache.snapshot()["ssd_hits"], 1)
        cache.close()
        restored = self.store()
        self.assertIsNotNone(restored.fetch_nearest_cache(("model", "tenant"), [1, 2, 3, 4])[0])
        self.assertIsNone(restored.fetch_nearest_cache(("model", "other"), [1, 2, 3, 4])[0])

    def test_hybrid_restore_requires_complete_checkpoint(self):
        cache = self.store()
        cache.insert_cache("hybrid", [1, 2, 3, 4], state(hybrid=True))
        cache.disk.flush()
        cache.trim_to(n_sequences=0)
        self.assertIsNone(cache.fetch_nearest_cache("hybrid", [1, 2, 3, 9])[0])
        hit, rest = cache.fetch_nearest_cache("hybrid", [1, 2, 3, 4, 9])
        self.assertEqual(rest, [9])
        self.assertIsInstance(hit[1], ArraysCache)
        self.assertTrue(mx.all(hit[1][0] == 1).item())

    def test_memory_disabled_still_uses_ssd(self):
        cache = self.store(memory_max_entries=0)
        cache.insert_cache("a", [1, 2, 3, 4], state())
        cache.disk.flush()
        self.assertEqual(len(cache), 0)
        self.assertIsNotNone(cache.fetch_nearest_cache("a", [1, 2, 3, 9])[0])
        self.assertEqual(len(cache), 0)

    def test_ssd_disabled_does_not_create_files(self):
        root = Path(self.temporary.name) / "disabled"
        cache = TieredPromptCache(CacheConfig(ssd_max_bytes=0), root)
        cache.insert_cache("a", [1, 2, 3, 4], state())
        self.assertIsNotNone(cache.fetch_nearest_cache("a", [1, 2, 3, 9])[0])
        self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
