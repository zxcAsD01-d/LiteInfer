import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from liteinfer_mlx.cache_settings import CacheConfig
from liteinfer_mlx.cached_server import ScopedDescription, ScopedGenerator, ScopedName, ScopedProvider, TENANT, model_fingerprint
from mlx_lm import server


class ServerAdapterTests(unittest.TestCase):
    def test_text_prefill_keeps_one_unprocessed_token_after_checkpoint(self):
        generator = object.__new__(ScopedGenerator)
        with patch.object(server.ResponseGenerator, "_tokenize",
                          return_value=([1, 2, 3, 4], [[1, 2, 3, 4]], ["user"], "normal", None)):
            _, segments, kinds, _, _ = generator._tokenize(None, None, None)
        self.assertEqual(segments, [[1, 2, 3], [4]])
        self.assertEqual(kinds, ["user", "assistant"])

    def test_tenant_is_part_of_batch_equality(self):
        context = TENANT.set("first")
        try:
            first = ScopedDescription("model", None, None)
            self.assertEqual(first, ScopedDescription("model", None, None))
            TENANT.set("second")
            second = ScopedDescription("model", None, None)
            self.assertNotEqual(first, second)
            self.assertEqual(first.model.tenant, "first")
            self.assertEqual(str(first.model), "model")
        finally:
            TENANT.reset(context)

    def test_tenant_switch_does_not_reload_weights(self):
        args = argparse.Namespace(pipeline=False, model="fixture", adapter_path=None,
                                  draft_model=None, trust_remote_code=False, chat_template=None)
        provider = ScopedProvider(args)
        def load(model, adapter, draft):
            provider.model_key = (model, adapter, draft)
            provider.model, provider.tokenizer = object(), object()
        with patch.object(provider, "_load", side_effect=load) as mocked, \
                patch("liteinfer_mlx.cached_server.model_fingerprint", return_value="fingerprint"):
            first = provider.load(ScopedName("fixture", "a"))
            self.assertEqual(provider.model_key, ("fingerprint", "a"))
            second = provider.load(ScopedName("fixture", "b"))
            self.assertEqual(provider.model_key, ("fingerprint", "b"))
            self.assertIs(first[0], second[0])
            self.assertEqual(mocked.call_count, 1)

    def test_model_config_and_weight_changes_invalidate_namespace(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            config, weights = directory / "config.json", directory / "model.safetensors"
            config.write_text('{"bits":4}')
            weights.write_bytes(b"fixture")
            paths = (root, None, None)
            original = model_fingerprint(paths)
            self.assertEqual(original, model_fingerprint(paths))
            config.write_text('{"bits":8}')
            self.assertNotEqual(original, model_fingerprint(paths))
            changed = model_fingerprint(paths)
            weights.write_bytes(b"different-weight")
            self.assertNotEqual(changed, model_fingerprint(paths))

    def test_failed_model_load_clears_identity_and_allows_retry(self):
        args = argparse.Namespace(pipeline=False, model="fixture", adapter_path=None,
                                  draft_model=None, trust_remote_code=False, chat_template=None)
        provider = ScopedProvider(args)
        with patch.object(provider, "_load", side_effect=RuntimeError("Simulated load failure")):
            with self.assertRaises(RuntimeError):
                provider.load(ScopedName("fixture", "tenant"))
        self.assertIsNone(provider.base_key)
        self.assertIsNone(provider.model_key)

    def test_invalid_cache_configuration_is_rejected(self):
        for values in ({"ssd_max_bytes": -1}, {"memory_max_entries": True},
                       {"ssd_max_bytes": 10, "write_queue_entries": 0},
                       {"ssd_max_bytes": 10, "ttl_seconds": 0}):
            with self.assertRaises(ValueError):
                CacheConfig(**values)


if __name__ == "__main__":
    unittest.main()
