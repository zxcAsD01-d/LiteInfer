from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import unittest

from liteinfer_mlx.cache_settings import CacheConfig
from liteinfer_mlx.disk_cache import DiskCache


class BytesCodec:
    def freeze(self, value: bytes) -> bytes:
        return value

    def save(self, path: Path, value: bytes) -> None:
        path.write_bytes(value)

    def load(self, path: Path) -> bytes:
        return path.read_bytes()


class DiskTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "ssd"
        self.config = CacheConfig(ssd_max_bytes=100, ssd_max_entries=2, min_tokens=1)

    def store(self, **changes):
        cache = DiskCache(self.root, replace(self.config, **changes), BytesCodec())
        self.addCleanup(cache.close)
        return cache

    def test_restart_lru_and_tenant_isolation(self):
        cache = self.store()
        self.assertTrue(cache.offer("tenant-a", (1, 2), b"first", 5, True))
        cache.flush()
        self.assertEqual(cache.fetch("tenant-b", (1, 2, 3)), (None, ()))
        self.assertEqual(cache.fetch("tenant-a", (1, 2, 3)), (b"first", (1, 2)))
        cache.close()
        restored = self.store()
        self.assertEqual(restored.fetch("tenant-a", (1, 2, 4)), (b"first", (1, 2)))
        self.assertEqual(restored.snapshot()["entries"], 1)
        self.assertEqual(self.root.stat().st_mode & 0o777, 0o700)
        self.assertEqual(next(self.root.glob("*.safetensors")).stat().st_mode & 0o777, 0o600)

    def test_capacity_and_access_order(self):
        cache = self.store(ssd_max_bytes=10)
        for token in (1, 2):
            cache.offer("a", (token,), b"12345", 5, True)
            cache.flush()
        cache.fetch("a", (1,))
        cache.offer("a", (3,), b"12345", 5, True)
        cache.flush()
        self.assertIsNone(cache.fetch("a", (2,))[0])
        self.assertEqual(cache.snapshot()["evictions"], 1)
        self.assertLessEqual(cache.nbytes, 10)

    def test_ttl_is_not_extended_by_reads(self):
        now = [10.]
        cache = DiskCache(self.root, replace(self.config, ttl_seconds=2), BytesCodec(), clock=lambda: now[0])
        self.addCleanup(cache.close)
        cache.offer("a", (1,), b"state", 5, True)
        cache.flush()
        now[0] = 11.
        self.assertIsNotNone(cache.fetch("a", (1,))[0])
        now[0] = 12.
        self.assertIsNone(cache.fetch("a", (1,))[0])
        self.assertEqual(cache.snapshot()["expirations"], 1)

    def test_corrupt_payload_is_rejected_and_removed(self):
        cache = self.store()
        cache.offer("a", (1,), b"state", 5, True)
        cache.flush()
        next(self.root.glob("*.safetensors")).write_bytes(b"WRONG")
        with self.assertLogs("liteinfer_mlx.disk_cache", level="WARNING"):
            self.assertIsNone(cache.fetch("a", (1,))[0])
        self.assertEqual(cache.snapshot()["read_errors"], 1)
        self.assertEqual(cache.snapshot()["entries"], 0)

    def test_pending_write_is_bounded_and_clear_cannot_resurrect_it(self):
        entered, release = threading.Event(), threading.Event()
        class SlowCodec(BytesCodec):
            def save(self, path, value):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("Writer test timed out")
                super().save(path, value)
        cache = DiskCache(self.root, replace(self.config, write_queue_entries=1), SlowCodec())
        try:
            self.assertTrue(cache.offer("a", (1,), b"state", 5, True))
            self.assertTrue(entered.wait(5))
            self.assertFalse(cache.offer("a", (2,), b"other", 5, True))
            self.assertEqual(cache.snapshot()["queue_drops"], 1)
            cache.clear()
            release.set()
            cache.flush()
            self.assertEqual(cache.snapshot()["entries"], 0)
        finally:
            release.set()
            cache.close()

    def test_write_failure_is_counted_and_writer_survives(self):
        class FailingCodec(BytesCodec):
            def save(self, path, value):
                raise OSError("Simulated disk full")
        cache = DiskCache(self.root, self.config, FailingCodec())
        self.addCleanup(cache.close)
        with self.assertLogs("liteinfer_mlx.disk_cache", level="ERROR"):
            cache.offer("a", (1,), b"state", 5, True)
            cache.flush()
        self.assertEqual(cache.snapshot()["write_errors"], 1)
        self.assertTrue(cache.writer.is_alive())
        self.assertEqual(cache.snapshot()["pending_bytes"], 0)

    def test_exclusive_owner_and_admission(self):
        cache = self.store()
        with self.assertRaises(BlockingIOError):
            DiskCache(self.root, self.config, BytesCodec())
        self.assertFalse(cache.offer("a", (), b"state", 5, True))
        self.assertFalse(cache.offer("a", (1,), b"x" * 101, 101, True))
        self.assertEqual(cache.snapshot()["admission_skips"], 2)

    def test_trimmable_prefixes_are_deduplicated_without_dropping_hybrid_checkpoints(self):
        cache = self.store(ssd_max_entries=8)
        for tokens in ((1, 2), (1, 2, 3, 4)):
            cache.offer("a", tokens, b"state", 5, True)
            cache.flush()
        self.assertEqual(cache.snapshot()["entries"], 1)
        self.assertFalse(cache.offer("a", (1, 2), b"state", 5, True))
        for tokens in ((1, 2), (1, 2, 3, 4)):
            cache.offer("hybrid", tokens, b"state", 5, False)
            cache.flush()
        self.assertEqual(cache.snapshot()["entries"], 3)
        self.assertEqual(cache.fetch("hybrid", (1, 2, 9))[1], (1, 2))

    def test_publish_failure_does_not_leave_unaccounted_payload_files(self):
        class FailingPublication(DiskCache):
            def _publish(self, job, temporary, name, size, digest):
                temporary.replace(self.directory / name)
                raise OSError("Simulated metadata publication failure")
        cache = FailingPublication(self.root, self.config, BytesCodec())
        self.addCleanup(cache.close)
        with self.assertLogs("liteinfer_mlx.disk_cache", level="ERROR"):
            cache.offer("a", (1,), b"state", 5, True)
            cache.flush()
        self.assertEqual(list(self.root.glob("*.safetensors")), [])
        self.assertEqual(cache.snapshot()["bytes"], 0)
        self.assertEqual(cache.snapshot()["write_errors"], 1)


if __name__ == "__main__":
    unittest.main()
