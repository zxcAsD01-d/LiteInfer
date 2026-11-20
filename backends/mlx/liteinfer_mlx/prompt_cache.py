"""Tiered MLX prompt cache; recurrent states are never partially trimmed."""
from __future__ import annotations

from collections import OrderedDict, deque
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Protocol

import mlx.core as mx
from mlx_lm.models.cache import can_trim_prompt_cache, load_prompt_cache, save_prompt_cache, trim_prompt_cache

from .cache_index import RadixIndex
from .cache_settings import CacheConfig
from .disk_cache import DiskCache, key_digest, latency

ModelKey = str | tuple[str | None, ...]


class CacheLayer(Protocol):
    @property
    def nbytes(self) -> int: ...
    @property
    def state(self) -> object: ...
    def is_trimmable(self) -> bool: ...


class PromptCodec:
    def freeze(self, value: list[CacheLayer]) -> list[CacheLayer]:
        frozen = copy.deepcopy(value)
        mx.eval([layer.state for layer in frozen])
        return frozen

    def save(self, path: Path, value: list[CacheLayer]) -> None:
        save_prompt_cache(str(path), value)

    def load(self, path: Path) -> list[CacheLayer]:
        return load_prompt_cache(str(path))


@dataclass
class MemoryEntry:
    scope: str
    tokens: tuple[int, ...]
    cache: list[CacheLayer]
    size: int
    cache_type: str
    trimmable: bool


class TieredPromptCache:
    def __init__(self, config: CacheConfig, directory: Path | None = None):
        self.config, self.codec = config, PromptCodec()
        self.lock = threading.RLock()
        self.entries: OrderedDict[str, MemoryEntry] = OrderedDict()
        self.index = RadixIndex()
        self.nbytes = 0
        self.counts = dict.fromkeys(("requests", "hits", "misses", "memory_hits", "ssd_hits",
                                    "exact_hits", "prefix_hits", "input_tokens", "saved_tokens",
                                    "evictions"), 0)
        self.lookup_ms: deque[float] = deque(maxlen=512)
        if config.ssd_max_bytes and directory is None:
            raise ValueError("Enabled SSD cache requires a dedicated directory")
        self.disk = DiskCache(directory, config, self.codec) if config.ssd_max_bytes else None

    def __len__(self) -> int:
        with self.lock:
            return len(self.entries)

    @staticmethod
    def scope(model: ModelKey) -> str:
        encoded = json.dumps(["liteinfer-prompt-v1", model], separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def fetch_nearest_cache(self, model: ModelKey, tokens: list[int]) -> tuple[list[CacheLayer] | None, list[int]]:
        started = time.perf_counter()
        sequence, scope = tuple(tokens), self.scope(model)
        with self.lock:
            self.counts["requests"] += 1
            self.counts["input_tokens"] += len(tokens)
            key, length = self.index.nearest(scope, sequence)
            entry = self.entries.get(key)
            cache = None
            source = "memory_hits"
            if self.disk is not None and length < len(sequence):
                cache, stored = self.disk.fetch(scope, sequence, minimum=length)
                if cache is not None:
                    entry = self._insert(scope, stored, cache, "restored")
                    length = self._common(stored, sequence)
                    source = "ssd_hits"
            if cache is None and entry is not None:
                self.entries.move_to_end(key)
                cache = entry.cache
                stored = entry.tokens
            result = self._reuse(cache, stored, length) if cache is not None else None
            self._record_hit(source, length, len(tokens), result is not None)
            self.lookup_ms.append((time.perf_counter() - started) * 1000)
            return (result, tokens[length:]) if result is not None else (None, tokens.copy())

    @staticmethod
    def _common(left: tuple[int, ...], right: tuple[int, ...]) -> int:
        count = 0
        for a, b in zip(left, right):
            if a != b:
                break
            count += 1
        return count

    def _reuse(self, cache: list[CacheLayer], stored: tuple[int, ...], length: int) -> list[CacheLayer]:
        cloned = self.codec.freeze(cache)
        if length < len(stored):
            if not can_trim_prompt_cache(cloned):
                raise ValueError("Recurrent cache cannot be trimmed")
            trim_prompt_cache(cloned, len(stored) - length)
        return cloned

    def _record_hit(self, source: str, length: int, total: int, hit: bool) -> None:
        self.counts["hits" if hit else "misses"] += 1
        if hit:
            self.counts[source] += 1
            self.counts["exact_hits" if length == total else "prefix_hits"] += 1
            self.counts["saved_tokens"] += length

    def _drop(self, key: str) -> None:
        entry = self.entries.pop(key)
        self.nbytes -= entry.size
        self.index.remove(entry.scope, entry.tokens)

    def _insert(self, scope: str, tokens: tuple[int, ...], cache: list[CacheLayer],
                cache_type: str) -> MemoryEntry | None:
        size = sum(layer.nbytes for layer in cache)
        if not tokens or not self.config.memory_max_entries or not self.config.memory_max_bytes or size > self.config.memory_max_bytes:
            return None
        key = key_digest(scope, tokens)
        if key in self.entries:
            self._drop(key)
        trimmable = can_trim_prompt_cache(cache)
        if trimmable:
            for prefix in self.index.prefixes(scope, tokens):
                self._drop(prefix)
        entry = MemoryEntry(scope, tokens, cache, size, cache_type, trimmable)
        self.entries[key] = entry
        self.nbytes += size
        self.index.add(scope, tokens, key, trimmable)
        self._trim(self.config.memory_max_entries, self.config.memory_max_bytes)
        return entry

    def insert_cache(self, model: ModelKey, tokens: list[int], cache: list[CacheLayer],
                     *, cache_type: str = "assistant") -> None:
        if not tokens or (not self.disk and not (self.config.memory_max_entries and self.config.memory_max_bytes)):
            return
        scope, sequence = self.scope(model), tuple(tokens)
        frozen = self.codec.freeze(cache)
        with self.lock:
            self._insert(scope, sequence, frozen, cache_type)
            if self.disk is not None:
                self.disk.offer(scope, sequence, frozen, sum(c.nbytes for c in frozen),
                                can_trim_prompt_cache(frozen), cache_type)

    def _trim(self, entries: int, size: int) -> None:
        while self.entries and (len(self.entries) > entries or self.nbytes > size):
            self._drop(next(iter(self.entries)))
            self.counts["evictions"] += 1

    def trim_to(self, *, n_sequences: int | None = None, n_bytes: int | None = None) -> None:
        with self.lock:
            self._trim(self.config.memory_max_entries if n_sequences is None else max(0, n_sequences),
                       self.config.memory_max_bytes if n_bytes is None else max(0, n_bytes))

    def stats_by_type(self) -> dict[str, dict[str, int]]:
        with self.lock:
            stats: dict[str, dict[str, int]] = {}
            for entry in self.entries.values():
                row = stats.setdefault(entry.cache_type, {"n_sequences": 0, "n_bytes": 0})
                row["n_sequences"] += 1
                row["n_bytes"] += entry.size
            return stats

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            requests, inputs = self.counts["requests"], self.counts["input_tokens"]
            return {**self.counts, "entries": len(self.entries), "bytes": self.nbytes,
                    "hit_rate": self.counts["hits"] / requests if requests else 0.0,
                    "token_hit_rate": self.counts["saved_tokens"] / inputs if inputs else 0.0,
                    "lookup_latency": latency(self.lookup_ms), "index": self.index.snapshot(),
                    "ssd": self.disk.snapshot() if self.disk else {"enabled": False},
                    "settings": self.config.as_dict(), "analysis": self.analysis()}

    def analysis(self) -> list[str]:
        notes = []
        if self.counts["requests"] < 100:
            notes.append("请求样本不足 100，命中率暂不代表稳定工作负载")
        if self.counts["evictions"] and self.counts["misses"]:
            notes.append("内存发生淘汰并存在未命中，可结合 SSD 命中率评估容量")
        if self.disk:
            stats = self.disk.snapshot()
            if stats["queue_drops"]:
                notes.append("SSD 写入队列发生丢弃，检查队列预算和写入延迟")
            if stats["read_errors"] or stats["write_errors"]:
                notes.append("SSD 有读写错误，检查磁盘空间、权限及缓存文件完整性")
            if stats["admission_skips"]:
                notes.append("部分 SSD 写入被准入策略跳过，检查最小 token 与条目容量")
        return notes

    def clear(self) -> None:
        with self.lock:
            self.entries.clear()
            self.index, self.nbytes = RadixIndex(), 0
            if self.disk:
                self.disk.clear()

    def close(self) -> None:
        if self.disk:
            self.disk.close()
