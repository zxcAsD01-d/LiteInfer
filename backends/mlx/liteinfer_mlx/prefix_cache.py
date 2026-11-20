"""Single-owner KV cache with tenant/model scoping and bounded LRU storage."""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json

import mlx.core as mx
from mlx_lm.models.cache import KVCache
from .cache_index import RadixIndex
from .disk_cache import key_digest


@dataclass(frozen=True)
class CacheScope:
    model: str
    tenant: str


@dataclass
class Entry:
    tokens: tuple[int, ...]
    cache: list[KVCache]
    nbytes: int


def compact_cache(cache: list[KVCache], length: int) -> list[KVCache]:
    if not cache or length <= 0:
        raise ValueError("Cache and sequence must not be empty")
    result = []
    for layer in cache:
        if type(layer) is not KVCache:
            raise TypeError("Prefix cache supports plain KVCache only")
        if layer.keys is None or layer.values is None or layer.offset < length:
            raise ValueError("KV cache is shorter than its token sequence")
        clone = KVCache()
        clone.keys = mx.array(mx.contiguous(layer.keys[..., :length, :]))
        clone.values = mx.array(mx.contiguous(layer.values[..., :length, :]))
        clone.offset = length
        result.append(clone)
    mx.eval([layer.state for layer in result])
    return result


class PrefixCache:
    def __init__(self, max_bytes: int = 256 * 1024 * 1024, max_entries: int = 64) -> None:
        if max_bytes < 0 or max_entries < 0:
            raise ValueError("Cache budgets must be nonnegative")
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.entries: OrderedDict[tuple[CacheScope, tuple[int, ...]], Entry] = OrderedDict()
        self.nbytes = self.hits = self.misses = self.saved_tokens = self.input_tokens = 0
        self.index = RadixIndex()
        self.index_keys: dict[str, tuple[CacheScope, tuple[int, ...]]] = {}

    @staticmethod
    def _scope(scope: CacheScope) -> str:
        return hashlib.sha256(json.dumps([scope.model, scope.tenant]).encode()).hexdigest()

    def _drop(self, key: tuple[CacheScope, tuple[int, ...]]) -> None:
        self.nbytes -= self.entries.pop(key).nbytes
        scope = self._scope(key[0])
        self.index.remove(scope, key[1])
        del self.index_keys[key_digest(scope, key[1])]

    def fetch(self, scope: CacheScope, tokens: list[int]) -> tuple[list[KVCache] | None, list[int]]:
        self.input_tokens += len(tokens)
        identifier, length = self.index.nearest(self._scope(scope), tuple(tokens[:-1]))
        if identifier is None:
            self.misses += 1
            return None, tokens.copy()
        best = self.index_keys[identifier]
        entry = self.entries[best]
        cache = compact_cache(entry.cache, length)
        self.entries.move_to_end(best)
        self.hits += 1
        self.saved_tokens += length
        return cache, tokens[length:]

    def save(self, scope: CacheScope, tokens: list[int], cache: list[KVCache]) -> bool:
        if not tokens or not self.max_entries or not self.max_bytes:
            return False
        sequence = tuple(tokens)
        namespace = self._scope(scope)
        identifier, length = self.index.nearest(namespace, sequence)
        if identifier is not None and length == len(sequence):
            self.entries.move_to_end(self.index_keys[identifier])
            return False
        compact = compact_cache(cache, len(tokens))
        size = sum(layer.nbytes for layer in compact)
        if size > self.max_bytes:
            return False
        for identifier in self.index.prefixes(namespace, sequence):
            self._drop(self.index_keys[identifier])
        self.entries[(scope, sequence)] = Entry(sequence, compact, size)
        identifier = key_digest(namespace, sequence)
        self.index_keys[identifier] = (scope, sequence)
        self.index.add(namespace, sequence, identifier, True)
        self.nbytes += size
        while len(self.entries) > self.max_entries or self.nbytes > self.max_bytes:
            self._drop(next(iter(self.entries)))
        return True

    def clear(self, scope: CacheScope | None = None) -> None:
        for key in list(self.entries):
            if scope is None or key[0] == scope:
                self._drop(key)

    def snapshot(self) -> dict[str, object]:
        return {"entries": len(self.entries), "bytes": self.nbytes,
                "max_bytes": self.max_bytes, "max_entries": self.max_entries,
                "hits": self.hits, "misses": self.misses, "saved_tokens": self.saved_tokens,
                "input_tokens": self.input_tokens,
                "token_hit_rate": self.saved_tokens / self.input_tokens if self.input_tokens else 0.0,
                "index": self.index.snapshot()}
