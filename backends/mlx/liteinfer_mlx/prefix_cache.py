"""Single-owner KV cache with tenant/model scoping and bounded LRU storage."""
from collections import OrderedDict
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.models.cache import KVCache


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

    def fetch(self, scope: CacheScope, tokens: list[int]) -> tuple[list[KVCache] | None, list[int]]:
        self.input_tokens += len(tokens)
        best, length = None, 0
        for key, entry in self.entries.items():
            if key[0] != scope:
                continue
            common = 0
            for left, right in zip(entry.tokens, tokens[:-1]):
                if left != right:
                    break
                common += 1
            if common > length:
                best, length = key, common
        if best is None:
            self.misses += 1
            return None, tokens.copy()
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
        for key, entry in self.entries.items():
            if key[0] == scope and entry.tokens[:len(sequence)] == sequence:
                self.entries.move_to_end(key)
                return False
        compact = compact_cache(cache, len(tokens))
        size = sum(layer.nbytes for layer in compact)
        if size > self.max_bytes:
            return False
        for key in list(self.entries):
            if key[0] == scope and sequence[:len(key[1])] == key[1]:
                self.nbytes -= self.entries.pop(key).nbytes
        self.entries[(scope, sequence)] = Entry(sequence, compact, size)
        self.nbytes += size
        while len(self.entries) > self.max_entries or self.nbytes > self.max_bytes:
            self.nbytes -= self.entries.popitem(last=False)[1].nbytes
        return True

    def clear(self, scope: CacheScope | None = None) -> None:
        for key in list(self.entries):
            if scope is None or key[0] == scope:
                self.nbytes -= self.entries.pop(key).nbytes

    def snapshot(self) -> dict[str, int | float]:
        return {"entries": len(self.entries), "bytes": self.nbytes,
                "max_bytes": self.max_bytes, "max_entries": self.max_entries,
                "hits": self.hits, "misses": self.misses, "saved_tokens": self.saved_tokens,
                "input_tokens": self.input_tokens,
                "token_hit_rate": self.saved_tokens / self.input_tokens if self.input_tokens else 0.0}
