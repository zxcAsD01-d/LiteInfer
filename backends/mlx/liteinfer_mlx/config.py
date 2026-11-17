"""Validated, model-independent inference settings."""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.7
    top_k: int = 64
    top_p: float = 0.9
    seed: int = 0

    def __post_init__(self) -> None:
        if not isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a uint32 integer")


@dataclass(frozen=True)
class BatchConfig:
    concurrency: int = 3
    queue_capacity: int = 24
    prefill_batch_size: int = 3
    prefill_step: int = 128
    idle_prefill_step: int = 512
    cache_max_bytes: int = 256 * 1024 * 1024
    cache_max_entries: int = 64
    prefix_cache: bool = True

    def __post_init__(self) -> None:
        for name in ("concurrency", "prefill_batch_size", "prefill_step", "idle_prefill_step"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("queue_capacity", "cache_max_bytes", "cache_max_entries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
