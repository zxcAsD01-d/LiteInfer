"""Continuous batching. Call all methods from one GPU-owner thread."""
from collections import deque
from dataclasses import dataclass

import mlx.nn as nn
from mlx_lm.generate import GenerationBatch
from mlx_lm.models.cache import KVCache, make_prompt_cache

from .adaptive_batch import AdaptiveBatchGenerator
from .config import BatchConfig, SamplingConfig
from .prefix_cache import CacheScope, PrefixCache
from .sampling import make_sampler, vocabulary_processor


@dataclass
class Job:
    id: int
    tokens: list[int]
    scope: CacheScope
    max_tokens: int
    sampling: SamplingConfig
    generated: int = 0
    cached_tokens: int = 0
    decoding: bool = False


@dataclass(frozen=True)
class TokenEvent:
    id: int
    token: int
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int


class BatchEngine:
    def __init__(self, model: nn.Module, model_id: str, config: BatchConfig = BatchConfig(),
                 stop_tokens: list[list[int]] | None = None, vocabulary_size: int | None = None) -> None:
        if config.prefix_cache and any(type(layer) is not KVCache for layer in make_prompt_cache(model)):
            raise TypeError("This architecture requires prefix_cache=False")
        self.config, self.model_id = config, model_id
        self.cache = PrefixCache(config.cache_max_bytes, config.cache_max_entries)
        self.batch = AdaptiveBatchGenerator(model, completion_batch_size=config.concurrency,
            prefill_batch_size=min(config.prefill_batch_size, config.concurrency),
            prefill_step_size=config.prefill_step, stop_tokens=stop_tokens,
            logits_processors=[vocabulary_processor(vocabulary_size)] if vocabulary_size is not None else None)
        self.pending: deque[Job] = deque()
        self.active: dict[int, Job] = {}
        self.prefilling: dict[int, tuple[CacheScope, tuple[int, ...]]] = {}
        self.next_id = self.peak_active = 0
        self.closed = False

    def submit(self, tokens: list[int], *, tenant: str, max_tokens: int = 1024,
               sampling: SamplingConfig = SamplingConfig()) -> int:
        if self.closed:
            raise RuntimeError("Engine is closed")
        if not tokens or max_tokens <= 0:
            raise ValueError("Prompt and output budget must be positive")
        if len(self.pending) + len(self.active) >= self.config.concurrency + self.config.queue_capacity:
            raise BufferError("Inference admission queue is full")
        job = Job(self.next_id, tokens.copy(), CacheScope(self.model_id, tenant), max_tokens, sampling)
        self.next_id += 1
        self.pending.append(job)
        return job.id

    def _admit(self) -> None:
        for _ in range(len(self.pending)):
            if len(self.active) >= self.config.concurrency:
                break
            job = self.pending.popleft()
            signature = (job.scope, tuple(job.tokens))
            if self.config.prefix_cache and signature in self.prefilling.values():
                self.pending.append(job)
                continue
            cache, suffix = self.cache.fetch(job.scope, job.tokens) if self.config.prefix_cache else (None, job.tokens)
            job.cached_tokens = len(job.tokens) - len(suffix)
            uid = self.batch.insert([suffix], [job.max_tokens], caches=[cache],
                all_tokens=[job.tokens[:job.cached_tokens]], samplers=[make_sampler(job.sampling)])[0]
            self.active[uid] = job
            self.prefilling[uid] = signature
            self.peak_active = max(self.peak_active, len(self.active))

    def step(self) -> list[TokenEvent]:
        if self.closed:
            raise RuntimeError("Engine is closed")
        self._admit()
        if not self.active:
            return []
        interactive = any(job.decoding or len(job.tokens) - job.cached_tokens <= self.config.prefill_step
                          for job in self.active.values())
        self.batch.set_prefill_step(self.config.prefill_step if interactive else self.config.idle_prefill_step)
        prompts, responses = self.batch.next()
        for response in prompts:
            if response.end_of_prompt:
                self.active[response.uid].decoding = True
                self.prefilling.pop(response.uid, None)
                if self.config.prefix_cache:
                    cache, tokens = self.batch.extract_cache([response.uid])[response.uid]
                    self.cache.save(self.active[response.uid].scope, tokens, cache)
        return [self._consume(response) for response in responses]

    def _consume(self, response: GenerationBatch.Response) -> TokenEvent:
        job = self.active[response.uid]
        job.generated += 1
        event = TokenEvent(job.id, response.token, response.finish_reason, len(job.tokens),
                           job.generated, job.cached_tokens)
        if response.finish_reason is not None:
            if self.config.prefix_cache:
                self.cache.save(job.scope, response.all_tokens, response.prompt_cache)
            del self.active[response.uid]
            self.prefilling.pop(response.uid, None)
        return event

    def cancel(self, job_id: int) -> bool:
        for job in self.pending:
            if job.id == job_id:
                self.pending.remove(job)
                return True
        for uid, job in self.active.items():
            if job.id == job_id:
                self.batch.remove([uid])
                del self.active[uid]
                self.prefilling.pop(uid, None)
                return True
        return False

    def close(self) -> None:
        if not self.closed:
            self.batch.close()
            self.pending.clear()
            self.active.clear()
            self.prefilling.clear()
            self.cache.clear()
            self.closed = True
