"""Compiled sparse sampling with stable candidates and independent request keys."""
from collections.abc import Callable
from functools import lru_cache

import mlx.core as mx
from mlx_lm.sample_utils import apply_top_p

from .config import SamplingConfig

SamplingKernel = Callable[[mx.array, mx.array], tuple[mx.array, mx.array]]


def vocabulary_processor(size: int) -> Callable[[mx.array, mx.array], mx.array]:
    """Mask padded model rows which have no token in the tokenizer vocabulary."""
    if size <= 0:
        raise ValueError("Tokenizer vocabulary must not be empty")

    def process(tokens: mx.array, logits: mx.array) -> mx.array:
        if logits.shape[-1] < size:
            raise ValueError("Model vocabulary is smaller than tokenizer vocabulary")
        return mx.where(mx.arange(logits.shape[-1]) < size, logits, -float("inf"))

    return process


def candidate_logits(logits: mx.array, temperature: float, top_k: int,
                     top_p: float) -> tuple[mx.array, mx.array | None]:
    if not logits.shape[-1]:
        raise ValueError("Logits vocabulary must not be empty")
    indices = None
    if top_k < logits.shape[-1]:
        # Positive temperature preserves rank: select before scaling the vocabulary.
        indices = mx.argpartition(-logits, kth=top_k - 1, axis=-1)[..., :top_k]
        indices = mx.sort(indices, axis=-1)
        logits = mx.take_along_axis(logits, indices, axis=-1)
    values = logits.astype(mx.float32)
    values = (values - mx.max(values, axis=-1, keepdims=True)) / max(temperature, 1e-30)
    if top_p < 1:
        values = values - mx.logsumexp(values, axis=-1, keepdims=True)
        values = apply_top_p(values, top_p)
    return values, indices


@lru_cache(maxsize=32)
def sampling_kernel(temperature: float, top_k: int, top_p: float) -> SamplingKernel:
    @mx.compile
    def sample(logits: mx.array, key: mx.array) -> tuple[mx.array, mx.array]:
        key, sample_key = mx.random.split(key)
        values, indices = candidate_logits(logits, temperature, top_k, top_p)
        sampled = mx.random.categorical(values, key=sample_key)
        if indices is not None:
            sampled = mx.take_along_axis(indices, sampled[..., None], axis=-1).squeeze(-1)
        return key, sampled
    return sample


class CompiledSampler:
    def __init__(self, config: SamplingConfig) -> None:
        self.key = mx.random.key(config.seed)
        self.sample = sampling_kernel(config.temperature, config.top_k, config.top_p)

    def __call__(self, logits: mx.array) -> mx.array:
        self.key, sampled = self.sample(logits, self.key)
        mx.async_eval(self.key)
        return sampled


def make_sampler(config: SamplingConfig) -> Callable[[mx.array], mx.array]:
    return (lambda logits: mx.argmax(logits, axis=-1)) if config.temperature == 0 else CompiledSampler(config)
