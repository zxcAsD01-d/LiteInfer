"""Small adapter for the pinned mlx-lm 0.32.2 prompt batch contract."""
from mlx_lm.generate import BatchGenerator, PromptProcessingBatch


class AdaptiveBatchGenerator(BatchGenerator):
    def set_prefill_step(self, tokens: int) -> None:
        if tokens <= 0:
            raise ValueError("Prefill step must be positive")
        # Both fields must change: _next slices prompts and prompt() executes them.
        if not isinstance(self._prompt_batch, PromptProcessingBatch):
            raise RuntimeError("mlx-lm prompt batch contract changed; review adaptive batching")
        self.prefill_step_size = tokens
        self._prompt_batch.prefill_step_size = tokens
