"""Hold possible stop prefixes so streamed stop strings never leak."""


class StopFilter:
    def __init__(self, sequences: tuple[str, ...]) -> None:
        if any(not stop for stop in sequences):
            raise ValueError("Stop strings must not be empty")
        self.sequences = sequences
        self.pending = ""
        self.text = ""
        self.stopped = False

    def push(self, delta: str, *, final: bool = False) -> str:
        if self.stopped:
            return ""
        self.pending += delta
        matches = [self.pending.find(stop) for stop in self.sequences if stop in self.pending]
        if matches:
            result = self.pending[:min(matches)]
            self.pending = ""
            self.stopped = True
        else:
            held = 0 if final else self._prefix_length()
            count = len(self.pending) - held
            result, self.pending = self.pending[:count], self.pending[count:]
        self.text += result
        return result

    def _prefix_length(self) -> int:
        return max((length for stop in self.sequences
                    for length in range(1, min(len(stop), len(self.pending) + 1))
                    if self.pending.endswith(stop[:length])), default=0)
