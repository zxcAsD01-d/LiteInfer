"""Dependency-free settings shared by the installed and embedded MLX cache."""
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


@dataclass(frozen=True)
class CacheConfig:
    memory_max_entries: int = 64
    memory_max_bytes: int = 2 * 1024**3
    ssd_max_entries: int = 512
    ssd_max_bytes: int = 0
    ttl_seconds: int = 3600
    min_tokens: int = 128
    write_queue_entries: int = 2
    write_queue_bytes: int = 256 * 1024**2

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{item.name} must be a nonnegative integer")
        if self.ssd_max_bytes and not all((self.ssd_max_entries, self.ttl_seconds,
                                          self.write_queue_entries, self.write_queue_bytes)):
            raise ValueError("Enabled SSD cache requires positive entry, TTL and queue budgets")

    @classmethod
    def from_file(cls, path: Path) -> "CacheConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Cache configuration must be a JSON object")
        return cls(**data)

    def as_dict(self) -> dict[str, int]:
        return asdict(self)
