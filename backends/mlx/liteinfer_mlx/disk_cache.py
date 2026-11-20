"""Bounded asynchronous SSD cache with a resident radix/SQLite index."""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
import fcntl
import hashlib
import heapq
import json
import logging
import os
from pathlib import Path
from queue import Queue
import re
import sqlite3
import threading
import time
from typing import Callable, Generic, Protocol, TypeVar
import uuid

from .cache_index import RadixIndex
from .cache_settings import CacheConfig

LOG = logging.getLogger(__name__)
T = TypeVar("T")
FILE_NAME = re.compile(r"[a-f0-9]{64}-[a-f0-9]{32}\.safetensors")


class Codec(Protocol[T]):
    def freeze(self, value: T) -> T: ...
    def save(self, path: Path, value: T) -> None: ...
    def load(self, path: Path) -> T: ...


@dataclass(frozen=True)
class Record:
    key: str
    scope: str
    tokens: tuple[int, ...]
    trimmable: bool
    cache_type: str
    filename: str
    size: int
    expires: float
    checksum: str


@dataclass(frozen=True)
class WriteJob(Generic[T]):
    key: str
    scope: str
    tokens: tuple[int, ...]
    value: T
    size: int
    trimmable: bool
    cache_type: str
    expires: float
    epoch: int


def key_digest(scope: str, tokens: tuple[int, ...]) -> str:
    data = json.dumps([scope, tokens], separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def checksum(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def latency(values: deque[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"samples": 0, "p50_ms": 0.0, "p95_ms": 0.0}
    return {"samples": len(ordered), "p50_ms": ordered[len(ordered) // 2],
            "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * .95))]}


class DiskCache(Generic[T]):
    def __init__(self, directory: Path, config: CacheConfig, codec: Codec[T],
                 clock: Callable[[], float] = time.time):
        if not config.ssd_max_bytes:
            raise ValueError("SSD cache is disabled")
        self.directory, self.config, self.codec, self.clock = directory, config, codec, clock
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
        self.guard = (directory / "cache.lock").open("a+b")
        os.chmod(self.guard.name, 0o600)
        try:
            fcntl.flock(self.guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.database = sqlite3.connect(directory / "index.sqlite3", check_same_thread=False)
            self._schema()
            self._initialize()
        except (OSError, sqlite3.Error, ValueError):
            if hasattr(self, "database"):
                self.database.close()
            self.guard.close()
            raise
        self.writer = threading.Thread(target=self._write_loop, name="ssd-cache-writer", daemon=True)
        self.writer.start()

    def _schema(self) -> None:
        self.database.execute("PRAGMA journal_mode=WAL")
        self.database.execute("PRAGMA synchronous=NORMAL")
        self.database.execute("""CREATE TABLE IF NOT EXISTS entries(
            key TEXT PRIMARY KEY, scope TEXT NOT NULL, tokens TEXT NOT NULL,
            trimmable INTEGER NOT NULL, cache_type TEXT NOT NULL, filename TEXT NOT NULL,
            size INTEGER NOT NULL, expires REAL NOT NULL, checksum TEXT NOT NULL, accessed REAL NOT NULL)""")
        self.database.execute("CREATE INDEX IF NOT EXISTS by_scope ON entries(scope, accessed)")
        self.database.commit()
        os.chmod(self.directory / "index.sqlite3", 0o600)

    def _initialize(self) -> None:
        self.lock = threading.RLock()
        self.index = RadixIndex()
        self.records: OrderedDict[str, Record] = OrderedDict()
        self.expiry: list[tuple[float, str]] = []
        self.pending: set[str] = set()
        self.queue: Queue[WriteJob[T] | None] = Queue(self.config.write_queue_entries)
        self.nbytes = self.pending_bytes = self.epoch = 0
        self.closed = False
        self.counts = dict.fromkeys(("hits", "misses", "writes", "evictions", "expirations",
                                    "read_errors", "write_errors", "queue_drops", "admission_skips"), 0)
        self.read_ms: deque[float] = deque(maxlen=512)
        self.write_ms: deque[float] = deque(maxlen=512)
        self._restore()

    def _restore(self) -> None:
        rows = self.database.execute("SELECT * FROM entries ORDER BY accessed").fetchall()
        for row in rows:
            key, scope, encoded, trim, kind, name, size, expires, digest, _ = row
            tokens = tuple(json.loads(encoded))
            path = self.directory / name
            valid = (FILE_NAME.fullmatch(name) and tokens and all(type(t) is int for t in tokens)
                     and key == key_digest(scope, tokens) and expires > self.clock()
                     and size <= self.config.ssd_max_bytes and path.is_file() and path.stat().st_size == size)
            if not valid:
                self.database.execute("DELETE FROM entries WHERE key=?", (key,))
                if FILE_NAME.fullmatch(name):
                    path.unlink(missing_ok=True)
                continue
            record = Record(key, scope, tokens, bool(trim), kind, name, size, expires, digest)
            self._remember(record)
        self._evict()
        live = {r.filename for r in self.records.values()}
        for path in self.directory.iterdir():
            if ((FILE_NAME.fullmatch(path.name) and path.name not in live)
                    or path.name.endswith(".tmp.safetensors")):
                path.unlink()
        self.database.commit()

    def _remember(self, record: Record) -> None:
        self.records[record.key] = record
        self.nbytes += record.size
        self.index.add(record.scope, record.tokens, record.key, record.trimmable)
        heapq.heappush(self.expiry, (record.expires, record.key))

    def _drop(self, key: str) -> None:
        record = self.records.pop(key)
        self.index.remove(record.scope, record.tokens)
        self.nbytes -= record.size
        self.database.execute("DELETE FROM entries WHERE key=?", (key,))
        (self.directory / record.filename).unlink(missing_ok=True)

    def _expire(self) -> None:
        now = self.clock()
        changed = False
        while self.expiry and self.expiry[0][0] <= now:
            deadline, key = heapq.heappop(self.expiry)
            record = self.records.get(key)
            if record is not None and record.expires == deadline:
                self._drop(key)
                self.counts["expirations"] += 1
                changed = True
        if changed:
            self.database.commit()

    def _evict(self) -> None:
        while len(self.records) > self.config.ssd_max_entries or self.nbytes > self.config.ssd_max_bytes:
            self._drop(next(iter(self.records)))
            self.counts["evictions"] += 1
        if len(self.expiry) > 2 * len(self.records) + 64:
            self.expiry = [(record.expires, key) for key, record in self.records.items()]
            heapq.heapify(self.expiry)

    def offer(self, scope: str, tokens: tuple[int, ...], value: T, size: int,
              trimmable: bool, cache_type: str = "assistant") -> bool:
        if size < 0:
            raise ValueError("Cache accounting cannot be negative")
        key = key_digest(scope, tokens)
        with self.lock:
            if not self._reserve(key, scope, tokens, size):
                return False
            epoch = self.epoch
        try:
            frozen = self.codec.freeze(value)
        except (ValueError, TypeError, RuntimeError):
            with self.lock:
                self.pending.remove(key)
                self.pending_bytes -= size
                self.counts["write_errors"] += 1
            LOG.exception("SSD cache snapshot failed")
            return False
        job = WriteJob(key, scope, tokens, frozen, size, trimmable, cache_type,
                       self.clock() + self.config.ttl_seconds, epoch)
        with self.lock:
            if self.closed or self.epoch != epoch:
                self.pending.remove(key)
                self.pending_bytes -= size
                return False
            self.queue.put_nowait(job)
        return True

    def _reserve(self, key: str, scope: str, tokens: tuple[int, ...], size: int) -> bool:
        if self.closed or key in self.records or key in self.pending:
            return False
        if not tokens or len(tokens) < self.config.min_tokens or size > self.config.ssd_max_bytes:
            self.counts["admission_skips"] += 1
            return False
        covering, length = self.index.nearest(scope, tokens)
        if covering is not None and length == len(tokens) and self.records[covering].trimmable:
            self.counts["admission_skips"] += 1
            return False
        if (len(self.pending) >= self.config.write_queue_entries
                or self.pending_bytes + size > self.config.write_queue_bytes):
            self.counts["queue_drops"] += 1
            return False
        self.pending.add(key)
        self.pending_bytes += size
        return True

    def fetch(self, scope: str, tokens: tuple[int, ...], minimum: int = 0) -> tuple[T | None, tuple[int, ...]]:
        started = time.perf_counter()
        with self.lock:
            try:
                self._expire()
                key, length = self.index.nearest(scope, tokens)
                if key is None or length <= minimum:
                    self.counts["misses"] += 1
                    return None, ()
                record = self.records[key]
                value = self._load(record)
            except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError, sqlite3.Error):
                self.counts["read_errors"] += 1
                self.counts["misses"] += 1
                LOG.warning("SSD cache read rejected; input will be recomputed", exc_info=True)
                return None, ()
            self.records.move_to_end(key)
            self.counts["hits"] += 1
            self.read_ms.append((time.perf_counter() - started) * 1000)
            return value, record.tokens

    def _load(self, record: Record) -> T:
        path = self.directory / record.filename
        try:
            if path.stat().st_size != record.size or checksum(path) != record.checksum:
                raise ValueError("SSD cache checksum mismatch")
            return self.codec.load(path)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError):
            self._drop(record.key)
            self.database.commit()
            raise

    def _write_loop(self) -> None:
        while True:
            job = self.queue.get()
            try:
                if job is None:
                    return
                self._write(job)
            finally:
                if job is not None:
                    with self.lock:
                        self.pending.remove(job.key)
                        self.pending_bytes -= job.size
                self.queue.task_done()

    def _write(self, job: WriteJob[T]) -> None:
        started = time.perf_counter()
        name = f"{job.key}-{uuid.uuid4().hex}.safetensors"
        temporary = self.directory / f"{name}.tmp.safetensors"
        try:
            self.codec.save(temporary, job.value)
            temporary.chmod(0o600)
            size, digest = temporary.stat().st_size, checksum(temporary)
            with self.lock:
                if self._publish(job, temporary, name, size, digest):
                    self.counts["writes"] += 1
                    self.write_ms.append((time.perf_counter() - started) * 1000)
        except (OSError, ValueError, TypeError, RuntimeError, sqlite3.Error):
            LOG.exception("SSD cache write failed; inference continues with memory cache")
            self._reject_write(job.key, name)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                with self.lock:
                    self.counts["write_errors"] += 1
                LOG.exception("SSD cache temporary file cleanup failed")

    def _reject_write(self, key: str, name: str) -> None:
        with self.lock:
            self.counts["write_errors"] += 1
            record = self.records.get(key)
            if record is not None and record.filename == name:
                self.records.pop(key)
                self.index.remove(record.scope, record.tokens)
                self.nbytes -= record.size
            try:
                self.database.rollback()
            except sqlite3.Error:
                LOG.exception("SSD cache metadata rollback failed")
            try:
                (self.directory / name).unlink(missing_ok=True)
            except OSError:
                LOG.exception("SSD cache failed payload cleanup failed")

    def _publish(self, job: WriteJob[T], temporary: Path, name: str, size: int, digest: str) -> bool:
        if job.epoch != self.epoch or job.expires <= self.clock() or size > self.config.ssd_max_bytes:
            self.counts["admission_skips"] += 1
            return False
        temporary.replace(self.directory / name)
        record = Record(job.key, job.scope, job.tokens, job.trimmable, job.cache_type,
                        name, size, job.expires, digest)
        self._expire()
        self.database.execute("INSERT OR REPLACE INTO entries VALUES(?,?,?,?,?,?,?,?,?,?)",
            (job.key, job.scope, json.dumps(job.tokens), job.trimmable, job.cache_type,
             name, size, job.expires, digest, self.clock()))
        self._remember(record)
        if record.trimmable:
            for prefix in self.index.prefixes(record.scope, record.tokens):
                if prefix != record.key and self.records[prefix].trimmable:
                    self._drop(prefix)
        self._evict()
        self.database.commit()
        return True

    def flush(self) -> None:
        self.queue.join()

    def clear(self) -> None:
        with self.lock:
            self.epoch += 1
            for key in list(self.records):
                self._drop(key)
            self.expiry.clear()
            self.database.commit()

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            total = self.counts["hits"] + self.counts["misses"]
            return {**self.counts, "entries": len(self.records), "bytes": self.nbytes,
                    "max_bytes": self.config.ssd_max_bytes, "pending_writes": len(self.pending),
                    "pending_bytes": self.pending_bytes,
                    "hit_rate": self.counts["hits"] / total if total else 0.0,
                    "read_latency": latency(self.read_ms), "write_latency": latency(self.write_ms),
                    "index": self.index.snapshot()}

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
        self.flush()
        self.queue.put(None)
        self.writer.join()
        self.database.close()
        self.guard.close()
