"""Compressed token radix index with constant-time subtree representatives."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class Node:
    edge: tuple[int, ...] = ()
    children: dict[int, Node] = field(default_factory=dict)
    key: str | None = None
    trimmable: bool = False
    sample: str | None = None

    def refresh(self) -> None:
        self.sample = self.key if self.trimmable else None
        if self.sample is None:
            self.sample = next((c.sample for c in self.children.values() if c.sample), None)


class RadixIndex:
    def __init__(self) -> None:
        self.roots: dict[str, Node] = {}
        self.entries = self.lookups = self.visited = self.last_visited = 0
        self.lookup_ns: deque[int] = deque(maxlen=512)

    def add(self, scope: str, tokens: tuple[int, ...], key: str, trimmable: bool) -> None:
        if not tokens:
            raise ValueError("Cannot index an empty prefix")
        node = self.roots.setdefault(scope, Node())
        path, offset = [node], 0
        while offset < len(tokens):
            child = node.children.get(tokens[offset])
            if child is None:
                child = Node(tokens[offset:])
                node.children[tokens[offset]] = child
            common = self._common(child.edge, tokens, offset)
            if common < len(child.edge):
                tail = Node(child.edge[common:], child.children, child.key, child.trimmable, child.sample)
                child.edge = child.edge[:common]
                child.children, child.key, child.trimmable = {tail.edge[0]: tail}, None, False
            offset += common
            node = child
            path.append(node)
        self.entries += int(node.key is None)
        node.key, node.trimmable = key, trimmable
        for item in reversed(path):
            item.refresh()

    @staticmethod
    def _common(edge: tuple[int, ...], tokens: tuple[int, ...], offset: int) -> int:
        count = 0
        for index in range(min(len(edge), len(tokens) - offset)):
            if edge[index] != tokens[offset + index]:
                break
            count += 1
        return count

    def nearest(self, scope: str, tokens: tuple[int, ...]) -> tuple[str | None, int]:
        started = time.perf_counter_ns()
        node, offset, best, length = self.roots.get(scope), 0, None, 0
        visits = 0
        while node is not None and offset < len(tokens):
            node = node.children.get(tokens[offset])
            if node is None:
                break
            visits += 1
            common = self._common(node.edge, tokens, offset)
            offset += common
            if node.sample is not None and offset > length:
                best, length = node.sample, offset
            if common < len(node.edge):
                break
            if node.key is not None and offset >= length:
                best, length = node.key, offset
        self.lookups += 1
        self.last_visited = visits
        self.visited += visits
        self.lookup_ns.append(time.perf_counter_ns() - started)
        return best, length

    def prefixes(self, scope: str, tokens: tuple[int, ...]) -> list[str]:
        node, offset, found = self.roots.get(scope), 0, []
        while node is not None and offset < len(tokens):
            node = node.children.get(tokens[offset])
            if node is None or self._common(node.edge, tokens, offset) != len(node.edge):
                break
            offset += len(node.edge)
            if node.key is not None:
                found.append(node.key)
        return found

    def remove(self, scope: str, tokens: tuple[int, ...]) -> None:
        root = self.roots.get(scope)
        if root is None:
            return
        node, offset, path = root, 0, []
        while offset < len(tokens):
            child = node.children.get(tokens[offset])
            if child is None or self._common(child.edge, tokens, offset) != len(child.edge):
                return
            path.append((node, child.edge[0], child))
            node, offset = child, offset + len(child.edge)
        if node.key is None:
            return
        node.key, node.trimmable = None, False
        self.entries -= 1
        for parent, token, child in reversed(path):
            if child.key is None and not child.children:
                del parent.children[token]
            elif child.key is None and len(child.children) == 1:
                tail = next(iter(child.children.values()))
                child.edge += tail.edge
                child.children, child.key, child.trimmable = tail.children, tail.key, tail.trimmable
            child.refresh()
            parent.refresh()
        if not root.children:
            del self.roots[scope]

    def snapshot(self) -> dict[str, int | float]:
        stack, nodes, edge_tokens = list(self.roots.values()), 0, 0
        while stack:
            node = stack.pop()
            nodes += 1
            edge_tokens += len(node.edge)
            stack.extend(node.children.values())
        durations = sorted(self.lookup_ns)
        return {"entries": self.entries, "scopes": len(self.roots), "nodes": nodes,
                "edge_tokens": edge_tokens, "lookups": self.lookups,
                "last_nodes_visited": self.last_visited,
                "mean_nodes_visited": self.visited / self.lookups if self.lookups else 0.0,
                "lookup_p50_us": durations[len(durations) // 2] / 1000 if durations else 0.0,
                "lookup_p95_us": durations[min(len(durations) - 1, int(len(durations) * .95))] / 1000 if durations else 0.0}
