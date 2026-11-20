import random
import unittest

from liteinfer_mlx.cache_index import RadixIndex


class IndexTests(unittest.TestCase):
    def test_partial_prefix_scope_removal_and_compression(self):
        index = RadixIndex()
        index.add("a", (1, 2, 3, 4), "one", True)
        index.add("a", (1, 2, 5, 6), "two", True)
        index.add("b", (1, 2, 3, 4), "other-tenant", True)
        self.assertEqual(index.nearest("a", (1, 2, 3, 9)), ("one", 3))
        self.assertEqual(index.prefixes("a", (1, 2, 3, 4, 5)), ["one"])
        index.remove("a", (1, 2, 3, 4))
        self.assertEqual(index.nearest("a", (1, 2, 3)), ("two", 2))
        self.assertEqual(index.nearest("b", (1, 2, 3, 4)), ("other-tenant", 4))
        self.assertLessEqual(index.snapshot()["nodes"], 4)

    def test_recurrent_state_only_matches_complete_saved_prefix(self):
        index = RadixIndex()
        index.add("a", (1, 2, 3, 4), "hybrid", False)
        self.assertEqual(index.nearest("a", (1, 2, 3, 8)), (None, 0))
        self.assertEqual(index.nearest("a", (1, 2, 3, 4, 8)), ("hybrid", 4))
        index.add("a", (1, 2), "checkpoint", False)
        self.assertEqual(index.nearest("a", (1, 2, 3, 8)), ("checkpoint", 2))

    def test_random_operations_match_linear_reference(self):
        rng, index, entries = random.Random(42), RadixIndex(), {}
        for step in range(1200):
            scope = rng.choice(("a", "b", "c"))
            tokens = tuple(rng.randrange(5) for _ in range(rng.randrange(1, 12)))
            if rng.random() < .7:
                trim = bool(rng.randrange(2))
                key = f"{scope}:{tokens}"
                index.add(scope, tokens, key, trim)
                entries[scope, tokens] = (key, trim)
            else:
                index.remove(scope, tokens)
                entries.pop((scope, tokens), None)
            query = tuple(rng.randrange(5) for _ in range(rng.randrange(1, 14)))
            expected = 0
            for (namespace, prefix), (_, trim) in entries.items():
                if namespace != scope:
                    continue
                common = next((i for i, (a, b) in enumerate(zip(prefix, query)) if a != b),
                              min(len(prefix), len(query)))
                if trim or common == len(prefix):
                    expected = max(expected, common)
            key, length = index.nearest(scope, query)
            self.assertEqual(length, expected, f"step={step}")
            self.assertEqual(key is None, expected == 0)
        self.assertEqual(index.entries, len(entries))

    def test_lookup_does_not_scan_unrelated_entries(self):
        index = RadixIndex()
        for value in range(8192):
            index.add("tenant", (value, 1, 2, 3, 4), str(value), True)
        self.assertEqual(index.nearest("tenant", (7000, 1, 2, 3, 9)), ("7000", 4))
        self.assertEqual(index.last_visited, 1)


if __name__ == "__main__":
    unittest.main()
