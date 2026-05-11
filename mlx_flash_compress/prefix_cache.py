"""RadixAttention-style prefix trie for KV cache sharing across sessions.

Stores KV tensors keyed by token prefixes in a compressed trie (radix
tree).  When a new prompt shares a common prefix with a previous one
the cached KV data is reused instead of recomputed.

Trie nodes with single children are collapsed into a single node (radix
compression) to reduce memory overhead for long unique prefixes.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import mlx.core as mx


class TrieNode:
    __slots__ = ("tokens", "children", "kv_data", "last_access", "ref_count")

    def __init__(self, tokens: list[int] | None = None):
        self.tokens: list[int] = tokens or []
        self.children: dict[int, TrieNode] = {}
        self.kv_data: list[tuple[mx.array, mx.array]] | None = None
        self.last_access: float = 0.0
        self.ref_count: int = 0


class PrefixCacheTrie:
    """Thread-safe radix trie mapping token prefixes to cached KV data.

    Parameters
    ----------
    max_entries:
        Maximum number of trie nodes that store KV data.  When exceeded
        LRU eviction removes the least-recently-used entries.
    """

    def __init__(self, max_entries: int = 1024):
        self.max_entries = max_entries
        self._root = TrieNode()
        self._lock = threading.RLock()
        self._total_lookups = 0
        self._total_hits = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(self, token_ids: list[int], kv_data: list[tuple[mx.array, mx.array]]) -> None:
        """Store *kv_data* for the given token prefix."""
        with self._lock:
            self._insert(token_ids, kv_data)
            if self._count_kv_nodes(self._root) > self.max_entries:
                self.evict_lru(self.max_entries)

    def lookup(self, token_ids: list[int]) -> tuple[int, list[tuple[mx.array, mx.array]] | None]:
        """Return ``(match_length, kv_data)`` for the longest matching prefix.

        If no prefix matches, returns ``(0, None)``.
        """
        with self._lock:
            self._total_lookups += 1
            length, kv = self._lookup(token_ids)
            if length > 0:
                self._total_hits += 1
            return length, kv

    def evict_lru(self, max_entries: int) -> None:
        """Remove least-recently-used KV entries until at most *max_entries* remain."""
        with self._lock:
            nodes = self._collect_kv_nodes(self._root)
            nodes.sort(key=lambda n: n.last_access)
            to_evict = len(nodes) - max_entries
            for i in range(max(to_evict, 0)):
                nodes[i].kv_data = None
                nodes[i].ref_count = 0
            self._compress(self._root)

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            kv_nodes = self._collect_kv_nodes(self._root)
            total_tokens = sum(len(n.tokens) for n in self._all_nodes(self._root))
            mem = sum(self._node_memory(n) for n in kv_nodes)
            return {
                "entries": len(kv_nodes),
                "total_tokens": total_tokens,
                "hit_rate": self._total_hits / max(self._total_lookups, 1),
                "memory_bytes": mem,
                "total_lookups": self._total_lookups,
                "total_hits": self._total_hits,
            }

    # ------------------------------------------------------------------
    # Internal: insert
    # ------------------------------------------------------------------

    def _insert(self, tokens: list[int], kv_data: list[tuple[mx.array, mx.array]]) -> None:
        node = self._root
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok not in node.children:
                child = TrieNode(tokens[i:])
                child.kv_data = kv_data
                child.last_access = time.monotonic()
                child.ref_count = 1
                node.children[tok] = child
                return

            child = node.children[tok]
            prefix_len = self._common_prefix_len(child.tokens, tokens[i:])

            if prefix_len < len(child.tokens):
                # Split the existing node
                split = TrieNode(child.tokens[:prefix_len])
                child.tokens = child.tokens[prefix_len:]
                if child.tokens:
                    split.children[child.tokens[0]] = child
                else:
                    split.kv_data = child.kv_data
                    split.last_access = child.last_access
                    split.ref_count = child.ref_count
                node.children[tok] = split

                remaining = tokens[i + prefix_len :]
                if remaining:
                    new_leaf = TrieNode(remaining)
                    new_leaf.kv_data = kv_data
                    new_leaf.last_access = time.monotonic()
                    new_leaf.ref_count = 1
                    split.children[remaining[0]] = new_leaf
                else:
                    split.kv_data = kv_data
                    split.last_access = time.monotonic()
                    split.ref_count += 1
                return

            i += prefix_len
            node = child

        # Exact match with existing path
        node.kv_data = kv_data
        node.last_access = time.monotonic()
        node.ref_count += 1

    # ------------------------------------------------------------------
    # Internal: lookup
    # ------------------------------------------------------------------

    def _lookup(self, tokens: list[int]) -> tuple[int, list[tuple[mx.array, mx.array]] | None]:
        node = self._root
        matched = 0
        best_len = 0
        best_kv: list[tuple[mx.array, mx.array]] | None = None

        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok not in node.children:
                break
            child = node.children[tok]
            prefix_len = self._common_prefix_len(child.tokens, tokens[i:])
            if prefix_len == 0:
                break
            matched += prefix_len
            if prefix_len < len(child.tokens):
                # Partial node match -- kv_data belongs to the full node prefix
                break
            if child.kv_data is not None:
                best_len = matched
                best_kv = child.kv_data
                child.last_access = time.monotonic()
                child.ref_count += 1
            i += prefix_len
            node = child

        return best_len, best_kv

    # ------------------------------------------------------------------
    # Internal: compression & helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _common_prefix_len(a: list[int], b: list[int]) -> int:
        n = min(len(a), len(b))
        for j in range(n):
            if a[j] != b[j]:
                return j
        return n

    def _compress(self, node: TrieNode) -> None:
        """Collapse single-child chains (radix compression)."""
        for tok in list(node.children):
            child = node.children[tok]
            self._compress(child)
            if len(child.children) == 1 and child.kv_data is None:
                grandchild_tok = next(iter(child.children))
                grandchild = child.children[grandchild_tok]
                grandchild.tokens = child.tokens + grandchild.tokens
                node.children[tok] = grandchild

    def _collect_kv_nodes(self, node: TrieNode) -> list[TrieNode]:
        result: list[TrieNode] = []
        if node.kv_data is not None:
            result.append(node)
        for child in node.children.values():
            result.extend(self._collect_kv_nodes(child))
        return result

    def _all_nodes(self, node: TrieNode) -> list[TrieNode]:
        result = [node]
        for child in node.children.values():
            result.extend(self._all_nodes(child))
        return result

    def _count_kv_nodes(self, node: TrieNode) -> int:
        count = 1 if node.kv_data is not None else 0
        for child in node.children.values():
            count += self._count_kv_nodes(child)
        return count

    @staticmethod
    def _node_memory(node: TrieNode) -> int:
        if node.kv_data is None:
            return 0
        total = 0
        for k, v in node.kv_data:
            total += k.nbytes + v.nbytes
        return total


def get_or_compute_prefix(
    trie: PrefixCacheTrie,
    token_ids: list[int],
    compute_fn: Any,
    num_layers: int,
) -> tuple[int, list[tuple[mx.array, mx.array]]]:
    """Check the trie first; compute and store if missing.

    Parameters
    ----------
    trie:
        The prefix trie to query / populate.
    token_ids:
        Full token sequence for the current request.
    compute_fn:
        Callable ``(token_ids) -> list[tuple[mx.array, mx.array]]``
        that produces per-layer ``(keys, values)`` for the given tokens.
    num_layers:
        Number of transformer layers (used only for documentation /
        future assertions).

    Returns
    -------
    (cache_hit_length, kv_list):
        How many prefix tokens were served from cache, and the full
        per-layer KV list (from cache or freshly computed).
    """
    hit_len, cached = trie.lookup(token_ids)
    if cached is not None:
        return hit_len, cached

    kv_list = compute_fn(token_ids)
    trie.insert(token_ids, kv_list)
    return 0, kv_list
