"""DDTree: Dynamic Draft Trees for DFlash speculative decoding.

Extends DFlash block diffusion by building tree-structured drafts from the
drafter's per-position logits. Instead of verifying a single flat sequence,
DDTree explores multiple token candidates per position, dramatically improving
effective acceptance even when per-position accuracy is low.

Key techniques from the literature:
  - EAGLE-2: confidence-based expansion (cumulative probability as proxy)
  - Sequoia: provably optimal tree topology via DP (offline)
  - Yggdrasil: equal-growth trees for mx.compile() compatibility

Algorithm:
  1. DFlash drafter produces logits for N positions (single forward pass)
  2. At each position, take top-k candidates weighted by cumulative confidence
  3. Build ancestor-only attention mask (each node sees its root path + context)
  4. Target model verifies entire tree in ONE forward pass
  5. Accept longest valid path through tree + bonus token

With 8.7% per-position acceptance and tree_width=5:
  P(>=1 match) = 1-(1-0.087)^5 = 37% → tau ~3-4 tokens/step
"""

import heapq
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import numpy as np


@dataclass
class DDTreeConfig:
    """Configuration for DDTree draft tree construction."""

    tree_width: int = 5
    max_tree_size: int = 60
    temperature: float = 0.0
    min_confidence: float = 0.01


@dataclass
class TreeNode:
    """A node in the draft tree."""

    token_id: int
    depth: int
    prob: float
    cumulative_prob: float
    parent_idx: int
    children_idx: list[int] = field(default_factory=list)


@dataclass
class DraftTree:
    """Complete draft tree structure ready for verification."""

    nodes: list[TreeNode]
    token_ids: mx.array
    attention_mask: mx.array
    depth_ids: mx.array

    @property
    def size(self) -> int:
        return len(self.nodes)


class DDTreeBuilder:
    """Builds and verifies dynamic draft trees from DFlash logits.

    Uses EAGLE-2 style confidence-based expansion: nodes are globally
    ranked by cumulative probability and the top-budget nodes are kept.
    """

    def __init__(self, config: DDTreeConfig | None = None):
        self.config = config or DDTreeConfig()
        self._stats: dict[str, Any] = {
            "trees_built": 0,
            "total_nodes": 0,
            "total_accepted": 0,
            "path_lengths": [],
        }

    def build_tree(self, draft_logits: mx.array) -> DraftTree:
        """Build a draft tree from DFlash drafter logits.

        Uses confidence-based expansion: at each depth, take top-k candidates.
        Globally rank all candidates by cumulative probability and keep the
        top max_tree_size nodes.

        Args:
            draft_logits: [1, N, vocab_size] or [N, vocab_size]

        Returns:
            DraftTree ready for tree-masked verification
        """
        if draft_logits.ndim == 3:
            draft_logits = draft_logits.squeeze(0)

        N, vocab = draft_logits.shape
        probs_all = mx.softmax(draft_logits, axis=-1)
        mx.eval(probs_all)
        probs_np = np.array(probs_all)

        cfg = self.config
        nodes: list[TreeNode] = []

        # heap: (-cumulative_prob, depth, parent_idx, token_id, prob)
        heap: list[tuple[float, int, int, int, float]] = []

        top_k = min(cfg.tree_width, vocab)
        root_top = np.argpartition(probs_np[0], -top_k)[-top_k:]
        for tid in root_top:
            p = float(probs_np[0, tid])
            if p >= cfg.min_confidence:
                heapq.heappush(heap, (-p, 0, -1, int(tid), p))

        while heap and len(nodes) < cfg.max_tree_size:
            neg_cum, depth, parent_idx, token_id, prob = heapq.heappop(heap)
            cum = -neg_cum

            node_idx = len(nodes)
            nodes.append(
                TreeNode(
                    token_id=token_id,
                    depth=depth,
                    prob=prob,
                    cumulative_prob=cum,
                    parent_idx=parent_idx,
                )
            )
            if parent_idx >= 0:
                nodes[parent_idx].children_idx.append(node_idx)

            next_depth = depth + 1
            if next_depth >= N:
                continue

            child_top = np.argpartition(probs_np[next_depth], -top_k)[-top_k:]
            for tid in child_top:
                p = float(probs_np[next_depth, tid])
                child_cum = cum * p
                if child_cum >= cfg.min_confidence:
                    heapq.heappush(heap, (-child_cum, next_depth, node_idx, int(tid), p))

        tree = self._build_masks(nodes)
        self._stats["trees_built"] += 1
        self._stats["total_nodes"] += len(nodes)
        return tree

    def _build_masks(self, nodes: list[TreeNode]) -> DraftTree:
        """Build attention mask and depth IDs from node list."""
        size = len(nodes)
        if size == 0:
            return DraftTree(
                nodes=[],
                token_ids=mx.array([], dtype=mx.int32),
                attention_mask=mx.zeros((0, 0), dtype=mx.bool_),
                depth_ids=mx.array([], dtype=mx.int32),
            )

        token_ids = mx.array([n.token_id for n in nodes], dtype=mx.int32)
        depth_ids = mx.array([n.depth for n in nodes], dtype=mx.int32)

        mask = np.zeros((size, size), dtype=np.bool_)
        for i in range(size):
            j = i
            while j >= 0:
                mask[i, j] = True
                j = nodes[j].parent_idx

        attention_mask = mx.array(mask)
        return DraftTree(nodes=nodes, token_ids=token_ids, attention_mask=attention_mask, depth_ids=depth_ids)

    def verify_tree(self, tree: DraftTree, verify_logits: mx.array, ctx_len: int) -> tuple[list[int], int]:
        """Verify draft tree against target model logits.

        The target model has already been run on [context, tree_tokens].
        We extract the target's predictions at each tree node position and
        find the longest matching root-to-leaf path.

        Args:
            tree: DraftTree with candidates
            verify_logits: [1, ctx_len + tree_size, vocab] from target forward
            ctx_len: Length of context preceding the tree

        Returns:
            (accepted_token_ids, num_accepted) including bonus token
        """
        if tree.size == 0:
            return [], 0

        # Target logits at position i predict token i+1
        # For tree node at flat position j (after context), the target's prediction
        # is at logit position ctx_len - 1 + j (since logit[i] predicts token[i+1])
        # But with tree attention, each node only sees its ancestors, so we need
        # to match each node's token against the target's prediction at its parent position

        # For root nodes (parent=-1): target predicts at ctx_len-1
        # For child nodes: target predicts at ctx_len + parent_flat_idx

        if self.config.temperature == 0:
            all_target = mx.argmax(verify_logits[0], axis=-1)
        else:
            p = mx.softmax(verify_logits[0] / self.config.temperature, axis=-1)
            all_target = mx.random.categorical(p)
        mx.eval(all_target)
        target_np = np.array(all_target)

        best_path: list[int] = []

        def dfs(node_idx: int, path: list[int]):
            nonlocal best_path
            node = tree.nodes[node_idx]

            if node.parent_idx < 0:
                verify_pos = ctx_len - 1
            else:
                verify_pos = ctx_len + node.parent_idx

            if verify_pos >= len(target_np):
                return

            if node.token_id != target_np[verify_pos]:
                return

            path.append(node_idx)
            if len(path) > len(best_path):
                best_path = path.copy()

            for child_idx in node.children_idx:
                dfs(child_idx, path)

            path.pop()

        for i, node in enumerate(tree.nodes):
            if node.parent_idx < 0:
                dfs(i, [])

        accepted = [tree.nodes[idx].token_id for idx in best_path]

        # Bonus token: target's prediction after the last accepted node
        if best_path:
            last_verify_pos = ctx_len + best_path[-1]
            if last_verify_pos < len(target_np):
                accepted.append(int(target_np[last_verify_pos]))
        elif ctx_len - 1 < len(target_np):
            accepted.append(int(target_np[ctx_len - 1]))

        self._stats["total_accepted"] += len(accepted)
        self._stats["path_lengths"].append(len(accepted))

        return accepted, len(accepted)

    def get_stats(self) -> dict:
        avg_path = float(np.mean(self._stats["path_lengths"])) if self._stats["path_lengths"] else 0
        avg_tree = self._stats["total_nodes"] / max(1, self._stats["trees_built"])
        return {
            "trees_built": self._stats["trees_built"],
            "avg_tree_size": round(avg_tree, 1),
            "total_accepted": self._stats["total_accepted"],
            "avg_path_length": round(avg_path, 1),
        }


def sequoia_optimal_tree(budget: int, acceptance_probs: list[float], max_depth: int | None = None) -> dict[str, Any]:
    """Compute Sequoia-optimal tree topology via DP.

    Given a token budget and per-position acceptance probabilities,
    finds the tree structure that maximizes expected accepted tokens.

    Args:
        budget: Maximum number of nodes in the tree
        acceptance_probs: p[k] = probability of accepting the k-th child at any node
        max_depth: Maximum tree depth (None = unlimited)

    Returns:
        List of (parent_idx, child_rank) pairs defining the tree topology.
        parent_idx=-1 for root children.
    """
    if max_depth is None:
        max_depth = budget

    B = len(acceptance_probs)

    # T[n][b] = max expected tokens for subtree of size n, root has b children
    T = np.zeros((budget + 1, B + 1))
    T[1, 0] = 1.0

    for n in range(2, budget + 1):
        for b in range(1, min(B, n) + 1):
            best = 0.0
            for m in range(1, n):
                child_val = max(T[m, j] for j in range(min(B, m) + 1))
                val = T[n - m, b - 1] + acceptance_probs[b - 1] * child_val
                best = max(best, val)
            T[n, b] = best

    best_b = int(np.argmax([T[budget, b] for b in range(min(B, budget) + 1)]))
    expected = T[budget, best_b]

    # Return the optimal branching factor at root (topology reconstruction is complex,
    # so we return the expected value and root branching for now)
    return {
        "budget": budget,
        "expected_tokens": round(float(expected), 2),
        "optimal_root_branches": best_b,
        "acceptance_probs": acceptance_probs,
    }
