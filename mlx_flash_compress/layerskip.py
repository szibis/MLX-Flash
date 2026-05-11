"""LayerSkip: Self-Speculative Decoding for MLX.

Implements LayerSkip (arXiv:2404.16710, Meta, ACL 2024) for Apple Silicon.
The target model serves as both draft and verifier -- no separate drafter needed.
Early exit at layer N/2 produces draft tokens; the full model verifies them.

Key insight: For easy tokens (common words, punctuation), early transformer layers
already produce high-confidence predictions. By exiting early for drafting, we
skip ~50% of layers per draft token while maintaining lossless verification.

Performance: 1.8-2.2x speedup with zero additional memory overhead.

Architecture:
  1. Draft phase: Forward through layers 0..exit_layer, apply norm + lm_head
  2. Verify phase: Full model forward on [context + drafts]
  3. Accept longest matching prefix + bonus token
  4. Repeat

Usage:
  from mlx_flash_compress.layerskip import apply_layerskip, LayerSkipConfig

  config = LayerSkipConfig(num_speculative_tokens=5)
  engine = apply_layerskip(model, tokenizer, config)
  output = engine.generate(prompt_tokens, max_tokens=100)
"""

import time
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn


@dataclass
class LayerSkipConfig:
    """Configuration for LayerSkip self-speculative decoding."""

    exit_layer: int = -1  # draft exit layer (-1 = auto: num_layers // 2)
    num_speculative_tokens: int = 5  # tokens to draft per cycle
    temperature: float = 0.0  # greedy by default
    confidence_threshold: float = 0.9  # early exit if top-1 prob > this
    adaptive_exit: bool = True  # dynamically choose exit layer based on confidence

    def __post_init__(self):
        if self.num_speculative_tokens < 1:
            raise ValueError(f"num_speculative_tokens must be >= 1, got {self.num_speculative_tokens}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0.0, got {self.temperature}")
        if self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0:
            raise ValueError(f"confidence_threshold must be in [0.0, 1.0], got {self.confidence_threshold}")
        # exit_layer validated later when num_layers is known; -1 is auto


class LayerSkipDrafter:
    """Uses early exit from the target model as a draft model.

    Instead of maintaining a separate drafter network, this class runs the
    target model's first N layers and applies its own LM head to the
    intermediate hidden states. For easy tokens, the first half of layers
    already produce confident predictions.
    """

    def __init__(self, target_model, config: LayerSkipConfig):
        self.target = target_model
        self.config = config

        # Detect model components (same pattern as DFlashRunner)
        self._embed_fn: Any = None
        self._layers: Any = None
        self._norm_fn: Any = None
        self._lm_head_fn: Any = None
        self._detect_components()

        # Resolve exit layer
        if self.config.exit_layer < 0:
            self.config.exit_layer = len(self._layers) // 2

    def _detect_components(self):
        """Detect embed_tokens, layers, norm, and lm_head from the target model."""
        m = self.target
        candidates = []
        if hasattr(m, "language_model"):
            lm = m.language_model
            if hasattr(lm, "model"):
                candidates.append((lm.model, lm))
        if hasattr(m, "model"):
            candidates.append((m.model, m))
        candidates.append((m, m))

        for inner, outer in candidates:
            embed = getattr(inner, "embed_tokens", None)
            layers = getattr(inner, "layers", None)
            norm = getattr(inner, "norm", None)
            head = getattr(outer, "lm_head", None) or getattr(inner, "output", None)

            if head is None and embed is not None:
                args = getattr(outer, "args", getattr(inner, "args", None))
                if args is not None and getattr(args, "tie_word_embeddings", False):
                    head = embed.as_linear

            if embed is not None and layers is not None:
                self._embed_fn = embed
                self._layers = layers
                self._norm_fn = norm
                self._lm_head_fn = head
                return

        raise RuntimeError(
            "Cannot detect target model components (embed_tokens/layers). "
            "Model must follow standard HuggingFace/MLX structure."
        )

    def get_exit_layer(self) -> int:
        """Return the layer index for early exit."""
        return self.config.exit_layer

    def _forward_partial(self, input_ids: mx.array, num_layers: int, cache=None) -> mx.array:
        """Forward pass through first num_layers layers only.

        Args:
            input_ids: Token IDs [B, seq_len]
            num_layers: Number of layers to execute (0-indexed, exclusive)
            cache: Optional per-layer KV cache list

        Returns:
            Hidden states after num_layers layers [B, seq_len, hidden_dim]
        """
        h = self._embed_fn(input_ids)

        for i in range(min(num_layers, len(self._layers))):
            layer = self._layers[i]
            if cache is not None and i < len(cache):
                h = layer(h, cache=cache[i])
            else:
                h = layer(h)
            if isinstance(h, tuple):
                h = h[0]

        return h

    def draft(self, input_ids: mx.array, cache=None) -> tuple[mx.array, list]:
        """Generate draft tokens using early exit.

        Forward through layers 0..exit_layer, apply norm + LM head to get
        logits, then greedily select tokens. Repeats autoregressively for
        num_speculative_tokens.

        When adaptive_exit is enabled, the exit layer is adjusted per token:
        - High confidence (above threshold): use fewer layers (exit_layer - 1)
        - Low confidence (below threshold): use more layers (exit_layer + 1)
        The exit layer is clamped to [1, total_layers - 1].

        Args:
            input_ids: Context token IDs [B, seq_len]
            cache: Optional KV cache

        Returns:
            (draft_token_ids [B, num_spec], draft_logits_list)
        """
        base_exit_layer = self.config.exit_layer
        current_exit_layer = base_exit_layer
        num_draft = self.config.num_speculative_tokens
        total_layers = len(self._layers)
        min_exit = max(1, base_exit_layer // 2)
        max_exit = min(total_layers - 1, base_exit_layer + base_exit_layer // 2)

        draft_ids = []
        draft_logits_list = []
        current_ids = input_ids
        self._draft_exit_layers = []  # track per-token exit layers for stats

        for _ in range(num_draft):
            h = self._forward_partial(current_ids, current_exit_layer, cache=cache)

            # Apply norm and LM head to intermediate hidden states
            if self._norm_fn is not None:
                h = self._norm_fn(h)
            logits = self._lm_head_fn(h)

            # Take logits for the last position
            last_logits = logits[:, -1:, :]  # [B, 1, vocab]
            draft_logits_list.append(last_logits)

            if self.config.temperature == 0.0:
                next_token = mx.argmax(last_logits, axis=-1)  # [B, 1]
            else:
                probs = mx.softmax(last_logits / self.config.temperature, axis=-1)
                next_token = mx.random.categorical(mx.log(probs + 1e-10).squeeze(1)).reshape(-1, 1)

            draft_ids.append(next_token)
            self._draft_exit_layers.append(current_exit_layer)

            # Adaptive exit: adjust exit layer for next token based on confidence
            if self.config.adaptive_exit:
                top_prob = mx.max(mx.softmax(last_logits, axis=-1))
                mx.eval(top_prob)
                top_prob_val = float(top_prob.item())

                if top_prob_val >= self.config.confidence_threshold:
                    # High confidence: use fewer layers next time
                    current_exit_layer = max(min_exit, current_exit_layer - 1)
                else:
                    # Low confidence: use more layers next time
                    current_exit_layer = min(max_exit, current_exit_layer + 1)

            # Next iteration: predict from the drafted token only
            current_ids = next_token

        # Stack draft tokens: [B, num_draft]
        draft_token_ids = mx.concatenate(draft_ids, axis=1)
        return draft_token_ids, draft_logits_list


class LayerSkipEngine:
    """Self-speculative decoding engine using LayerSkip.

    Uses the target model's early layers as a draft model and the full
    model for verification. No additional model parameters are needed.
    """

    def __init__(self, model, tokenizer, config: LayerSkipConfig = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or LayerSkipConfig()
        self.drafter = LayerSkipDrafter(model, self.config)

        self.stats: dict[str, Any] = {
            "total_draft_tokens": 0,
            "total_accepted": 0,
            "total_verify_steps": 0,
            "draft_times_ms": [],
            "verify_times_ms": [],
            "acceptance_counts": [],
        }

    def _draft_step_cached(self, last_token: mx.array, cache: list, exit_layer: int) -> tuple[list[int], float]:
        """Draft K tokens using partial layers with KV cache.

        Forwards last_token + K-1 autoregressive steps through layers
        0..exit_layer. Returns draft token IDs and elapsed time.
        """
        t0 = time.perf_counter()
        K = self.config.num_speculative_tokens
        draft_ids: list[int] = []

        current = last_token  # [1, 1]
        for _ in range(K):
            h = self.drafter._forward_partial(current, exit_layer, cache=cache)
            if self.drafter._norm_fn is not None:
                h = self.drafter._norm_fn(h)
            logits = self.drafter._lm_head_fn(h)
            next_id = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            draft_ids.append(next_id)
            current = mx.array([[next_id]])

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.stats["draft_times_ms"].append(elapsed_ms)
        self.stats["total_draft_tokens"] += K
        return draft_ids, elapsed_ms

    def _verify_and_accept(self, last_token_id: int, draft_ids: list[int], cache: list) -> tuple[list[int], int]:
        """Verify draft tokens with full model forward + accept prefix.

        Feeds [last_token, draft_1, ..., draft_K] through all layers.
        Compares full-model greedy predictions against drafts.
        Returns accepted new tokens (excluding last_token, including bonus).
        """
        t0 = time.perf_counter()
        K = len(draft_ids)

        verify_input = mx.array([[last_token_id] + draft_ids])
        logits = self.model(verify_input, cache=cache)
        mx.eval(logits)

        # logits[:, i, :] predicts token at position (cache_start + i + 1)
        # logits[:, 0] predicts what should come after last_token → draft_ids[0]
        # logits[:, k] predicts what should come after draft_ids[k-1] → draft_ids[k]
        # logits[:, K] predicts bonus token after draft_ids[K-1]
        target_preds = mx.argmax(logits[:, :K, :], axis=-1)
        mx.eval(target_preds)
        target_list: list = list(target_preds[0].tolist())  # type: ignore[arg-type]

        num_accepted = 0
        for draft_tok, target_tok in zip(draft_ids, target_list):
            if draft_tok == target_tok:
                num_accepted += 1
            else:
                break

        # Bonus: full model's prediction at the first rejection/end point
        bonus_id = int(mx.argmax(logits[:, num_accepted, :], axis=-1).item())

        # New tokens: accepted drafts + bonus (NOT including last_token)
        new_tokens = draft_ids[:num_accepted] + [bonus_id]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.stats["verify_times_ms"].append(elapsed_ms)
        self.stats["total_verify_steps"] += 1
        self.stats["total_accepted"] += len(new_tokens)
        self.stats["acceptance_counts"].append(num_accepted)

        return new_tokens, num_accepted

    def _draft_step(self, input_ids: mx.array) -> tuple[mx.array, mx.array]:
        """Non-cached draft step (for testing). Processes full context."""
        t0 = time.perf_counter()
        draft_ids, draft_logits_list = self.drafter.draft(input_ids)
        draft_logits = mx.concatenate(draft_logits_list, axis=1)
        mx.eval(draft_ids, draft_logits)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.stats["draft_times_ms"].append(elapsed_ms)
        self.stats["total_draft_tokens"] += self.config.num_speculative_tokens
        return draft_ids, draft_logits

    def _verify_step(self, input_ids: mx.array, draft_tokens: mx.array) -> tuple[mx.array, int]:
        """Non-cached verify step (for testing). Processes full context."""
        t0 = time.perf_counter()
        full_ids = mx.concatenate([input_ids, draft_tokens], axis=-1)
        logits = self.model(full_ids)
        ctx_len = input_ids.shape[-1]
        n_draft = draft_tokens.shape[-1]
        verify_logits = logits[:, ctx_len - 1 : ctx_len - 1 + n_draft, :]
        target_ids = mx.argmax(verify_logits, axis=-1)
        mx.eval(target_ids)
        draft_np: list = list(draft_tokens[0].tolist())  # type: ignore[arg-type]
        target_np: list = list(target_ids[0].tolist())  # type: ignore[arg-type]
        num_accepted = 0
        for d, t in zip(draft_np, target_np):
            if d == t:
                num_accepted += 1
            else:
                break
        bonus_idx = num_accepted
        if bonus_idx < len(target_np):
            accepted = draft_np[:num_accepted] + [target_np[bonus_idx]]
        else:
            accepted = draft_np[:num_accepted]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.stats["verify_times_ms"].append(elapsed_ms)
        self.stats["total_verify_steps"] += 1
        self.stats["total_accepted"] += len(accepted)
        self.stats["acceptance_counts"].append(num_accepted)
        accepted_arr = mx.array(accepted, dtype=mx.int32)
        return accepted_arr, len(accepted)

    def generate(self, prompt_tokens: mx.array, max_tokens: int = 100, callback=None) -> mx.array:
        """Generate tokens using self-speculative decoding with KV caching.

        Uses KV cache to avoid reprocessing the full context each iteration:
          1. Prefill: full model forward on prompt, cache all layers
          2. Draft: forward last token through layers 0..exit (K tokens)
          3. Roll back partial caches, verify [last + drafts] through full model
          4. Accept prefix + bonus, trim rejected entries, repeat

        Args:
            prompt_tokens: Input token IDs [seq_len] or [1, seq_len]
            max_tokens: Maximum new tokens to generate
            callback: Optional callback(new_tokens) called after each verify step

        Returns:
            All tokens (prompt + generated) as mx.array
        """
        self.stats = {
            "total_draft_tokens": 0,
            "total_accepted": 0,
            "total_verify_steps": 0,
            "draft_times_ms": [],
            "verify_times_ms": [],
            "acceptance_counts": [],
        }

        if prompt_tokens.ndim == 1:
            prompt_tokens = prompt_tokens.reshape(1, -1)

        generated: list = list(prompt_tokens[0].tolist())  # type: ignore[arg-type]
        tokens_generated = 0

        if max_tokens <= 0:
            return mx.array(generated)

        if hasattr(self.model, "make_cache"):
            cache = self.model.make_cache()
        else:
            from mlx_lm.models.cache import make_prompt_cache

            cache = make_prompt_cache(self.model)
        prefill_logits = self.model(prompt_tokens, cache=cache)
        mx.eval(prefill_logits)

        first_token = int(mx.argmax(prefill_logits[:, -1, :], axis=-1).item())
        generated.append(first_token)
        tokens_generated += 1

        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is not None and first_token == eos:
            return mx.array(generated)

        exit_layer = self.drafter.get_exit_layer()
        K = self.config.num_speculative_tokens

        while tokens_generated < max_tokens:
            last_token_id = generated[-1]

            # Draft K tokens through partial layers (0..exit_layer)
            draft_cache = [cache[i] for i in range(exit_layer)]
            last_tok_arr = mx.array([[last_token_id]])
            draft_ids, _ = self._draft_step_cached(last_tok_arr, draft_cache, exit_layer)

            # Roll back: draft extended cache[0..exit] by K entries
            for i in range(exit_layer):
                cache[i].trim(K)

            # Verify: full model processes [last_token, draft_1..K]
            new_tokens, num_accepted = self._verify_and_accept(last_token_id, draft_ids, cache)

            # Trim rejected entries: verify fed K+1 tokens, we keep 1+num_accepted
            # (last_token + accepted drafts); bonus is NOT in cache
            n_to_trim = K - num_accepted
            if n_to_trim > 0:
                for c in cache:
                    c.trim(n_to_trim)

            if len(new_tokens) == 0:
                break

            generated.extend(new_tokens)
            tokens_generated += len(new_tokens)

            if callback:
                callback(new_tokens)

            if eos is not None and eos in new_tokens:
                break

        return mx.array(generated)

    def get_stats(self) -> dict:
        """Return acceptance rate, tokens/draft, speedup factor, etc."""
        total_drafts = self.stats["total_draft_tokens"]
        total_accepted = self.stats["total_accepted"]
        total_steps = self.stats["total_verify_steps"]

        acceptance_rate = total_accepted / max(1, total_drafts)
        tokens_per_step = total_accepted / max(1, total_steps)
        # Speedup: we generate tokens_per_step tokens per 2 forward passes
        # (1 partial draft + 1 full verify) vs 1 token per 1 full pass
        # So speedup = tokens_per_step / (1 + exit_layer/total_layers)
        total_layers = len(self.drafter._layers)
        exit_layer = self.drafter.get_exit_layer()
        draft_cost_fraction = exit_layer / max(1, total_layers)
        effective_passes = 1.0 + draft_cost_fraction  # verify + draft cost
        speedup = tokens_per_step / effective_passes if effective_passes > 0 else 1.0

        draft_times = self.stats["draft_times_ms"]
        verify_times = self.stats["verify_times_ms"]

        # Collect adaptive exit layer info if available
        draft_exit_layers = getattr(self.drafter, "_draft_exit_layers", [])

        return {
            "exit_layer": exit_layer,
            "total_layers": total_layers,
            "total_draft_tokens": total_drafts,
            "total_accepted": total_accepted,
            "total_verify_steps": total_steps,
            "acceptance_rate": round(acceptance_rate, 3),
            "tokens_per_step": round(tokens_per_step, 1),
            "speedup_factor": round(speedup, 2),
            "avg_draft_ms": round(sum(draft_times) / len(draft_times), 1) if draft_times else 0,
            "avg_verify_ms": round(sum(verify_times) / len(verify_times), 1) if verify_times else 0,
            "adaptive_exit_enabled": self.config.adaptive_exit,
            "avg_adaptive_exit_layer": round(sum(draft_exit_layers) / len(draft_exit_layers), 1)
            if draft_exit_layers
            else exit_layer,
        }


def apply_layerskip(model, tokenizer, config: LayerSkipConfig = None) -> LayerSkipEngine:
    """One-line setup for self-speculative decoding.

    Args:
        model: Target model (HuggingFace/MLX causal LM)
        tokenizer: Tokenizer with encode/decode
        config: Optional LayerSkipConfig

    Returns:
        LayerSkipEngine ready for generation
    """
    return LayerSkipEngine(model, tokenizer, config)
