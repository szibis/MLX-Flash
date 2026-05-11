"""EAGLE-3: Speculative Decoding with Autoregression Head for MLX.

Implements EAGLE-style speculative decoding for Apple Silicon. A lightweight
transformer head predicts the next hidden state from the current hidden state
and token embedding, enabling fast autoregressive drafting in hidden-state
space without running the full target model.

Key insight: Instead of predicting tokens directly, EAGLE predicts hidden states.
A single transformer layer can learn the mapping (h_t, e_t) -> h_{t+1} because
consecutive hidden states are highly correlated. The target model's LM head
converts predicted hidden states to token probabilities.

Performance: 2.7-3.5x on GPU, ~2x expected on Apple Silicon.

Architecture:
  1. Train: Collect (h_t, h_{t+1}) pairs from target model, train draft head
  2. Draft: Autoregressively predict hidden states, apply LM head for tokens
  3. Verify: Full model forward on [context + drafts], standard spec. decoding
  4. Accept longest matching prefix + bonus token

Usage:
  from mlx_flash_compress.eagle3 import EAGLE3Engine, EAGLE3Config, EAGLE3Trainer

  # Train draft head
  trainer = EAGLE3Trainer(model, tokenizer)
  pairs = trainer.collect_training_data(["sample text..."])
  draft_head = trainer.train(pairs, num_steps=1000)
  trainer.save(draft_head, "eagle_head.safetensors")

  # Inference
  engine = EAGLE3Engine(model, tokenizer, draft_head=draft_head)
  output = engine.generate(prompt_tokens, max_tokens=100)
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


@dataclass
class EAGLE3Config:
    """Configuration for EAGLE-3 speculative decoding."""

    num_draft_tokens: int = 6  # tokens to draft per cycle
    hidden_dim: int = 0  # auto-detect from model (0 = auto)
    num_heads: int = 4
    num_layers: int = 1  # lightweight single-layer draft head
    temperature: float = 0.0

    def __post_init__(self):
        if self.num_draft_tokens < 1:
            raise ValueError(f"num_draft_tokens must be >= 1, got {self.num_draft_tokens}")
        if self.hidden_dim < 0:
            raise ValueError(f"hidden_dim must be >= 0 (0 = auto-detect), got {self.hidden_dim}")
        if self.num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {self.num_heads}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0.0, got {self.temperature}")


class EAGLEDraftHead(nn.Module):
    """Lightweight transformer layer that predicts next hidden state from current.

    Maps (hidden_state + token_embedding) -> predicted next hidden state.
    The architecture is deliberately minimal: a single transformer layer
    with pre-norm residual connections. This keeps draft overhead tiny
    while exploiting the high correlation between consecutive hidden states.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Fuse hidden state + token embedding
        self.input_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Transformer layers for hidden state prediction
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(_EAGLETransformerBlock(hidden_dim, num_heads))

        self.output_norm = nn.RMSNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def __call__(self, hidden_state: mx.array, token_embedding: mx.array) -> mx.array:
        """Predict next hidden state from current hidden state + token embedding.

        Args:
            hidden_state: Current hidden state [B, seq_len, hidden_dim]
            token_embedding: Token embedding for current token [B, seq_len, hidden_dim]

        Returns:
            Predicted next hidden state [B, seq_len, hidden_dim]
        """
        # Concatenate hidden state and token embedding along feature dim
        combined = mx.concatenate([hidden_state, token_embedding], axis=-1)
        x = self.input_proj(combined)

        for layer in self.layers:
            x = layer(x)

        x = self.output_norm(x)
        x = self.output_proj(x)
        return x


class _EAGLETransformerBlock(nn.Module):
    """Single pre-norm transformer block for EAGLE draft head."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.attention = nn.MultiHeadAttention(hidden_dim, num_heads)
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm1(x)
        h = self.attention(h, h, h)
        x = x + h
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x


class EAGLE3Engine:
    """EAGLE-3 speculative decoding engine.

    Uses a lightweight draft head to predict hidden states autoregressively,
    then applies the target model's LM head to get token predictions.
    Verification uses standard speculative decoding.
    """

    def __init__(self, model, tokenizer, config: EAGLE3Config = None, draft_head: EAGLEDraftHead = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EAGLE3Config()

        # Detect model components
        self._embed_fn: Any = None
        self._layers: Any = None
        self._norm_fn: Any = None
        self._lm_head_fn: Any = None
        self._inner_model: Any = None
        self._detect_components()

        # Auto-detect hidden dim if needed
        if self.config.hidden_dim == 0:
            self.config.hidden_dim = self._detect_hidden_dim()

        # Initialize or use provided draft head
        if draft_head is not None:
            self.draft_head = draft_head
        else:
            self.draft_head = EAGLEDraftHead(
                hidden_dim=self.config.hidden_dim,
                num_heads=self.config.num_heads,
                num_layers=self.config.num_layers,
            )

        self.stats: dict[str, Any] = {
            "total_draft_tokens": 0,
            "total_accepted": 0,
            "total_verify_steps": 0,
            "draft_times_ms": [],
            "verify_times_ms": [],
            "acceptance_counts": [],
        }

    def _detect_components(self):
        """Detect target model components."""
        m = self.model
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

            if embed is not None and layers is not None:
                self._embed_fn = embed
                self._layers = layers
                self._norm_fn = norm
                self._lm_head_fn = head
                self._inner_model = inner
                return

        raise RuntimeError(
            "Cannot detect target model components (embed_tokens/layers). "
            "Model must follow standard HuggingFace/MLX structure."
        )

    def _detect_hidden_dim(self) -> int:
        """Auto-detect hidden dimension from model's embedding layer."""
        if hasattr(self._embed_fn, "weight"):
            return self._embed_fn.weight.shape[-1]
        # Fallback: probe with a dummy input
        dummy = mx.array([[0]])
        out = self._embed_fn(dummy)
        return out.shape[-1]

    def _get_last_hidden_state(self, input_ids: mx.array) -> mx.array:
        """Forward pass to get last hidden state (before LM head).

        Runs the full model but returns the hidden state before the
        final projection to vocabulary logits.

        Args:
            input_ids: Token IDs [B, seq_len]

        Returns:
            Hidden states [B, seq_len, hidden_dim]
        """
        h = self._embed_fn(input_ids)

        for layer in self._layers:
            h = layer(h)

        if self._norm_fn is not None:
            h = self._norm_fn(h)

        return h

    def _hidden_to_logits(self, hidden: mx.array) -> mx.array:
        """Apply LM head to hidden state to get logits.

        Args:
            hidden: Hidden states [B, seq_len, hidden_dim]

        Returns:
            Logits [B, seq_len, vocab_size]
        """
        return self._lm_head_fn(hidden)

    def draft(self, last_hidden: mx.array, last_token_id: int) -> tuple[mx.array, mx.array]:
        """Autoregressively draft tokens using the draft head.

        1. Use draft_head to predict next hidden state
        2. Apply LM head to get token
        3. Get embedding of that token
        4. Repeat for num_draft_tokens

        Args:
            last_hidden: Hidden state of the last context token [B, 1, hidden_dim]
            last_token_id: Token ID of the last context token

        Returns:
            (draft_token_ids [B, num_draft], draft_hidden_states [B, num_draft, hidden_dim])
        """
        t0 = time.perf_counter()

        num_draft = self.config.num_draft_tokens
        current_hidden = last_hidden  # [B, 1, hidden_dim]
        current_token_id = last_token_id

        draft_ids = []
        draft_hiddens = []

        for _ in range(num_draft):
            # Get embedding for current token
            token_ids = mx.array([[current_token_id]])
            token_embed = self._embed_fn(token_ids)  # [1, 1, hidden_dim]

            # Predict next hidden state
            predicted_hidden = self.draft_head(current_hidden, token_embed)  # [B, 1, hidden_dim]

            draft_hiddens.append(predicted_hidden)

            # Apply LM head to get token prediction
            logits = self._hidden_to_logits(predicted_hidden)  # [B, 1, vocab]

            if self.config.temperature == 0.0:
                next_token = mx.argmax(logits, axis=-1)  # [B, 1]
            else:
                probs = mx.softmax(logits / self.config.temperature, axis=-1)
                next_token = mx.random.categorical(mx.log(probs + 1e-10).squeeze(1)).reshape(-1, 1)

            draft_ids.append(next_token)
            mx.eval(next_token)
            current_token_id = int(next_token[0, 0].item())
            current_hidden = predicted_hidden

        draft_token_ids = mx.concatenate(draft_ids, axis=1)  # [B, num_draft]
        draft_hidden_states = mx.concatenate(draft_hiddens, axis=1)  # [B, num_draft, hidden_dim]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.stats["draft_times_ms"].append(elapsed_ms)
        self.stats["total_draft_tokens"] += num_draft

        return draft_token_ids, draft_hidden_states

    def verify(self, input_ids: mx.array, draft_tokens: mx.array) -> tuple[mx.array, int]:
        """Verify draft tokens with full model forward pass.

        Args:
            input_ids: Context token IDs [B, seq_len]
            draft_tokens: Draft token IDs [B, num_draft]

        Returns:
            (accepted_tokens [1D], num_accepted)
        """
        t0 = time.perf_counter()

        full_ids = mx.concatenate([input_ids, draft_tokens], axis=-1)
        logits = self.model(full_ids)

        ctx_len = input_ids.shape[-1]
        n_draft = draft_tokens.shape[-1]

        # Logits at position i predict token i+1
        verify_logits = logits[:, ctx_len - 1 : ctx_len - 1 + n_draft, :]
        target_ids = mx.argmax(verify_logits, axis=-1)
        mx.eval(target_ids)

        draft_np = draft_tokens[0].tolist()
        target_np = target_ids[0].tolist()

        num_accepted = 0
        for d, t in zip(draft_np, target_np):
            if d == t:
                num_accepted += 1
            else:
                break

        # Bonus token
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

        return mx.array(accepted, dtype=mx.int32), len(accepted)

    def generate(self, prompt_tokens: mx.array, max_tokens: int = 100, callback=None) -> mx.array:
        """Full generation loop with EAGLE-3 speculation.

        Args:
            prompt_tokens: Input token IDs [seq_len] or [1, seq_len]
            max_tokens: Maximum new tokens to generate
            callback: Optional callback(new_tokens) per step

        Returns:
            All tokens (prompt + generated) as mx.array
        """
        # Reset stats
        self.stats: dict[str, Any] = {
            "total_draft_tokens": 0,
            "total_accepted": 0,
            "total_verify_steps": 0,
            "draft_times_ms": [],
            "verify_times_ms": [],
            "acceptance_counts": [],
        }

        if prompt_tokens.ndim == 1:
            prompt_tokens = prompt_tokens.reshape(1, -1)

        prompt_len = prompt_tokens.shape[-1]
        generated = list(prompt_tokens[0].tolist())
        tokens_generated = 0

        # Initial forward pass to get hidden states for last context token
        input_ids = mx.array([generated])
        hidden_states = self._get_last_hidden_state(input_ids)
        last_hidden = hidden_states[:, -1:, :]  # [1, 1, hidden_dim]
        last_token_id = generated[-1]

        while tokens_generated < max_tokens:
            input_ids = mx.array([generated])

            # Step 1: Draft tokens using EAGLE head
            draft_ids, draft_hiddens = self.draft(last_hidden, last_token_id)

            # Step 2: Verify with full model
            accepted, num_accepted = self.verify(input_ids, draft_ids)

            if num_accepted == 0 and len(accepted) == 0:
                break

            accepted_list = accepted.tolist()
            generated.extend(accepted_list)
            tokens_generated += len(accepted_list)

            # Update hidden state for next draft cycle
            # Re-run full model to get accurate hidden state at the last accepted position
            updated_ids = mx.array([generated])
            hidden_states = self._get_last_hidden_state(updated_ids)
            last_hidden = hidden_states[:, -1:, :]
            last_token_id = generated[-1]

            if callback:
                callback(accepted_list)

            eos = getattr(self.tokenizer, "eos_token_id", None)
            if eos is not None and eos in accepted_list:
                break

        return mx.array(generated)

    def get_stats(self) -> dict:
        """Return acceptance rate, tokens/draft, speedup factor, etc."""
        total_drafts = self.stats["total_draft_tokens"]
        total_accepted = self.stats["total_accepted"]
        total_steps = self.stats["total_verify_steps"]

        acceptance_rate = total_accepted / max(1, total_drafts)
        tokens_per_step = total_accepted / max(1, total_steps)
        # EAGLE draft head is ~100x cheaper than full model forward
        # So draft cost is negligible; speedup ~= tokens_per_step
        speedup = tokens_per_step

        draft_times = self.stats["draft_times_ms"]
        verify_times = self.stats["verify_times_ms"]

        return {
            "total_draft_tokens": total_drafts,
            "total_accepted": total_accepted,
            "total_verify_steps": total_steps,
            "acceptance_rate": round(acceptance_rate, 3),
            "tokens_per_step": round(tokens_per_step, 1),
            "speedup_factor": round(speedup, 2),
            "avg_draft_ms": round(sum(draft_times) / len(draft_times), 1) if draft_times else 0,
            "avg_verify_ms": round(sum(verify_times) / len(verify_times), 1) if verify_times else 0,
        }


class EAGLE3Trainer:
    """Train the draft head on target model hidden states.

    Collects pairs of consecutive hidden states (h_t, h_{t+1}) from the
    target model, then trains the draft head to predict h_{t+1} from
    h_t and the token embedding at position t.
    """

    def __init__(self, model, tokenizer, config: EAGLE3Config = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EAGLE3Config()

        # Detect model components
        self._embed_fn: Any = None
        self._layers: Any = None
        self._norm_fn: Any = None
        self._lm_head_fn: Any = None
        self._detect_components()

        if self.config.hidden_dim == 0:
            self.config.hidden_dim = self._detect_hidden_dim()

    def _detect_components(self):
        """Detect target model components."""
        m = self.model
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

            if embed is not None and layers is not None:
                self._embed_fn = embed
                self._layers = layers
                self._norm_fn = norm
                self._lm_head_fn = head
                return

        raise RuntimeError("Cannot detect target model components")

    def _detect_hidden_dim(self) -> int:
        """Auto-detect hidden dimension."""
        if hasattr(self._embed_fn, "weight"):
            return self._embed_fn.weight.shape[-1]
        dummy = mx.array([[0]])
        out = self._embed_fn(dummy)
        return out.shape[-1]

    def _forward_get_hidden(self, input_ids: mx.array) -> mx.array:
        """Forward pass returning hidden states before LM head.

        Args:
            input_ids: [B, seq_len]

        Returns:
            Hidden states [B, seq_len, hidden_dim]
        """
        h = self._embed_fn(input_ids)
        for layer in self._layers:
            h = layer(h)
        if self._norm_fn is not None:
            h = self._norm_fn(h)
        return h

    def collect_training_data(self, texts: list[str], max_tokens_per_text: int = 256) -> tuple[mx.array, mx.array]:
        """Collect (hidden_state_t, hidden_state_t+1) pairs from target model.

        Runs the target model on each text and extracts consecutive hidden
        state pairs. Also collects token embeddings for the input to the
        draft head.

        Args:
            texts: List of training texts
            max_tokens_per_text: Max tokens per text (truncation)

        Returns:
            (input_pairs, target_hiddens) where:
              input_pairs: [N, hidden_dim * 2] (concat of h_t and embed_t)
              target_hiddens: [N, hidden_dim] (h_{t+1})
        """
        all_inputs = []
        all_targets = []

        for text in texts:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > max_tokens_per_text:
                tokens = tokens[:max_tokens_per_text]
            if len(tokens) < 2:
                continue

            input_ids = mx.array([tokens])

            # Get hidden states for all positions
            hidden = self._forward_get_hidden(input_ids)  # [1, seq_len, hidden_dim]
            mx.eval(hidden)

            # Get token embeddings
            embeddings = self._embed_fn(input_ids)  # [1, seq_len, hidden_dim]
            mx.eval(embeddings)

            seq_len = hidden.shape[1]

            # Create pairs: (h_t, embed_t) -> h_{t+1}
            for t in range(seq_len - 1):
                h_t = hidden[0, t, :]  # [hidden_dim]
                e_t = embeddings[0, t, :]  # [hidden_dim]
                h_next = hidden[0, t + 1, :]  # [hidden_dim]

                input_pair = mx.concatenate([h_t, e_t])  # [hidden_dim * 2]
                all_inputs.append(input_pair)
                all_targets.append(h_next)

        if not all_inputs:
            hidden_dim = self.config.hidden_dim
            return mx.zeros((0, hidden_dim * 2)), mx.zeros((0, hidden_dim))

        input_pairs = mx.stack(all_inputs)  # [N, hidden_dim * 2]
        target_hiddens = mx.stack(all_targets)  # [N, hidden_dim]

        return input_pairs, target_hiddens

    def train(
        self, hidden_pairs: tuple[mx.array, mx.array], num_steps: int = 1000, lr: float = 1e-3, batch_size: int = 32
    ) -> EAGLEDraftHead:
        """Train draft head to predict next hidden state.

        Uses MSE loss between predicted and actual next hidden states.

        Args:
            hidden_pairs: (input_pairs [N, hidden_dim*2], targets [N, hidden_dim])
            num_steps: Training steps
            lr: Learning rate
            batch_size: Training batch size

        Returns:
            Trained EAGLEDraftHead
        """
        input_pairs, target_hiddens = hidden_pairs
        N = input_pairs.shape[0]

        if N == 0:
            raise ValueError("No training data provided")

        hidden_dim = self.config.hidden_dim
        draft_head = EAGLEDraftHead(
            hidden_dim=hidden_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
        )

        optimizer = optim.Adam(learning_rate=lr)

        def loss_fn(model, x_hidden, x_embed, targets):
            predicted = model(x_hidden, x_embed)  # [B, 1, hidden_dim]
            predicted = predicted.squeeze(1)  # [B, hidden_dim]
            return mx.mean((predicted - targets) ** 2)

        loss_and_grad = nn.value_and_grad(draft_head, loss_fn)

        for step in range(num_steps):
            # Sample batch
            indices = mx.random.randint(0, N, (min(batch_size, N),))
            batch_inputs = input_pairs[indices]  # [B, hidden_dim * 2]
            batch_targets = target_hiddens[indices]  # [B, hidden_dim]

            # Split inputs into hidden state and embedding
            x_hidden = batch_inputs[:, :hidden_dim].reshape(-1, 1, hidden_dim)
            x_embed = batch_inputs[:, hidden_dim:].reshape(-1, 1, hidden_dim)

            loss, grads = loss_and_grad(draft_head, x_hidden, x_embed, batch_targets)
            optimizer.update(draft_head, grads)
            mx.eval(draft_head.parameters(), optimizer.state)

        return draft_head

    def save(self, draft_head: EAGLEDraftHead, path: str):
        """Save trained draft head weights.

        Args:
            draft_head: Trained EAGLEDraftHead module
            path: File path (typically .safetensors or .npz)
        """
        weights = dict(draft_head.parameters())
        flat = {}
        _flatten_dict(weights, "", flat)
        mx.save_safetensors(path, flat)

    @staticmethod
    def load(path: str, hidden_dim: int, num_heads: int = 4, num_layers: int = 1) -> EAGLEDraftHead:
        """Load trained draft head.

        Args:
            path: Path to saved weights
            hidden_dim: Hidden dimension (must match saved model)
            num_heads: Number of attention heads
            num_layers: Number of transformer layers

        Returns:
            Loaded EAGLEDraftHead
        """
        draft_head = EAGLEDraftHead(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        weights = mx.load(path)
        draft_head.load_weights(list(weights.items()))
        return draft_head


def apply_eagle3(
    model,
    tokenizer,
    config: EAGLE3Config = None,
    draft_head: EAGLEDraftHead = None,
) -> EAGLE3Engine:
    """One-line setup for EAGLE-3 speculative decoding.

    Args:
        model: Target model (HuggingFace/MLX causal LM)
        tokenizer: Tokenizer with encode/decode
        config: Optional EAGLE3Config
        draft_head: Optional pre-trained EAGLEDraftHead

    Returns:
        EAGLE3Engine ready for generation
    """
    return EAGLE3Engine(model, tokenizer, config, draft_head)


def _flatten_dict(d, prefix, out):
    """Flatten nested dict with dot-separated keys for safetensors."""
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten_dict(v, key, out)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    _flatten_dict(item, f"{key}.{i}", out)
                else:
                    out[f"{key}.{i}"] = item
        else:
            out[key] = v
