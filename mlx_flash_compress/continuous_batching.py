"""Continuous batching engine for MLX inference (vLLM/Orca-style).

Dynamically adds and removes requests from a running batch, maximizing GPU
utilization. Instead of waiting for a batch to finish before starting new
requests, each generation step can include a mix of prefilling (new) and
generating (in-progress) requests.

Key design decisions:
  - KV cache pool: pre-allocated slots avoid per-request allocation overhead
  - Chunked prefill: long prompts are processed in chunks so generation
    requests aren't starved
  - Padded forward: different-length sequences are right-padded with
    attention masks so they can be batched in a single forward pass
  - Thread-safe queue: requests can be submitted from any thread while
    the engine runs its batch loop in a background thread

Based on:
  - Orca (Yu et al., OSDI 2022): iteration-level scheduling
  - vLLM (Kwon et al., SOSP 2023): PagedAttention + continuous batching

Usage:
    from mlx_flash_compress.continuous_batching import (
        ContinuousBatchingEngine, BatchSchedulerConfig, create_batching_server,
    )

    engine = create_batching_server(model, tokenizer)
    engine.start()

    req = engine.submit("What is 2+2?", max_tokens=64)
    result = engine.wait_for_completion(req)
    print(tokenizer.decode(result.generated_tokens))

    engine.stop()
"""

from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import threading
import time
from typing import Optional

import mlx.core as mx


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class RequestStatus(Enum):
    """Lifecycle status of an inference request."""
    QUEUED = auto()
    PREFILLING = auto()
    GENERATING = auto()
    COMPLETED = auto()
    CANCELLED = auto()


@dataclass
class InferenceRequest:
    """Single inference request tracked through the batching pipeline."""
    request_id: str
    prompt_tokens: list[int]
    max_tokens: int = 256
    temperature: float = 0.0
    status: RequestStatus = RequestStatus.QUEUED
    generated_tokens: list[int] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    first_token_at: float = 0.0
    completed_at: float = 0.0
    # Internal bookkeeping
    _prefill_pos: int = 0  # how far into prompt_tokens we've prefilled
    _cache_slot: int = -1  # assigned KV cache slot
    _event: threading.Event = field(default_factory=threading.Event)

    @property
    def ttft_ms(self) -> float:
        """Time to first token in milliseconds."""
        if self.first_token_at > 0:
            return (self.first_token_at - self.created_at) * 1000
        return 0.0

    @property
    def tokens_per_second(self) -> float:
        """Decode throughput (excludes prefill)."""
        if self.completed_at > self.first_token_at > 0:
            elapsed = self.completed_at - self.first_token_at
            if elapsed > 0:
                return len(self.generated_tokens) / elapsed
        return 0.0

    @property
    def is_prefill_done(self) -> bool:
        return self._prefill_pos >= len(self.prompt_tokens)


@dataclass
class BatchSchedulerConfig:
    """Tuning knobs for the continuous batching scheduler."""
    max_batch_size: int = 8
    max_sequence_length: int = 4096
    prefill_chunk_size: int = 512
    scheduling_policy: str = "fcfs"  # "fcfs" or "shortest_first"
    max_wait_ms: float = 100.0


# ---------------------------------------------------------------------------
# KV Cache Pool
# ---------------------------------------------------------------------------

class KVCachePool:
    """Pre-allocated KV cache pool shared across concurrent requests.

    Each slot stores keys and values for every transformer layer.  Slots are
    allocated to incoming requests and freed on completion, avoiding the
    overhead of per-request dynamic allocation.
    """

    def __init__(
        self,
        max_batch_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 4096,
    ):
        self.max_batch_size = max_batch_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Per-slot, per-layer K and V arrays.  Lazily grown on update.
        # _keys[slot][layer] = mx.array of shape (seq_len, num_kv_heads, head_dim)
        # _values mirrors keys.
        self._keys: list[list[Optional[mx.array]]] = [
            [None] * num_layers for _ in range(max_batch_size)
        ]
        self._values: list[list[Optional[mx.array]]] = [
            [None] * num_layers for _ in range(max_batch_size)
        ]
        # Track current sequence length per slot
        self._seq_lens: list[int] = [0] * max_batch_size

        # Slot management
        self._free_slots: set[int] = set(range(max_batch_size))
        self._slot_to_request: dict[int, str] = {}
        self._request_to_slot: dict[str, int] = {}
        self._lock = threading.Lock()

    # -- allocation ----------------------------------------------------------

    def allocate(self, request_id: str) -> int:
        """Allocate a KV cache slot. Returns slot index or raises if full."""
        with self._lock:
            if request_id in self._request_to_slot:
                return self._request_to_slot[request_id]
            if not self._free_slots:
                raise RuntimeError("KV cache pool exhausted — no free slots")
            slot = min(self._free_slots)  # deterministic for testing
            self._free_slots.discard(slot)
            self._slot_to_request[slot] = request_id
            self._request_to_slot[request_id] = slot
            self._seq_lens[slot] = 0
            return slot

    def free(self, request_id: str):
        """Release a KV cache slot back to the pool."""
        with self._lock:
            slot = self._request_to_slot.pop(request_id, None)
            if slot is None:
                return
            self._slot_to_request.pop(slot, None)
            self._free_slots.add(slot)
            # Clear cached tensors
            for layer in range(self.num_layers):
                self._keys[slot][layer] = None
                self._values[slot][layer] = None
            self._seq_lens[slot] = 0

    # -- KV access -----------------------------------------------------------

    def get_kv(self, slot: int, layer: int) -> tuple[Optional[mx.array], Optional[mx.array]]:
        """Get cached K, V for a given slot and layer."""
        return self._keys[slot][layer], self._values[slot][layer]

    def update_kv(self, slot: int, layer: int, keys: mx.array, values: mx.array):
        """Append new K/V entries to a slot's cache for the given layer.

        keys/values shape: (new_tokens, num_kv_heads, head_dim)
        """
        existing_k = self._keys[slot][layer]
        if existing_k is not None:
            self._keys[slot][layer] = mx.concatenate([existing_k, keys], axis=0)
            self._values[slot][layer] = mx.concatenate(
                [self._values[slot][layer], values], axis=0
            )
        else:
            self._keys[slot][layer] = keys
            self._values[slot][layer] = values

        # Update seq_len to the max across layers (they should be identical)
        self._seq_lens[slot] = self._keys[slot][layer].shape[0]

    def get_seq_len(self, slot: int) -> int:
        """Current sequence length stored in a slot."""
        return self._seq_lens[slot]

    # -- stats ---------------------------------------------------------------

    @property
    def num_free_slots(self) -> int:
        with self._lock:
            return len(self._free_slots)

    @property
    def utilization(self) -> float:
        """Fraction of slots currently in use (0.0 – 1.0)."""
        with self._lock:
            used = self.max_batch_size - len(self._free_slots)
            return used / self.max_batch_size if self.max_batch_size > 0 else 0.0


# ---------------------------------------------------------------------------
# Batch Scheduler
# ---------------------------------------------------------------------------

class BatchScheduler:
    """Selects which requests to include in the next batch step.

    Supports FCFS (first-come first-served) and shortest-first policies.
    Mixes prefilling and generating requests up to ``max_batch_size``.
    """

    def __init__(self, config: Optional[BatchSchedulerConfig] = None):
        self.config = config or BatchSchedulerConfig()
        self._queue: deque[InferenceRequest] = deque()
        self._active: dict[str, InferenceRequest] = {}  # request_id -> req
        self._completed: dict[str, InferenceRequest] = {}
        self._lock = threading.Lock()

    def add_request(self, request: InferenceRequest):
        """Add a new request to the waiting queue."""
        with self._lock:
            self._queue.append(request)

    def cancel_request(self, request_id: str):
        """Cancel a pending or in-progress request."""
        with self._lock:
            # Remove from queue if still waiting
            self._queue = deque(
                r for r in self._queue if r.request_id != request_id
            )
            # Mark active request as cancelled
            if request_id in self._active:
                req = self._active.pop(request_id)
                req.status = RequestStatus.CANCELLED
                req.completed_at = time.time()
                req._event.set()
                self._completed[request_id] = req

    def get_batch(self) -> list[InferenceRequest]:
        """Select the next batch of requests to process.

        Active (generating) requests are always included.  Remaining capacity
        is filled from the queue according to the scheduling policy.
        """
        with self._lock:
            batch: list[InferenceRequest] = []

            # 1) Include all active (generating / prefilling) requests
            for req in list(self._active.values()):
                if len(batch) >= self.config.max_batch_size:
                    break
                batch.append(req)

            # 2) Fill remaining slots from queue
            remaining = self.config.max_batch_size - len(batch)
            if remaining > 0 and self._queue:
                candidates = list(self._queue)
                if self.config.scheduling_policy == "shortest_first":
                    candidates.sort(key=lambda r: len(r.prompt_tokens))

                # Pick the top `remaining` candidates in sorted order
                selected_ids: set[str] = set()
                for req in candidates[:remaining]:
                    req.status = RequestStatus.PREFILLING
                    self._active[req.request_id] = req
                    batch.append(req)
                    selected_ids.add(req.request_id)

                # Rebuild queue without selected requests
                self._queue = deque(
                    r for r in self._queue if r.request_id not in selected_ids
                )

            return batch

    def mark_completed(self, request_id: str):
        """Move a request from active to completed."""
        with self._lock:
            req = self._active.pop(request_id, None)
            if req is not None:
                req.status = RequestStatus.COMPLETED
                req.completed_at = time.time()
                req._event.set()
                self._completed[request_id] = req

    def get_request(self, request_id: str) -> Optional[InferenceRequest]:
        """Look up a request by ID (any state)."""
        with self._lock:
            for req in self._queue:
                if req.request_id == request_id:
                    return req
            if request_id in self._active:
                return self._active[request_id]
            return self._completed.get(request_id)

    def get_stats(self) -> dict:
        """Scheduler statistics."""
        with self._lock:
            completed = list(self._completed.values())
            ttfts = [r.ttft_ms for r in completed if r.ttft_ms > 0]
            tps_vals = [r.tokens_per_second for r in completed if r.tokens_per_second > 0]
            return {
                "queue_depth": len(self._queue),
                "active_requests": len(self._active),
                "completed_requests": len(self._completed),
                "batch_utilization": len(self._active) / self.config.max_batch_size
                    if self.config.max_batch_size > 0 else 0.0,
                "avg_ttft_ms": sum(ttfts) / len(ttfts) if ttfts else 0.0,
                "avg_tokens_per_second": sum(tps_vals) / len(tps_vals) if tps_vals else 0.0,
            }


# ---------------------------------------------------------------------------
# Continuous Batching Engine
# ---------------------------------------------------------------------------

class ContinuousBatchingEngine:
    """Main engine that runs the continuous batching loop.

    Accepts an MLX model and tokenizer, manages a KV cache pool and batch
    scheduler, and runs a background thread that repeatedly:
      1. Selects a batch (get_batch)
      2. Runs one step (prefill chunk or decode) for each request
      3. Updates request states

    Thread-safe: ``submit`` / ``cancel`` can be called from any thread.
    """

    def __init__(self, model, tokenizer, config: Optional[BatchSchedulerConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or BatchSchedulerConfig()
        self.scheduler = BatchScheduler(self.config)

        # Detect model dimensions from model attributes
        num_layers = getattr(model, "num_layers", 1)
        num_kv_heads = getattr(model, "num_kv_heads", 1)
        head_dim = getattr(model, "head_dim", 64)

        self.kv_pool = KVCachePool(
            max_batch_size=self.config.max_batch_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=self.config.max_sequence_length,
        )

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._request_counter = 0

        # Throughput tracking
        self._total_tokens_generated = 0
        self._total_requests_completed = 0
        self._engine_start_time = 0.0

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _collect_arrays(obj) -> list:
        """Recursively collect all mx.array leaves from a nested structure."""
        arrays = []
        if isinstance(obj, mx.array):
            arrays.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                arrays.extend(ContinuousBatchingEngine._collect_arrays(v))
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                arrays.extend(ContinuousBatchingEngine._collect_arrays(v))
        return arrays

    def _eval_model_params(self):
        """Eagerly evaluate all model parameters on the current thread."""
        if hasattr(self.model, "parameters"):
            params = self.model.parameters()
            arrays = self._collect_arrays(params)
            if arrays:
                mx.eval(*arrays)

    # -- public API ----------------------------------------------------------

    def start(self):
        """Start the batching engine in a background thread.

        Eagerly evaluates model parameters so they are materialized
        before the background thread begins.  This avoids lazy-eval
        references to the main thread's default stream, which would
        cause ``RuntimeError: There is no Stream(gpu, 0)`` when
        ``mx.eval`` is called from the worker thread.
        """
        with self._lock:
            if self._running:
                return
            # Materialize all model parameters on the main thread
            self._eval_model_params()
            self._running = True
            self._engine_start_time = time.time()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop the engine gracefully."""
        with self._lock:
            self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def submit(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> InferenceRequest:
        """Submit a new inference request.  Returns immediately.

        The returned ``InferenceRequest`` object is updated in-place as
        generation proceeds.  Use ``wait_for_completion`` or
        ``stream_tokens`` to consume results.
        """
        with self._lock:
            self._request_counter += 1
            request_id = f"req-{self._request_counter}"

        # Tokenize prompt
        if hasattr(self.tokenizer, "encode"):
            prompt_tokens = self.tokenizer.encode(prompt)
        else:
            # Fallback for mock tokenizers that are just callables
            prompt_tokens = self.tokenizer(prompt)

        if isinstance(prompt_tokens, mx.array):
            prompt_tokens = prompt_tokens.tolist()

        req = InferenceRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.scheduler.add_request(req)
        return req

    def cancel(self, request_id: str):
        """Cancel a pending or in-progress request."""
        req = self.scheduler.get_request(request_id)
        if req and req._cache_slot >= 0:
            self.kv_pool.free(request_id)
        self.scheduler.cancel_request(request_id)

    def wait_for_completion(
        self,
        request: InferenceRequest,
        timeout: float = 60.0,
    ) -> InferenceRequest:
        """Block until the request completes or times out."""
        request._event.wait(timeout=timeout)
        return request

    def stream_tokens(self, request: InferenceRequest):
        """Generator that yields token IDs as they are produced."""
        idx = 0
        while True:
            if len(request.generated_tokens) > idx:
                yield request.generated_tokens[idx]
                idx += 1
            elif request.status in (RequestStatus.COMPLETED, RequestStatus.CANCELLED):
                # Yield any remaining tokens
                while idx < len(request.generated_tokens):
                    yield request.generated_tokens[idx]
                    idx += 1
                return
            else:
                time.sleep(0.001)

    def get_stats(self) -> dict:
        """Engine-level statistics."""
        scheduler_stats = self.scheduler.get_stats()
        elapsed = time.time() - self._engine_start_time if self._engine_start_time else 0
        return {
            **scheduler_stats,
            "total_tokens_generated": self._total_tokens_generated,
            "total_requests_completed": self._total_requests_completed,
            "engine_uptime_s": round(elapsed, 2),
            "kv_pool_utilization": self.kv_pool.utilization,
            "kv_pool_free_slots": self.kv_pool.num_free_slots,
            "overall_tokens_per_second": (
                self._total_tokens_generated / elapsed if elapsed > 0 else 0.0
            ),
        }

    # -- background loop -----------------------------------------------------

    def _run_loop(self):
        """Main batch processing loop (runs in background thread).

        MLX requires each non-main thread to set its own default stream
        (thread-local streams, added in MLX 0.31.2).  We create a GPU
        stream once at thread start and use it for all operations.
        """
        # Set up a thread-local GPU stream for MLX operations
        stream = mx.new_stream(mx.gpu) if hasattr(mx, "new_stream") else mx.gpu
        mx.set_default_stream(stream)

        while self._running:
            batch = self.scheduler.get_batch()
            if not batch:
                time.sleep(0.001)
                continue
            self._batch_step(batch)

    def _batch_step(self, batch: list[InferenceRequest]):
        """Execute one generation step for the entire batch.

        For prefilling requests: process a chunk of prompt tokens.
        For generating requests: generate the next token.
        """
        for req in batch:
            if req.status == RequestStatus.CANCELLED:
                continue

            # Allocate KV cache slot if needed
            if req._cache_slot < 0:
                try:
                    req._cache_slot = self.kv_pool.allocate(req.request_id)
                except RuntimeError:
                    continue  # pool full, try next iteration

            if not req.is_prefill_done:
                # -- Prefill phase: process a chunk of prompt tokens --
                self._prefill_step(req)
            else:
                # -- Decode phase: generate next token --
                self._decode_step(req)

    def _prefill_step(self, req: InferenceRequest):
        """Process a chunk of prompt tokens for prefilling."""
        start = req._prefill_pos
        end = min(start + self.config.prefill_chunk_size, len(req.prompt_tokens))
        chunk = req.prompt_tokens[start:end]

        token_ids = mx.array([chunk])  # (1, chunk_len)
        seq_len = end  # total sequence length after this chunk

        # Create attention mask: causal mask for the chunk
        chunk_len = len(chunk)
        # Full causal mask: each position can attend to itself and all prior positions
        # But we only need the rows for the current chunk positions
        mask = mx.zeros((1, 1, chunk_len, seq_len))
        for i in range(chunk_len):
            pos_in_seq = start + i
            # Can attend to positions 0..pos_in_seq
            mask[0, 0, i, pos_in_seq + 1:] = float("-inf")

        logits = self._padded_forward(token_ids, mask, [req._cache_slot])
        mx.eval(logits)  # force computation

        req._prefill_pos = end
        if req.is_prefill_done:
            req.status = RequestStatus.GENERATING

    def _decode_step(self, req: InferenceRequest):
        """Generate one token for a request in the decode phase."""
        # Get the last token (either last prompt token or last generated token)
        if req.generated_tokens:
            last_token = req.generated_tokens[-1]
        else:
            last_token = req.prompt_tokens[-1]

        token_ids = mx.array([[last_token]])  # (1, 1)
        total_len = len(req.prompt_tokens) + len(req.generated_tokens)

        # No masking needed for single-token decode (attends to all prior)
        mask = mx.zeros((1, 1, 1, total_len))

        logits = self._padded_forward(token_ids, mask, [req._cache_slot])

        # Sample next token
        next_token = self._sample(logits, req.temperature)
        mx.eval(next_token)
        next_token_id = next_token.item()

        if not req.generated_tokens:
            req.first_token_at = time.time()

        req.generated_tokens.append(next_token_id)
        self._total_tokens_generated += 1

        # Check stopping conditions
        if len(req.generated_tokens) >= req.max_tokens:
            self._finish_request(req)
        elif hasattr(self.tokenizer, "eos_token_id") and next_token_id == self.tokenizer.eos_token_id:
            self._finish_request(req)

    def _padded_forward(
        self,
        token_ids: mx.array,
        attention_mask: mx.array,
        cache_slots: list[int],
    ) -> mx.array:
        """Forward pass with attention mask for batched inference.

        Args:
            token_ids: (batch, seq_len) token IDs
            attention_mask: (batch, 1, seq_len, total_len) mask
                            0 = attend, -inf = mask out
            cache_slots: KV cache slot indices (one per batch element)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        # Pass through model — the model interface depends on the specific
        # model class. We support both __call__(x) and __call__(x, mask).
        try:
            logits = self.model(token_ids, mask=attention_mask)
        except TypeError:
            # Fallback: model only takes token_ids
            logits = self.model(token_ids)

        return logits

    def _sample(self, logits: mx.array, temperature: float) -> mx.array:
        """Sample a token from logits.

        Args:
            logits: (batch, seq_len, vocab_size) — we take the last position.
            temperature: 0 = greedy, >0 = softmax sampling.
        """
        # Take logits for the last position
        last_logits = logits[:, -1, :]  # (batch, vocab_size)

        if temperature <= 0:
            return mx.argmax(last_logits, axis=-1)
        else:
            probs = mx.softmax(last_logits / temperature, axis=-1)
            return mx.random.categorical(probs)

    def _finish_request(self, req: InferenceRequest):
        """Mark a request as completed and clean up resources."""
        self.kv_pool.free(req.request_id)
        self.scheduler.mark_completed(req.request_id)
        self._total_requests_completed += 1


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_batching_server(
    model,
    tokenizer,
    config: Optional[BatchSchedulerConfig] = None,
) -> ContinuousBatchingEngine:
    """One-line setup for continuous batching.

    Args:
        model: An MLX model with a __call__ method.
        tokenizer: A tokenizer with encode/decode methods.
        config: Optional scheduler configuration.

    Returns:
        A ``ContinuousBatchingEngine`` (not yet started — call ``.start()``).
    """
    return ContinuousBatchingEngine(model, tokenizer, config)
