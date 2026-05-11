"""Tests for continuous batching engine (no real model loading)."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except (ImportError, ModuleNotFoundError):
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="requires mlx")

# Import after guard so collection doesn't fail without mlx
if HAS_MLX:
    from mlx_flash_compress.continuous_batching import (
        BatchScheduler,
        BatchSchedulerConfig,
        ContinuousBatchingEngine,
        InferenceRequest,
        KVCachePool,
        RequestStatus,
        create_batching_server,
    )


# ---------------------------------------------------------------------------
# Mock model: a simple linear layer that produces logits
# ---------------------------------------------------------------------------


class MockModel:
    """Minimal model that returns deterministic logits of the right shape.

    Uses a simple linear projection.  All operations are pure MLX ops
    so they work correctly with thread-local streams.
    """

    def __init__(self, vocab_size: int = 32, num_layers: int = 2, num_kv_heads: int = 2, head_dim: int = 8):
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self._linear = nn.Linear(vocab_size, vocab_size)

    def parameters(self):
        return self._linear.parameters()

    def __call__(self, token_ids: mx.array, **kwargs) -> mx.array:
        """Forward pass: return logits of shape (batch, seq_len, vocab_size).

        Creates a simple embedding via one-hot encoding (vectorized)
        and projects through a linear layer.
        """
        batch, seq_len = token_ids.shape
        ids_clamped = mx.clip(token_ids, 0, self.vocab_size - 1)
        # Vectorized one-hot: use eye matrix as lookup table
        eye = mx.eye(self.vocab_size)
        flat_ids = ids_clamped.reshape(-1)
        one_hot = eye[flat_ids]  # (batch*seq_len, vocab_size)
        logits = self._linear(one_hot)
        return logits.reshape(batch, seq_len, self.vocab_size)


class MockTokenizer:
    """Tokenizer that maps each character to its ordinal."""

    def __init__(self, vocab_size: int = 32):
        self.vocab_size = vocab_size
        self.eos_token_id = 0  # we'll never hit this in tests

    def encode(self, text: str) -> list[int]:
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(t + 65) for t in token_ids)


def _make_model(vocab_size=32):
    return MockModel(vocab_size=vocab_size)


def _make_tokenizer(vocab_size=32):
    return MockTokenizer(vocab_size=vocab_size)


# ===========================================================================
# InferenceRequest
# ===========================================================================


class TestInferenceRequest:
    def test_initial_status_is_queued(self):
        req = InferenceRequest(request_id="r1", prompt_tokens=[1, 2, 3])
        assert req.status == RequestStatus.QUEUED

    def test_ttft_zero_before_first_token(self):
        req = InferenceRequest(request_id="r1", prompt_tokens=[1])
        assert req.ttft_ms == 0.0

    def test_ttft_calculated_after_first_token(self):
        req = InferenceRequest(request_id="r1", prompt_tokens=[1])
        req.created_at = 100.0
        req.first_token_at = 100.05
        assert abs(req.ttft_ms - 50.0) < 1e-6

    def test_tokens_per_second_zero_when_incomplete(self):
        req = InferenceRequest(request_id="r1", prompt_tokens=[1])
        req.generated_tokens = [10, 11, 12]
        assert req.tokens_per_second == 0.0

    def test_tokens_per_second_calculated(self):
        req = InferenceRequest(request_id="r1", prompt_tokens=[1])
        req.first_token_at = 100.0
        req.completed_at = 102.0
        req.generated_tokens = [10, 11, 12, 13]
        assert abs(req.tokens_per_second - 2.0) < 1e-6

    def test_is_prefill_done(self):
        req = InferenceRequest(request_id="r1", prompt_tokens=[1, 2, 3])
        assert not req.is_prefill_done
        req._prefill_pos = 3
        assert req.is_prefill_done

    def test_default_max_tokens(self):
        req = InferenceRequest(request_id="r1", prompt_tokens=[1])
        assert req.max_tokens == 256

    def test_lifecycle_queued_to_completed(self):
        req = InferenceRequest(request_id="r1", prompt_tokens=[1, 2])
        assert req.status == RequestStatus.QUEUED
        req.status = RequestStatus.PREFILLING
        assert req.status == RequestStatus.PREFILLING
        req.status = RequestStatus.GENERATING
        assert req.status == RequestStatus.GENERATING
        req.status = RequestStatus.COMPLETED
        assert req.status == RequestStatus.COMPLETED


# ===========================================================================
# KV Cache Pool
# ===========================================================================


class TestKVCachePool:
    def test_allocate_returns_slot(self):
        pool = KVCachePool(max_batch_size=4, num_layers=2, num_kv_heads=2, head_dim=8)
        slot = pool.allocate("r1")
        assert isinstance(slot, int)
        assert 0 <= slot < 4

    def test_allocate_idempotent(self):
        pool = KVCachePool(max_batch_size=4, num_layers=2, num_kv_heads=2, head_dim=8)
        s1 = pool.allocate("r1")
        s2 = pool.allocate("r1")
        assert s1 == s2

    def test_free_returns_slot(self):
        pool = KVCachePool(max_batch_size=2, num_layers=1, num_kv_heads=1, head_dim=4)
        pool.allocate("r1")
        assert pool.num_free_slots == 1
        pool.free("r1")
        assert pool.num_free_slots == 2

    def test_free_nonexistent_is_noop(self):
        pool = KVCachePool(max_batch_size=2, num_layers=1, num_kv_heads=1, head_dim=4)
        pool.free("nonexistent")  # should not raise
        assert pool.num_free_slots == 2

    def test_exhaust_pool_raises(self):
        pool = KVCachePool(max_batch_size=2, num_layers=1, num_kv_heads=1, head_dim=4)
        pool.allocate("r1")
        pool.allocate("r2")
        with pytest.raises(RuntimeError, match="exhausted"):
            pool.allocate("r3")

    def test_utilization(self):
        pool = KVCachePool(max_batch_size=4, num_layers=1, num_kv_heads=1, head_dim=4)
        assert pool.utilization == 0.0
        pool.allocate("r1")
        assert abs(pool.utilization - 0.25) < 1e-6
        pool.allocate("r2")
        assert abs(pool.utilization - 0.5) < 1e-6

    def test_update_and_get_kv(self):
        pool = KVCachePool(max_batch_size=2, num_layers=2, num_kv_heads=2, head_dim=4)
        slot = pool.allocate("r1")
        keys = mx.ones((3, 2, 4))  # 3 tokens
        values = mx.zeros((3, 2, 4))
        pool.update_kv(slot, layer=0, keys=keys, values=values)

        k, v = pool.get_kv(slot, layer=0)
        assert k.shape == (3, 2, 4)
        assert v.shape == (3, 2, 4)

    def test_update_kv_appends(self):
        pool = KVCachePool(max_batch_size=2, num_layers=1, num_kv_heads=1, head_dim=4)
        slot = pool.allocate("r1")

        pool.update_kv(slot, 0, mx.ones((2, 1, 4)), mx.ones((2, 1, 4)))
        pool.update_kv(slot, 0, mx.zeros((3, 1, 4)), mx.zeros((3, 1, 4)))

        k, v = pool.get_kv(slot, 0)
        assert k.shape == (5, 1, 4)  # 2 + 3
        assert pool.get_seq_len(slot) == 5

    def test_get_kv_empty_slot(self):
        pool = KVCachePool(max_batch_size=2, num_layers=1, num_kv_heads=1, head_dim=4)
        slot = pool.allocate("r1")
        k, v = pool.get_kv(slot, 0)
        assert k is None
        assert v is None

    def test_free_clears_cache(self):
        pool = KVCachePool(max_batch_size=2, num_layers=1, num_kv_heads=1, head_dim=4)
        slot = pool.allocate("r1")
        pool.update_kv(slot, 0, mx.ones((2, 1, 4)), mx.ones((2, 1, 4)))
        pool.free("r1")

        # Re-allocate same slot, should be clean
        slot2 = pool.allocate("r2")
        assert slot2 == slot  # deterministic: picks min free slot
        k, v = pool.get_kv(slot2, 0)
        assert k is None

    def test_concurrent_allocate(self):
        """Multiple threads allocating should not produce conflicts."""
        pool = KVCachePool(max_batch_size=16, num_layers=1, num_kv_heads=1, head_dim=4)
        results = {}
        errors = []

        def alloc(rid):
            try:
                slot = pool.allocate(rid)
                results[rid] = slot
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=alloc, args=(f"r{i}",)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(set(results.values())) == 16  # all unique slots


# ===========================================================================
# Batch Scheduler
# ===========================================================================


class TestBatchScheduler:
    def test_add_and_get_batch(self):
        sched = BatchScheduler(BatchSchedulerConfig(max_batch_size=4))
        req = InferenceRequest(request_id="r1", prompt_tokens=[1, 2])
        sched.add_request(req)
        batch = sched.get_batch()
        assert len(batch) == 1
        assert batch[0].request_id == "r1"
        assert batch[0].status == RequestStatus.PREFILLING

    def test_fcfs_ordering(self):
        sched = BatchScheduler(BatchSchedulerConfig(max_batch_size=4))
        for i in range(3):
            sched.add_request(InferenceRequest(request_id=f"r{i}", prompt_tokens=[1] * (10 - i)))
        batch = sched.get_batch()
        assert [r.request_id for r in batch] == ["r0", "r1", "r2"]

    def test_shortest_first_ordering(self):
        config = BatchSchedulerConfig(max_batch_size=4, scheduling_policy="shortest_first")
        sched = BatchScheduler(config)
        sched.add_request(InferenceRequest(request_id="long", prompt_tokens=[1] * 100))
        sched.add_request(InferenceRequest(request_id="short", prompt_tokens=[1] * 5))
        sched.add_request(InferenceRequest(request_id="mid", prompt_tokens=[1] * 50))

        batch = sched.get_batch()
        ids = [r.request_id for r in batch]
        assert ids == ["short", "mid", "long"]

    def test_max_batch_size_respected(self):
        sched = BatchScheduler(BatchSchedulerConfig(max_batch_size=2))
        for i in range(5):
            sched.add_request(InferenceRequest(request_id=f"r{i}", prompt_tokens=[1]))
        batch = sched.get_batch()
        assert len(batch) == 2
        # Remaining 3 still in queue
        stats = sched.get_stats()
        assert stats["queue_depth"] == 3

    def test_active_requests_stay_in_batch(self):
        sched = BatchScheduler(BatchSchedulerConfig(max_batch_size=4))
        r1 = InferenceRequest(request_id="r1", prompt_tokens=[1])
        sched.add_request(r1)

        batch1 = sched.get_batch()
        assert len(batch1) == 1
        # r1 is now active; add another request
        sched.add_request(InferenceRequest(request_id="r2", prompt_tokens=[2]))
        batch2 = sched.get_batch()
        assert len(batch2) == 2
        assert {r.request_id for r in batch2} == {"r1", "r2"}

    def test_mark_completed(self):
        sched = BatchScheduler(BatchSchedulerConfig(max_batch_size=4))
        req = InferenceRequest(request_id="r1", prompt_tokens=[1])
        sched.add_request(req)
        sched.get_batch()  # moves r1 to active
        sched.mark_completed("r1")
        assert req.status == RequestStatus.COMPLETED
        assert req.completed_at > 0

    def test_cancel_queued_request(self):
        sched = BatchScheduler(BatchSchedulerConfig(max_batch_size=4))
        req = InferenceRequest(request_id="r1", prompt_tokens=[1])
        sched.add_request(req)
        sched.cancel_request("r1")
        batch = sched.get_batch()
        assert len(batch) == 0

    def test_cancel_active_request(self):
        sched = BatchScheduler(BatchSchedulerConfig(max_batch_size=4))
        req = InferenceRequest(request_id="r1", prompt_tokens=[1])
        sched.add_request(req)
        sched.get_batch()  # moves to active
        sched.cancel_request("r1")
        assert req.status == RequestStatus.CANCELLED
        assert req.completed_at > 0

    def test_get_request_any_state(self):
        sched = BatchScheduler(BatchSchedulerConfig(max_batch_size=4))
        r1 = InferenceRequest(request_id="r1", prompt_tokens=[1])
        r2 = InferenceRequest(request_id="r2", prompt_tokens=[2])
        sched.add_request(r1)
        sched.add_request(r2)

        # r1 is queued
        assert sched.get_request("r1") is r1

        # Move r1 to active
        sched.get_batch()
        assert sched.get_request("r1") is r1

        # Complete r1
        sched.mark_completed("r1")
        assert sched.get_request("r1") is r1
        assert sched.get_request("r1").status == RequestStatus.COMPLETED

    def test_get_request_nonexistent(self):
        sched = BatchScheduler()
        assert sched.get_request("nope") is None

    def test_stats_accuracy(self):
        sched = BatchScheduler(BatchSchedulerConfig(max_batch_size=4))
        stats = sched.get_stats()
        assert stats["queue_depth"] == 0
        assert stats["active_requests"] == 0
        assert stats["completed_requests"] == 0
        assert stats["batch_utilization"] == 0.0

        # Add requests
        for i in range(3):
            sched.add_request(InferenceRequest(request_id=f"r{i}", prompt_tokens=[1]))
        stats = sched.get_stats()
        assert stats["queue_depth"] == 3

        # Get batch (moves to active)
        sched.get_batch()
        stats = sched.get_stats()
        assert stats["active_requests"] == 3
        assert stats["queue_depth"] == 0
        assert abs(stats["batch_utilization"] - 0.75) < 1e-6

    def test_stats_avg_ttft(self):
        sched = BatchScheduler(BatchSchedulerConfig(max_batch_size=4))
        r1 = InferenceRequest(request_id="r1", prompt_tokens=[1])
        r1.created_at = 100.0
        r1.first_token_at = 100.050
        sched.add_request(r1)
        sched.get_batch()
        sched.mark_completed("r1")

        stats = sched.get_stats()
        assert abs(stats["avg_ttft_ms"] - 50.0) < 1e-3

    def test_empty_batch_when_no_requests(self):
        sched = BatchScheduler()
        assert sched.get_batch() == []


# ===========================================================================
# ContinuousBatchingEngine
# ===========================================================================


class TestContinuousBatchingEngine:
    def _make_engine(self, max_batch=4, vocab_size=32, max_tokens_default=8):
        model = _make_model(vocab_size=vocab_size)
        tokenizer = _make_tokenizer(vocab_size=vocab_size)
        config = BatchSchedulerConfig(
            max_batch_size=max_batch,
            prefill_chunk_size=64,
        )
        return ContinuousBatchingEngine(model, tokenizer, config)

    def test_submit_returns_request(self):
        engine = self._make_engine()
        req = engine.submit("hello", max_tokens=10)
        assert isinstance(req, InferenceRequest)
        assert req.status == RequestStatus.QUEUED
        assert len(req.prompt_tokens) > 0

    def test_unique_request_ids(self):
        engine = self._make_engine()
        r1 = engine.submit("a")
        r2 = engine.submit("b")
        assert r1.request_id != r2.request_id

    def test_start_stop(self):
        engine = self._make_engine()
        engine.start()
        assert engine._running
        engine.stop()
        assert not engine._running

    def test_start_idempotent(self):
        engine = self._make_engine()
        engine.start()
        engine.start()  # should not raise or create duplicate threads
        engine.stop()

    def test_single_request_completion(self):
        engine = self._make_engine()
        engine.start()
        try:
            req = engine.submit("hi", max_tokens=4)
            result = engine.wait_for_completion(req, timeout=10.0)
            assert result.status == RequestStatus.COMPLETED
            assert len(result.generated_tokens) > 0
            assert len(result.generated_tokens) <= 4
            assert result.first_token_at > 0
            assert result.completed_at > 0
            assert result.ttft_ms > 0
        finally:
            engine.stop()

    def test_multiple_concurrent_requests(self):
        engine = self._make_engine(max_batch=4)
        engine.start()
        try:
            requests = [engine.submit(f"prompt {i}", max_tokens=4) for i in range(4)]
            results = [engine.wait_for_completion(r, timeout=15.0) for r in requests]
            for r in results:
                assert r.status == RequestStatus.COMPLETED
                assert len(r.generated_tokens) > 0
        finally:
            engine.stop()

    def test_cancel_request(self):
        engine = self._make_engine()
        # Don't start engine so the request stays queued
        req = engine.submit("cancel me", max_tokens=10)
        engine.cancel(req.request_id)
        # After cancel, the request should be cancelled or removed
        found = engine.scheduler.get_request(req.request_id)
        # If found, it should be cancelled; if not found, it was removed from queue
        if found:
            assert found.status == RequestStatus.CANCELLED

    def test_stream_tokens(self):
        engine = self._make_engine()
        engine.start()
        try:
            req = engine.submit("stream test", max_tokens=3)
            tokens = list(engine.stream_tokens(req))
            assert len(tokens) > 0
            assert len(tokens) <= 3
        finally:
            engine.stop()

    def test_stats(self):
        engine = self._make_engine()
        engine.start()
        try:
            req = engine.submit("stats test", max_tokens=2)
            engine.wait_for_completion(req, timeout=10.0)
            stats = engine.get_stats()
            assert stats["total_requests_completed"] >= 1
            assert stats["total_tokens_generated"] >= 1
            assert stats["kv_pool_free_slots"] == engine.config.max_batch_size
            assert "engine_uptime_s" in stats
            assert "overall_tokens_per_second" in stats
        finally:
            engine.stop()

    def test_kv_pool_freed_after_completion(self):
        engine = self._make_engine(max_batch=2)
        engine.start()
        try:
            req = engine.submit("free test", max_tokens=2)
            engine.wait_for_completion(req, timeout=10.0)
            assert engine.kv_pool.num_free_slots == 2
        finally:
            engine.stop()

    def test_request_exceeding_pool_waits(self):
        """More requests than pool slots should still complete (via queueing)."""
        engine = self._make_engine(max_batch=2)
        engine.start()
        try:
            requests = [engine.submit(f"req {i}", max_tokens=2) for i in range(4)]
            for r in requests:
                engine.wait_for_completion(r, timeout=20.0)
            completed = sum(1 for r in requests if r.status == RequestStatus.COMPLETED)
            assert completed == 4
        finally:
            engine.stop()


# ===========================================================================
# Factory function
# ===========================================================================


class TestCreateBatchingServer:
    def test_returns_engine(self):
        model = _make_model()
        tokenizer = _make_tokenizer()
        engine = create_batching_server(model, tokenizer)
        assert isinstance(engine, ContinuousBatchingEngine)

    def test_accepts_custom_config(self):
        model = _make_model()
        tokenizer = _make_tokenizer()
        config = BatchSchedulerConfig(max_batch_size=16)
        engine = create_batching_server(model, tokenizer, config)
        assert engine.config.max_batch_size == 16

    def test_not_started_by_default(self):
        engine = create_batching_server(_make_model(), _make_tokenizer())
        assert not engine._running


# ===========================================================================
# BatchSchedulerConfig
# ===========================================================================


class TestBatchSchedulerConfig:
    def test_defaults(self):
        cfg = BatchSchedulerConfig()
        assert cfg.max_batch_size == 8
        assert cfg.max_sequence_length == 4096
        assert cfg.prefill_chunk_size == 512
        assert cfg.scheduling_policy == "fcfs"
        assert cfg.max_wait_ms == 100.0

    def test_custom_values(self):
        cfg = BatchSchedulerConfig(
            max_batch_size=16,
            scheduling_policy="shortest_first",
        )
        assert cfg.max_batch_size == 16
        assert cfg.scheduling_policy == "shortest_first"


# ===========================================================================
# RequestStatus enum
# ===========================================================================


class TestRequestStatus:
    def test_all_statuses_exist(self):
        assert RequestStatus.QUEUED
        assert RequestStatus.PREFILLING
        assert RequestStatus.GENERATING
        assert RequestStatus.COMPLETED
        assert RequestStatus.CANCELLED

    def test_statuses_are_distinct(self):
        statuses = [
            RequestStatus.QUEUED,
            RequestStatus.PREFILLING,
            RequestStatus.GENERATING,
            RequestStatus.COMPLETED,
            RequestStatus.CANCELLED,
        ]
        assert len(set(statuses)) == 5


# ===========================================================================
# Concurrency & Stress Tests
# ===========================================================================


class TestBatchingConcurrency:
    """Concurrency, stress, and edge-case tests for the batching engine."""

    def _make_engine(self, max_batch=8, vocab_size=32):
        model = _make_model(vocab_size=vocab_size)
        tokenizer = _make_tokenizer(vocab_size=vocab_size)
        config = BatchSchedulerConfig(
            max_batch_size=max_batch,
            prefill_chunk_size=64,
        )
        return ContinuousBatchingEngine(model, tokenizer, config)

    # -- test_concurrent_submit_no_crash ------------------------------------

    def test_concurrent_submit_no_crash(self):
        """Submit 20+ requests concurrently using threads; verify no exceptions."""
        engine = self._make_engine(max_batch=4)
        engine.start()
        try:
            num_requests = 24
            errors = []
            results = []

            def submit_and_wait(idx):
                try:
                    req = engine.submit(f"concurrent prompt {idx}", max_tokens=3)
                    result = engine.wait_for_completion(req, timeout=30.0)
                    return result
                except Exception as e:
                    errors.append(e)
                    return None

            with ThreadPoolExecutor(max_workers=12) as pool:
                futures = [pool.submit(submit_and_wait, i) for i in range(num_requests)]
                for f in as_completed(futures):
                    r = f.result()
                    if r is not None:
                        results.append(r)

            assert not errors, f"Got {len(errors)} errors: {errors}"
            # All requests should complete
            assert len(results) == num_requests
            for r in results:
                assert r.status == RequestStatus.COMPLETED
                assert len(r.generated_tokens) > 0
        finally:
            engine.stop()

    # -- test_queue_ordering_fifo -------------------------------------------

    def test_queue_ordering_fifo(self):
        """Verify requests are processed in FIFO order by the scheduler."""
        config = BatchSchedulerConfig(max_batch_size=2, scheduling_policy="fcfs")
        sched = BatchScheduler(config)

        # Add 5 requests in order
        request_ids = [f"fifo-{i}" for i in range(5)]
        for rid in request_ids:
            sched.add_request(InferenceRequest(request_id=rid, prompt_tokens=[1, 2, 3]))

        # First batch should pick the first 2 (max_batch_size=2)
        batch1 = sched.get_batch()
        batch1_ids = [r.request_id for r in batch1]
        assert batch1_ids == ["fifo-0", "fifo-1"]

        # Complete them, then get next batch
        for r in batch1:
            sched.mark_completed(r.request_id)

        batch2 = sched.get_batch()
        batch2_ids = [r.request_id for r in batch2]
        assert batch2_ids == ["fifo-2", "fifo-3"]

        # Complete, get last
        for r in batch2:
            sched.mark_completed(r.request_id)

        batch3 = sched.get_batch()
        batch3_ids = [r.request_id for r in batch3]
        assert batch3_ids == ["fifo-4"]

    # -- test_batch_formation -----------------------------------------------

    def test_batch_formation(self):
        """Requests arriving within the same scheduling cycle are grouped together."""
        config = BatchSchedulerConfig(max_batch_size=4)
        sched = BatchScheduler(config)

        # Simulate several requests arriving before the scheduler gets a batch
        for i in range(4):
            sched.add_request(InferenceRequest(request_id=f"batch-{i}", prompt_tokens=[1] * (i + 1)))

        batch = sched.get_batch()
        # All 4 should be in the same batch
        assert len(batch) == 4
        assert {r.request_id for r in batch} == {"batch-0", "batch-1", "batch-2", "batch-3"}
        # All should be PREFILLING (just promoted from queue)
        for r in batch:
            assert r.status == RequestStatus.PREFILLING

    def test_batch_formation_partial(self):
        """When more requests arrive than max_batch_size, only max_batch_size are batched."""
        config = BatchSchedulerConfig(max_batch_size=3)
        sched = BatchScheduler(config)

        for i in range(6):
            sched.add_request(InferenceRequest(request_id=f"partial-{i}", prompt_tokens=[1]))

        batch = sched.get_batch()
        assert len(batch) == 3
        # Remaining 3 stay in queue
        stats = sched.get_stats()
        assert stats["queue_depth"] == 3

    # -- test_scheduler_config_validation -----------------------------------

    def test_scheduler_config_validation_zero_batch_size(self):
        """BatchSchedulerConfig with max_batch_size=0 should not crash the scheduler."""
        config = BatchSchedulerConfig(max_batch_size=0)
        sched = BatchScheduler(config)
        sched.add_request(InferenceRequest(request_id="r1", prompt_tokens=[1]))
        # With batch size 0, no requests can be scheduled
        batch = sched.get_batch()
        assert len(batch) == 0

    def test_scheduler_config_validation_negative_batch_size(self):
        """Negative max_batch_size should result in empty batches (no crash)."""
        config = BatchSchedulerConfig(max_batch_size=-1)
        sched = BatchScheduler(config)
        sched.add_request(InferenceRequest(request_id="r1", prompt_tokens=[1]))
        batch = sched.get_batch()
        assert len(batch) == 0

    def test_scheduler_config_validation_unknown_policy(self):
        """Unknown scheduling policy falls back to FCFS behavior (no crash)."""
        config = BatchSchedulerConfig(max_batch_size=4, scheduling_policy="unknown_policy")
        sched = BatchScheduler(config)
        for i in range(3):
            sched.add_request(InferenceRequest(request_id=f"r{i}", prompt_tokens=[1] * (10 - i)))
        batch = sched.get_batch()
        # Should be in insertion order (FCFS) since unknown policy doesn't sort
        assert [r.request_id for r in batch] == ["r0", "r1", "r2"]

    def test_scheduler_config_validation_zero_wait_ms(self):
        """Zero max_wait_ms is valid and shouldn't crash."""
        config = BatchSchedulerConfig(max_wait_ms=0.0)
        assert config.max_wait_ms == 0.0

    def test_scheduler_config_validation_large_sequence_length(self):
        """Very large sequence length should be accepted without error."""
        config = BatchSchedulerConfig(max_sequence_length=1_000_000)
        assert config.max_sequence_length == 1_000_000

    # -- test_engine_shutdown_drains_queue -----------------------------------

    def test_engine_shutdown_drains_queue(self):
        """Pending requests should complete (or be in progress) when engine stops."""
        engine = self._make_engine(max_batch=4)
        engine.start()
        try:
            requests = [engine.submit(f"drain-{i}", max_tokens=3) for i in range(6)]
            # Wait for all requests to complete before stopping
            for r in requests:
                engine.wait_for_completion(r, timeout=30.0)

            completed = [r for r in requests if r.status == RequestStatus.COMPLETED]
            assert len(completed) == 6
        finally:
            engine.stop()

        # After stop, all should be completed
        for r in requests:
            assert r.status == RequestStatus.COMPLETED
            assert len(r.generated_tokens) > 0

    def test_engine_stop_is_safe_when_idle(self):
        """Stopping an idle engine should not hang or raise."""
        engine = self._make_engine()
        engine.start()
        time.sleep(0.05)  # let the loop spin a few times
        engine.stop()
        assert not engine._running

    def test_engine_stop_without_start(self):
        """Calling stop without start should be a no-op."""
        engine = self._make_engine()
        engine.stop()  # should not raise
        assert not engine._running

    # -- test_engine_handles_slow_model -------------------------------------

    def test_engine_handles_slow_model(self):
        """Mock a slow model (adds delay per forward pass); verify timeout behavior."""
        call_count = [0]

        class SlowModel(MockModel):
            def __call__(self, token_ids, **kwargs):
                call_count[0] += 1
                time.sleep(0.05)  # 50ms per forward pass
                return super().__call__(token_ids, **kwargs)

        model = SlowModel(vocab_size=32)
        tokenizer = _make_tokenizer(vocab_size=32)
        config = BatchSchedulerConfig(max_batch_size=4, prefill_chunk_size=64)

        engine = ContinuousBatchingEngine(model, tokenizer, config)
        engine.start()
        try:
            # Short timeout should still work because max_tokens is small
            req = engine.submit("slow model test", max_tokens=2)
            result = engine.wait_for_completion(req, timeout=10.0)
            assert result.status == RequestStatus.COMPLETED
            assert len(result.generated_tokens) > 0
            assert call_count[0] > 0  # model was actually called
        finally:
            engine.stop()

    def test_engine_slow_model_timeout(self):
        """With a very slow model and a very short timeout, wait should time out."""

        class VerySlowModel(MockModel):
            def __call__(self, token_ids, **kwargs):
                time.sleep(2.0)  # 2 seconds per forward pass
                return mx.zeros((token_ids.shape[0], token_ids.shape[1], 32))

        model = VerySlowModel(vocab_size=32)
        tokenizer = _make_tokenizer(vocab_size=32)
        config = BatchSchedulerConfig(max_batch_size=4, prefill_chunk_size=64)

        engine = ContinuousBatchingEngine(model, tokenizer, config)
        engine.start()
        try:
            req = engine.submit("timeout test", max_tokens=10)
            # Very short timeout -- the request won't finish in time
            result = engine.wait_for_completion(req, timeout=0.1)
            # The request should NOT be completed (still generating or prefilling)
            assert result.status != RequestStatus.COMPLETED
        finally:
            engine.stop()

    # -- additional concurrency edge cases ----------------------------------

    def test_concurrent_submit_and_cancel(self):
        """Submit and cancel requests concurrently without crashes."""
        engine = self._make_engine(max_batch=4)
        engine.start()
        try:
            errors = []

            def submit_work(idx):
                try:
                    req = engine.submit(f"cancel-test-{idx}", max_tokens=3)
                    if idx % 2 == 0:
                        # Cancel even-numbered requests
                        engine.cancel(req.request_id)
                    else:
                        engine.wait_for_completion(req, timeout=15.0)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=submit_work, args=(i,)) for i in range(16)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30.0)

            assert not errors, f"Got errors: {errors}"
        finally:
            engine.stop()

    def test_rapid_start_stop_cycles(self):
        """Rapidly starting and stopping the engine should not crash."""
        engine = self._make_engine()
        for _ in range(5):
            engine.start()
            engine.submit("cycle test", max_tokens=1)
            engine.stop()
        assert not engine._running

    def test_kv_pool_reclaimed_under_load(self):
        """KV pool slots are properly reclaimed when processing many requests."""
        engine = self._make_engine(max_batch=2)
        engine.start()
        try:
            # Submit more requests than pool slots, sequentially
            for i in range(8):
                req = engine.submit(f"reclaim-{i}", max_tokens=2)
                engine.wait_for_completion(req, timeout=15.0)
                assert req.status == RequestStatus.COMPLETED

            # All slots should be free after all requests complete
            assert engine.kv_pool.num_free_slots == 2
        finally:
            engine.stop()
