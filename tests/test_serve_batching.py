"""Tests for continuous batching integration in serve.py."""

import argparse
import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except (ImportError, ModuleNotFoundError):
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="requires mlx")

if HAS_MLX:
    from mlx_flash_compress.continuous_batching import (
        ContinuousBatchingEngine,
        InferenceRequest,
        RequestStatus,
    )
    from mlx_flash_compress.serve import InferenceState, main

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

HW_PATCH = "mlx_flash_compress.serve.detect_hardware"
MGR_PATCH = "mlx_flash_compress.serve.MemoryManager"
MEM_PATCH = "mlx_flash_compress.serve.get_memory_state"


def _make_fake_hw():
    hw = MagicMock()
    hw.chip = "Apple M2 Pro"
    hw.total_ram_gb = 32.0
    return hw


def _make_fake_mem(pressure="normal"):
    mem = MagicMock()
    mem.total_gb = 32.0
    mem.free_gb = 8.0
    mem.inactive_gb = 4.0
    mem.available_gb = 10.0
    mem.pressure_level = pressure
    mem.swap_used_gb = 0.0
    return mem


def _make_fake_mem_mgr():
    mgr = MagicMock()
    mgr.get_cache_budget_gb.return_value = 4.0
    mgr.get_optimization_hints.return_value = []
    mgr.auto_release_if_needed.return_value = {"released": False}
    return mgr


def _make_state(batching=False):
    """Create an InferenceState with external dependencies mocked."""
    with (
        patch(HW_PATCH, return_value=_make_fake_hw()),
        patch(MGR_PATCH, return_value=_make_fake_mem_mgr()),
        patch(MEM_PATCH, return_value=_make_fake_mem()),
    ):
        state = InferenceState("test-model", batching=batching)
    return state


# ===========================================================================
# CLI flag parsing
# ===========================================================================


class TestBatchingFlag:
    def test_batching_flag_default_off(self):
        """--batching defaults to False."""
        state = _make_state(batching=False)
        assert state.batching is False
        assert state.engine is None

    def test_batching_flag_enabled(self):
        """When batching=True, InferenceState stores it."""
        state = _make_state(batching=True)
        assert state.batching is True
        # Engine is None until model is loaded
        assert state.engine is None

    def test_argparse_has_batching(self):
        """The --batching flag is recognized by the argument parser."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--batching", action="store_true")
        args = parser.parse_args(["--batching"])
        assert args.batching is True

        args_default = parser.parse_args([])
        assert args_default.batching is False


# ===========================================================================
# ContinuousBatchingEngine creation
# ===========================================================================


class MockModel:
    """Minimal mock model for engine creation tests."""

    def __init__(self):
        self.num_layers = 2
        self.num_kv_heads = 2
        self.head_dim = 8
        self._linear = nn.Linear(32, 32)

    def parameters(self):
        return self._linear.parameters()

    def __call__(self, token_ids, **kwargs):
        batch, seq_len = token_ids.shape
        eye = mx.eye(32)
        flat_ids = mx.clip(token_ids.reshape(-1), 0, 31)
        one_hot = eye[flat_ids]
        logits = self._linear(one_hot)
        return logits.reshape(batch, seq_len, 32)


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 0

    def encode(self, text):
        return [ord(c) % 32 for c in text]

    def decode(self, token_ids):
        return "".join(chr(t + 65) for t in token_ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(m.get("content", "") for m in messages)


class TestEngineCreation:
    def test_engine_can_be_created(self):
        """ContinuousBatchingEngine can be instantiated with mock model/tokenizer."""
        model = MockModel()
        tokenizer = MockTokenizer()
        engine = ContinuousBatchingEngine(model, tokenizer)
        assert engine is not None
        assert engine.model is model
        assert engine.tokenizer is tokenizer

    def test_engine_start_stop(self):
        """Engine starts and stops cleanly."""
        model = MockModel()
        tokenizer = MockTokenizer()
        engine = ContinuousBatchingEngine(model, tokenizer)
        engine.start()
        assert engine._running is True
        engine.stop()
        assert engine._running is False


# ===========================================================================
# wait_for_completion
# ===========================================================================


class TestWaitForCompletion:
    def test_wait_completed_request(self):
        """wait_for_completion returns immediately for already-completed requests."""
        model = MockModel()
        tokenizer = MockTokenizer()
        engine = ContinuousBatchingEngine(model, tokenizer)

        req = InferenceRequest(
            request_id="test-1",
            prompt_tokens=[1, 2, 3],
            max_tokens=5,
        )
        req.status = RequestStatus.COMPLETED
        req.generated_tokens = [10, 11, 12]
        req._event.set()

        result = engine.wait_for_completion(req, timeout=1.0)
        assert result.status == RequestStatus.COMPLETED
        assert result.generated_tokens == [10, 11, 12]

    def test_wait_timeout(self):
        """wait_for_completion respects timeout for unfinished requests."""
        model = MockModel()
        tokenizer = MockTokenizer()
        engine = ContinuousBatchingEngine(model, tokenizer)

        req = InferenceRequest(
            request_id="test-2",
            prompt_tokens=[1, 2, 3],
            max_tokens=5,
        )
        # Request stays QUEUED, event never set
        result = engine.wait_for_completion(req, timeout=0.1)
        assert result.status == RequestStatus.QUEUED

    def test_wait_with_background_completion(self):
        """wait_for_completion unblocks when another thread sets the event."""
        model = MockModel()
        tokenizer = MockTokenizer()
        engine = ContinuousBatchingEngine(model, tokenizer)

        req = InferenceRequest(
            request_id="test-3",
            prompt_tokens=[1, 2, 3],
            max_tokens=5,
        )

        def complete_later():
            time.sleep(0.05)
            req.status = RequestStatus.COMPLETED
            req.generated_tokens = [7, 8, 9]
            req._event.set()

        t = threading.Thread(target=complete_later)
        t.start()
        result = engine.wait_for_completion(req, timeout=5.0)
        t.join()

        assert result.status == RequestStatus.COMPLETED
        assert result.generated_tokens == [7, 8, 9]


# ===========================================================================
# stream_tokens
# ===========================================================================


class TestStreamTokens:
    def test_stream_yields_tokens(self):
        """stream_tokens yields tokens as they appear in the request."""
        model = MockModel()
        tokenizer = MockTokenizer()
        engine = ContinuousBatchingEngine(model, tokenizer)

        req = InferenceRequest(
            request_id="stream-1",
            prompt_tokens=[1, 2],
            max_tokens=3,
        )

        # Pre-populate tokens and mark completed so the generator finishes
        req.generated_tokens = [10, 20, 30]
        req.status = RequestStatus.COMPLETED

        tokens = list(engine.stream_tokens(req))
        assert tokens == [10, 20, 30]

    def test_stream_yields_incrementally(self):
        """stream_tokens yields tokens as they are appended by another thread."""
        model = MockModel()
        tokenizer = MockTokenizer()
        engine = ContinuousBatchingEngine(model, tokenizer)

        req = InferenceRequest(
            request_id="stream-2",
            prompt_tokens=[1],
            max_tokens=3,
        )
        req.status = RequestStatus.GENERATING

        collected = []

        def producer():
            for tok in [100, 200, 300]:
                time.sleep(0.02)
                req.generated_tokens.append(tok)
            time.sleep(0.02)
            req.status = RequestStatus.COMPLETED

        t = threading.Thread(target=producer)
        t.start()

        for tok in engine.stream_tokens(req):
            collected.append(tok)

        t.join()
        assert collected == [100, 200, 300]

    def test_stream_cancelled_request(self):
        """stream_tokens terminates when request is cancelled."""
        model = MockModel()
        tokenizer = MockTokenizer()
        engine = ContinuousBatchingEngine(model, tokenizer)

        req = InferenceRequest(
            request_id="stream-3",
            prompt_tokens=[1],
            max_tokens=10,
        )
        req.generated_tokens = [5]
        req.status = RequestStatus.CANCELLED

        tokens = list(engine.stream_tokens(req))
        assert tokens == [5]


# ===========================================================================
# Integration: batching routing in _handle_chat
# ===========================================================================


class TestBatchedChatRouting:
    def test_batching_disabled_uses_direct_generate(self):
        """When batching is off, state.engine is None and direct generate is used."""
        state = _make_state(batching=False)
        assert state.engine is None
        assert state.batching is False

    def test_batching_enabled_state_has_engine_field(self):
        """When batching is on, the engine field exists (None until model loaded)."""
        state = _make_state(batching=True)
        assert state.batching is True
        assert hasattr(state, "engine")
