"""Tests for batch profiling endpoints (no model loading)."""

import json
import os
import tempfile
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

try:
    import mlx_flash_compress.serve

    HAS_SERVE = True
except (ImportError, ModuleNotFoundError):
    HAS_SERVE = False

pytestmark = pytest.mark.skipif(not HAS_SERVE, reason="serve module requires mlx")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_REGISTRY = {
    "cache_dir": "~/.cache/huggingface/hub",
    "models": [
        {
            "id": "mlx-community/TestModel-3B-4bit",
            "category_expected": "small_dense",
            "size_gb_approx": 2,
            "notes": "Test small model",
        },
        {
            "id": "mlx-community/TestModel-30B-4bit",
            "category_expected": "large_dense",
            "size_gb_approx": 18,
            "notes": "Test large model",
        },
        {
            "id": "mlx-community/SkippedModel",
            "category_expected": "small_dense",
            "size_gb_approx": 1,
            "notes": "Should be skipped",
            "skip": True,
        },
    ],
}


HW_PATCH = "mlx_flash_compress.serve.detect_hardware"
MGR_PATCH = "mlx_flash_compress.serve.MemoryManager"
MEM_PATCH = "mlx_flash_compress.serve.get_memory_state"
MX_PATCH = "mlx_flash_compress.serve.mx"
LOAD_PATCH = "mlx_flash_compress.serve.load"
GEN_PATCH = "mlx_flash_compress.serve.generate"
REGISTRY_PATCH = "mlx_flash_compress.serve.load_model_registry"


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


def _make_state(model_name="test-model"):
    """Create an InferenceState with all external dependencies mocked."""
    fake_mgr = _make_fake_mem_mgr()
    fake_mem = _make_fake_mem()

    with (
        patch(HW_PATCH, return_value=_make_fake_hw()),
        patch(MGR_PATCH, return_value=fake_mgr),
        patch(MEM_PATCH, return_value=fake_mem),
    ):
        from mlx_flash_compress.serve import InferenceState

        state = InferenceState(model_name)

    state._fake_mgr = fake_mgr
    state._fake_mem = fake_mem
    return state


class FakeRequest:
    """Simulate an HTTP request for testing handler methods."""

    def __init__(self, body=b"", method="GET", path="/"):
        self.body = body
        self.method = method
        self.path = path
        self.headers = {"Content-Length": str(len(body))}
        self.rfile = BytesIO(body)


class FakeHandler:
    """Minimal stand-in for ChatHandler to test handler methods in isolation."""

    def __init__(self, state, body=b"", path="/"):
        from mlx_flash_compress.serve import ChatHandler

        self.server_state = state
        self.headers = {"Content-Length": str(len(body))}
        self.rfile = BytesIO(body)
        self.path = path
        self._response_status = None
        self._response_body = None
        self._handler_cls = ChatHandler

    def _send_json(self, data, status=200):
        self._response_status = status
        self._response_body = data


# ---------------------------------------------------------------------------
# Tests for GET /profile/models
# ---------------------------------------------------------------------------


class TestProfileModels:
    def test_returns_model_list(self):
        state = _make_state()
        handler = FakeHandler(state, path="/profile/models")

        with patch(REGISTRY_PATCH, return_value=SAMPLE_REGISTRY):
            from mlx_flash_compress.serve import ChatHandler

            ChatHandler._handle_profile_models(handler)

        assert handler._response_status == 200
        body = handler._response_body
        assert "models" in body
        assert body["total_models"] == 3
        assert len(body["models"]) == 3

    def test_model_fields_present(self):
        state = _make_state()
        handler = FakeHandler(state, path="/profile/models")

        with patch(REGISTRY_PATCH, return_value=SAMPLE_REGISTRY):
            from mlx_flash_compress.serve import ChatHandler

            ChatHandler._handle_profile_models(handler)

        model = handler._response_body["models"][0]
        assert model["id"] == "mlx-community/TestModel-3B-4bit"
        assert model["category_expected"] == "small_dense"
        assert model["size_gb_approx"] == 2
        assert model["skip"] is False
        assert "notes" in model

    def test_skipped_model_included_in_list(self):
        """GET /profile/models returns all models including skipped ones."""
        state = _make_state()
        handler = FakeHandler(state, path="/profile/models")

        with patch(REGISTRY_PATCH, return_value=SAMPLE_REGISTRY):
            from mlx_flash_compress.serve import ChatHandler

            ChatHandler._handle_profile_models(handler)

        ids = [m["id"] for m in handler._response_body["models"]]
        assert "mlx-community/SkippedModel" in ids

    def test_registry_read_error(self):
        state = _make_state()
        handler = FakeHandler(state, path="/profile/models")

        with patch(REGISTRY_PATCH, side_effect=FileNotFoundError("no file")):
            from mlx_flash_compress.serve import ChatHandler

            ChatHandler._handle_profile_models(handler)

        assert handler._response_status == 500
        assert "error" in handler._response_body

    def test_cache_dir_in_response(self):
        state = _make_state()
        handler = FakeHandler(state, path="/profile/models")

        with patch(REGISTRY_PATCH, return_value=SAMPLE_REGISTRY):
            from mlx_flash_compress.serve import ChatHandler

            ChatHandler._handle_profile_models(handler)

        assert "cache_dir" in handler._response_body


# ---------------------------------------------------------------------------
# Tests for POST /profile/batch
# ---------------------------------------------------------------------------


class TestProfileBatch:
    def _make_batch_handler(self, body_dict=None):
        state = _make_state()
        body = json.dumps(body_dict or {}).encode()
        handler = FakeHandler(state, body=body, path="/profile/batch")
        return handler, state

    def test_profiles_all_non_skipped_models(self):
        handler, state = self._make_batch_handler()

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_tokenizer.encode.return_value = list(range(32))

        with (
            patch(REGISTRY_PATCH, return_value=SAMPLE_REGISTRY),
            patch(LOAD_PATCH, return_value=(fake_model, fake_tokenizer)),
            patch(GEN_PATCH, return_value="generated output text"),
            patch(MEM_PATCH, return_value=_make_fake_mem()),
            patch(MX_PATCH),
        ):
            from mlx_flash_compress.serve import ChatHandler

            ChatHandler._handle_profile_batch(handler)

        assert handler._response_status == 200
        results = handler._response_body
        assert isinstance(results, list)
        # Should profile 2 models (the skipped one excluded)
        assert len(results) == 2
        assert results[0]["model"] == "mlx-community/TestModel-3B-4bit"
        assert results[1]["model"] == "mlx-community/TestModel-30B-4bit"

    def test_profiles_specific_models(self):
        body = {"models": ["mlx-community/TestModel-30B-4bit"]}
        handler, state = self._make_batch_handler(body)

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_tokenizer.encode.return_value = list(range(32))

        with (
            patch(REGISTRY_PATCH, return_value=SAMPLE_REGISTRY),
            patch(LOAD_PATCH, return_value=(fake_model, fake_tokenizer)),
            patch(GEN_PATCH, return_value="generated output text"),
            patch(MEM_PATCH, return_value=_make_fake_mem()),
            patch(MX_PATCH),
        ):
            from mlx_flash_compress.serve import ChatHandler

            ChatHandler._handle_profile_batch(handler)

        results = handler._response_body
        assert len(results) == 1
        assert results[0]["model"] == "mlx-community/TestModel-30B-4bit"

    def test_result_fields(self):
        handler, state = self._make_batch_handler()

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_tokenizer.encode.return_value = list(range(32))

        with (
            patch(REGISTRY_PATCH, return_value=SAMPLE_REGISTRY),
            patch(LOAD_PATCH, return_value=(fake_model, fake_tokenizer)),
            patch(GEN_PATCH, return_value="generated output text"),
            patch(MEM_PATCH, return_value=_make_fake_mem()),
            patch(MX_PATCH),
        ):
            from mlx_flash_compress.serve import ChatHandler

            ChatHandler._handle_profile_batch(handler)

        result = handler._response_body[0]
        assert "model" in result
        assert "tokens_per_sec" in result
        assert "memory_gb" in result
        assert "load_time_s" in result
        assert "category" in result

    def test_model_load_failure_recorded(self):
        handler, state = self._make_batch_handler()

        with (
            patch(REGISTRY_PATCH, return_value=SAMPLE_REGISTRY),
            patch(LOAD_PATCH, side_effect=Exception("download failed")),
            patch(MEM_PATCH, return_value=_make_fake_mem()),
            patch(MX_PATCH),
        ):
            from mlx_flash_compress.serve import ChatHandler

            ChatHandler._handle_profile_batch(handler)

        results = handler._response_body
        assert len(results) == 2
        # Both should have error since load always fails
        for r in results:
            assert "error" in r
            assert "download failed" in r["error"]

    def test_invalid_json_returns_400(self):
        state = _make_state()
        handler = FakeHandler(state, body=b"not json", path="/profile/batch")

        from mlx_flash_compress.serve import ChatHandler

        ChatHandler._handle_profile_batch(handler)

        assert handler._response_status == 400
        assert "error" in handler._response_body

    def test_empty_body_profiles_all(self):
        """Empty body should profile all non-skipped models."""
        state = _make_state()
        handler = FakeHandler(state, body=b"", path="/profile/batch")
        # Patch Content-Length to 0 for empty body
        handler.headers = {"Content-Length": "0"}

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_tokenizer.encode.return_value = list(range(32))

        with (
            patch(REGISTRY_PATCH, return_value=SAMPLE_REGISTRY),
            patch(LOAD_PATCH, return_value=(fake_model, fake_tokenizer)),
            patch(GEN_PATCH, return_value="generated output text"),
            patch(MEM_PATCH, return_value=_make_fake_mem()),
            patch(MX_PATCH),
        ):
            from mlx_flash_compress.serve import ChatHandler

            ChatHandler._handle_profile_batch(handler)

        results = handler._response_body
        assert isinstance(results, list)
        assert len(results) == 2  # skipped model excluded

    def test_registry_read_error_returns_500(self):
        handler, state = self._make_batch_handler()

        with patch(REGISTRY_PATCH, side_effect=FileNotFoundError("missing")):
            from mlx_flash_compress.serve import ChatHandler

            ChatHandler._handle_profile_batch(handler)

        assert handler._response_status == 500
        assert "error" in handler._response_body

    def test_original_model_name_restored(self):
        """After batch profiling, the original model name should be restored."""
        state = _make_state("my-original-model")
        body = json.dumps({"models": ["mlx-community/TestModel-3B-4bit"]}).encode()
        handler = FakeHandler(state, body=body, path="/profile/batch")
        handler.server_state = state

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_tokenizer.encode.return_value = list(range(32))

        with (
            patch(REGISTRY_PATCH, return_value=SAMPLE_REGISTRY),
            patch(LOAD_PATCH, return_value=(fake_model, fake_tokenizer)),
            patch(GEN_PATCH, return_value="generated output text"),
            patch(MEM_PATCH, return_value=_make_fake_mem()),
            patch(MX_PATCH),
        ):
            from mlx_flash_compress.serve import ChatHandler

            ChatHandler._handle_profile_batch(handler)

        assert state.model_name == "my-original-model"

    def test_custom_max_tokens_and_prompt(self):
        """Custom max_tokens and prompt should be forwarded to generate()."""
        body = {
            "models": ["mlx-community/TestModel-3B-4bit"],
            "max_tokens": 64,
            "prompt": "custom prompt here",
        }
        handler, state = self._make_batch_handler(body)

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_tokenizer.encode.return_value = list(range(64))

        with (
            patch(REGISTRY_PATCH, return_value=SAMPLE_REGISTRY),
            patch(LOAD_PATCH, return_value=(fake_model, fake_tokenizer)),
            patch(GEN_PATCH, return_value="generated output text") as mock_gen,
            patch(MEM_PATCH, return_value=_make_fake_mem()),
            patch(MX_PATCH),
        ):
            from mlx_flash_compress.serve import ChatHandler

            ChatHandler._handle_profile_batch(handler)

        # Verify generate was called with the custom prompt and max_tokens
        call_kwargs = mock_gen.call_args
        assert (
            call_kwargs.kwargs.get("prompt") == "custom prompt here"
            or call_kwargs[1].get("prompt") == "custom prompt here"
        )
        assert call_kwargs.kwargs.get("max_tokens") == 64 or call_kwargs[1].get("max_tokens") == 64

    def test_partial_failure_still_returns_all_results(self):
        """If one model fails and another succeeds, both are in results."""
        handler, state = self._make_batch_handler()

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_tokenizer.encode.return_value = list(range(32))

        call_count = 0

        def load_side_effect(model_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("first model failed")
            return (fake_model, fake_tokenizer)

        with (
            patch(REGISTRY_PATCH, return_value=SAMPLE_REGISTRY),
            patch(LOAD_PATCH, side_effect=load_side_effect),
            patch(GEN_PATCH, return_value="generated output text"),
            patch(MEM_PATCH, return_value=_make_fake_mem()),
            patch(MX_PATCH),
        ):
            from mlx_flash_compress.serve import ChatHandler

            ChatHandler._handle_profile_batch(handler)

        results = handler._response_body
        assert len(results) == 2
        assert "error" in results[0]
        assert "tokens_per_sec" in results[1]
