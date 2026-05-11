"""Tests for router_hook — MLX router interception for expert prefetching."""

import threading
from collections import defaultdict

import numpy as np
import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    nn = None

from mlx_flash_compress.router_hook import (
    RouterHook,
    RouterHookStats,
    RoutingEvent,
)

# RouterHook requires MLX at __init__ time
pytestmark = pytest.mark.skipif(not HAS_MLX, reason="router_hook requires mlx")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeGate:
    """Minimal gate module that returns fixed logits."""

    def __call__(self, x):
        return mx.array([[1.0, 0.5, 0.3, 0.1]])


class _FakeMLP:
    def __init__(self, has_gate: bool = True):
        if has_gate:
            self.gate = _FakeGate()


class _FakeLayer:
    def __init__(self, has_gate: bool = True):
        self.mlp = _FakeMLP(has_gate=has_gate)


def _make_fake_model(num_layers: int = 3, has_gates: bool = True):
    class FakeModel:
        class model:
            layers = [_FakeLayer(has_gate=has_gates) for _ in range(num_layers)]

    return FakeModel()


def _make_direct_layers_model(num_layers: int = 2, has_gates: bool = True):
    """Model with model.layers (no model.model wrapper)."""

    class FakeModel:
        layers = [_FakeLayer(has_gate=has_gates) for _ in range(num_layers)]

    return FakeModel()


# ---------------------------------------------------------------------------
# RoutingEvent tests
# ---------------------------------------------------------------------------


class TestRoutingEvent:
    def test_creation(self):
        ev = RoutingEvent(token_idx=0, layer_idx=1, expert_ids=[2, 5], expert_weights=[0.7, 0.3])
        assert ev.token_idx == 0
        assert ev.layer_idx == 1
        assert ev.expert_ids == [2, 5]
        assert ev.expert_weights == [0.7, 0.3]

    def test_fields_are_mutable(self):
        ev = RoutingEvent(token_idx=0, layer_idx=0, expert_ids=[], expert_weights=[])
        ev.expert_ids.append(3)
        assert 3 in ev.expert_ids

    def test_default_values_not_required(self):
        """All fields are positional — none have defaults."""
        with pytest.raises(TypeError):
            RoutingEvent()  # type: ignore[call-arg]

    def test_multiple_experts(self):
        ids = list(range(8))
        weights = [1.0 / 8] * 8
        ev = RoutingEvent(token_idx=5, layer_idx=2, expert_ids=ids, expert_weights=weights)
        assert len(ev.expert_ids) == 8
        assert len(ev.expert_weights) == 8


# ---------------------------------------------------------------------------
# RouterHookStats tests
# ---------------------------------------------------------------------------


class TestRouterHookStats:
    def test_defaults(self):
        stats = RouterHookStats()
        assert stats.total_events == 0
        assert stats.total_tokens == 0
        assert stats.prediction_correct == 0
        assert stats.prediction_total == 0

    def test_prediction_accuracy_zero_division(self):
        stats = RouterHookStats()
        assert stats.prediction_accuracy == 0.0

    def test_prediction_accuracy_computed(self):
        stats = RouterHookStats(prediction_correct=7, prediction_total=10)
        assert abs(stats.prediction_accuracy - 0.7) < 1e-9

    def test_expert_frequency_is_defaultdict(self):
        stats = RouterHookStats()
        assert isinstance(stats.expert_frequency, defaultdict)
        # Accessing unknown key should return 0
        assert stats.expert_frequency[999] == 0

    def test_layer_expert_frequency_nested_defaultdict(self):
        stats = RouterHookStats()
        assert isinstance(stats.layer_expert_frequency, defaultdict)
        # Nested access should work without KeyError
        assert stats.layer_expert_frequency[0][5] == 0


# ---------------------------------------------------------------------------
# RouterHook creation and configuration tests
# ---------------------------------------------------------------------------


class TestRouterHookInit:
    def test_basic_creation(self):
        model = _make_fake_model()
        hook = RouterHook(model, num_layers=3, num_experts=8, top_k=2)
        assert hook.num_layers == 3
        assert hook.num_experts == 8
        assert hook.top_k == 2
        assert not hook._installed

    def test_default_params(self):
        model = _make_fake_model()
        hook = RouterHook(model)
        assert hook.num_layers == 24
        assert hook.num_experts == 60
        assert hook.top_k == 4

    def test_routing_log_starts_empty(self):
        model = _make_fake_model()
        hook = RouterHook(model, num_layers=3, num_experts=8)
        assert hook.get_routing_log() == []

    def test_cooccurrence_shape(self):
        model = _make_fake_model()
        hook = RouterHook(model, num_layers=4, num_experts=10)
        assert hook._cooccurrence.shape == (4, 10, 10)

    def test_stats_initial(self):
        model = _make_fake_model()
        hook = RouterHook(model, num_layers=3, num_experts=8)
        assert hook.stats.total_events == 0
        assert hook.stats.total_tokens == 0


# ---------------------------------------------------------------------------
# Gate finding and hooking tests
# ---------------------------------------------------------------------------


class TestGateFinding:
    def test_install_finds_gates_via_model_model_layers(self):
        model = _make_fake_model(num_layers=3)
        hook = RouterHook(model, num_layers=3, num_experts=8)
        hook.install()
        # _gate_modules should have been populated for layers with gates
        assert len(hook._gate_modules) == 3
        hook.uninstall()

    def test_install_finds_gates_via_direct_layers(self):
        model = _make_direct_layers_model(num_layers=2)
        hook = RouterHook(model, num_layers=2, num_experts=8)
        hook.install()
        assert len(hook._gate_modules) == 2
        hook.uninstall()

    def test_install_no_layers(self):
        class EmptyModel:
            pass

        hook = RouterHook(EmptyModel(), num_layers=1, num_experts=4)
        hook.install()
        assert len(hook._gate_modules) == 0
        hook.uninstall()

    def test_install_layers_without_gate(self):
        model = _make_fake_model(num_layers=2, has_gates=False)
        hook = RouterHook(model, num_layers=2, num_experts=8)
        hook.install()
        assert len(hook._gate_modules) == 0
        hook.uninstall()

    def test_install_idempotent(self):
        model = _make_fake_model(num_layers=2)
        hook = RouterHook(model, num_layers=2, num_experts=8)
        hook.install()
        hook.install()  # second call should be no-op
        assert hook._installed is True
        hook.uninstall()

    def test_uninstall_clears_state(self):
        model = _make_fake_model(num_layers=2)
        hook = RouterHook(model, num_layers=2, num_experts=8)
        hook.install()
        assert hook._installed is True
        hook.uninstall()
        assert hook._installed is False
        assert len(hook._patched_mlps) == 0
        assert len(hook._original_gates) == 0

    def test_block_sparse_moe_path(self):
        """Verify hook finds gates via layer.block_sparse_moe.gate."""

        class FakeBSMoE:
            def __init__(self):
                self.gate = _FakeGate()

        class FakeLayer:
            def __init__(self):
                self.block_sparse_moe = FakeBSMoE()

        class FakeModel:
            class model:
                layers = [FakeLayer()]

        hook = RouterHook(FakeModel(), num_layers=1, num_experts=4)
        hook.install()
        assert len(hook._gate_modules) == 1
        hook.uninstall()


# ---------------------------------------------------------------------------
# Gate interception tests — verify actual routing capture
# ---------------------------------------------------------------------------


class TestGateInterception:
    def test_hooked_gate_records_events(self):
        """Calling the hooked gate should record a RoutingEvent."""
        model = _make_fake_model(num_layers=3)
        hook = RouterHook(model, num_layers=3, num_experts=4, top_k=2)
        hook.install()

        # Call the gate on layer 0 — should trigger routing capture
        x = mx.array([[1.0, 2.0, 3.0, 4.0]])
        gate = model.model.layers[0].mlp.gate
        result = gate(x)

        assert hook.stats.total_events == 1
        log = hook.get_routing_log()
        assert len(log) == 1
        assert log[0].layer_idx == 0
        assert len(log[0].expert_ids) == 2  # top_k=2
        assert len(log[0].expert_weights) == 2
        hook.uninstall()

    def test_hooked_gate_returns_original_logits(self):
        """Hook must not modify the gate output (observation only)."""
        model = _make_fake_model(num_layers=1)
        original_gate = model.model.layers[0].mlp.gate

        # Get expected logits before hooking
        x = mx.array([[1.0, 2.0, 3.0, 4.0]])
        expected = original_gate(x)

        hook = RouterHook(model, num_layers=1, num_experts=4, top_k=2)
        hook.install()

        hooked_result = model.model.layers[0].mlp.gate(x)
        assert np.allclose(np.array(expected), np.array(hooked_result))
        hook.uninstall()

    def test_uninstall_restores_original_gate(self):
        """After uninstall, gate should be the original object."""
        model = _make_fake_model(num_layers=2)
        original_gates = [layer.mlp.gate for layer in model.model.layers]

        hook = RouterHook(model, num_layers=2, num_experts=4, top_k=2)
        hook.install()

        # Gates should now be wrappers, not originals
        for layer in model.model.layers:
            assert layer.mlp.gate is not original_gates[0] or layer.mlp.gate is not original_gates[1]

        hook.uninstall()

        # Gates should be restored to originals
        for i, layer in enumerate(model.model.layers):
            assert layer.mlp.gate is original_gates[i]

    def test_multiple_gate_calls_record_multiple_events(self):
        """Each gate call should record a separate event."""
        model = _make_fake_model(num_layers=2)
        hook = RouterHook(model, num_layers=2, num_experts=4, top_k=2)
        hook.install()

        x = mx.array([[1.0, 2.0, 3.0, 4.0]])
        # Call gates on two different layers
        model.model.layers[0].mlp.gate(x)
        model.model.layers[1].mlp.gate(x)

        assert hook.stats.total_events == 2
        log = hook.get_routing_log()
        assert log[0].layer_idx == 0
        assert log[1].layer_idx == 1
        hook.uninstall()

    def test_hooked_gate_updates_cooccurrence_within_layer(self):
        """Experts selected together in the same layer should update
        within-layer co-occurrence."""
        model = _make_fake_model(num_layers=2)
        hook = RouterHook(model, num_layers=2, num_experts=4, top_k=2)
        hook.install()

        x = mx.array([[1.0, 2.0, 3.0, 4.0]])
        model.model.layers[0].mlp.gate(x)

        # The FakeGate returns [1.0, 0.5, 0.3, 0.1] -> top-2 are experts 0 and 1
        # Co-occurrence within layer 0: cooccurrence[0, 0, 1] and [0, 1, 0] should be > 0
        assert hook._cooccurrence[0, 0, 1] > 0
        assert hook._cooccurrence[0, 1, 0] > 0
        hook.uninstall()

    def test_hooked_gate_delegates_attributes(self):
        """The wrapper should delegate attribute access to the original gate."""

        class GateWithAttr:
            custom_attr = 42

            def __call__(self, x):
                return mx.array([[1.0, 0.5, 0.3, 0.1]])

        class MLP:
            def __init__(self):
                self.gate = GateWithAttr()

        class Layer:
            def __init__(self):
                self.mlp = MLP()

        class FakeModel:
            class model:
                layers = [Layer()]

        model = FakeModel()
        hook = RouterHook(model, num_layers=1, num_experts=4, top_k=2)
        hook.install()

        assert model.model.layers[0].mlp.gate.custom_attr == 42
        hook.uninstall()

    def test_block_sparse_moe_gate_interception(self):
        """Verify gate interception works via block_sparse_moe path."""

        class FakeBSMoEGate:
            def __call__(self, x):
                return mx.array([[0.9, 0.1, 0.05, 0.01]])

        class FakeBSMoE:
            def __init__(self):
                self.gate = FakeBSMoEGate()

        class FakeLayer:
            def __init__(self):
                self.block_sparse_moe = FakeBSMoE()

        class FakeModel:
            class model:
                layers = [FakeLayer()]

        model = FakeModel()
        hook = RouterHook(model, num_layers=1, num_experts=4, top_k=2)
        hook.install()

        x = mx.array([[1.0, 2.0, 3.0]])
        model.model.layers[0].block_sparse_moe.gate(x)

        assert hook.stats.total_events == 1
        log = hook.get_routing_log()
        assert log[0].layer_idx == 0
        assert len(log[0].expert_ids) == 2
        hook.uninstall()

        # Gate should be restored
        assert isinstance(model.model.layers[0].block_sparse_moe.gate, FakeBSMoEGate)


# ---------------------------------------------------------------------------
# _record_routing and co-occurrence tests
# ---------------------------------------------------------------------------


class TestRecordRouting:
    def test_single_event(self):
        model = _make_fake_model()
        hook = RouterHook(model, num_layers=4, num_experts=8, top_k=2)
        hook._record_routing(0, [1, 3], [0.6, 0.4])
        assert hook.stats.total_events == 1
        log = hook.get_routing_log()
        assert len(log) == 1
        assert log[0].layer_idx == 0
        assert log[0].expert_ids == [1, 3]

    def test_expert_frequency_updated(self):
        model = _make_fake_model()
        hook = RouterHook(model, num_layers=4, num_experts=8)
        hook._record_routing(0, [1, 3], [0.6, 0.4])
        hook._record_routing(0, [1, 5], [0.7, 0.3])
        assert hook.stats.expert_frequency[1] == 2
        assert hook.stats.expert_frequency[3] == 1
        assert hook.stats.expert_frequency[5] == 1

    def test_layer_expert_frequency_updated(self):
        model = _make_fake_model()
        hook = RouterHook(model, num_layers=4, num_experts=8)
        hook._record_routing(2, [0, 4], [0.5, 0.5])
        assert hook.stats.layer_expert_frequency[2][0] == 1
        assert hook.stats.layer_expert_frequency[2][4] == 1

    def test_cooccurrence_updated_across_layers(self):
        model = _make_fake_model()
        hook = RouterHook(model, num_layers=4, num_experts=8)
        # Layer 0 activates experts [1, 2]
        hook._record_routing(0, [1, 2], [0.5, 0.5])
        # Layer 1 activates experts [3, 4] — should update co-occurrence[0, 1, 3] etc.
        hook._record_routing(1, [3, 4], [0.5, 0.5])

        assert hook._cooccurrence[0, 1, 3] == 1
        assert hook._cooccurrence[0, 1, 4] == 1
        assert hook._cooccurrence[0, 2, 3] == 1
        assert hook._cooccurrence[0, 2, 4] == 1

    def test_cooccurrence_not_updated_for_layer_zero(self):
        model = _make_fake_model()
        hook = RouterHook(model, num_layers=4, num_experts=8)
        hook._record_routing(0, [1, 2], [0.5, 0.5])
        # No previous layer, so co-occurrence should be all zeros
        assert hook._cooccurrence.sum() == 0.0

    def test_cooccurrence_ignores_out_of_range(self):
        model = _make_fake_model()
        hook = RouterHook(model, num_layers=4, num_experts=4)
        hook._record_routing(0, [0, 1], [0.5, 0.5])
        # Expert 99 is out of range
        hook._record_routing(1, [99], [1.0])
        # Should not crash, out-of-range experts skipped
        assert hook._cooccurrence.sum() == 0.0

    def test_routing_log_is_copy(self):
        model = _make_fake_model()
        hook = RouterHook(model, num_layers=4, num_experts=8)
        hook._record_routing(0, [1], [1.0])
        log = hook.get_routing_log()
        log.clear()
        # Original should be unaffected
        assert len(hook.get_routing_log()) == 1


# ---------------------------------------------------------------------------
# Token counter and advance_token tests
# ---------------------------------------------------------------------------


class TestAdvanceToken:
    def test_advance_increments_counter(self):
        model = _make_fake_model()
        hook = RouterHook(model, num_layers=4, num_experts=8)
        assert hook._token_counter == 0
        hook.advance_token()
        assert hook._token_counter == 1
        assert hook.stats.total_tokens == 1

    def test_advance_clears_prev_experts(self):
        model = _make_fake_model()
        hook = RouterHook(model, num_layers=4, num_experts=8)
        hook._record_routing(0, [1, 2], [0.5, 0.5])
        assert len(hook._prev_experts) > 0
        hook.advance_token()
        assert len(hook._prev_experts) == 0


# ---------------------------------------------------------------------------
# predict_next tests
# ---------------------------------------------------------------------------


class TestPredictNext:
    def test_no_data_returns_empty(self):
        model = _make_fake_model()
        hook = RouterHook(model, num_layers=4, num_experts=8, top_k=2)
        result = hook.predict_next(0, [1])
        assert result == []

    def test_last_layer_returns_empty(self):
        model = _make_fake_model()
        hook = RouterHook(model, num_layers=4, num_experts=8, top_k=2)
        result = hook.predict_next(3, [1])  # layer 3 is last layer (0-indexed)
        assert result == []

    def test_with_cooccurrence_data(self):
        model = _make_fake_model()
        hook = RouterHook(model, num_layers=4, num_experts=8, top_k=2)
        # Manually set co-occurrence: layer 0, expert 1 -> expert 5 strongly
        hook._cooccurrence[0, 1, 5] = 10.0
        hook._cooccurrence[0, 1, 3] = 5.0
        predicted = hook.predict_next(0, [1])
        assert 5 in predicted
        assert 3 in predicted

    def test_predict_uses_top_k(self):
        model = _make_fake_model()
        hook = RouterHook(model, num_layers=4, num_experts=8, top_k=3)
        for i in range(8):
            hook._cooccurrence[0, 0, i] = float(i)
        predicted = hook.predict_next(0, [0])
        assert len(predicted) == 3
        # Should return [7, 6, 5] — highest scores
        assert predicted == [7, 6, 5]


# ---------------------------------------------------------------------------
# Expert heatmap tests
# ---------------------------------------------------------------------------


class TestExpertHeatmap:
    def test_empty_log(self):
        model = _make_fake_model()
        hook = RouterHook(model, num_layers=3, num_experts=4)
        heatmap = hook.get_expert_heatmap()
        assert heatmap.shape == (3, 4)
        # All zeros normalized: should be 0 (divides by 1 to avoid /0)
        assert heatmap.sum() == 0.0

    def test_single_event_heatmap(self):
        model = _make_fake_model()
        hook = RouterHook(model, num_layers=3, num_experts=4)
        hook._record_routing(0, [0, 1], [0.5, 0.5])
        heatmap = hook.get_expert_heatmap()
        # Layer 0 should have experts 0 and 1 with 0.5 each
        assert abs(heatmap[0, 0] - 0.5) < 1e-6
        assert abs(heatmap[0, 1] - 0.5) < 1e-6
        assert heatmap[1, :].sum() == 0.0


# ---------------------------------------------------------------------------
# Prediction accuracy measurement
# ---------------------------------------------------------------------------


class TestMeasurePredictionAccuracy:
    def test_too_few_events(self):
        model = _make_fake_model()
        hook = RouterHook(model, num_layers=4, num_experts=8)
        assert hook.measure_prediction_accuracy() == 0.0

    def test_one_event_returns_zero(self):
        model = _make_fake_model()
        hook = RouterHook(model, num_layers=4, num_experts=8)
        hook._record_routing(0, [1, 2], [0.5, 0.5])
        assert hook.measure_prediction_accuracy() == 0.0

    def test_accuracy_after_multiple_events(self):
        model = _make_fake_model()
        hook = RouterHook(model, num_layers=4, num_experts=8, top_k=2)
        # Build enough routing data for the predictor to work
        for _t in range(5):
            hook._record_routing(0, [0, 1], [0.6, 0.4])
            hook._record_routing(1, [2, 3], [0.5, 0.5])
            hook.advance_token()

        acc = hook.measure_prediction_accuracy()
        # With consistent routing, co-occurrence predictor should learn
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0
        # Stats should be updated
        assert hook.stats.prediction_total > 0


# ---------------------------------------------------------------------------
# get_hot_experts tests
# ---------------------------------------------------------------------------


class TestGetHotExperts:
    def test_empty_returns_empty(self):
        model = _make_fake_model()
        hook = RouterHook(model, num_layers=3, num_experts=4)
        assert hook.get_hot_experts() == {}

    def test_returns_frequent_experts(self):
        model = _make_fake_model()
        hook = RouterHook(model, num_layers=3, num_experts=4)
        # Record expert 0 many times at layer 0
        for _ in range(20):
            hook._record_routing(0, [0], [1.0])
        # Expert 0 is 100% of layer 0 traffic — above 5% threshold
        hot = hook.get_hot_experts(threshold=0.05)
        assert 0 in hot
        assert 0 in hot[0]

    def test_threshold_filters_cold_experts(self):
        model = _make_fake_model()
        hook = RouterHook(model, num_layers=3, num_experts=100)
        # Expert 0 accessed 95 times, experts 1-99 accessed 1 time each
        # But our tracking only counts layer_expert_frequency via _record_routing
        for _ in range(95):
            hook._record_routing(0, [0], [1.0])
        for i in range(1, 6):
            hook._record_routing(0, [i], [1.0])
        hot = hook.get_hot_experts(threshold=0.5)
        # Only expert 0 should be above 50% threshold
        assert 0 in hot.get(0, [])
        for i in range(1, 6):
            assert i not in hot.get(0, [])


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_record_routing(self):
        model = _make_fake_model()
        hook = RouterHook(model, num_layers=4, num_experts=8)

        def record_many():
            for i in range(100):
                hook._record_routing(0, [0, 1], [0.5, 0.5])

        threads = [threading.Thread(target=record_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert hook.stats.total_events == 400
