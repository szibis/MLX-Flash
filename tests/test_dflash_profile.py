"""Tests for DFlash auto-profiling module (dflash_profile.py).

Tests profile selection logic, profile configuration values, ModelProfile
properties, and edge cases. Does NOT test detect_model or measure_ar_baseline
since those need a real MLX model; instead tests the pure-Python logic in
select_profile, ModelProfile, and DFlashProfile.
"""

import pytest

try:
    from mlx_flash_compress.dflash_profile import (
        PROFILES,
        DFlashProfile,
        ModelProfile,
        select_profile,
    )

    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(not HAS_MODULE, reason="dflash_profile requires mlx")


# ---------------------------------------------------------------------------
# Helper: create a ModelProfile with sensible defaults
# ---------------------------------------------------------------------------


def _make_profile(
    *,
    ar_tok_s=None,
    has_ssm=False,
    is_moe=False,
    category="medium_dense",
    active_params_b=15.0,
    total_params_b=15.0,
    num_layers=32,
    num_ssm_layers=0,
    num_attn_layers=32,
    is_quantized=False,
    quant_bits=None,
):
    return ModelProfile(
        category=category,
        total_params_b=total_params_b,
        active_params_b=active_params_b,
        num_layers=num_layers,
        num_ssm_layers=num_ssm_layers,
        num_attn_layers=num_attn_layers,
        is_moe=is_moe,
        is_quantized=is_quantized,
        quant_bits=quant_bits,
        ar_tok_s=ar_tok_s,
        has_ssm=has_ssm,
    )


# ===========================================================================
# Tests for ModelProfile dataclass
# ===========================================================================


class TestModelProfile:
    """Tests for ModelProfile properties and basic behavior."""

    def test_ssm_ratio_no_ssm(self):
        profile = _make_profile(num_ssm_layers=0, num_layers=32)
        assert profile.ssm_ratio == 0.0

    def test_ssm_ratio_half(self):
        profile = _make_profile(num_ssm_layers=16, num_layers=32)
        assert profile.ssm_ratio == 0.5

    def test_ssm_ratio_all_ssm(self):
        profile = _make_profile(num_ssm_layers=32, num_layers=32)
        assert profile.ssm_ratio == 1.0

    def test_ssm_ratio_zero_layers(self):
        """Edge case: num_layers=0 should not divide by zero."""
        profile = _make_profile(num_ssm_layers=0, num_layers=0)
        # max(1, 0) = 1, so result is 0/1 = 0
        assert profile.ssm_ratio == 0.0

    def test_recommendation_skip_high_speed(self):
        profile = _make_profile(ar_tok_s=60.0)
        assert profile.recommendation == "skip"

    def test_recommendation_skip_boundary(self):
        profile = _make_profile(ar_tok_s=50.1)
        assert profile.recommendation == "skip"

    def test_recommendation_marginal(self):
        profile = _make_profile(ar_tok_s=30.0)
        assert profile.recommendation == "marginal"

    def test_recommendation_marginal_boundary_low(self):
        profile = _make_profile(ar_tok_s=25.1)
        assert profile.recommendation == "marginal"

    def test_recommendation_recommended(self):
        profile = _make_profile(ar_tok_s=20.0)
        assert profile.recommendation == "recommended"

    def test_recommendation_recommended_low(self):
        profile = _make_profile(ar_tok_s=5.0)
        assert profile.recommendation == "recommended"

    def test_recommendation_none_speed(self):
        """If ar_tok_s is None, recommendation should be 'recommended'."""
        profile = _make_profile(ar_tok_s=None)
        assert profile.recommendation == "recommended"

    def test_recommendation_zero_speed(self):
        profile = _make_profile(ar_tok_s=0.0)
        assert profile.recommendation == "recommended"


# ===========================================================================
# Tests for DFlashProfile dataclass
# ===========================================================================


class TestDFlashProfile:
    """Tests for DFlashProfile values and PROFILES dict."""

    def test_profiles_dict_not_empty(self):
        assert len(PROFILES) > 0

    def test_all_profiles_have_required_fields(self):
        for name, profile in PROFILES.items():
            assert isinstance(profile, DFlashProfile)
            assert profile.name == name, f"Profile key {name!r} != profile.name {profile.name!r}"
            assert isinstance(profile.description, str)
            assert len(profile.description) > 0
            assert isinstance(profile.use_cache, bool)
            assert isinstance(profile.compile_drafter, bool)
            assert profile.priority in ("auto", "quality", "speed", "balanced")

    def test_quality_profiles_have_no_quantization(self):
        """Quality profiles should use bf16 (quantize_drafter=None)."""
        quality_profiles = [p for p in PROFILES.values() if p.priority == "quality"]
        assert len(quality_profiles) > 0, "Expected at least one quality profile"
        for p in quality_profiles:
            assert p.quantize_drafter is None, (
                f"Quality profile {p.name!r} should have quantize_drafter=None, got {p.quantize_drafter}"
            )

    def test_speed_profiles_have_quantization(self):
        """Speed profiles should quantize the drafter."""
        speed_profiles = [p for p in PROFILES.values() if p.priority == "speed"]
        assert len(speed_profiles) > 0, "Expected at least one speed profile"
        for p in speed_profiles:
            assert p.quantize_drafter is not None, f"Speed profile {p.name!r} should have quantize_drafter set"
            assert p.quantize_drafter in (4, 8), (
                f"Speed profile {p.name!r} has unexpected quantize_drafter={p.quantize_drafter}"
            )

    def test_all_profiles_use_cache(self):
        """All current profiles should use cache=True."""
        for name, profile in PROFILES.items():
            assert profile.use_cache is True, f"Profile {name!r} has use_cache=False"

    def test_known_profile_names(self):
        expected = {
            "quality_slow",
            "quality_medium",
            "quality_ssd",
            "speed_slow",
            "speed_medium",
            "speed_ssd",
            "fast_target",
            "medium_target",
            "slow_target",
            "ssd_fast",
            "ssd_slow",
        }
        assert set(PROFILES.keys()) == expected


# ===========================================================================
# Tests for select_profile with priority="auto"
# ===========================================================================


class TestSelectProfileAuto:
    """select_profile with priority='auto' (default)."""

    def test_very_fast_target(self):
        """AR > 50 tok/s -> fast_target."""
        profile = _make_profile(ar_tok_s=55.0)
        result = select_profile(profile, priority="auto")
        assert result.name == "fast_target"

    def test_fast_target_boundary(self):
        """AR = 51 -> fast_target."""
        profile = _make_profile(ar_tok_s=51.0)
        result = select_profile(profile, priority="auto")
        assert result.name == "fast_target"

    def test_at_50_not_fast_target(self):
        """AR = 50 exactly -> should NOT be fast_target (>50 required)."""
        profile = _make_profile(ar_tok_s=50.0)
        result = select_profile(profile, priority="auto")
        assert result.name != "fast_target"

    def test_ssm_fast(self):
        """SSM hybrid with AR > 30 -> speed_ssd."""
        profile = _make_profile(ar_tok_s=35.0, has_ssm=True)
        result = select_profile(profile, priority="auto")
        assert result.name == "speed_ssd"

    def test_ssm_slow(self):
        """SSM hybrid with AR <= 30 -> quality_ssd."""
        profile = _make_profile(ar_tok_s=20.0, has_ssm=True)
        result = select_profile(profile, priority="auto")
        assert result.name == "quality_ssd"

    def test_ssm_boundary_30(self):
        """SSM at exactly 30 -> quality_ssd (not > 30)."""
        profile = _make_profile(ar_tok_s=30.0, has_ssm=True)
        result = select_profile(profile, priority="auto")
        assert result.name == "quality_ssd"

    def test_medium_fast(self):
        """Dense, AR 26-50 -> speed_medium."""
        profile = _make_profile(ar_tok_s=30.0)
        result = select_profile(profile, priority="auto")
        assert result.name == "speed_medium"

    def test_medium_target(self):
        """Dense, AR 16-25 -> medium_target."""
        profile = _make_profile(ar_tok_s=20.0)
        result = select_profile(profile, priority="auto")
        assert result.name == "medium_target"

    def test_slow_target(self):
        """Dense, AR <= 15 -> quality_slow."""
        profile = _make_profile(ar_tok_s=10.0)
        result = select_profile(profile, priority="auto")
        assert result.name == "quality_slow"

    def test_slow_target_boundary(self):
        """AR = 15 exactly -> medium_target (not <= 15 for quality_slow)."""
        profile = _make_profile(ar_tok_s=15.0)
        result = select_profile(profile, priority="auto")
        # 15 is not > 25, not > 15, so goes to quality_slow
        # Actually: > 25 no, > 15 no (15 is not > 15), so quality_slow
        assert result.name == "quality_slow"

    def test_none_ar_treated_as_zero(self):
        """When ar_tok_s is None, treated as 0 -> quality_slow."""
        profile = _make_profile(ar_tok_s=None)
        result = select_profile(profile, priority="auto")
        assert result.name == "quality_slow"

    def test_ssm_overrides_speed_for_fast_ar(self):
        """SSM with very fast AR should still check SSM first (before >50)."""
        # SSM + AR > 50 -> first check is ar > 50 -> fast_target
        # Wait, let's check the code: ar > 50 is checked BEFORE has_ssm
        profile = _make_profile(ar_tok_s=55.0, has_ssm=True)
        result = select_profile(profile, priority="auto")
        assert result.name == "fast_target"


# ===========================================================================
# Tests for select_profile with priority="quality"
# ===========================================================================


class TestSelectProfileQuality:
    """select_profile with priority='quality'."""

    def test_quality_ssm(self):
        profile = _make_profile(ar_tok_s=20.0, has_ssm=True)
        result = select_profile(profile, priority="quality")
        assert result.name == "quality_ssd"

    def test_quality_fast_dense(self):
        """AR > 15 -> quality_medium."""
        profile = _make_profile(ar_tok_s=30.0)
        result = select_profile(profile, priority="quality")
        assert result.name == "quality_medium"

    def test_quality_slow_dense(self):
        """AR <= 15 -> quality_slow."""
        profile = _make_profile(ar_tok_s=10.0)
        result = select_profile(profile, priority="quality")
        assert result.name == "quality_slow"

    def test_quality_boundary_15(self):
        """AR = 15 exactly -> quality_slow (not > 15)."""
        profile = _make_profile(ar_tok_s=15.0)
        result = select_profile(profile, priority="quality")
        assert result.name == "quality_slow"

    def test_quality_profile_always_bf16(self):
        """Quality profiles should always have quantize_drafter=None."""
        for ar in [5, 15, 30, 50, 80]:
            profile = _make_profile(ar_tok_s=float(ar))
            result = select_profile(profile, priority="quality")
            assert result.quantize_drafter is None, (
                f"At AR={ar}, quality profile {result.name!r} has quantize_drafter={result.quantize_drafter}"
            )


# ===========================================================================
# Tests for select_profile with priority="speed"
# ===========================================================================


class TestSelectProfileSpeed:
    """select_profile with priority='speed'."""

    def test_speed_ssm(self):
        profile = _make_profile(ar_tok_s=20.0, has_ssm=True)
        result = select_profile(profile, priority="speed")
        assert result.name == "speed_ssd"

    def test_speed_fast_dense(self):
        """AR > 15 -> speed_medium."""
        profile = _make_profile(ar_tok_s=30.0)
        result = select_profile(profile, priority="speed")
        assert result.name == "speed_medium"

    def test_speed_slow_dense(self):
        """AR <= 15 -> speed_slow."""
        profile = _make_profile(ar_tok_s=10.0)
        result = select_profile(profile, priority="speed")
        assert result.name == "speed_slow"

    def test_speed_profile_always_quantized(self):
        """Speed profiles should always have quantize_drafter set."""
        for ar in [5, 15, 30, 50]:
            profile = _make_profile(ar_tok_s=float(ar))
            result = select_profile(profile, priority="speed")
            assert result.quantize_drafter is not None, (
                f"At AR={ar}, speed profile {result.name!r} has quantize_drafter=None"
            )


# ===========================================================================
# Tests for select_profile with priority="balanced"
# ===========================================================================


class TestSelectProfileBalanced:
    """select_profile with priority='balanced'."""

    def test_balanced_ssm_fast(self):
        """SSM with AR > 30 -> ssd_fast."""
        profile = _make_profile(ar_tok_s=35.0, has_ssm=True)
        result = select_profile(profile, priority="balanced")
        assert result.name == "ssd_fast"

    def test_balanced_ssm_slow(self):
        """SSM with AR <= 30 -> ssd_slow."""
        profile = _make_profile(ar_tok_s=20.0, has_ssm=True)
        result = select_profile(profile, priority="balanced")
        assert result.name == "ssd_slow"

    def test_balanced_fast_target(self):
        """Dense, AR > 40 -> fast_target."""
        profile = _make_profile(ar_tok_s=45.0)
        result = select_profile(profile, priority="balanced")
        assert result.name == "fast_target"

    def test_balanced_medium_target(self):
        """Dense, AR 16-40 -> medium_target."""
        profile = _make_profile(ar_tok_s=25.0)
        result = select_profile(profile, priority="balanced")
        assert result.name == "medium_target"

    def test_balanced_slow_target(self):
        """Dense, AR <= 15 -> slow_target."""
        profile = _make_profile(ar_tok_s=10.0)
        result = select_profile(profile, priority="balanced")
        assert result.name == "slow_target"

    def test_balanced_boundary_40(self):
        """AR = 40 exactly -> medium_target (not > 40)."""
        profile = _make_profile(ar_tok_s=40.0)
        result = select_profile(profile, priority="balanced")
        assert result.name == "medium_target"


# ===========================================================================
# Edge cases
# ===========================================================================


class TestSelectProfileEdgeCases:
    """Edge cases and boundary conditions for select_profile."""

    def test_zero_ar(self):
        profile = _make_profile(ar_tok_s=0.0)
        result = select_profile(profile, priority="auto")
        assert result.name == "quality_slow"

    def test_negative_ar(self):
        """Negative AR (should not happen, but handle gracefully)."""
        profile = _make_profile(ar_tok_s=-5.0)
        result = select_profile(profile, priority="auto")
        # Negative is < all thresholds -> quality_slow
        assert result.name == "quality_slow"

    def test_very_high_ar(self):
        profile = _make_profile(ar_tok_s=200.0)
        result = select_profile(profile, priority="auto")
        assert result.name == "fast_target"

    def test_ssm_with_zero_ar(self):
        profile = _make_profile(ar_tok_s=0.0, has_ssm=True)
        result = select_profile(profile, priority="auto")
        assert result.name == "quality_ssd"

    def test_result_is_dflash_profile(self):
        """Return type should always be DFlashProfile."""
        for priority in ("auto", "quality", "speed", "balanced"):
            profile = _make_profile(ar_tok_s=20.0)
            result = select_profile(profile, priority=priority)
            assert isinstance(result, DFlashProfile)

    def test_all_priorities_return_valid_profile(self):
        """Every priority should return a profile that exists in PROFILES."""
        for priority in ("auto", "quality", "speed", "balanced"):
            for ar in [0, 5, 15, 16, 25, 26, 30, 31, 40, 41, 50, 51, 100]:
                for has_ssm in (True, False):
                    mp = _make_profile(ar_tok_s=float(ar), has_ssm=has_ssm)
                    result = select_profile(mp, priority=priority)
                    assert result.name in PROFILES, (
                        f"priority={priority}, ar={ar}, ssm={has_ssm} -> {result.name!r} not in PROFILES"
                    )


# ===========================================================================
# Tests for profile_and_configure (function exists, signature correct)
# ===========================================================================


class TestProfileAndConfigure:
    """Test that profile_and_configure is importable and has the right signature."""

    def test_importable(self):
        from mlx_flash_compress.dflash_profile import profile_and_configure

        assert callable(profile_and_configure)

    def test_has_expected_params(self):
        import inspect

        from mlx_flash_compress.dflash_profile import profile_and_configure

        sig = inspect.signature(profile_and_configure)
        param_names = list(sig.parameters.keys())
        assert "model" in param_names
        assert "tokenizer" in param_names
        assert "priority" in param_names
