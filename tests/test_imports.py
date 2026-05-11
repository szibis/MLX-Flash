"""Tests for lazy imports in mlx_flash_compress.__init__.py.

Verifies that all public names exposed via __getattr__ resolve correctly,
that __version__ is set, and that __all__ is consistent if defined.
"""

import importlib
import types

import pytest

# All lazy import names and (source_module, expected_type_name) pairs
# derived from the __getattr__ blocks in __init__.py.
LAZY_IMPORTS = {
    # cache.py
    "ExpertCacheManager": ("mlx_flash_compress.cache", "ExpertCacheManager"),
    "CacheTier": ("mlx_flash_compress.cache", "CacheTier"),
    "CacheStats": ("mlx_flash_compress.cache", "CacheStats"),
    # engine.py
    "MoEInferenceEngine": ("mlx_flash_compress.engine", "MoEInferenceEngine"),
    "InferenceMode": ("mlx_flash_compress.engine", "InferenceMode"),
    # dflash.py
    "DFlashEngine": ("mlx_flash_compress.dflash", "DFlashEngine"),
    "DFlashConfig": ("mlx_flash_compress.dflash", "DFlashConfig"),
    # ddtree.py
    "DDTreeBuilder": ("mlx_flash_compress.ddtree", "DDTreeBuilder"),
    "DDTreeConfig": ("mlx_flash_compress.ddtree", "DDTreeConfig"),
    # dflash_model.py
    "DFlashDraftModel": ("mlx_flash_compress.dflash_model", "DFlashDraftModel"),
    "DFlashModelConfig": ("mlx_flash_compress.dflash_model", "DFlashModelConfig"),
    "DFlashRunner": ("mlx_flash_compress.dflash_model", "DFlashRunner"),
    # dflash_profile.py
    "profile_and_configure": ("mlx_flash_compress.dflash_profile", "profile_and_configure"),
    "detect_model": ("mlx_flash_compress.dflash_profile", "detect_model"),
    "select_profile": ("mlx_flash_compress.dflash_profile", "select_profile"),
    # expert_pruning.py
    "ExpertPruner": ("mlx_flash_compress.expert_pruning", "ExpertPruner"),
    "ExpertPruningConfig": ("mlx_flash_compress.expert_pruning", "ExpertPruningConfig"),
    "install_expert_pruning": ("mlx_flash_compress.expert_pruning", "install_expert_pruning"),
    # shared_expert_pinning.py
    "SharedExpertDetector": ("mlx_flash_compress.shared_expert_pinning", "SharedExpertDetector"),
    "SharedExpertPinner": ("mlx_flash_compress.shared_expert_pinning", "SharedExpertPinner"),
    "detect_and_pin_shared_experts": ("mlx_flash_compress.shared_expert_pinning", "detect_and_pin_shared_experts"),
    # streaming_llm.py
    "StreamingLLMCache": ("mlx_flash_compress.streaming_llm", "StreamingLLMCache"),
    "StreamingLLMConfig": ("mlx_flash_compress.streaming_llm", "StreamingLLMConfig"),
    "apply_streaming_llm": ("mlx_flash_compress.streaming_llm", "apply_streaming_llm"),
    # quantized_kv_cache.py
    "QuantizedKVCacheManager": ("mlx_flash_compress.quantized_kv_cache", "QuantizedKVCacheManager"),
    "QuantizedKVConfig": ("mlx_flash_compress.quantized_kv_cache", "QuantizedKVConfig"),
    # kv_compression.py
    "CompressedKVCache": ("mlx_flash_compress.kv_compression", "CompressedKVCache"),
    "KVCompressionConfig": ("mlx_flash_compress.kv_compression", "KVCompressionConfig"),
    "apply_kv_compression": ("mlx_flash_compress.kv_compression", "apply_kv_compression"),
    # layerskip.py
    "LayerSkipEngine": ("mlx_flash_compress.layerskip", "LayerSkipEngine"),
    "LayerSkipConfig": ("mlx_flash_compress.layerskip", "LayerSkipConfig"),
    "apply_layerskip": ("mlx_flash_compress.layerskip", "apply_layerskip"),
    # eagle3.py
    "EAGLE3Engine": ("mlx_flash_compress.eagle3", "EAGLE3Engine"),
    "EAGLE3Config": ("mlx_flash_compress.eagle3", "EAGLE3Config"),
    "EAGLEDraftHead": ("mlx_flash_compress.eagle3", "EAGLEDraftHead"),
    "EAGLE3Trainer": ("mlx_flash_compress.eagle3", "EAGLE3Trainer"),
    "apply_eagle3": ("mlx_flash_compress.eagle3", "apply_eagle3"),
    # layer_quantization.py
    "LayerQuantizer": ("mlx_flash_compress.layer_quantization", "LayerQuantizer"),
    "LayerQuantConfig": ("mlx_flash_compress.layer_quantization", "LayerQuantConfig"),
    "apply_layer_quantization": ("mlx_flash_compress.layer_quantization", "apply_layer_quantization"),
    # sequoia.py
    "SequoiaEngine": ("mlx_flash_compress.sequoia", "SequoiaEngine"),
    "SequoiaConfig": ("mlx_flash_compress.sequoia", "SequoiaConfig"),
    "apply_sequoia": ("mlx_flash_compress.sequoia", "apply_sequoia"),
    # matformer.py
    "AdaptiveMatFormer": ("mlx_flash_compress.matformer", "AdaptiveMatFormer"),
    "MatFormerConfig": ("mlx_flash_compress.matformer", "MatFormerConfig"),
    "apply_matformer": ("mlx_flash_compress.matformer", "apply_matformer"),
    # continuous_batching.py
    "ContinuousBatchingEngine": ("mlx_flash_compress.continuous_batching", "ContinuousBatchingEngine"),
    "BatchSchedulerConfig": ("mlx_flash_compress.continuous_batching", "BatchSchedulerConfig"),
    "create_batching_server": ("mlx_flash_compress.continuous_batching", "create_batching_server"),
    # kv_cache_backend.py
    "KVCacheBackend": ("mlx_flash_compress.kv_cache_backend", "KVCacheBackend"),
    "PlainKVCache": ("mlx_flash_compress.kv_cache_backend", "PlainKVCache"),
    "StreamingKVCache": ("mlx_flash_compress.kv_cache_backend", "StreamingKVCache"),
    "QuantizedKVCache": ("mlx_flash_compress.kv_cache_backend", "QuantizedKVCache"),
    "HybridKVCache": ("mlx_flash_compress.kv_cache_backend", "HybridKVCache"),
    "create_kv_cache": ("mlx_flash_compress.kv_cache_backend", "create_kv_cache"),
    "install_kv_cache": ("mlx_flash_compress.kv_cache_backend", "install_kv_cache"),
}


class TestVersion:
    """Tests for __version__ attribute."""

    def test_version_exists(self):
        import mlx_flash_compress

        assert hasattr(mlx_flash_compress, "__version__")

    def test_version_is_string(self):
        import mlx_flash_compress

        assert isinstance(mlx_flash_compress.__version__, str)

    def test_version_is_semver_like(self):
        import mlx_flash_compress

        parts = mlx_flash_compress.__version__.split(".")
        assert len(parts) >= 2, f"Version should have at least major.minor, got {mlx_flash_compress.__version__!r}"
        # Major and minor should be numeric
        assert parts[0].isdigit(), f"Major version not numeric: {parts[0]!r}"
        assert parts[1].isdigit(), f"Minor version not numeric: {parts[1]!r}"


class TestAllAttribute:
    """Tests for __all__ if it exists."""

    def test_all_is_list_or_absent(self):
        import mlx_flash_compress

        if hasattr(mlx_flash_compress, "__all__"):
            assert isinstance(mlx_flash_compress.__all__, (list, tuple))

    def test_all_entries_are_importable(self):
        """If __all__ is defined, every name in it should be importable."""
        import mlx_flash_compress

        if not hasattr(mlx_flash_compress, "__all__"):
            pytest.skip("__all__ not defined")
        for name in mlx_flash_compress.__all__:
            try:
                getattr(mlx_flash_compress, name)
            except (ImportError, AttributeError) as exc:
                pytest.fail(f"__all__ contains {name!r} which cannot be imported: {exc}")


class TestUnknownAttribute:
    """Test that accessing unknown names raises AttributeError."""

    def test_nonexistent_attribute_raises(self):
        import mlx_flash_compress

        with pytest.raises(AttributeError, match="no attribute"):
            _ = mlx_flash_compress.ThisDoesNotExist

    def test_nonexistent_with_similar_name(self):
        import mlx_flash_compress

        with pytest.raises(AttributeError):
            _ = mlx_flash_compress.ExpertPrunerXYZ  # typo, not a real name


class TestLazyImports:
    """Test that each lazy import resolves to the correct object."""

    @pytest.mark.parametrize("name,source", list(LAZY_IMPORTS.items()))
    def test_lazy_import_resolves(self, name, source):
        """Verify that importing `name` from mlx_flash_compress succeeds
        and the object comes from the expected source module."""
        source_module_path, expected_type_name = source
        try:
            import mlx_flash_compress

            obj = getattr(mlx_flash_compress, name)
        except ImportError:
            pytest.skip(f"Import of {name} failed (likely missing mlx dependency)")
            return

        # Object should not be None
        assert obj is not None, f"{name} resolved to None"

        # Verify it has the right name
        actual_name = getattr(obj, "__name__", None) or type(obj).__name__
        assert actual_name == expected_type_name, f"{name} resolved to {actual_name!r}, expected {expected_type_name!r}"

    @pytest.mark.parametrize("name,source", list(LAZY_IMPORTS.items()))
    def test_lazy_import_matches_direct_import(self, name, source):
        """Verify the lazy-imported object is the same as direct import."""
        source_module_path, expected_type_name = source
        try:
            import mlx_flash_compress

            lazy_obj = getattr(mlx_flash_compress, name)
        except ImportError:
            pytest.skip(f"Lazy import of {name} failed")
            return

        try:
            source_mod = importlib.import_module(source_module_path)
            direct_obj = getattr(source_mod, expected_type_name)
        except ImportError:
            pytest.skip(f"Direct import from {source_module_path} failed")
            return

        assert lazy_obj is direct_obj, (
            f"Lazy import of {name} returned different object than "
            f"direct import from {source_module_path}.{expected_type_name}"
        )

    @pytest.mark.parametrize("name,source", list(LAZY_IMPORTS.items()))
    def test_lazy_import_is_callable_or_class(self, name, source):
        """Every public name should be either a class (type) or a callable (function)."""
        try:
            import mlx_flash_compress

            obj = getattr(mlx_flash_compress, name)
        except ImportError:
            pytest.skip(f"Import of {name} failed")
            return

        assert callable(obj) or isinstance(obj, type), f"{name} should be callable or a type, got {type(obj).__name__}"


class TestLazyImportCoverage:
    """Verify that the test covers all lazy imports in __init__.py."""

    def test_all_getattr_names_tested(self):
        """Parse __init__.py to extract all names from if name == '...' blocks
        and verify each is in LAZY_IMPORTS."""
        import inspect
        import re

        import mlx_flash_compress

        source = inspect.getsource(mlx_flash_compress)
        # Match patterns like: if name == "SomeName":
        pattern = re.compile(r'if\s+name\s*==\s*"(\w+)"')
        getattr_names = set(pattern.findall(source))

        tested_names = set(LAZY_IMPORTS.keys())
        untested = getattr_names - tested_names
        assert not untested, f"These lazy-import names in __init__.py are not tested: {sorted(untested)}"

    def test_no_extra_test_names(self):
        """LAZY_IMPORTS should not contain names that don't exist in __init__.py."""
        import inspect
        import re

        import mlx_flash_compress

        source = inspect.getsource(mlx_flash_compress)
        pattern = re.compile(r'if\s+name\s*==\s*"(\w+)"')
        getattr_names = set(pattern.findall(source))

        tested_names = set(LAZY_IMPORTS.keys())
        extra = tested_names - getattr_names
        assert not extra, f"These names are in LAZY_IMPORTS but not in __init__.py: {sorted(extra)}"
