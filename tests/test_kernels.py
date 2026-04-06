"""Tests for Metal kernel compilation and loading."""

import sys

import pytest

from mlx_flash_compress.kernels.loader import (
    MetalKernelLoader,
    get_kernel_loader,
    _has_metal_compiler,
    KERNEL_DIR,
)


class TestMetalKernelLoader:
    def test_init(self):
        loader = MetalKernelLoader()
        assert loader.compiled == {}

    def test_available_on_macos(self):
        loader = MetalKernelLoader()
        if sys.platform == "darwin":
            # May or may not be available depending on Xcode
            assert isinstance(loader.available, bool)
        else:
            assert loader.available is False

    def test_compile_nonexistent(self):
        loader = MetalKernelLoader()
        result = loader.compile_shader("nonexistent_shader_xyz")
        assert result is None

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
    def test_compile_flash_dequant(self):
        loader = MetalKernelLoader()
        if not loader.available:
            pytest.skip("Metal compiler not available")
        result = loader.compile_shader("flash_dequant")
        assert result is not None
        assert result.exists()
        assert result.suffix == ".metallib"

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
    def test_compile_all(self):
        loader = MetalKernelLoader()
        if not loader.available:
            pytest.skip("Metal compiler not available")
        results = loader.compile_all()
        assert isinstance(results, dict)
        assert "flash_dequant" in results

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
    def test_cached_compilation(self):
        loader = MetalKernelLoader()
        if not loader.available:
            pytest.skip("Metal compiler not available")
        # Compile twice — second should use cache
        loader.compile_shader("flash_dequant")
        loader.compile_shader("flash_dequant")
        assert "flash_dequant" in loader.compiled

    def test_get_compiled_path_triggers_compile(self):
        loader = MetalKernelLoader()
        # For nonexistent shader, returns None
        result = loader.get_compiled_path("nonexistent")
        assert result is None


class TestGetKernelLoader:
    def test_singleton(self):
        loader1 = get_kernel_loader()
        loader2 = get_kernel_loader()
        assert loader1 is loader2

    def test_is_loader_instance(self):
        loader = get_kernel_loader()
        assert isinstance(loader, MetalKernelLoader)


class TestHasMetalCompiler:
    def test_returns_bool(self):
        result = _has_metal_compiler()
        assert isinstance(result, bool)


class TestKernelDir:
    def test_kernel_dir_exists(self):
        assert KERNEL_DIR.exists()

    def test_metal_files_exist(self):
        metal_files = list(KERNEL_DIR.glob("*.metal"))
        assert len(metal_files) >= 1
        names = [f.stem for f in metal_files]
        assert "flash_dequant" in names
