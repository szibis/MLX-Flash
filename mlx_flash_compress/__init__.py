"""MLX-Flash: Tiered compressed expert cache for MoE inference on Apple Silicon."""

__version__ = "0.6.2"


def __getattr__(name):
    """Lazy imports to avoid pulling in all dependencies on import."""
    if name == "ExpertCacheManager":
        from mlx_flash_compress.cache import ExpertCacheManager
        return ExpertCacheManager
    if name == "CacheTier":
        from mlx_flash_compress.cache import CacheTier
        return CacheTier
    if name == "CacheStats":
        from mlx_flash_compress.cache import CacheStats
        return CacheStats
    if name == "MoEInferenceEngine":
        from mlx_flash_compress.engine import MoEInferenceEngine
        return MoEInferenceEngine
    if name == "InferenceMode":
        from mlx_flash_compress.engine import InferenceMode
        return InferenceMode
    if name == "DFlashEngine":
        from mlx_flash_compress.dflash import DFlashEngine
        return DFlashEngine
    if name == "DFlashConfig":
        from mlx_flash_compress.dflash import DFlashConfig
        return DFlashConfig
    if name == "DDTreeBuilder":
        from mlx_flash_compress.ddtree import DDTreeBuilder
        return DDTreeBuilder
    if name == "DDTreeConfig":
        from mlx_flash_compress.ddtree import DDTreeConfig
        return DDTreeConfig
    raise AttributeError(f"module 'mlx_flash_compress' has no attribute {name!r}")
