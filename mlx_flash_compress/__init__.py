"""MLX-Flash: Tiered compressed expert cache for MoE inference on Apple Silicon."""

__version__ = "0.1.0"

from mlx_flash_compress.cache import ExpertCacheManager, CacheTier, CacheStats
from mlx_flash_compress.engine import MoEInferenceEngine, InferenceMode
