"""MLX-Flash: Tiered compressed expert cache for MoE inference on Apple Silicon."""

__version__ = "0.7.0"


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
    if name == "DFlashDraftModel":
        from mlx_flash_compress.dflash_model import DFlashDraftModel
        return DFlashDraftModel
    if name == "DFlashModelConfig":
        from mlx_flash_compress.dflash_model import DFlashModelConfig
        return DFlashModelConfig
    if name == "DFlashRunner":
        from mlx_flash_compress.dflash_model import DFlashRunner
        return DFlashRunner
    if name == "profile_and_configure":
        from mlx_flash_compress.dflash_profile import profile_and_configure
        return profile_and_configure
    if name == "detect_model":
        from mlx_flash_compress.dflash_profile import detect_model
        return detect_model
    if name == "select_profile":
        from mlx_flash_compress.dflash_profile import select_profile
        return select_profile
    if name == "ExpertPruner":
        from mlx_flash_compress.expert_pruning import ExpertPruner
        return ExpertPruner
    if name == "ExpertPruningConfig":
        from mlx_flash_compress.expert_pruning import ExpertPruningConfig
        return ExpertPruningConfig
    if name == "install_expert_pruning":
        from mlx_flash_compress.expert_pruning import install_expert_pruning
        return install_expert_pruning
    if name == "SharedExpertDetector":
        from mlx_flash_compress.shared_expert_pinning import SharedExpertDetector
        return SharedExpertDetector
    if name == "SharedExpertPinner":
        from mlx_flash_compress.shared_expert_pinning import SharedExpertPinner
        return SharedExpertPinner
    if name == "detect_and_pin_shared_experts":
        from mlx_flash_compress.shared_expert_pinning import detect_and_pin_shared_experts
        return detect_and_pin_shared_experts
    if name == "StreamingLLMCache":
        from mlx_flash_compress.streaming_llm import StreamingLLMCache
        return StreamingLLMCache
    if name == "StreamingLLMConfig":
        from mlx_flash_compress.streaming_llm import StreamingLLMConfig
        return StreamingLLMConfig
    if name == "apply_streaming_llm":
        from mlx_flash_compress.streaming_llm import apply_streaming_llm
        return apply_streaming_llm
    if name == "QuantizedKVCacheManager":
        from mlx_flash_compress.quantized_kv_cache import QuantizedKVCacheManager
        return QuantizedKVCacheManager
    if name == "QuantizedKVConfig":
        from mlx_flash_compress.quantized_kv_cache import QuantizedKVConfig
        return QuantizedKVConfig
    if name == "CompressedKVCache":
        from mlx_flash_compress.kv_compression import CompressedKVCache
        return CompressedKVCache
    if name == "KVCompressionConfig":
        from mlx_flash_compress.kv_compression import KVCompressionConfig
        return KVCompressionConfig
    if name == "apply_kv_compression":
        from mlx_flash_compress.kv_compression import apply_kv_compression
        return apply_kv_compression
    if name == "LayerSkipEngine":
        from mlx_flash_compress.layerskip import LayerSkipEngine
        return LayerSkipEngine
    if name == "LayerSkipConfig":
        from mlx_flash_compress.layerskip import LayerSkipConfig
        return LayerSkipConfig
    if name == "apply_layerskip":
        from mlx_flash_compress.layerskip import apply_layerskip
        return apply_layerskip
    if name == "EAGLE3Engine":
        from mlx_flash_compress.eagle3 import EAGLE3Engine
        return EAGLE3Engine
    if name == "EAGLE3Config":
        from mlx_flash_compress.eagle3 import EAGLE3Config
        return EAGLE3Config
    if name == "EAGLEDraftHead":
        from mlx_flash_compress.eagle3 import EAGLEDraftHead
        return EAGLEDraftHead
    if name == "EAGLE3Trainer":
        from mlx_flash_compress.eagle3 import EAGLE3Trainer
        return EAGLE3Trainer
    if name == "LayerQuantizer":
        from mlx_flash_compress.layer_quantization import LayerQuantizer
        return LayerQuantizer
    if name == "LayerQuantConfig":
        from mlx_flash_compress.layer_quantization import LayerQuantConfig
        return LayerQuantConfig
    if name == "apply_layer_quantization":
        from mlx_flash_compress.layer_quantization import apply_layer_quantization
        return apply_layer_quantization
    if name == "SequoiaEngine":
        from mlx_flash_compress.sequoia import SequoiaEngine
        return SequoiaEngine
    if name == "SequoiaConfig":
        from mlx_flash_compress.sequoia import SequoiaConfig
        return SequoiaConfig
    if name == "apply_sequoia":
        from mlx_flash_compress.sequoia import apply_sequoia
        return apply_sequoia
    if name == "AdaptiveMatFormer":
        from mlx_flash_compress.matformer import AdaptiveMatFormer
        return AdaptiveMatFormer
    if name == "MatFormerConfig":
        from mlx_flash_compress.matformer import MatFormerConfig
        return MatFormerConfig
    if name == "apply_matformer":
        from mlx_flash_compress.matformer import apply_matformer
        return apply_matformer
    if name == "ContinuousBatchingEngine":
        from mlx_flash_compress.continuous_batching import ContinuousBatchingEngine
        return ContinuousBatchingEngine
    if name == "BatchSchedulerConfig":
        from mlx_flash_compress.continuous_batching import BatchSchedulerConfig
        return BatchSchedulerConfig
    if name == "create_batching_server":
        from mlx_flash_compress.continuous_batching import create_batching_server
        return create_batching_server
    raise AttributeError(f"module 'mlx_flash_compress' has no attribute {name!r}")
