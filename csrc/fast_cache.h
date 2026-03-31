/*
 * fast_cache.h — GCD-accelerated expert cache for MoE inference on Apple Silicon
 *
 * This replaces Python's ThreadPoolExecutor (~50us per dispatch) with
 * GCD dispatch_async (<1us per dispatch) for the hot path:
 *   - Expert lookup in hash map
 *   - LZ4 decompression from RAM cache
 *   - Async SSD prefetch via dispatch_io
 *   - Parallel expert reads via dispatch_group
 *
 * All functions are C with Objective-C for GCD/dispatch APIs.
 * Exposed to Python via ctypes.
 */

#ifndef FAST_CACHE_H
#define FAST_CACHE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Cache entry metadata */
typedef struct {
    int32_t  layer_idx;
    int32_t  expert_id;
    uint64_t frequency;
    uint64_t last_step;
    uint64_t size_bytes;
    uint8_t *data;
} CacheEntry;

/* Cache statistics */
typedef struct {
    uint64_t cache_hits;
    uint64_t prefetch_hits;
    uint64_t cold_loads;
    uint64_t skip_fallbacks;
    uint64_t total_requests;
    uint64_t evictions;
    double   total_lookup_us;   /* microseconds in lookups */
    double   total_read_us;     /* microseconds in SSD reads */
    double   total_decomp_us;   /* microseconds in decompression */
} CacheStats;

/* Opaque cache handle */
typedef struct FastCache FastCache;

/* Create a new cache instance.
 * expert_dir: path to directory containing layer_NNN/expert_NNNN.bin files
 * capacity_bytes: maximum RAM for cached expert data
 * lcp_base: LCP decay base (0.25 recommended)
 * lcp_decay: LCP decay constant (128 recommended)
 */
FastCache* fc_create(
    const char *expert_dir,
    uint64_t capacity_bytes,
    double lcp_base,
    int32_t lcp_decay,
    int32_t num_workers
);

/* Destroy cache and free all resources. */
void fc_destroy(FastCache *cache);

/* Advance the step counter (call once per generated token). */
void fc_advance_step(FastCache *cache);

/* Fetch a single expert. Returns pointer to cached/loaded data.
 * out_size: receives the size of returned data in bytes.
 * out_source: receives 0=cache, 1=prefetch, 2=cold, 3=skip
 * Returns NULL if skip_fallback is used and expert not cached.
 */
const uint8_t* fc_fetch_one(
    FastCache *cache,
    int32_t layer_idx,
    int32_t expert_id,
    uint64_t *out_size,
    int32_t *out_source
);

/* Fetch K experts in parallel using GCD dispatch_group.
 * expert_ids: array of K expert IDs
 * k: number of experts
 * out_sizes: array of K uint64_t receiving sizes
 * out_sources: array of K int32_t receiving sources
 * out_ptrs: array of K pointers receiving data pointers
 */
void fc_fetch_parallel(
    FastCache *cache,
    int32_t layer_idx,
    const int32_t *expert_ids,
    int32_t k,
    uint64_t *out_sizes,
    int32_t *out_sources,
    const uint8_t **out_ptrs
);

/* Kick off async prefetch for predicted experts. Non-blocking. */
void fc_prefetch(
    FastCache *cache,
    int32_t layer_idx,
    const int32_t *expert_ids,
    int32_t k
);

/* Get cache statistics. */
CacheStats fc_get_stats(const FastCache *cache);

/* Reset statistics counters. */
void fc_reset_stats(FastCache *cache);

/* Benchmark: measure dispatch overhead for N lookups. Returns avg microseconds. */
double fc_benchmark_dispatch(FastCache *cache, int32_t iterations);

#ifdef __cplusplus
}
#endif

#endif /* FAST_CACHE_H */
