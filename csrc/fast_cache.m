/*
 * fast_cache.m — GCD-accelerated expert cache implementation
 *
 * Uses Objective-C for GCD (Grand Central Dispatch) APIs:
 *   - dispatch_async: <1us per dispatch (vs Python's ~50us)
 *   - dispatch_group: parallel expert reads with sync barrier
 *   - dispatch_io: async SSD reads (non-blocking)
 *
 * Compiled with: clang -O2 -framework Foundation -shared -o libfastcache.dylib fast_cache.m
 */

#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>
#import <os/lock.h>
#import <sys/stat.h>
#import <fcntl.h>
#import <unistd.h>
#import <mach/mach_time.h>
#import <string.h>
#import <stdlib.h>

#include "fast_cache.h"

/* ── Hash map for O(1) cache lookup ────────────────────────── */

#define HASH_BUCKETS 16384
#define MAX_KEY(l, e) (((uint64_t)(l) << 32) | (uint32_t)(e))

typedef struct HashNode {
    uint64_t key;
    CacheEntry entry;
    struct HashNode *next;
} HashNode;

typedef struct {
    HashNode *buckets[HASH_BUCKETS];
    os_unfair_lock lock;
} HashMap;

static inline uint32_t hash_key(uint64_t key) {
    key = (key ^ (key >> 16)) * 0x45d9f3b;
    key = (key ^ (key >> 16)) * 0x45d9f3b;
    key = key ^ (key >> 16);
    return (uint32_t)(key & (HASH_BUCKETS - 1));
}

/* ── Cache structure ───────────────────────────────────────── */

struct FastCache {
    char expert_dir[1024];
    uint64_t capacity;
    uint64_t current_bytes;
    uint64_t step;

    double lcp_base;
    int32_t lcp_decay;

    HashMap map;
    CacheStats stats;

    dispatch_queue_t io_queue;        /* concurrent queue for SSD reads */
    dispatch_queue_t cache_queue;     /* serial queue for cache mutations */
    os_unfair_lock stats_lock;

    mach_timebase_info_data_t timebase;
};

/* ── Time helpers ──────────────────────────────────────────── */

static inline double ticks_to_us(uint64_t ticks, mach_timebase_info_data_t tb) {
    return (double)(ticks * tb.numer) / (double)(tb.denom * 1000);
}

/* ── HashMap operations ────────────────────────────────────── */

static CacheEntry* map_get(HashMap *m, int32_t layer, int32_t expert) {
    uint64_t key = MAX_KEY(layer, expert);
    uint32_t idx = hash_key(key);

    os_unfair_lock_lock(&m->lock);
    HashNode *node = m->buckets[idx];
    while (node) {
        if (node->key == key) {
            os_unfair_lock_unlock(&m->lock);
            return &node->entry;
        }
        node = node->next;
    }
    os_unfair_lock_unlock(&m->lock);
    return NULL;
}

static CacheEntry* map_insert(HashMap *m, int32_t layer, int32_t expert) {
    uint64_t key = MAX_KEY(layer, expert);
    uint32_t idx = hash_key(key);

    os_unfair_lock_lock(&m->lock);
    /* Check if exists */
    HashNode *node = m->buckets[idx];
    while (node) {
        if (node->key == key) {
            os_unfair_lock_unlock(&m->lock);
            return &node->entry;
        }
        node = node->next;
    }

    /* Insert new */
    HashNode *new_node = (HashNode *)calloc(1, sizeof(HashNode));
    new_node->key = key;
    new_node->entry.layer_idx = layer;
    new_node->entry.expert_id = expert;
    new_node->next = m->buckets[idx];
    m->buckets[idx] = new_node;
    os_unfair_lock_unlock(&m->lock);
    return &new_node->entry;
}

static void map_remove(HashMap *m, int32_t layer, int32_t expert) {
    uint64_t key = MAX_KEY(layer, expert);
    uint32_t idx = hash_key(key);

    os_unfair_lock_lock(&m->lock);
    HashNode **pp = &m->buckets[idx];
    while (*pp) {
        if ((*pp)->key == key) {
            HashNode *doomed = *pp;
            *pp = doomed->next;
            if (doomed->entry.data) free(doomed->entry.data);
            free(doomed);
            os_unfair_lock_unlock(&m->lock);
            return;
        }
        pp = &(*pp)->next;
    }
    os_unfair_lock_unlock(&m->lock);
}

static void map_destroy(HashMap *m) {
    for (int i = 0; i < HASH_BUCKETS; i++) {
        HashNode *node = m->buckets[i];
        while (node) {
            HashNode *next = node->next;
            if (node->entry.data) free(node->entry.data);
            free(node);
            node = next;
        }
        m->buckets[i] = NULL;
    }
}

/* ── File I/O ──────────────────────────────────────────────── */

static uint8_t* read_expert_file(const char *dir, int32_t layer, int32_t expert,
                                  uint64_t *out_size) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/layer_%03d/expert_%04d.bin", dir, layer, expert);

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        *out_size = 0;
        return NULL;
    }

    struct stat st;
    fstat(fd, &st);
    uint64_t size = (uint64_t)st.st_size;

    uint8_t *buf = (uint8_t *)malloc(size);
    if (!buf) {
        close(fd);
        *out_size = 0;
        return NULL;
    }

    ssize_t total = 0;
    while (total < (ssize_t)size) {
        ssize_t n = pread(fd, buf + total, size - total, total);
        if (n <= 0) break;
        total += n;
    }
    close(fd);

    *out_size = (uint64_t)total;
    return buf;
}

/* ── LCP priority ──────────────────────────────────────────── */

static double lcp_priority(FastCache *c, CacheEntry *e) {
    int64_t steps_since = (int64_t)c->step - (int64_t)e->last_step;
    if (steps_since < 0) steps_since = 0;
    return (double)e->frequency * pow(c->lcp_base, (double)steps_since / (double)c->lcp_decay);
}

/* ── Eviction ──────────────────────────────────────────────── */

static void evict_lowest(FastCache *c, uint64_t needed) {
    while (c->current_bytes + needed > c->capacity) {
        /* Find lowest priority entry */
        double min_pri = 1e30;
        int32_t min_layer = -1, min_expert = -1;
        uint64_t min_size = 0;

        os_unfair_lock_lock(&c->map.lock);
        for (int i = 0; i < HASH_BUCKETS; i++) {
            HashNode *node = c->map.buckets[i];
            while (node) {
                if (node->entry.data) {
                    double p = lcp_priority(c, &node->entry);
                    if (p < min_pri) {
                        min_pri = p;
                        min_layer = node->entry.layer_idx;
                        min_expert = node->entry.expert_id;
                        min_size = node->entry.size_bytes;
                    }
                }
                node = node->next;
            }
        }
        os_unfair_lock_unlock(&c->map.lock);

        if (min_layer < 0) break;

        map_remove(&c->map, min_layer, min_expert);
        c->current_bytes -= min_size;
        c->stats.evictions++;
    }
}

/* ── Public API ────────────────────────────────────────────── */

FastCache* fc_create(const char *expert_dir, uint64_t capacity_bytes,
                     double lcp_base, int32_t lcp_decay, int32_t num_workers) {
    FastCache *c = (FastCache *)calloc(1, sizeof(FastCache));
    strncpy(c->expert_dir, expert_dir, sizeof(c->expert_dir) - 1);
    c->capacity = capacity_bytes;
    c->lcp_base = lcp_base;
    c->lcp_decay = lcp_decay;
    c->step = 0;
    c->current_bytes = 0;

    memset(&c->map, 0, sizeof(c->map));
    c->map.lock = OS_UNFAIR_LOCK_INIT;
    c->stats_lock = OS_UNFAIR_LOCK_INIT;
    memset(&c->stats, 0, sizeof(c->stats));

    mach_timebase_info(&c->timebase);

    /* Create GCD queues */
    c->io_queue = dispatch_queue_create("com.mlx-flash.io",
        dispatch_queue_attr_make_with_qos_class(
            DISPATCH_QUEUE_CONCURRENT, QOS_CLASS_USER_INITIATED, 0));
    c->cache_queue = dispatch_queue_create("com.mlx-flash.cache",
        DISPATCH_QUEUE_SERIAL);

    return c;
}

void fc_destroy(FastCache *c) {
    if (!c) return;
    map_destroy(&c->map);
    if (c->io_queue) dispatch_release(c->io_queue);
    if (c->cache_queue) dispatch_release(c->cache_queue);
    free(c);
}

void fc_advance_step(FastCache *c) {
    __atomic_add_fetch(&c->step, 1, __ATOMIC_RELAXED);
}

const uint8_t* fc_fetch_one(FastCache *c, int32_t layer, int32_t expert,
                            uint64_t *out_size, int32_t *out_source) {
    uint64_t t0 = mach_absolute_time();

    os_unfair_lock_lock(&c->stats_lock);
    c->stats.total_requests++;
    os_unfair_lock_unlock(&c->stats_lock);

    /* Fast path: cache lookup */
    CacheEntry *entry = map_get(&c->map, layer, expert);
    if (entry && entry->data) {
        entry->frequency++;
        entry->last_step = c->step;
        *out_size = entry->size_bytes;
        *out_source = 0; /* cache hit */

        uint64_t t1 = mach_absolute_time();
        os_unfair_lock_lock(&c->stats_lock);
        c->stats.cache_hits++;
        c->stats.total_lookup_us += ticks_to_us(t1 - t0, c->timebase);
        os_unfair_lock_unlock(&c->stats_lock);

        return entry->data;
    }

    /* Cold path: read from SSD */
    uint64_t t_read0 = mach_absolute_time();
    uint64_t size = 0;
    uint8_t *data = read_expert_file(c->expert_dir, layer, expert, &size);
    uint64_t t_read1 = mach_absolute_time();

    if (!data) {
        *out_size = 0;
        *out_source = 3; /* skip */
        os_unfair_lock_lock(&c->stats_lock);
        c->stats.skip_fallbacks++;
        os_unfair_lock_unlock(&c->stats_lock);
        return NULL;
    }

    /* Insert into cache */
    evict_lowest(c, size);
    CacheEntry *new_entry = map_insert(&c->map, layer, expert);
    new_entry->data = data;
    new_entry->size_bytes = size;
    new_entry->frequency = 1;
    new_entry->last_step = c->step;
    c->current_bytes += size;

    *out_size = size;
    *out_source = 2; /* cold load */

    os_unfair_lock_lock(&c->stats_lock);
    c->stats.cold_loads++;
    c->stats.total_read_us += ticks_to_us(t_read1 - t_read0, c->timebase);
    os_unfair_lock_unlock(&c->stats_lock);

    return data;
}

void fc_fetch_parallel(FastCache *c, int32_t layer, const int32_t *expert_ids,
                       int32_t k, uint64_t *out_sizes, int32_t *out_sources,
                       const uint8_t **out_ptrs) {
    dispatch_group_t group = dispatch_group_create();

    for (int i = 0; i < k; i++) {
        const int idx = i;
        const int32_t eid = expert_ids[i];

        /* Check cache first (fast path, no dispatch needed) */
        CacheEntry *entry = map_get(&c->map, layer, eid);
        if (entry && entry->data) {
            entry->frequency++;
            entry->last_step = c->step;
            out_sizes[idx] = entry->size_bytes;
            out_sources[idx] = 0;
            out_ptrs[idx] = entry->data;

            os_unfair_lock_lock(&c->stats_lock);
            c->stats.cache_hits++;
            c->stats.total_requests++;
            os_unfair_lock_unlock(&c->stats_lock);
            continue;
        }

        /* Cold path: dispatch to GCD for parallel SSD read */
        dispatch_group_async(group, c->io_queue, ^{
            uint64_t size = 0;
            uint8_t *data = read_expert_file(c->expert_dir, layer, eid, &size);
            if (data) {
                /* Insert into cache on serial queue */
                dispatch_sync(c->cache_queue, ^{
                    evict_lowest(c, size);
                    CacheEntry *e = map_insert(&c->map, layer, eid);
                    e->data = data;
                    e->size_bytes = size;
                    e->frequency = 1;
                    e->last_step = c->step;
                    c->current_bytes += size;
                });
                out_sizes[idx] = size;
                out_sources[idx] = 2;
                out_ptrs[idx] = data;
            } else {
                out_sizes[idx] = 0;
                out_sources[idx] = 3;
                out_ptrs[idx] = NULL;
            }

            os_unfair_lock_lock(&c->stats_lock);
            c->stats.cold_loads++;
            c->stats.total_requests++;
            os_unfair_lock_unlock(&c->stats_lock);
        });
    }

    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    dispatch_release(group);
}

void fc_prefetch(FastCache *c, int32_t layer, const int32_t *expert_ids, int32_t k) {
    for (int i = 0; i < k; i++) {
        const int32_t eid = expert_ids[i];

        /* Skip if already cached */
        if (map_get(&c->map, layer, eid)) continue;

        /* Fire-and-forget async read */
        dispatch_async(c->io_queue, ^{
            uint64_t size = 0;
            uint8_t *data = read_expert_file(c->expert_dir, layer, eid, &size);
            if (data) {
                dispatch_sync(c->cache_queue, ^{
                    /* Double-check not already inserted by another path */
                    CacheEntry *existing = map_get(&c->map, layer, eid);
                    if (existing && existing->data) {
                        free(data);
                        return;
                    }
                    evict_lowest(c, size);
                    CacheEntry *e = map_insert(&c->map, layer, eid);
                    e->data = data;
                    e->size_bytes = size;
                    e->frequency = 1;
                    e->last_step = c->step;
                    c->current_bytes += size;
                });
            }
        });
    }
}

CacheStats fc_get_stats(const FastCache *c) {
    return c->stats;
}

void fc_reset_stats(FastCache *c) {
    os_unfair_lock_lock(&c->stats_lock);
    memset(&c->stats, 0, sizeof(c->stats));
    os_unfair_lock_unlock(&c->stats_lock);
}

double fc_benchmark_dispatch(FastCache *c, int32_t iterations) {
    uint64_t t0 = mach_absolute_time();

    for (int i = 0; i < iterations; i++) {
        dispatch_group_t g = dispatch_group_create();
        dispatch_group_async(g, c->io_queue, ^{
            /* no-op — measuring pure dispatch overhead */
        });
        dispatch_group_wait(g, DISPATCH_TIME_FOREVER);
        dispatch_release(g);
    }

    uint64_t t1 = mach_absolute_time();
    return ticks_to_us(t1 - t0, c->timebase) / (double)iterations;
}
