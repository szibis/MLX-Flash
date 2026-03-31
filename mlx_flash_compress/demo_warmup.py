"""Progressive warm-up demo — ISP-like caching for MoE inference.

Shows the warm-up behavior in real-time:
  1. Cold start: first tokens are slow (all experts from SSD)
  2. Warming: cache fills, hit rate climbs, speed increases
  3. Steady state: 85%+ cache hits, near-full speed
  4. Topic change: brief re-warming, then fast again
  5. Same topic: stays fast (experts already cached)

Like ISP prefetch caching — slow start, progressively faster.

Usage:
  python -m mlx_flash_compress.demo_warmup
  python -m mlx_flash_compress.demo_warmup --tokens 200 --ssd-latency 0.6
  python -m mlx_flash_compress.demo_warmup --experts 512 --layers 60  # simulate large model
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from mlx_flash_compress.lcp_cache import LCPCache
from mlx_flash_compress.hardware import detect_hardware


@dataclass
class TokenMetrics:
    """Metrics for a single generated token."""
    token_idx: int
    time_ms: float
    cache_hits: int
    cache_misses: int
    hit_rate: float
    cumulative_hit_rate: float
    topic: str = ""


@dataclass
class WarmupSession:
    """A session of tokens with warm-up tracking."""
    topic: str
    token_metrics: list[TokenMetrics] = field(default_factory=list)

    @property
    def avg_time_first_10(self) -> float:
        return np.mean([t.time_ms for t in self.token_metrics[:10]]) if len(self.token_metrics) >= 10 else 0

    @property
    def avg_time_last_10(self) -> float:
        return np.mean([t.time_ms for t in self.token_metrics[-10:]]) if len(self.token_metrics) >= 10 else 0

    @property
    def speedup(self) -> float:
        first = self.avg_time_first_10
        last = self.avg_time_last_10
        return first / last if last > 0 else 0


def create_expert_files(work_dir: str, num_layers: int, num_experts: int,
                        expert_size_bytes: int = 256 * 1024) -> Path:
    """Create synthetic expert weight files on disk."""
    expert_dir = Path(work_dir) / "warmup_experts"
    if expert_dir.exists():
        shutil.rmtree(expert_dir)

    rng = np.random.default_rng(42)
    for layer in range(num_layers):
        layer_dir = expert_dir / f"layer_{layer:03d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        for expert in range(num_experts):
            data = rng.integers(0, 256, size=expert_size_bytes, dtype=np.uint8).tobytes()
            (layer_dir / f"expert_{expert:04d}.bin").write_bytes(data)

    return expert_dir


def make_topic_routing(topic: str, num_experts: int, rng: np.random.Generator) -> np.ndarray:
    """Create a topic-specific expert routing distribution.

    Different topics activate different subsets of experts.
    This simulates real MoE behavior where coding activates different
    experts than writing or math.
    """
    # Use topic hash to seed a deterministic but topic-specific distribution
    topic_seed = hash(topic) % 2**31
    topic_rng = np.random.default_rng(topic_seed)

    # Select ~25% of experts as "hot" for this topic
    n_hot = max(4, num_experts // 4)
    hot_experts = topic_rng.choice(num_experts, size=n_hot, replace=False)

    probs = np.ones(num_experts) * 0.001  # cold experts get tiny probability
    for e in hot_experts:
        probs[e] = topic_rng.uniform(0.5, 2.0)  # hot experts get high probability
    probs /= probs.sum()
    return probs


def simulate_token(
    cache: LCPCache,
    layer_count: int,
    expert_probs: np.ndarray,
    num_experts: int,
    k: int,
    rng: np.random.Generator,
    ssd_latency_ms: float,
) -> tuple[int, int, float]:
    """Simulate one token of MoE inference through the cache.

    Returns (hits, misses, time_ms).
    """
    hits = 0
    misses = 0

    cache.advance_step()
    t0 = time.monotonic()

    prev_experts = None
    for layer in range(layer_count):
        experts = rng.choice(num_experts, size=k, replace=False, p=expert_probs).tolist()

        # Prefetch predicted experts for this layer
        if prev_experts is not None:
            predicted = cache.predict_next(layer - 1, prev_experts)
            if predicted:
                cache.prefetch(layer, predicted)

        # Fetch experts (cache hit = fast, miss = SSD read)
        results = cache.fetch(layer, experts, allow_skip=False)
        for data, source in results:
            if source in ("cache", "prefetch"):
                hits += 1
            else:
                misses += 1
                # Simulate SSD latency for cold loads
                if ssd_latency_ms > 0:
                    time.sleep(ssd_latency_ms / 1000)

        prev_experts = experts

    elapsed_ms = (time.monotonic() - t0) * 1000
    return hits, misses, elapsed_ms


def print_token_bar(idx: int, time_ms: float, hit_rate: float, cum_hit_rate: float,
                    topic: str, max_time_ms: float):
    """Print a single token's metrics as a visual bar."""
    # Speed bar (inverse of time)
    if max_time_ms > 0:
        speed_frac = 1.0 - min(time_ms / max_time_ms, 1.0)
    else:
        speed_frac = 1.0
    bar_len = int(speed_frac * 30)
    bar = "#" * bar_len + "." * (30 - bar_len)

    # Hit rate indicator
    if hit_rate >= 0.9:
        status = "FAST"
    elif hit_rate >= 0.5:
        status = "warm"
    else:
        status = "cold"

    sys.stdout.write(
        f"\r  [{idx:>4d}] {time_ms:>6.1f}ms  hit:{cum_hit_rate:>5.1%}  {bar}  {status}  [{topic}]"
    )
    sys.stdout.flush()


def run_warmup_session(
    cache: LCPCache,
    topic: str,
    num_tokens: int,
    num_layers: int,
    num_experts: int,
    k: int,
    ssd_latency_ms: float,
    rng: np.random.Generator,
    start_idx: int = 0,
    show_every: int = 1,
) -> WarmupSession:
    """Run a session of tokens for one topic, showing warm-up progress."""
    session = WarmupSession(topic=topic)
    probs = make_topic_routing(topic, num_experts, rng)

    cum_hits = 0
    cum_total = 0
    max_time = 0

    for i in range(num_tokens):
        hits, misses, time_ms = simulate_token(
            cache, num_layers, probs, num_experts, k, rng, ssd_latency_ms
        )
        cum_hits += hits
        cum_total += hits + misses
        cum_hit_rate = cum_hits / cum_total if cum_total > 0 else 0
        hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0
        max_time = max(max_time, time_ms)

        metric = TokenMetrics(
            token_idx=start_idx + i,
            time_ms=time_ms,
            cache_hits=hits,
            cache_misses=misses,
            hit_rate=hit_rate,
            cumulative_hit_rate=cum_hit_rate,
            topic=topic,
        )
        session.token_metrics.append(metric)

        if (i % show_every == 0) or i == num_tokens - 1:
            print_token_bar(start_idx + i, time_ms, hit_rate, cum_hit_rate, topic, max_time)

    print()  # newline after progress bar
    return session


def print_session_summary(sessions: list[WarmupSession]):
    """Print summary of all sessions showing warm-up effect."""
    print()
    print("=" * 72)
    print("  WARM-UP ANALYSIS")
    print("=" * 72)

    for s in sessions:
        if len(s.token_metrics) < 10:
            continue
        first = s.avg_time_first_10
        last = s.avg_time_last_10
        speedup = s.speedup
        final_hit = s.token_metrics[-1].cumulative_hit_rate

        bar_first = int(max(0, 30 - first / max(first, 0.1) * 30))
        bar_last = int(max(0, 30 - last / max(first, 0.1) * 30))

        print(f"\n  Topic: {s.topic}")
        print(f"    First 10 tokens:  {first:>6.1f}ms avg  {'.' * (30 - bar_first)}{'#' * bar_first}")
        print(f"    Last 10 tokens:   {last:>6.1f}ms avg  {'.' * (30 - bar_last)}{'#' * bar_last}")
        print(f"    Speedup: {speedup:.1f}x (from warm-up)")
        print(f"    Final hit rate: {final_hit:.1%}")

    # Cross-topic analysis
    if len(sessions) > 1:
        print("\n  -- TOPIC SWITCHING --")
        for i in range(1, len(sessions)):
            prev = sessions[i - 1]
            curr = sessions[i]
            if not curr.token_metrics:
                continue
            first_after_switch = curr.token_metrics[0].time_ms
            last_before_switch = prev.token_metrics[-1].time_ms if prev.token_metrics else 0
            print(f"\n    {prev.topic} -> {curr.topic}:")
            print(f"      Last token before switch: {last_before_switch:.1f}ms")
            print(f"      First token after switch: {first_after_switch:.1f}ms")
            if last_before_switch > 0:
                slowdown = first_after_switch / last_before_switch
                print(f"      Temporary slowdown: {slowdown:.1f}x (re-warming to new topic)")
            warmup_tokens = 0
            for m in curr.token_metrics:
                if m.cumulative_hit_rate >= 0.7:
                    warmup_tokens = m.token_idx - curr.token_metrics[0].token_idx
                    break
            if warmup_tokens > 0:
                print(f"      Tokens to re-warm (>70% hit): {warmup_tokens}")

    # The story
    print("\n  -- THE ISP ANALOGY --")
    print()
    print("  Like ISP prefetch caching:")
    print("    1. First request: full SSD read (cold)")
    print("    2. Hot experts cached in RAM (warming)")
    print("    3. Steady state: 85%+ from RAM (fast)")
    print("    4. Topic change: brief re-warm, then fast again")
    print("    5. Return to same topic: instant fast (still cached)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="MLX-Flash: Progressive Warm-up Demo"
    )
    parser.add_argument("--layers", type=int, default=24, help="Number of layers")
    parser.add_argument("--experts", type=int, default=60, help="Experts per layer")
    parser.add_argument("--k", type=int, default=4, help="Top-K routing")
    parser.add_argument("--tokens", type=int, default=100, help="Tokens per topic")
    parser.add_argument("--ssd-latency", type=float, default=0.6,
                        help="Simulated SSD read latency per expert (ms)")
    parser.add_argument("--cache-mb", type=int, default=512, help="Cache size in MB")
    parser.add_argument("--topics", nargs="+",
                        default=["coding", "writing", "coding", "math"],
                        help="Topic sequence to demonstrate")
    parser.add_argument("--expert-size-kb", type=int, default=256,
                        help="Expert weight size in KB")
    parser.add_argument("--work-dir", default="/tmp/mlx_warmup_demo")
    args = parser.parse_args()

    print()
    print("=" * 72)
    print("  MLX-Flash: Progressive Warm-up Demo")
    print("=" * 72)

    hw = detect_hardware()
    print(f"\n  Hardware: {hw.chip}, {hw.total_ram_gb:.0f}GB RAM")
    print(f"  Config: {args.layers} layers, {args.experts} experts, top-{args.k}")
    print(f"  Cache: {args.cache_mb}MB, SSD latency: {args.ssd_latency}ms/expert")
    print(f"  Topics: {' -> '.join(args.topics)}")

    # Create expert files
    print(f"\n  Creating {args.layers * args.experts} expert files...")
    expert_dir = create_expert_files(
        args.work_dir, args.layers, args.experts,
        args.expert_size_kb * 1024,
    )
    total_mb = args.layers * args.experts * args.expert_size_kb / 1024
    print(f"  Total expert data: {total_mb:.0f} MB on disk")
    print(f"  Cache can hold: {args.cache_mb / (args.expert_size_kb / 1024):.0f} experts "
          f"({args.cache_mb / total_mb * 100:.0f}% of total)")

    # Create cache
    cache = LCPCache(
        expert_dir=str(expert_dir),
        capacity_bytes=args.cache_mb * 1024 * 1024,
        enable_dendritic=False,
        enable_skip_fallback=False,
        simulated_ssd_latency_ms=0,  # we add latency in simulate_token
    )

    rng = np.random.default_rng(42)
    sessions = []
    token_offset = 0

    for topic in args.topics:
        print(f"\n  --- Topic: {topic} ({args.tokens} tokens) ---")
        session = run_warmup_session(
            cache=cache,
            topic=topic,
            num_tokens=args.tokens,
            num_layers=args.layers,
            num_experts=args.experts,
            k=args.k,
            ssd_latency_ms=args.ssd_latency,
            rng=rng,
            start_idx=token_offset,
            show_every=max(1, args.tokens // 20),
        )
        sessions.append(session)
        token_offset += args.tokens

    cache.shutdown()

    # Summary
    print_session_summary(sessions)

    # Cleanup
    if Path(args.work_dir).exists():
        shutil.rmtree(args.work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
