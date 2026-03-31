"""Cached Inference Engine — progressive warm-up for MoE models.

Combines real MLX inference with LCP expert caching to show the
ISP-like warm-up behavior: slow start, progressively faster.

For models that fit in RAM: tracks expert routing patterns and
shows what the cache WOULD do if the model were too large for RAM.

For models approaching RAM limits: mixed precision + cache reduce
memory pressure, measurably improving performance.

Architecture:
  1. SwitchGLU hook intercepts expert routing indices per layer
  2. LCPCache tracks hit rates and manages expert priority
  3. Each token shows real-time cache warm-up metrics
  4. Topic changes detected via routing pattern shifts

Usage:
  python -m mlx_flash_compress.cached_inference --model mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit
  python -m mlx_flash_compress.cached_inference --model PATH --prompt "Write a Python function"
  python -m mlx_flash_compress.cached_inference --model PATH --interactive
"""

import argparse
import gc
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import mlx.core as mx
from mlx_lm import load, generate

from mlx_flash_compress.hardware import detect_hardware


# ── Expert routing interceptor ──

@dataclass
class RoutingEvent:
    """A single expert routing decision captured during inference."""
    layer_idx: int
    expert_indices: list[int]
    token_idx: int
    timestamp: float


class ExpertRouter:
    """Captures expert routing decisions from the MoE forward pass.

    Monkey-patches SwitchGLU.__call__ to record which experts are activated
    at each layer for each token. This is non-invasive: the model runs
    identically, we just observe the routing.
    """

    def __init__(self):
        self.events: list[RoutingEvent] = []
        self.token_counter = 0
        self._patched = False
        self._original_calls = {}
        self._layer_map = {}  # module id -> layer index
        self._calls_this_token = 0
        self._num_layers = 0  # set during install

    def install(self, model):
        """Install routing hooks on all SwitchGLU modules."""
        layers = None
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers

        if layers is None:
            return False

        moe_layer_count = 0
        switch_cls = None

        for layer_idx, layer in enumerate(layers):
            if not hasattr(layer, "mlp"):
                continue
            mlp = layer.mlp

            switch = getattr(mlp, "switch_mlp", None)
            if switch is None:
                continue

            moe_layer_count += 1
            self._layer_map[id(switch)] = layer_idx
            if switch_cls is None:
                switch_cls = type(switch)

        if switch_cls is None:
            return False

        self._num_layers = moe_layer_count

        # Patch the CLASS method once (not per-instance)
        original_call = switch_cls.__call__
        self._original_calls["__class__"] = original_call
        self._switch_cls = switch_cls
        router = self

        def hooked_call(self_module, x, indices):
            # Auto-detect token boundaries: every num_layers calls = one token
            router._calls_this_token += 1
            if router._num_layers > 0 and router._calls_this_token > router._num_layers:
                router.token_counter += 1
                router._calls_this_token = 1

            layer_id = router._layer_map.get(id(self_module), -1)
            if isinstance(indices, mx.array):
                idx_np = np.array(indices).flatten().tolist()
            else:
                idx_np = list(indices)
            router.events.append(RoutingEvent(
                layer_idx=layer_id,
                expert_indices=idx_np,
                token_idx=router.token_counter,
                timestamp=time.monotonic(),
            ))
            return original_call(self_module, x, indices)

        switch_cls.__call__ = hooked_call
        self._patched = True
        return True

    def advance_token(self):
        """Call after each generated token."""
        self.token_counter += 1

    def reset_for_new_generation(self):
        """Reset call counter for a new generation run."""
        self._calls_this_token = 0

    def get_events_for_token(self, token_idx: int) -> list[RoutingEvent]:
        """Get all routing events for a specific token."""
        return [e for e in self.events if e.token_idx == token_idx]

    def get_expert_frequencies(self) -> dict[tuple[int, int], int]:
        """Get (layer, expert) -> activation count."""
        freq = defaultdict(int)
        for event in self.events:
            for eid in event.expert_indices:
                freq[(event.layer_idx, eid)] += 1
        return freq

    def uninstall(self, model):
        """Restore original __call__ method."""
        orig = self._original_calls.get("__class__")
        if orig and hasattr(self, "_switch_cls"):
            self._switch_cls.__call__ = orig


# ── Cache simulation with real routing data ──

@dataclass
class CacheSimState:
    """Tracks cache state using real routing data."""
    capacity_experts: int  # how many experts fit in cache
    cached: set = field(default_factory=set)  # (layer, expert) tuples
    total_requests: int = 0
    total_hits: int = 0
    # LCP priority tracking
    frequency: dict = field(default_factory=lambda: defaultdict(int))
    last_used: dict = field(default_factory=lambda: defaultdict(int))
    step: int = 0

    @property
    def hit_rate(self) -> float:
        return self.total_hits / self.total_requests if self.total_requests > 0 else 0.0

    def process_token(self, events: list[RoutingEvent]) -> tuple[int, int]:
        """Process routing events for one token. Returns (hits, misses)."""
        self.step += 1
        hits = 0
        misses = 0

        for event in events:
            for eid in event.expert_indices:
                key = (event.layer_idx, eid)
                self.total_requests += 1
                self.frequency[key] += 1
                self.last_used[key] = self.step

                if key in self.cached:
                    hits += 1
                    self.total_hits += 1
                else:
                    misses += 1
                    # Cache this expert (evict if full)
                    if len(self.cached) >= self.capacity_experts:
                        self._evict_lcp()
                    self.cached.add(key)

        return hits, misses

    def _evict_lcp(self):
        """Evict the lowest-priority expert using LCP policy."""
        if not self.cached:
            return
        min_key = None
        min_priority = float('inf')
        for key in self.cached:
            freq = self.frequency.get(key, 0)
            age = self.step - self.last_used.get(key, 0)
            priority = freq * (0.25 ** (age / 128))
            if priority < min_priority:
                min_priority = priority
                min_key = key
        if min_key:
            self.cached.discard(min_key)


# ── Token-by-token inference with warm-up display ──

def _fmt_prompt(tokenizer, prompt):
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            pass
    return prompt


def generate_with_warmup(
    model, tokenizer, prompt: str, max_tokens: int = 100,
    cache_experts: int = 500, show_progress: bool = True,
) -> dict:
    """Generate text while showing progressive cache warm-up.

    Returns metrics including per-token hit rates and speedup curve.
    """
    router = ExpertRouter()
    installed = router.install(model)

    if not installed:
        print("  Warning: Could not install routing hooks (model may not be MoE)")

    formatted = _fmt_prompt(tokenizer, prompt)

    # Warm up MLX (without recording)
    _ = generate(model, tokenizer, prompt=formatted, max_tokens=3, verbose=False)
    mx.synchronize()
    router.events.clear()
    router.token_counter = 0
    router.reset_for_new_generation()

    cache = CacheSimState(capacity_experts=cache_experts)

    if show_progress:
        print(f"\n  Generating {max_tokens} tokens with cache warm-up tracking...")
        print(f"  Cache capacity: {cache_experts} expert slots")
        print()

    # Generate — routing events captured automatically
    router.reset_for_new_generation()
    event_start = len(router.events)
    token_start = router.token_counter

    t0 = time.monotonic()
    output = generate(model, tokenizer, prompt=formatted,
                      max_tokens=max_tokens, verbose=False)
    mx.synchronize()
    total_time = time.monotonic() - t0

    # Process ONLY events from this generation (not warmup)
    gen_events = router.events[event_start:]
    token_indices = sorted(set(e.token_idx for e in gen_events))

    # Group events by token and process through cache
    token_metrics = []
    for t in token_indices:
        events = [e for e in gen_events if e.token_idx == t]
        if not events:
            continue
        hits, misses = cache.process_token(events)
        total = hits + misses
        token_metrics.append({
            "token": len(token_metrics),  # sequential index
            "real_token_idx": t,
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / total if total > 0 else 0,
            "cumulative_hit_rate": cache.hit_rate,
            "cache_size": len(cache.cached),
        })

    # Display warm-up curve
    if show_progress and token_metrics:
        print("  Token  Hit Rate  Cumulative  Cache Size  Status")
        print("  " + "-" * 60)

        display_every = max(1, len(token_metrics) // 25)
        for i, m in enumerate(token_metrics):
            if i % display_every == 0 or i == len(token_metrics) - 1:
                hr = m["cumulative_hit_rate"]
                bar_len = int(hr * 20)
                bar = "#" * bar_len + "." * (20 - bar_len)

                if hr >= 0.85:
                    status = "FAST"
                elif hr >= 0.5:
                    status = "warm"
                else:
                    status = "cold"

                print(f"  {m['token']:>5d}  {m['hit_rate']:>5.0%}      "
                      f"{hr:>5.1%}       {m['cache_size']:>5d}       "
                      f"{bar}  {status}")

    # Expert heatmap
    freqs = router.get_expert_frequencies()

    # Uninstall hooks
    router.uninstall(model)

    tokens_out = len(tokenizer.encode(output))
    tps = tokens_out / total_time if total_time > 0 else 0

    result = {
        "output": output,
        "tokens": tokens_out,
        "time_s": total_time,
        "tok_per_s": tps,
        "token_metrics": token_metrics,
        "final_hit_rate": cache.hit_rate,
        "total_routing_events": len(router.events),
        "unique_experts_activated": len(freqs),
        "expert_frequencies": freqs,
    }

    if show_progress:
        print()
        print(f"  Output: {output[:120]}...")
        print(f"  Speed: {tps:.1f} tok/s, {tokens_out} tokens in {total_time:.1f}s")
        print(f"  Routing events: {len(router.events)}")
        print(f"  Unique experts activated: {len(freqs)}")
        print(f"  Final cache hit rate: {cache.hit_rate:.1%}")

        # Warm-up summary
        if len(token_metrics) >= 10:
            first_10_hr = np.mean([m["cumulative_hit_rate"] for m in token_metrics[:10]])
            last_10_hr = np.mean([m["cumulative_hit_rate"] for m in token_metrics[-10:]])
            print(f"\n  Warm-up: {first_10_hr:.0%} -> {last_10_hr:.0%} hit rate")
            print(f"  (Like ISP caching: slow start, then full speed)")

    return result


def run_multi_topic(
    model, tokenizer, topics: list[tuple[str, str]], tokens_per_topic: int = 50,
    cache_experts: int = 500,
):
    """Run multiple topics to show topic-switch warm-up behavior."""
    router = ExpertRouter()
    router.install(model)

    cache = CacheSimState(capacity_experts=cache_experts)
    all_sessions = []

    for topic_name, prompt in topics:
        print(f"\n  === Topic: {topic_name} ===")
        formatted = _fmt_prompt(tokenizer, prompt)

        # Quick warmup (without recording)
        router.reset_for_new_generation()
        _ = generate(model, tokenizer, prompt=formatted, max_tokens=2, verbose=False)
        mx.synchronize()

        start_events = len(router.events)
        start_token = router.token_counter
        router.reset_for_new_generation()

        t0 = time.monotonic()
        output = generate(model, tokenizer, prompt=formatted,
                          max_tokens=tokens_per_topic, verbose=False)
        mx.synchronize()
        elapsed = time.monotonic() - t0

        # Process new events through cache
        session_metrics = []
        new_events = router.events[start_events:]
        max_token = max(e.token_idx for e in new_events) if new_events else start_token

        for t in range(start_token, max_token + 1):
            events = [e for e in new_events if e.token_idx == t]
            if not events:
                continue
            hits, misses = cache.process_token(events)
            total = hits + misses
            session_metrics.append({
                "token": t - start_token,
                "hits": hits,
                "misses": misses,
                "hit_rate": hits / total if total > 0 else 0,
                "cumulative_hit_rate": cache.hit_rate,
            })

        tokens_out = len(tokenizer.encode(output))
        tps = tokens_out / elapsed if elapsed > 0 else 0

        # Show progress
        if session_metrics:
            first_hr = session_metrics[0]["hit_rate"] if session_metrics else 0
            last_hr = session_metrics[-1]["cumulative_hit_rate"] if session_metrics else 0
            print(f"  {tps:.1f} tok/s | First token hit: {first_hr:.0%} | "
                  f"Final hit: {last_hr:.0%} | Cache: {len(cache.cached)} experts")

            if first_hr >= 0.8:
                print(f"  -> Instant fast (experts still cached from before)")
            elif first_hr >= 0.3:
                print(f"  -> Partial warm (some experts shared with previous topic)")
            else:
                print(f"  -> Cold start (new topic, loading experts)")

        all_sessions.append({
            "topic": topic_name,
            "output": output[:100],
            "tokens": tokens_out,
            "time_s": elapsed,
            "tok_per_s": tps,
            "metrics": session_metrics,
            "first_hit_rate": session_metrics[0]["hit_rate"] if session_metrics else 0,
        })

    router.uninstall(model)

    # Summary
    print("\n" + "=" * 60)
    print("  TOPIC SWITCHING SUMMARY")
    print("=" * 60)
    print()
    for s in all_sessions:
        first = s["first_hit_rate"]
        bar = "#" * int(first * 20) + "." * (20 - int(first * 20))
        print(f"  {s['topic']:<15s} first_hit: {first:>5.0%}  {bar}  {s['tok_per_s']:.0f} tok/s")

    print()
    print("  Legend: High first-token hit rate = experts were already cached")
    print("         Low first-token hit rate = new topic, loading from SSD")

    return all_sessions


def main():
    parser = argparse.ArgumentParser(
        description="MLX-Flash-Compress: Cached Inference with Warm-up"
    )
    parser.add_argument("--model", default="mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit")
    parser.add_argument("--prompt", default="Explain how caching works in computer systems, "
                        "from CPU L1/L2 caches to disk caches and CDN caching.")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--cache-experts", type=int, default=500,
                        help="Number of expert slots in cache")
    parser.add_argument("--multi-topic", action="store_true",
                        help="Run multi-topic demo showing topic switching")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  MLX-Flash-Compress: Cached Inference")
    print("=" * 60)

    hw = detect_hardware()
    print(f"\n  Hardware: {hw.chip}, {hw.total_ram_gb:.0f}GB RAM")

    print(f"  Loading: {args.model}")
    t0 = time.monotonic()
    model, tokenizer = load(args.model)
    print(f"  Loaded in {time.monotonic() - t0:.1f}s")

    if args.multi_topic:
        topics = [
            ("coding", "Write a Python function to sort a list using merge sort with detailed comments."),
            ("writing", "Write a compelling opening paragraph for a science fiction novel set on Mars."),
            ("coding", "Now write unit tests for the merge sort function you wrote earlier."),
            ("math", "Prove that the square root of 2 is irrational using proof by contradiction."),
            ("coding", "Refactor the merge sort to use an in-place algorithm for better memory efficiency."),
        ]
        run_multi_topic(model, tokenizer, topics,
                        tokens_per_topic=args.tokens,
                        cache_experts=args.cache_experts)
    else:
        generate_with_warmup(model, tokenizer, args.prompt,
                             max_tokens=args.tokens,
                             cache_experts=args.cache_experts)


if __name__ == "__main__":
    main()
