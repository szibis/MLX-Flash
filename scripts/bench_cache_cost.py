#!/usr/bin/env python3
"""Measure cache deepcopy cost and individual verify/replay forward pass times."""

import copy
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))
from mlx_flash_compress.dflash_model import DFlashDraftModel, DFlashModelConfig, DFlashRunner


def main():
    print("Loading models...")
    from huggingface_hub import snapshot_download
    from mlx_lm import load

    model, tokenizer = load("mlx-community/Qwen3.6-35B-A3B-4bit")
    drafter_path = snapshot_download("z-lab/Qwen3.6-35B-A3B-DFlash")
    drafter, config = DFlashDraftModel.from_pretrained(drafter_path)
    print("  Done.\n")

    runner = DFlashRunner(model, tokenizer, drafter, config)

    prompt = "def binary_search(arr, target):\n    "
    tokens = tokenizer.encode(prompt)

    cache = runner._make_cache()
    input_ids = mx.array([tokens])

    # Prefill
    logits, hidden = runner._forward_target_cached(input_ids, cache)
    mx.eval(logits, hidden)

    # Measure cache deepcopy cost
    times_copy = []
    for _ in range(20):
        t0 = time.perf_counter()
        snap = copy.deepcopy(cache)
        times_copy.append(time.perf_counter() - t0)
    times_copy.sort()

    # Draft some tokens
    all_ids = mx.array([tokens])
    draft_ids, _ = runner.draft_tokens(all_ids, target_hidden=hidden)
    mx.eval(draft_ids)
    n_draft = draft_ids.shape[-1]

    # Measure verify forward pass (processing draft tokens)
    times_verify = []
    for _ in range(5):
        snap = copy.deepcopy(cache)
        t0 = time.perf_counter()
        v_logits, v_hidden = runner._forward_target_cached(draft_ids, snap)
        mx.eval(v_logits)
        times_verify.append(time.perf_counter() - t0)
    times_verify.sort()

    # Measure replay forward pass (3 tokens = typical accepted)
    replay_ids = mx.array([[1, 2, 3]])
    times_replay = []
    for _ in range(5):
        snap = copy.deepcopy(cache)
        t0 = time.perf_counter()
        r_logits, r_hidden = runner._forward_target_cached(replay_ids, snap)
        mx.eval(r_logits)
        times_replay.append(time.perf_counter() - t0)
    times_replay.sort()

    # Measure trim cost
    times_trim = []
    for _ in range(20):
        snap = copy.deepcopy(cache)
        # Simulate: verify pushed D tokens, now trim D-3 of them
        for c in snap:
            if hasattr(c, "trim"):
                t0 = time.perf_counter()
                c.trim(n_draft - 3)
                times_trim.append(time.perf_counter() - t0)
                break
    times_trim.sort()

    print("=" * 60)
    print("DFlash Cache & Verify Overhead Analysis")
    print("=" * 60)
    print(f"Context length: {len(tokens)} tokens")
    print(f"Draft length: {n_draft} tokens")
    print()
    print(f"cache deepcopy:        {times_copy[len(times_copy) // 2] * 1000:.2f}ms median")
    print(f"verify forward ({n_draft} tok):  {times_verify[len(times_verify) // 2] * 1000:.2f}ms median")
    print(f"replay forward (3 tok):  {times_replay[len(times_replay) // 2] * 1000:.2f}ms median")
    if times_trim:
        print(f"KV cache trim:         {times_trim[len(times_trim) // 2] * 1000:.4f}ms median")
    print()

    total_per_step = (
        times_copy[len(times_copy) // 2] + times_verify[len(times_verify) // 2] + times_replay[len(times_replay) // 2]
    )
    print(f"Total verify+replay:   {total_per_step * 1000:.2f}ms")
    print("  = deepcopy + verify + replay")
    print()

    trim_verify = times_verify[len(times_verify) // 2] + (times_trim[len(times_trim) // 2] if times_trim else 0)
    print(f"With trim (no replay): {trim_verify * 1000:.2f}ms")
    print("  = verify + trim (skip deepcopy + replay)")
    savings = (total_per_step - trim_verify) / total_per_step * 100
    print(f"Potential savings:     {savings:.0f}%")


if __name__ == "__main__":
    main()
