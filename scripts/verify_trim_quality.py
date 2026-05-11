#!/usr/bin/env python3
"""Verify that trim-verify produces coherent output vs replay baseline."""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from mlx_flash_compress.dflash_model import DFlashDraftModel, DFlashModelConfig, DFlashRunner


def main():
    print("Loading models...")
    from huggingface_hub import snapshot_download
    from mlx_lm import load

    model, tokenizer = load("mlx-community/Qwen3.6-35B-A3B-4bit")
    drafter_path = snapshot_download("z-lab/Qwen3.6-35B-A3B-DFlash")
    print("  Done.\n")

    prompts = [
        "def binary_search(arr, target):\n    ",
        "The transformer architecture consists of",
        "To solve the equation 3x + 5 = 20,",
    ]

    for prompt in prompts:
        print(f"Prompt: {prompt[:50]}...")
        print()

        # Replay (baseline)
        d1, c1 = DFlashDraftModel.from_pretrained(drafter_path)
        nn.quantize(d1, group_size=64, bits=8)
        mx.eval(d1.parameters())
        r1 = DFlashRunner(model, tokenizer, d1, c1, trim_verify=False)
        text_replay, s1 = r1.generate(prompt, max_tokens=64, use_cache=True)

        # Trim
        d2, c2 = DFlashDraftModel.from_pretrained(drafter_path)
        nn.quantize(d2, group_size=64, bits=8)
        mx.eval(d2.parameters())
        r2 = DFlashRunner(model, tokenizer, d2, c2, trim_verify=True)
        text_trim, s2 = r2.generate(prompt, max_tokens=64, use_cache=True)

        print(f"  REPLAY ({s1['tok_per_sec']:.0f} tok/s): {text_replay[:100]}")
        print(f"  TRIM   ({s2['tok_per_sec']:.0f} tok/s): {text_trim[:100]}")
        match = "MATCH" if text_replay == text_trim else "DIFFER"
        print(f"  Output: {match}")
        print()


if __name__ == "__main__":
    main()
