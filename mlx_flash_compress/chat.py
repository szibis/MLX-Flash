"""Interactive chat with memory-aware warm-up display.

Shows real-time memory status, cache warm-up, and performance
metrics as you chat with the model.

Usage:
  python -m mlx_flash_compress.chat
  python -m mlx_flash_compress.chat --model mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit
"""

import argparse
import os
import sys
import time

import mlx.core as mx
from mlx_lm import load, generate

from mlx_flash_compress.hardware import detect_hardware
from mlx_flash_compress.memory_manager import MemoryManager, get_memory_state


def _fmt_prompt(tokenizer, messages):
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def _memory_bar(used_pct: float) -> str:
    """Create a memory usage bar."""
    bar_len = 20
    filled = int(used_pct / 100 * bar_len)
    bar = "#" * filled + "." * (bar_len - filled)
    if used_pct >= 90:
        status = "CRITICAL"
    elif used_pct >= 75:
        status = "tight"
    elif used_pct >= 50:
        status = "ok"
    else:
        status = "free"
    return f"[{bar}] {used_pct:.0f}% {status}"


def print_status(hw, mem, request_num: int, total_tokens: int):
    """Print compact status line."""
    used_pct = (1 - mem.available_gb / mem.total_gb) * 100
    bar = _memory_bar(used_pct)
    print(f"\n  RAM: {bar}  "
          f"({mem.available_gb:.1f}GB free / {mem.total_gb:.0f}GB)")
    if mem.pressure_level in ("warning", "critical"):
        print(f"  ** Memory pressure: {mem.pressure_level.upper()} **")
        if mem.pressure_level == "critical":
            print("  Close apps to prevent slowdown!")
    if request_num > 0:
        print(f"  Session: {request_num} messages, {total_tokens} tokens generated")


def main():
    parser = argparse.ArgumentParser(description="MLX-Flash: Interactive Chat")
    parser.add_argument("--model", default="mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--system", default="You are a helpful AI assistant.",
                        help="System prompt")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  MLX-Flash: Interactive Chat")
    print("=" * 60)

    hw = detect_hardware()
    mem = get_memory_state()
    print(f"\n  {hw.chip}, {hw.total_ram_gb:.0f}GB RAM")
    used_pct = (1 - mem.available_gb / mem.total_gb) * 100
    print(f"  Memory: {_memory_bar(used_pct)}")

    print(f"\n  Loading: {args.model}")
    t0 = time.monotonic()
    model, tokenizer = load(args.model)
    mx.synchronize()
    print(f"  Ready in {time.monotonic() - t0:.1f}s")

    mem_after = get_memory_state()
    model_gb = mem.available_gb - mem_after.available_gb
    print(f"  Model: ~{model_gb:.1f}GB, {mem_after.available_gb:.1f}GB remaining")

    if mem_after.pressure_level in ("warning", "critical"):
        print(f"\n  ** Memory is {mem_after.pressure_level.upper()} **")
        print("  Performance may be degraded. Close other apps to help.")
        print("  Or try a smaller model with: --model mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit")

    messages = [{"role": "system", "content": args.system}]
    request_num = 0
    total_tokens = 0

    print(f"\n  Type your message (or 'quit' to exit, '/status' for memory info)")
    print("  " + "-" * 56)

    while True:
        try:
            user_input = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "/quit", "/exit"):
            print("  Goodbye!")
            break

        if user_input.lower() in ("/status", "/mem", "/memory"):
            mem = get_memory_state()
            print_status(hw, mem, request_num, total_tokens)
            continue

        if user_input.lower() == "/help":
            print("  Commands:")
            print("    /status  - Show memory and session info")
            print("    /clear   - Clear conversation history")
            print("    /quit    - Exit")
            continue

        if user_input.lower() == "/clear":
            messages = [{"role": "system", "content": args.system}]
            print("  Conversation cleared.")
            continue

        # Check memory before generation
        mem = get_memory_state()
        if mem.pressure_level == "critical":
            print("\n  ** Memory pressure CRITICAL! **")
            print("  Close applications to avoid slowdown.")
            print("  Generating anyway (may be slow)...")

        messages.append({"role": "user", "content": user_input})
        formatted = _fmt_prompt(tokenizer, messages)

        # Generate
        t0 = time.monotonic()
        output = generate(
            model, tokenizer,
            prompt=formatted,
            max_tokens=args.max_tokens,
            verbose=False,
        )
        mx.synchronize()
        elapsed = time.monotonic() - t0

        tokens = len(tokenizer.encode(output))
        tps = tokens / elapsed if elapsed > 0 else 0
        request_num += 1
        total_tokens += tokens

        messages.append({"role": "assistant", "content": output})

        # Display response
        print(f"\n  Assistant: {output}")
        print(f"  [{tokens} tokens, {tps:.0f} tok/s, {elapsed:.1f}s]", end="")

        # Compact memory status
        mem_after = get_memory_state()
        if mem_after.pressure_level != "normal":
            print(f"  [RAM: {mem_after.pressure_level}]", end="")
        print()


if __name__ == "__main__":
    main()
