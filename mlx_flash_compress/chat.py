"""Interactive chat with colorful console UI, download progress, and memory status.

Usage:
  mlx-flash-chat
  mlx-flash-chat --model mlx-community/Qwen3-30B-A3B-4bit
"""

import argparse
import os
import sys
import time

import mlx.core as mx
from mlx_lm import load, generate

from mlx_flash_compress.hardware import detect_hardware
from mlx_flash_compress.memory_manager import get_memory_state


# -- ANSI colors --

class C:
    """ANSI color codes for terminal output."""
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    WHITE = "\033[97m"

    @staticmethod
    def enabled():
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def c(code, text):
    """Colorize text if terminal supports it."""
    if C.enabled():
        return f"{code}{text}{C.RESET}"
    return text


# -- UI components --

def memory_bar(used_pct: float) -> str:
    """Colored memory usage bar."""
    bar_len = 25
    filled = int(used_pct / 100 * bar_len)
    empty = bar_len - filled

    if used_pct >= 90:
        color = C.RED
        label = "CRITICAL"
    elif used_pct >= 75:
        color = C.YELLOW
        label = "tight"
    elif used_pct >= 50:
        color = C.GREEN
        label = "ok"
    else:
        color = C.CYAN
        label = "free"

    bar = c(color, "█" * filled) + c(C.DIM, "░" * empty)
    return f"{bar} {c(color, f'{used_pct:.0f}%')} {c(C.DIM, label)}"


def print_header():
    print()
    print(c(C.CYAN, "  ╔══════════════════════════════════════════════════════╗"))
    print(c(C.CYAN, "  ║") + c(C.BOLD, "           ⚡ MLX-Flash Interactive Chat ⚡           ") + c(C.CYAN, "║"))
    print(c(C.CYAN, "  ╚══════════════════════════════════════════════════════╝"))


def print_hw_info(hw, mem):
    print(f"\n  {c(C.BOLD, hw.chip)}, {hw.total_ram_gb:.0f}GB RAM")
    used_pct = (1 - mem.available_gb / max(mem.total_gb, 1)) * 100
    print(f"  Memory: {memory_bar(used_pct)}")


def print_status(hw, mem, request_num: int, total_tokens: int):
    """Print status block."""
    used_pct = (1 - mem.available_gb / max(mem.total_gb, 1)) * 100
    print(f"\n  {c(C.BOLD, '📊 Status')}")
    print(f"  RAM:     {memory_bar(used_pct)}")
    print(f"  Free:    {c(C.GREEN, f'{mem.available_gb:.1f}GB')} / {mem.total_gb:.0f}GB")
    print(f"  Pressure: {_pressure_color(mem.pressure_level)}")
    if request_num > 0:
        print(f"  Session:  {request_num} messages, {total_tokens} tokens")


def _pressure_color(level: str) -> str:
    colors = {"normal": C.GREEN, "warning": C.YELLOW, "critical": C.RED}
    return c(colors.get(level, C.WHITE), level.upper())


def print_help():
    print(f"\n  {c(C.BOLD, '📋 Commands')}")
    print(f"  {c(C.CYAN, '/status')}   Show memory and session info")
    print(f"  {c(C.CYAN, '/clear')}    Clear conversation history")
    print(f"  {c(C.CYAN, '/help')}     Show this help")
    print(f"  {c(C.CYAN, '/quit')}     Exit")


def _fmt_prompt(tokenizer, messages):
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


# -- Main --

def main():
    parser = argparse.ArgumentParser(description="MLX-Flash: Interactive Chat")
    parser.add_argument("--model", default="mlx-community/Qwen3-8B-4bit",
                        help="MLX model to chat with")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--system", default="You are a helpful AI assistant.",
                        help="System prompt")
    args = parser.parse_args()

    print_header()

    hw = detect_hardware()
    mem = get_memory_state()
    print_hw_info(hw, mem)

    # Download with progress
    model_name = args.model
    if not os.path.isdir(model_name):
        try:
            from huggingface_hub import snapshot_download
            print(f"\n  {c(C.YELLOW, '⬇')}  Downloading {c(C.BOLD, model_name)}...")
            snapshot_download(
                model_name,
                allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
            )
            print(f"  {c(C.GREEN, '✓')}  Download complete")
        except Exception:
            pass

    print(f"\n  {c(C.YELLOW, '⏳')} Loading model...")
    t0 = time.monotonic()
    model, tokenizer = load(model_name)
    mx.synchronize()
    load_time = time.monotonic() - t0
    print(f"  {c(C.GREEN, '✓')}  {c(C.BOLD, model_name.split('/')[-1])} loaded in {load_time:.1f}s")

    mem_after = get_memory_state()
    model_gb = max(mem.available_gb - mem_after.available_gb, 0)
    print(f"  {c(C.DIM, f'   Model: ~{model_gb:.1f}GB, {mem_after.available_gb:.1f}GB remaining')}")

    if mem_after.pressure_level in ("warning", "critical"):
        print(f"\n  {c(C.RED, '⚠')}  Memory is {_pressure_color(mem_after.pressure_level)}")
        print(f"  {c(C.DIM, '   Try: --model mlx-community/Qwen3-4B-4bit')}")

    messages = [{"role": "system", "content": args.system}]
    request_num = 0
    total_tokens = 0

    print(f"\n  {c(C.DIM, 'Type a message to start chatting. /help for commands.')}")
    print(c(C.DIM, "  " + "─" * 56))

    while True:
        try:
            user_input = input(f"\n  {c(C.GREEN, '▶ You:')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n  {c(C.CYAN, 'Goodbye! 👋')}")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "/quit", "/exit"):
            print(f"  {c(C.CYAN, 'Goodbye! 👋')}")
            break

        if user_input.lower() in ("/status", "/mem", "/memory"):
            mem = get_memory_state()
            print_status(hw, mem, request_num, total_tokens)
            continue

        if user_input.lower() == "/help":
            print_help()
            continue

        if user_input.lower() == "/clear":
            messages = [{"role": "system", "content": args.system}]
            print(f"  {c(C.GREEN, '✓')}  Conversation cleared")
            continue

        # Memory check
        mem = get_memory_state()
        if mem.pressure_level == "critical":
            print(f"\n  {c(C.RED, '⚠  Memory pressure CRITICAL — may be slow')}")

        messages.append({"role": "user", "content": user_input})
        formatted = _fmt_prompt(tokenizer, messages)

        # Generate
        print(f"\n  {c(C.MAGENTA, '◆ Assistant:')}", end=" ", flush=True)
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
        print(output)
        stats = f"{tokens} tok, {tps:.0f} tok/s, {elapsed:.1f}s"
        print(f"  {c(C.DIM, f'   [{stats}]')}", end="")

        mem_after = get_memory_state()
        if mem_after.pressure_level != "normal":
            print(f"  {c(C.YELLOW, f'[RAM: {mem_after.pressure_level}]')}", end="")
        print()


if __name__ == "__main__":
    main()
