"""Interactive chat with colorful console UI, model switching, and memory status.

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
from mlx_flash_compress.web_search import (
    web_search, format_search_results, build_search_context, MemoryStore,
)


# -- Model catalog --

MODELS = [
    # (name, total_params, active_params, size_gb, type, description)
    ("mlx-community/Qwen3-4B-4bit", "4B", "4B", 2.5, "dense", "Fast, fits 8GB"),
    ("mlx-community/Qwen3-8B-4bit", "8B", "8B", 5.0, "dense", "Great quality, fits 8GB"),
    ("mlx-community/Qwen3-14B-4bit", "14B", "14B", 8.5, "dense", "Strong, needs 12GB+"),
    ("mlx-community/Qwen3-30B-A3B-4bit", "30B", "3B", 18.0, "MoE", "Best MoE, needs 24GB+ or streaming"),
    ("mlx-community/Qwen3.5-35B-A3B-4bit", "35B", "3B", 20.0, "MoE", "Latest MoE, needs 24GB+"),
    ("mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit", "30B", "3B", 18.0, "MoE", "Best for coding"),
    ("mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit", "47B", "13B", 26.0, "MoE", "Classic MoE, needs 32GB+"),
]


# -- ANSI colors --

class C:
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
    if C.enabled():
        return f"{code}{text}{C.RESET}"
    return str(text)


# -- UI components --

def memory_bar(used_pct: float) -> str:
    bar_len = 25
    filled = int(used_pct / 100 * bar_len)
    empty = bar_len - filled
    if used_pct >= 90:
        color, label = C.RED, "CRITICAL"
    elif used_pct >= 75:
        color, label = C.YELLOW, "tight"
    elif used_pct >= 50:
        color, label = C.GREEN, "ok"
    else:
        color, label = C.CYAN, "free"
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


def print_models(current_model: str, ram_gb: float):
    """Show available models with size and fit info."""
    print(f"\n  {c(C.BOLD, '📦 Available Models')}")
    print(f"  {c(C.DIM, '─' * 56)}")

    for i, (name, total, active, size, mtype, desc) in enumerate(MODELS, 1):
        short = name.split("/")[-1]
        fits = "✓" if size < ram_gb * 0.85 else "⚡" if mtype == "MoE" else "✗"

        if fits == "✓":
            fit_color = C.GREEN
        elif fits == "⚡":
            fit_color = C.YELLOW
        else:
            fit_color = C.RED

        active_str = c(C.CYAN, f"{active} active") if mtype == "MoE" else c(C.DIM, "dense")
        current = c(C.GREEN, " ◄ current") if name == current_model else ""

        print(f"  {c(fit_color, fits)} {c(C.BOLD, f'{i}.')} {short}")
        print(f"     {total} params ({active_str}) · {size:.1f}GB · {c(C.DIM, desc)}{current}")

    print(f"\n  {c(C.DIM, 'Legend: ✓ fits in RAM  ⚡ needs streaming  ✗ too large')}")
    print(f"  {c(C.DIM, 'Switch: /model <number> or /model <name>')}")


def print_status(hw, mem, request_num: int, total_tokens: int, model_name: str):
    used_pct = (1 - mem.available_gb / max(mem.total_gb, 1)) * 100
    print(f"\n  {c(C.BOLD, '📊 Status')}")
    print(f"  Model:    {c(C.CYAN, model_name.split('/')[-1])}")
    print(f"  RAM:      {memory_bar(used_pct)}")
    print(f"  Free:     {c(C.GREEN, f'{mem.available_gb:.1f}GB')} / {mem.total_gb:.0f}GB")
    print(f"  Pressure: {_pressure_color(mem.pressure_level)}")
    if request_num > 0:
        print(f"  Session:  {request_num} messages, {total_tokens} tokens")


def _pressure_color(level: str) -> str:
    colors = {"normal": C.GREEN, "warning": C.YELLOW, "critical": C.RED}
    return c(colors.get(level, C.WHITE), level.upper())


def print_help():
    print(f"\n  {c(C.BOLD, '📋 Commands')}")
    print(f"  {c(C.BOLD, 'Models')}")
    print(f"  {c(C.CYAN, '/models')}          List available models with size info")
    print(f"  {c(C.CYAN, '/model N')}        Switch to model N (number or name)")
    print(f"  {c(C.BOLD, 'Search & Memory')}")
    print(f"  {c(C.CYAN, '/search <query>')} Search the web (DuckDuckGo)")
    print(f"  {c(C.CYAN, '/ask <question>')} Search + auto-answer using results")
    print(f"  {c(C.CYAN, '/remember <fact>')} Save a fact to persistent memory")
    print(f"  {c(C.CYAN, '/memories')}       Show saved memories")
    print(f"  {c(C.CYAN, '/forget N')}       Delete memory by number")
    print(f"  {c(C.BOLD, 'Session')}")
    print(f"  {c(C.CYAN, '/status')}          Show memory and session info")
    print(f"  {c(C.CYAN, '/clear')}           Clear conversation history")
    print(f"  {c(C.CYAN, '/help')}            Show this help")
    print(f"  {c(C.CYAN, '/quit')}            Exit")


def _fmt_prompt(tokenizer, messages):
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def load_model(model_name: str):
    """Download and load a model with progress."""
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

    print(f"  {c(C.YELLOW, '⏳')} Loading {c(C.BOLD, model_name.split('/')[-1])}...")
    mem_before = get_memory_state()
    t0 = time.monotonic()
    model, tokenizer = load(model_name)
    mx.synchronize()
    load_time = time.monotonic() - t0

    mem_after = get_memory_state()
    model_gb = max(mem_before.available_gb - mem_after.available_gb, 0)
    print(f"  {c(C.GREEN, '✓')}  Loaded in {load_time:.1f}s (~{model_gb:.1f}GB, {mem_after.available_gb:.1f}GB remaining)")

    if mem_after.pressure_level in ("warning", "critical"):
        print(f"  {c(C.RED, '⚠')}  Memory {_pressure_color(mem_after.pressure_level)} — try a smaller model")

    return model, tokenizer


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

    model_name = args.model
    model, tokenizer = load_model(model_name)

    # Initialize memory store
    memory_store = MemoryStore()
    if memory_store.count() > 0:
        print(f"  {c(C.BLUE, '🧠')} {memory_store.count()} memories loaded")

    messages = [{"role": "system", "content": args.system}]
    request_num = 0
    total_tokens = 0

    print(f"\n  {c(C.DIM, 'Type a message to chat. /help for commands. /search for web.')}")
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

        if user_input.lower() in ("/models", "/model"):
            print_models(model_name, mem.total_gb)
            continue

        if user_input.lower().startswith("/model "):
            choice = user_input[7:].strip()
            new_name = None

            # Try as number
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(MODELS):
                    new_name = MODELS[idx][0]
            except ValueError:
                pass

            # Try as name (exact or partial match)
            if new_name is None:
                for m_name, *_ in MODELS:
                    if choice in m_name or choice == m_name.split("/")[-1]:
                        new_name = m_name
                        break

            # Try as arbitrary HF model name
            if new_name is None and "/" in choice:
                new_name = choice

            if new_name is None:
                print(f"  {c(C.RED, '✗')}  Unknown model: {choice}")
                print(f"  {c(C.DIM, '   Use /models to see available options, or pass a full HF name')}")
                continue

            if new_name == model_name:
                print(f"  {c(C.DIM, '   Already using this model')}")
                continue

            # Release old model
            print(f"  {c(C.YELLOW, '↻')}  Switching model...")
            del model, tokenizer
            try:
                mx.clear_cache()
            except AttributeError:
                pass

            model_name = new_name
            model, tokenizer = load_model(model_name)
            messages = [{"role": "system", "content": args.system}]
            request_num = 0
            total_tokens = 0
            print(f"  {c(C.GREEN, '✓')}  Conversation cleared for new model")
            continue

        if user_input.lower() in ("/status", "/mem", "/memory"):
            mem = get_memory_state()
            print_status(hw, mem, request_num, total_tokens, model_name)
            continue

        if user_input.lower() == "/help":
            print_help()
            continue

        if user_input.lower() == "/clear":
            messages = [{"role": "system", "content": args.system}]
            request_num = 0
            total_tokens = 0
            print(f"  {c(C.GREEN, '✓')}  Conversation cleared")
            continue

        # -- Web search --
        if user_input.lower().startswith("/search "):
            query = user_input[8:].strip()
            if not query:
                print(f"  {c(C.DIM, 'Usage: /search <query>')}")
                continue
            print(f"  {c(C.YELLOW, '🔍')} Searching: {c(C.BOLD, query)}...")
            results = web_search(query)
            for i, r in enumerate(results, 1):
                print(f"  {c(C.CYAN, f'{i}.')} {c(C.BOLD, r.title)}")
                if r.snippet:
                    print(f"     {c(C.DIM, r.snippet[:120])}")
                if r.url:
                    print(f"     {c(C.BLUE, r.url)}")
            print(f"\n  {c(C.DIM, 'Use /ask <question> to search + get an AI answer')}")
            continue

        # -- Search + auto-answer --
        if user_input.lower().startswith("/ask "):
            query = user_input[5:].strip()
            if not query:
                print(f"  {c(C.DIM, 'Usage: /ask <question>')}")
                continue
            print(f"  {c(C.YELLOW, '🔍')} Searching: {c(C.BOLD, query)}...")
            results = web_search(query)
            if results:
                for i, r in enumerate(results[:3], 1):
                    print(f"  {c(C.CYAN, f'{i}.')} {r.title}")
                search_ctx = build_search_context(query, results)
                # Inject search results as a system message
                messages.append({"role": "system", "content": search_ctx})
                messages.append({"role": "user", "content": query})
                formatted = _fmt_prompt(tokenizer, messages)

                print(f"\n  {c(C.MAGENTA, '◆ Assistant:')}", end=" ", flush=True)
                t0 = time.monotonic()
                output = generate(model, tokenizer, prompt=formatted,
                                  max_tokens=args.max_tokens, verbose=False)
                mx.synchronize()
                elapsed = time.monotonic() - t0

                tokens = len(tokenizer.encode(output))
                tps = tokens / elapsed if elapsed > 0 else 0
                request_num += 1
                total_tokens += tokens
                messages.append({"role": "assistant", "content": output})

                print(output)
                print(f"  {c(C.DIM, f'   [{tokens} tok, {tps:.0f} tok/s, {elapsed:.1f}s — with web context]')}")
            else:
                print(f"  {c(C.RED, '✗')}  No results found")
            continue

        # -- Remember --
        if user_input.lower().startswith("/remember "):
            fact = user_input[10:].strip()
            if not fact:
                print(f"  {c(C.DIM, 'Usage: /remember <fact>')}")
                continue
            idx = memory_store.add(fact)
            print(f"  {c(C.BLUE, '🧠')} Saved memory #{idx}: {c(C.BOLD, fact)}")
            continue

        # -- List memories --
        if user_input.lower() in ("/memories", "/memory", "/mem-list"):
            mems = memory_store.list_all()
            if not mems:
                print(f"  {c(C.DIM, 'No memories saved. Use /remember <fact> to save one.')}")
            else:
                print(f"\n  {c(C.BOLD, '🧠 Saved Memories')}")
                for i, m in enumerate(mems, 1):
                    age = time.time() - m.timestamp
                    if age < 3600:
                        age_str = f"{age / 60:.0f}m ago"
                    elif age < 86400:
                        age_str = f"{age / 3600:.0f}h ago"
                    else:
                        age_str = f"{age / 86400:.0f}d ago"
                    print(f"  {c(C.CYAN, f'{i}.')} {m.fact} {c(C.DIM, f'({age_str})')}")
                print(f"\n  {c(C.DIM, 'Use /forget N to delete a memory')}")
            continue

        # -- Forget --
        if user_input.lower().startswith("/forget "):
            try:
                idx = int(user_input[8:].strip()) - 1
                mems = memory_store.list_all()
                if 0 <= idx < len(mems):
                    fact = mems[idx].fact
                    memory_store.remove(idx)
                    print(f"  {c(C.GREEN, '✓')}  Forgot: {fact}")
                else:
                    print(f"  {c(C.RED, '✗')}  Invalid number. Use /memories to see list.")
            except ValueError:
                print(f"  {c(C.DIM, 'Usage: /forget <number>')}")
            continue

        # Memory check
        mem = get_memory_state()
        if mem.pressure_level == "critical":
            print(f"\n  {c(C.RED, '⚠  Memory pressure CRITICAL — may be slow')}")

        messages.append({"role": "user", "content": user_input})

        # Inject memories into context if any exist
        prompt_messages = messages.copy()
        mem_ctx = memory_store.build_context()
        if mem_ctx:
            prompt_messages.insert(1, {"role": "system", "content": mem_ctx})
        formatted = _fmt_prompt(tokenizer, prompt_messages)

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
