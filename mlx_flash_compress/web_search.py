"""Web search and knowledge memory for MLX-Flash chat.

Provides:
1. Web search via DuckDuckGo (no API key needed)
2. Persistent memory (JSON file) for facts the user wants remembered
3. Context injection — search results and memories added to prompts

Usage in chat:
  /search <query>     — search the web, show results
  /remember <fact>    — save a fact to persistent memory
  /memories           — show all saved memories
  /forget <number>    — delete a memory by number
"""

import json
import os
import time
import urllib.parse
import urllib.request
import re
from dataclasses import dataclass, field
from pathlib import Path


# -- Web Search (DuckDuckGo HTML, no API key) --

@dataclass
class SearchResult:
    title: str = ""
    url: str = ""
    snippet: str = ""


def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search the web via DuckDuckGo HTML (no API key needed)."""
    results = []

    try:
        encoded = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"

        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
        })

        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        # Parse results from DuckDuckGo HTML
        # Each result is in a <div class="result"> block
        result_blocks = re.findall(
            r'<a rel="nofollow" class="result__a" href="(.*?)">(.*?)</a>.*?'
            r'<a class="result__snippet".*?>(.*?)</a>',
            html, re.DOTALL
        )

        for href, title, snippet in result_blocks[:max_results]:
            # Clean HTML tags
            title = re.sub(r"<.*?>", "", title).strip()
            snippet = re.sub(r"<.*?>", "", snippet).strip()

            # Decode DuckDuckGo redirect URL
            if "uddg=" in href:
                match = re.search(r"uddg=([^&]+)", href)
                if match:
                    href = urllib.parse.unquote(match.group(1))

            results.append(SearchResult(title=title, url=href, snippet=snippet))

    except Exception as e:
        results.append(SearchResult(
            title="Search failed",
            snippet=str(e),
        ))

    return results


def format_search_results(results: list[SearchResult]) -> str:
    """Format search results for injection into LLM context."""
    if not results:
        return "No results found."

    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"{i}. {r.title}")
        if r.snippet:
            parts.append(f"   {r.snippet}")
        if r.url:
            parts.append(f"   Source: {r.url}")
    return "\n".join(parts)


def build_search_context(query: str, results: list[SearchResult]) -> str:
    """Build a context string for the LLM from search results."""
    formatted = format_search_results(results)
    return (
        f"[Web Search Results for: {query}]\n"
        f"{formatted}\n"
        f"[End of search results. Use this information to answer the user's question.]"
    )


# -- Persistent Memory --

@dataclass
class Memory:
    fact: str
    timestamp: float = 0.0
    source: str = "user"  # "user" or "auto"

    def to_dict(self):
        return {"fact": self.fact, "timestamp": self.timestamp, "source": self.source}

    @classmethod
    def from_dict(cls, d):
        return cls(fact=d["fact"], timestamp=d.get("timestamp", 0), source=d.get("source", "user"))


class MemoryStore:
    """Persistent JSON-based memory for chat facts."""

    def __init__(self, path: str = None):
        if path is None:
            config_dir = Path.home() / ".config" / "mlx-flash"
            config_dir.mkdir(parents=True, exist_ok=True)
            path = str(config_dir / "memories.json")
        self.path = path
        self.memories: list[Memory] = []
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path) as f:
                    data = json.load(f)
                self.memories = [Memory.from_dict(m) for m in data]
        except (json.JSONDecodeError, KeyError):
            self.memories = []

    def _save(self):
        with open(self.path, "w") as f:
            json.dump([m.to_dict() for m in self.memories], f, indent=2)

    def add(self, fact: str, source: str = "user") -> int:
        mem = Memory(fact=fact, timestamp=time.time(), source=source)
        self.memories.append(mem)
        self._save()
        return len(self.memories)

    def remove(self, index: int) -> bool:
        if 0 <= index < len(self.memories):
            self.memories.pop(index)
            self._save()
            return True
        return False

    def list_all(self) -> list[Memory]:
        return self.memories

    def build_context(self) -> str:
        """Build a context string from all memories for LLM injection."""
        if not self.memories:
            return ""
        parts = ["[User's persistent memories — use these as context:]"]
        for i, m in enumerate(self.memories, 1):
            parts.append(f"- {m.fact}")
        parts.append("[End of memories]")
        return "\n".join(parts)

    def count(self) -> int:
        return len(self.memories)
