"""Tests for web search and memory store."""
import json
import os
import time
import pytest
from mlx_flash_compress.web_search import (
    SearchResult,
    format_search_results,
    build_search_context,
    MemoryStore,
)


class TestSearchResult:
    def test_format_empty(self):
        assert format_search_results([]) == "No results found."

    def test_format_results(self):
        results = [
            SearchResult(title="Test Page", url="https://example.com", snippet="A test"),
            SearchResult(title="Another", snippet="More info"),
        ]
        formatted = format_search_results(results)
        assert "Test Page" in formatted
        assert "https://example.com" in formatted
        assert "Another" in formatted

    def test_build_context(self):
        results = [SearchResult(title="Result 1", snippet="Info")]
        ctx = build_search_context("test query", results)
        assert "test query" in ctx
        assert "Result 1" in ctx
        assert "[Web Search Results" in ctx


class TestMemoryStore:
    def test_add_and_list(self, tmp_path):
        store = MemoryStore(path=str(tmp_path / "mem.json"))
        store.add("The sky is blue")
        store.add("Pi is 3.14159")
        assert store.count() == 2
        mems = store.list_all()
        assert mems[0].fact == "The sky is blue"
        assert mems[1].fact == "Pi is 3.14159"

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "mem.json")
        store1 = MemoryStore(path=path)
        store1.add("Remember this")

        # Reload from disk
        store2 = MemoryStore(path=path)
        assert store2.count() == 1
        assert store2.list_all()[0].fact == "Remember this"

    def test_remove(self, tmp_path):
        store = MemoryStore(path=str(tmp_path / "mem.json"))
        store.add("Fact A")
        store.add("Fact B")
        store.add("Fact C")
        assert store.remove(1)  # remove "Fact B"
        assert store.count() == 2
        assert store.list_all()[0].fact == "Fact A"
        assert store.list_all()[1].fact == "Fact C"

    def test_remove_invalid(self, tmp_path):
        store = MemoryStore(path=str(tmp_path / "mem.json"))
        assert not store.remove(0)  # empty
        store.add("X")
        assert not store.remove(5)  # out of range

    def test_build_context_empty(self, tmp_path):
        store = MemoryStore(path=str(tmp_path / "mem.json"))
        assert store.build_context() == ""

    def test_build_context_with_memories(self, tmp_path):
        store = MemoryStore(path=str(tmp_path / "mem.json"))
        store.add("User likes Python")
        store.add("User works on MLX")
        ctx = store.build_context()
        assert "User likes Python" in ctx
        assert "User works on MLX" in ctx
        assert "[User's persistent memories" in ctx

    def test_timestamp(self, tmp_path):
        store = MemoryStore(path=str(tmp_path / "mem.json"))
        before = time.time()
        store.add("Timed fact")
        after = time.time()
        mem = store.list_all()[0]
        assert before <= mem.timestamp <= after
