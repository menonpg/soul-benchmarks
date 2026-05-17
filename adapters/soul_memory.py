"""Adapter wrapping soul.py HybridAgent for benchmarks.

Uses the local HybridAgent (RAG + RLM) from the soul.py repo directly.
No external API needed — BM25 for RAG, local LLM calls for RLM.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Literal

# Add soul.py repo to path
SOUL_REPO = Path(__file__).resolve().parents[1].parent / "soul.py"
sys.path.insert(0, str(SOUL_REPO))

from hybrid_agent import HybridAgent

MemoryMode = Literal["rag", "rlm", "auto"]


class SoulMemoryAdapter:
    """Wraps HybridAgent for benchmark evaluation."""

    def __init__(
        self,
        provider: str = "gemini",
        model: str | None = None,
        mode: MemoryMode = "auto",
        api_key: str | None = None,
    ):
        self.mode = mode
        self.provider = provider
        self._tmpdir = tempfile.mkdtemp(prefix="soul-bench-")

        soul_path = Path(self._tmpdir) / "SOUL.md"
        memory_path = Path(self._tmpdir) / "MEMORY.md"
        soul_path.write_text(
            "You are a memory-enabled assistant for benchmark evaluation. "
            "Answer questions precisely and concisely based on what you remember. "
            "If you don't have enough information, say so.\n"
        )
        memory_path.write_text("# MEMORY.md\n")

        kwargs = dict(
            soul_path=str(soul_path),
            memory_path=str(memory_path),
            provider=provider,
            mode=mode,
        )
        if model:
            kwargs["chat_model"] = model
        if api_key:
            kwargs["api_key"] = api_key

        self.agent = HybridAgent(**kwargs)

    def add_memory(self, conversation_text: str) -> None:
        """Ingest conversation text into memory."""
        self.agent.remember(conversation_text)

    def query_memory(self, question: str) -> str:
        """Query memory and return answer."""
        result = self.agent.ask(question, remember=False)
        if isinstance(result, dict):
            return result.get("answer", result.get("response", str(result)))
        return str(result)

    def reset(self) -> None:
        """Reset memory state."""
        memory_path = Path(self._tmpdir) / "MEMORY.md"
        memory_path.write_text("# MEMORY.md\n")
        self.agent.reset_conversation()


def create_gemini_adapter(mode: MemoryMode = "auto") -> SoulMemoryAdapter:
    """Create adapter using Gemini."""
    return SoulMemoryAdapter(
        provider="gemini",
        mode=mode,
        api_key=os.environ.get("GEMINI_API_KEY"),
    )


def create_azure_adapter(mode: MemoryMode = "auto") -> SoulMemoryAdapter:
    """Create adapter using Azure OpenAI via openai-compatible."""
    from openai import AzureOpenAI

    # For Azure we need to use the basic Agent with monkey-patched client
    # since HybridAgent uses its own REST clients
    # Actually, let's use openai-compatible provider
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    key = os.environ["AZURE_OPENAI_KEY"]
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5-chat")

    return SoulMemoryAdapter(
        provider="openai-compatible",
        model=deployment,
        mode=mode,
        api_key=key,
    )
