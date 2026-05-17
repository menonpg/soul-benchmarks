"""Adapter wrapping soul-agent as a memory backend for benchmarks."""

from __future__ import annotations

from typing import Literal

from soul import Soul


MemoryMode = Literal["rag", "rlm", "auto"]


class SoulMemoryAdapter:
    """Thin wrapper around soul.py for benchmark integration."""

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
        mode: MemoryMode = "auto",
        user_id: str = "benchmark-user",
    ):
        self.mode = mode
        self.soul = Soul(
            provider=provider,
            model=model,
            user_id=user_id,
        )

    def add_memory(self, conversation_text: str) -> None:
        """Ingest conversation text into memory via soul.remember()."""
        self.soul.remember(conversation_text)

    def query_memory(self, question: str) -> str:
        """Query memory and return answer via soul.ask()."""
        response = self.soul.ask(question)
        return response if isinstance(response, str) else str(response)

    def reset(self) -> None:
        """Reset memory state for a fresh benchmark run."""
        # Re-initialize to clear state
        self.soul = Soul(
            provider=self.soul.provider,
            model=self.soul.model,
            user_id=self.soul.user_id,
        )
