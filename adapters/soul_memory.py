"""Adapter wrapping soul.py HybridAgent for benchmarks.

Uses the local HybridAgent (RAG + RLM) from the soul.py repo directly.
Supports BM25, Qdrant (semantic), RLM, Qdrant+RLM, and Auto modes.
"""

from __future__ import annotations

import hashlib
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


def _load_secrets():
    """Load secrets from azure-openai.env and api_keys.json."""
    env_file = Path.home() / "clawd" / "secrets" / "azure-openai.env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    import json
    keys_file = Path.home() / "clawd" / "secrets" / "api_keys.json"
    if keys_file.exists():
        data = json.loads(keys_file.read_text())
        qdrant = data.get("qdrant", {})
        if qdrant.get("url"):
            os.environ.setdefault("QDRANT_URL", qdrant["url"] + ":6333")
        if qdrant.get("api_key"):
            os.environ.setdefault("QDRANT_API_KEY", qdrant["api_key"])


# Load secrets on import
_load_secrets()


class SoulMemoryAdapter:
    """Wraps HybridAgent for benchmark evaluation."""

    def __init__(
        self,
        provider: str = "gemini",
        model: str | None = None,
        mode: MemoryMode = "auto",
        api_key: str | None = None,
        use_qdrant: bool = False,
        collection_name: str | None = None,
    ):
        self.mode = mode
        self.provider = provider
        self.use_qdrant = use_qdrant
        self._tmpdir = tempfile.mkdtemp(prefix="soul-bench-")

        soul_path = Path(self._tmpdir) / "SOUL.md"
        memory_path = Path(self._tmpdir) / "MEMORY.md"
        soul_path.write_text(
            "You are a memory-enabled assistant for benchmark evaluation. "
            "Answer questions precisely and concisely based on what you remember. "
            "Give short, factual answers. If you don't know, say so.\n"
        )
        memory_path.write_text("# MEMORY.md\n")

        # Generate unique collection name for this benchmark run
        if not collection_name:
            uid = hashlib.md5(self._tmpdir.encode()).hexdigest()[:8]
            collection_name = f"locomo_bench_{uid}"
        self.collection_name = collection_name

        kwargs = dict(
            soul_path=str(soul_path),
            memory_path=str(memory_path),
            provider=provider,
            mode=mode,
            collection_name=collection_name,
        )
        if model:
            kwargs["chat_model"] = model
        if api_key:
            kwargs["api_key"] = api_key

        # If using Qdrant, pass credentials
        if use_qdrant:
            kwargs["qdrant_url"] = os.environ.get("QDRANT_URL", "")
            kwargs["qdrant_api_key"] = os.environ.get("QDRANT_API_KEY", "")
            kwargs["azure_embedding_endpoint"] = os.environ.get("AZURE_EMBEDDING_ENDPOINT", "")
            kwargs["azure_embedding_key"] = os.environ.get("AZURE_EMBEDDING_KEY", "")

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

    def cleanup_collection(self):
        """Delete the Qdrant collection created for this benchmark run."""
        if self.use_qdrant and self.collection_name.startswith("locomo_bench_"):
            try:
                import requests
                qdrant_url = os.environ.get("QDRANT_URL", "")
                qdrant_key = os.environ.get("QDRANT_API_KEY", "")
                if qdrant_url:
                    requests.delete(
                        f"{qdrant_url}/collections/{self.collection_name}",
                        headers={"api-key": qdrant_key},
                        timeout=10,
                    )
                    print(f"  Cleaned up collection: {self.collection_name}")
            except Exception as e:
                print(f"  Warning: failed to cleanup collection: {e}")


# ── Factory functions for each benchmark configuration ──

def create_bm25_adapter() -> SoulMemoryAdapter:
    """Config 1: BM25 keyword search only (no ML baseline)."""
    return SoulMemoryAdapter(
        provider="gemini",
        mode="rag",  # RAG mode but falls back to BM25 without Qdrant
        use_qdrant=False,
    )


def create_qdrant_adapter(collection_suffix: str = "") -> SoulMemoryAdapter:
    """Config 2: Qdrant semantic search only (no RLM)."""
    uid = hashlib.md5(os.urandom(8)).hexdigest()[:8]
    return SoulMemoryAdapter(
        provider="gemini",
        mode="rag",
        use_qdrant=True,
        collection_name=f"locomo_bench_qdrant_{uid}{collection_suffix}",
    )


def create_rlm_adapter() -> SoulMemoryAdapter:
    """Config 3: RLM only (pure LLM reasoning, no retrieval)."""
    return SoulMemoryAdapter(
        provider="gemini",
        mode="rlm",
        use_qdrant=False,
    )


def create_hybrid_adapter(collection_suffix: str = "") -> SoulMemoryAdapter:
    """Config 4: Qdrant + RLM (full hybrid — our flagship)."""
    uid = hashlib.md5(os.urandom(8)).hexdigest()[:8]
    return SoulMemoryAdapter(
        provider="gemini",
        mode="auto",
        use_qdrant=True,
        collection_name=f"locomo_bench_hybrid_{uid}{collection_suffix}",
    )


def create_auto_adapter(collection_suffix: str = "") -> SoulMemoryAdapter:
    """Config 5: Auto mode with Qdrant (router picks per query)."""
    uid = hashlib.md5(os.urandom(8)).hexdigest()[:8]
    return SoulMemoryAdapter(
        provider="gemini",
        mode="auto",
        use_qdrant=True,
        collection_name=f"locomo_bench_auto_{uid}{collection_suffix}",
    )


# Convenience aliases
CONFIGS = {
    "bm25": create_bm25_adapter,
    "qdrant": create_qdrant_adapter,
    "rlm": create_rlm_adapter,
    "hybrid": create_hybrid_adapter,
    "auto": create_auto_adapter,
}
