# Ingestion Pipeline Overhaul Plan for soul.py

## Problem Statement

soul.py currently stores raw conversation text into MEMORY.md. When asked to recall facts, it searches through verbatim dialog. This is why our LoCoMo benchmark scores (~63%) trail competitors like Mem0 (91.6%) and Xmem (91.5%).

**The bottleneck is NOT retrieval — it's ingestion.** Our RAG+RLM retrieval architecture is solid. The problem is what we're storing.

## What Competitors Do (Research Summary)

### Mem0's Architecture (from their docs + arXiv paper 2504.19413)
Five-stage extraction pipeline:
1. **Store New Memories** — Conversation enters asynchronously (after agent responds)
2. **Context Lookup** — Find related existing memories to avoid duplicates
3. **Distill Memories** — Single-pass LLM extraction produces ADD-only facts from input + context
4. **Deduplicate + Embed** — Hash-based dedup, then vectorize new memories
5. **Entity Linking** — Identify entities (proper nouns, quoted text) and link across memories

Three storage layers:
- **Vector DB**: Memory text + embeddings + metadata (timestamps, hash, categories)
- **Entity Store**: Entities + embeddings + linked memory IDs
- **SQL DB**: History log (ADD events) + rolling message window

Multi-signal retrieval (semantic + BM25 keyword + entity graph matching), fused via rank scoring.

**Key design choice:** ADD-only architecture. Never UPDATE or DELETE memories — preserves temporal context. Old algorithm lost 29.6% on temporal by updating/deleting.

### Xmem's Architecture
- **Multi-domain memory agents** — Specialized stores for personal facts, events, code, summaries, images
- **Judge-before-write** — Every memory passes through Judge agent: add, update, delete, or skip
- **Structured extraction** — Entities, dates, relationships extracted before storing

## Proposed Changes to soul.py

### Phase 1: Memory Extraction (Highest Impact)
**What:** Add an extraction step in `remember()` that converts raw conversation into atomic facts.

```python
# Before (current):
agent.remember("Alice: Hey Bob, I just got a job at Google! Starting next month in Tokyo.")

# Stored as: verbatim text

# After (proposed):
agent.remember("Alice: Hey Bob, I just got a job at Google! Starting next month in Tokyo.")

# Extracted and stored as:
# - Alice got a new job at Google
# - Alice will start working at Google next month
# - Alice is moving to/working in Tokyo
# - Alice told Bob about the job
```

**Implementation:**
- Add `MemoryExtractor` class in new file `memory_extractor.py`
- Uses LLM to extract atomic facts from conversation text
- Each fact gets metadata: `{source_text, timestamp, entities, confidence}`
- Called inside `remember()` before storage

**Prompt template:**
```
Extract atomic facts from this conversation. Each fact should be:
- One clear statement (subject-verb-object)
- Include temporal info if mentioned (dates, "next month", "yesterday")
- Include named entities (people, places, organizations)
- Include relationships between entities

Conversation:
{text}

Return as JSON array of facts.
```

### Phase 2: Deduplication
**What:** Before storing a new fact, check if it already exists (or contradicts existing facts).

**Implementation:**
- Semantic similarity check against existing memories (threshold ~0.9 for dupes)
- If near-duplicate found: skip or merge
- If contradiction found: keep both, mark older as `superseded_by` (from fengtrace discussion)
- Hash-based fast dedup for exact matches

### Phase 3: Temporal Indexing
**What:** Tag every memory with structured temporal metadata.

**Implementation:**
- Extract dates/times from conversation context (session timestamps, mentioned dates)
- Store as structured metadata: `{absolute_date, relative_context, session_id}`
- Enable temporal queries: "What happened in May 2023?" → filter by date range first, then semantic search

### Phase 4: Entity Linking
**What:** Extract entities and build a lightweight entity graph.

**Implementation:**
- NER on extracted facts (people, places, organizations, events)
- Store entity → memory_id mappings
- Query-time entity boost: if query mentions "Alice", boost memories linked to Alice entity
- Similar to Mem0's entity store but lighter weight

### Phase 5: Multi-Signal Retrieval
**What:** Combine BM25 + semantic + entity signals (we already have BM25 + semantic, just add entity).

**Implementation:**
- Rank fusion: weighted combination of BM25 score, vector similarity, entity match score
- Tunable weights per query type (factual → BM25, conceptual → semantic, entity-centric → entity)

## Implementation Priority

| Phase | Impact on LoCoMo | Effort | Dependencies |
|-------|-----------------|--------|-------------|
| 1. Memory Extraction | HIGH (est. +15-20%) | Medium | None |
| 2. Deduplication | MEDIUM (est. +5%) | Low | Phase 1 |
| 3. Temporal Indexing | HIGH for temporal category (est. +20% on temporal) | Medium | Phase 1 |
| 4. Entity Linking | MEDIUM (est. +5-10%) | Medium | Phase 1 |
| 5. Multi-Signal Retrieval | LOW (already have 2/3 signals) | Low | Phase 4 |

**Phase 1 alone should move us from ~63% to ~78-83%.** Phases 1-3 together could get us to ~85-90%.

## Backward Compatibility

- `remember(text)` API stays the same — extraction happens internally
- Add `remember(text, extract=True)` flag to opt in/out
- Raw text still stored alongside extracted facts (for RLM to reason over)
- Existing MEMORY.md format preserved; extracted facts get `## Fact:` prefix

## Cost Considerations

- Phase 1 adds 1 LLM call per `remember()` for extraction — acceptable for Gemini Flash (free) or Haiku (cheap)
- Phases 2-4 add retrieval calls but no extra LLM calls
- Net effect: slightly more expensive ingestion, same or cheaper retrieval (smaller, cleaner memory)

## Files to Create/Modify

| File | Action |
|------|--------|
| `memory_extractor.py` | NEW — fact extraction + dedup logic |
| `entity_store.py` | NEW — entity linking + graph |
| `temporal_index.py` | NEW — temporal metadata + filtering |
| `hybrid_agent.py` | MODIFY — wire extraction into remember() |
| `rag_memory.py` | MODIFY — add entity signal to retrieval |
| `soul.py` (base agent) | MODIFY — add extraction option |

## References

- Mem0 paper: https://arxiv.org/abs/2504.19413
- Mem0 docs: https://docs.mem0.ai/core-concepts/memory-evaluation
- Xmem repo: https://github.com/XortexAI/Xmem
- fengtrace discussion: https://github.com/menonpg/soul.py/issues/4
- LoCoMo: https://github.com/snap-research/locomo
- LongMemEval: https://github.com/xiaowu0162/LongMemEval
