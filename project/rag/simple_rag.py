"""Simple local RAG service backed by workspace documents."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import math
import re
from typing import Any

from project.rag.chunking import chunk_document

SUPPORTED_SUFFIXES = {".txt", ".md", ".html", ".htm", ".pdf"}
MAX_FILES = 40
BM25_K1 = 1.4
BM25_B = 0.72
RERANK_POOL_MIN = 30


@dataclass(slots=True)
class IndexedChunk:
    """Minimal local retrieval chunk."""

    id: str
    source: str
    text: str
    doc_type: str
    tokens: dict[str, float]
    metadata: dict[str, Any]


@dataclass(slots=True)
class RankedChunk:
    """Chunk plus retrieval-stage scores."""

    item: IndexedChunk
    bm25_score: float
    metadata_boost: float
    rerank_score: float
    final_score: float


class SimpleRAGService:
    """Small local lexical RAG with chunk and document-level retrieval."""

    @classmethod
    def from_config(cls) -> "SimpleRAGService":
        return cls()

    def query(
        self,
        question: str,
        *,
        top_k: int = 5,
        mode: str | None = None,
        dataset_scope: str | None = None,
        filters: dict[str, Any] | None = None,
        user_id: str | None = None,
        banned_doc_ids: list[str] | None = None,
        banned_chunk_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        query_mode = (mode or "mix").lower()
        chunks = list(_load_index())
        chunks = [
            item
            for item in chunks
            if _is_accessible(item, user_id or "demo_user")
            and not _is_banned(item, set(banned_doc_ids or []), set(banned_chunk_ids or []))
        ]
        if dataset_scope:
            filtered = [item for item in chunks if dataset_scope.lower() in item.doc_type or dataset_scope.lower() in item.source.lower()]
            chunks = filtered or chunks
        if filters:
            chunks = [item for item in chunks if _matches_filters(item, filters)]
        if not chunks:
            return {
                "question": question,
                "answer": "No relevant local study materials were found for this scope.",
                "retrieved_docs": [],
                "backend": "simple_rag",
                "query_mode": query_mode,
                "mode": "local",
                "message": "local_scope_empty",
            }

        selected = _rank_chunks(question, chunks, top_k, query_mode)

        retrieved_docs = [
            {
                "id": ranked.item.id,
                "source": ranked.item.source,
                "chunks": [ranked.item.text],
                "score": round(ranked.final_score, 4),
                "metadata": ranked.item.metadata,
                "ranking": {
                    "bm25": round(ranked.bm25_score, 4),
                    "metadata_boost": round(ranked.metadata_boost, 4),
                    "rerank": round(ranked.rerank_score, 4),
                },
            }
            for ranked in selected
        ]
        answer = _build_answer(question, [ranked.item for ranked in selected])
        return {
            "question": question,
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "backend": "simple_rag",
            "query_mode": query_mode,
            "mode": "local",
            "ranker": "bm25_metadata_rerank",
            "message": "OK",
        }


@lru_cache(maxsize=1)
def _load_index() -> tuple[IndexedChunk, ...]:
    from project.rag.local_index import load_persistent_index

    persistent_index = load_persistent_index()
    if persistent_index:
        return persistent_index

    return _build_runtime_index()


def _build_runtime_index() -> tuple[IndexedChunk, ...]:
    roots = [Path.cwd() / "data", Path.cwd() / "project" / "data"]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.name.startswith("_") or any(part in {"rag_index", "chunk_previews", "chunk_eval_reports"} for part in path.parts):
                continue
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                files.append(path)
    files = sorted(files)[:MAX_FILES]

    indexed: list[IndexedChunk] = []
    for path in files:
        try:
            chunks = chunk_document(path, strategy="auto", chunk_size=900, overlap=120)
        except Exception:
            continue
        for chunk in chunks[:30]:
            text = chunk.content.strip()
            if not text:
                continue
            source = str(path.relative_to(Path.cwd())).replace("\\", "/")
            indexed.append(
                IndexedChunk(
                    id=chunk.chunk_id,
                    source=source,
                    text=text,
                    doc_type=_infer_doc_type(path, text),
                    tokens=_token_vector(text),
                    metadata=_build_chunk_metadata(path, chunk, text),
                )
            )
    return tuple(indexed)


def _build_chunk_metadata(path: Path, chunk: Any, text: str) -> dict[str, Any]:
    owner_id, visibility = _infer_owner_visibility(path)
    keywords = _top_keywords(text)
    publication, issue_date, issue_folder = _infer_publication_issue(path)
    return {
        "doc_id": path.stem,
        "chunk_id": chunk.chunk_id,
        "chunk_type": _infer_chunk_type(chunk.strategy, text),
        "strategy": chunk.strategy,
        "article_title": path.stem,
        "section_title": chunk.title,
        "publication": publication,
        "issue_date": issue_date,
        "issue_folder": issue_folder,
        "topic": _infer_topic(path, text, keywords),
        "paragraph_role": _infer_paragraph_role(text),
        "stance": _infer_stance(text),
        "register": _infer_register(text),
        "sentence_pattern": _infer_sentence_pattern(text),
        "keywords": keywords,
        "entities": _extract_entities(text),
        "visibility": visibility,
        "owner_id": owner_id,
        **dict(getattr(chunk, "metadata", {}) or {}),
    }


def _infer_doc_type(path: Path, text: str) -> str:
    lowered = path.name.lower()
    lowered_text = text.lower()
    if (
        "writing" in lowered
        or "descriptor" in lowered
        or "task 2" in lowered
        or "essay" in lowered_text
        or "to what extent" in lowered_text
        or "discuss both" in lowered_text
        or "give reasons" in lowered_text
        or "task response" in lowered_text
        or "coherence and cohesion" in lowered_text
        or "lexical resource" in lowered_text
        or "grammatical range" in lowered_text
    ):
        return "writing"
    if "reading" in lowered or "not given" in lowered_text or "true / false" in lowered_text:
        return "reading"
    if "mistake" in lowered or "review" in lowered:
        return "mistakes"
    return "general"


def _infer_owner_visibility(path: Path) -> tuple[str, str]:
    parts = [part.lower() for part in path.parts]
    if "users" in parts:
        index = parts.index("users")
        if index + 1 < len(path.parts):
            return path.parts[index + 1], "private"
    return "public", "public"


def _infer_publication_issue(path: Path) -> tuple[str, str, str]:
    parts = path.parts
    marker = "awesome-english-ebooks"
    if marker not in parts:
        return "", "", ""
    index = parts.index(marker)
    publication_folder = parts[index + 1] if index + 1 < len(parts) else ""
    issue_folder = parts[index + 2] if index + 2 < len(parts) else ""
    publication = re.sub(r"^\d+[_-]*", "", publication_folder).strip("_-").lower()
    issue_date = _extract_issue_date(issue_folder) or _extract_issue_date(path.name)
    return publication, issue_date, issue_folder


def _extract_issue_date(value: str) -> str:
    match = re.search(r"(20\d{2})[.-](\d{1,2})[.-](\d{1,2})", value)
    if not match:
        return ""
    year, month, day = match.groups()
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def _is_accessible(item: IndexedChunk, user_id: str) -> bool:
    visibility = str(item.metadata.get("visibility") or "public")
    owner_id = str(item.metadata.get("owner_id") or "public")
    return visibility == "public" or owner_id == user_id


def _is_banned(item: IndexedChunk, banned_doc_ids: set[str], banned_chunk_ids: set[str]) -> bool:
    if item.id in banned_doc_ids or item.source in banned_doc_ids:
        return True
    if item.id in banned_chunk_ids:
        return True
    prefixes = (f"{item.id}:", f"{item.source}:")
    return any(key.startswith(prefixes) for key in banned_chunk_ids)


def _matches_filters(item: IndexedChunk, filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if expected in (None, "", []):
            continue
        actual = item.metadata.get(key)
        if isinstance(expected, list):
            if actual not in expected and not (isinstance(actual, list) and set(actual) & set(expected)):
                return False
        elif isinstance(actual, list):
            if str(expected).lower() not in {str(value).lower() for value in actual}:
                return False
        elif str(actual or "").lower() != str(expected).lower():
            return False
    return True


def _infer_chunk_type(strategy: str, text: str) -> str:
    if strategy == "rubric_items":
        return "rubric"
    if strategy == "qa_pairs":
        return "qa_pair"
    if len(re.findall(r"[.!?。！？]", text)) <= 2 and len(text) < 320:
        return "sentence"
    return "paragraph"


def _infer_topic(path: Path, text: str, keywords: list[str]) -> str:
    lowered = f"{path.name} {text}".lower()
    topics = {
        "writing": ("writing", "task 2", "essay", "coherence", "lexical", "grammar"),
        "reading": ("reading", "passage", "true", "false", "not given"),
        "speaking": ("speaking", "fluency", "pronunciation"),
        "listening": ("listening", "audio", "section"),
    }
    for topic, markers in topics.items():
        if any(marker in lowered for marker in markers):
            return topic
    return keywords[0] if keywords else "general"


def _infer_paragraph_role(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("for example", "for instance", "such as", "例如")):
        return "example"
    if any(token in lowered for token in ("however", "although", "while", "nevertheless", "despite", "然而")):
        return "concession"
    if any(token in lowered for token in ("because", "therefore", "as a result", "原因", "因此")):
        return "reason"
    if any(token in lowered for token in ("in conclusion", "to conclude", "overall", "总结")):
        return "conclusion"
    if any(token in lowered for token in ("argue", "believe", "think", "claim", "观点")):
        return "claim"
    return "neutral"


def _infer_stance(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("however", "although", "on the other hand", "balanced", "partly")):
        return "balanced"
    if any(token in lowered for token in ("advantage", "benefit", "support", "agree")):
        return "pro"
    if any(token in lowered for token in ("disadvantage", "risk", "oppose", "disagree")):
        return "con"
    return "neutral"


def _infer_register(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("band", "criterion", "descriptor", "task response")):
        return "analytical"
    if any(token in lowered for token in ("study", "practice", "student", "ielts")):
        return "formal"
    return "neutral"


def _infer_sentence_pattern(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("because", "therefore", "as a result", "lead to")):
        return "cause-effect"
    if any(token in lowered for token in ("however", "although", "while", "despite")):
        return "concession"
    if any(token in lowered for token in ("for example", "for instance", "such as")):
        return "example"
    if any(token in lowered for token in ("compared with", "whereas", "than")):
        return "comparison"
    if any(token in lowered for token in ("in conclusion", "overall", "to sum up")):
        return "summary"
    return "general"


def _top_keywords(text: str, limit: int = 8) -> list[str]:
    stopwords = {
        "the", "and", "for", "that", "with", "this", "from", "are", "you", "your",
        "ielts", "task", "will", "can", "into", "about", "which", "their",
    }
    counts: dict[str, int] = {}
    for token in re.findall(r"[a-z][a-z'-]{2,}|[\u4e00-\u9fff]{2,}", text.lower()):
        if token in stopwords:
            continue
        counts[token] = counts.get(token, 0) + 1
    return [token for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def _extract_entities(text: str, limit: int = 8) -> list[str]:
    entities = re.findall(r"\b[A-Z][A-Za-z0-9&'-]*(?:\s+[A-Z][A-Za-z0-9&'-]*){0,3}\b", text)
    seen: list[str] = []
    for entity in entities:
        if entity not in seen and entity.lower() not in {"the", "this"}:
            seen.append(entity)
        if len(seen) >= limit:
            break
    return seen


def _rank_chunks(question: str, chunks: list[IndexedChunk], top_k: int, mode: str) -> list[RankedChunk]:
    query_terms = _tokenize(question)
    if not query_terms:
        return []

    stats = _bm25_stats(chunks, set(query_terms))
    prelim = [
        RankedChunk(
            item=item,
            bm25_score=_bm25_score(item, query_terms, stats),
            metadata_boost=_metadata_boost(question, query_terms, item),
            rerank_score=0.0,
            final_score=0.0,
        )
        for item in chunks
    ]
    for ranked in prelim:
        ranked.final_score = ranked.bm25_score + ranked.metadata_boost

    pool = _candidate_pool(prelim, max(top_k, 1), mode)
    if not pool:
        pool = sorted(prelim, key=lambda item: item.final_score, reverse=True)[: max(top_k, 1)]

    bm25_max = max((item.bm25_score for item in pool), default=0.0) or 1.0
    boost_max = max((item.metadata_boost for item in pool), default=0.0) or 1.0
    for ranked in pool:
        ranked.rerank_score = _rerank_score(question, query_terms, ranked.item)
    rerank_max = max((item.rerank_score for item in pool), default=0.0) or 1.0

    for ranked in pool:
        bm25 = ranked.bm25_score / bm25_max
        boost = ranked.metadata_boost / boost_max
        rerank = ranked.rerank_score / rerank_max
        ranked.final_score = _weighted_final_score(bm25, boost, rerank, mode)

    return _dedupe_ranked(sorted(pool, key=lambda item: item.final_score, reverse=True), top_k)


def _candidate_pool(scored: list[RankedChunk], top_k: int, mode: str) -> list[RankedChunk]:
    pool_size = max(RERANK_POOL_MIN, top_k * 8)
    sorted_scored = sorted(scored, key=lambda item: item.final_score, reverse=True)
    if mode == "local":
        return sorted_scored[:pool_size]

    source_scores: dict[str, float] = {}
    for ranked in sorted_scored:
        source_scores[ranked.item.source] = max(source_scores.get(ranked.item.source, 0.0), ranked.final_score)
    best_sources = {
        source
        for source, _ in sorted(source_scores.items(), key=lambda item: item[1], reverse=True)[: max(top_k * 3, 6)]
    }
    global_pool = [ranked for ranked in sorted_scored if ranked.item.source in best_sources][: pool_size * 2]
    if mode == "global":
        return global_pool

    merged: list[RankedChunk] = []
    seen: set[str] = set()
    for ranked in sorted_scored[:pool_size] + global_pool:
        key = f"{ranked.item.source}:{ranked.item.id}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(ranked)
        if len(merged) >= pool_size * 2:
            break
    return merged


def _bm25_stats(chunks: list[IndexedChunk], query_terms: set[str]) -> dict[str, Any]:
    doc_count = max(len(chunks), 1)
    lengths = [_doc_length(item) for item in chunks]
    avgdl = sum(lengths) / max(len(lengths), 1)
    dfs: dict[str, int] = {}
    for term in query_terms:
        dfs[term] = sum(1 for item in chunks if item.tokens.get(term, 0.0) > 0.0)
    return {"doc_count": doc_count, "avgdl": max(avgdl, 1.0), "dfs": dfs}


def _bm25_score(item: IndexedChunk, query_terms: list[str], stats: dict[str, Any]) -> float:
    doc_count = int(stats["doc_count"])
    avgdl = float(stats["avgdl"])
    doc_length = _doc_length(item)
    score = 0.0
    for term in query_terms:
        tf = float(item.tokens.get(term, 0.0))
        if tf <= 0.0:
            continue
        df = int(stats["dfs"].get(term, 0))
        idf = math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
        denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * doc_length / avgdl)
        score += idf * (tf * (BM25_K1 + 1.0)) / max(denom, 1e-9)
    return score


def _metadata_boost(question: str, query_terms: list[str], item: IndexedChunk) -> float:
    query_set = set(query_terms)
    metadata = item.metadata
    weighted_fields = (
        ("article_title", 0.45),
        ("section_title", 0.6),
        ("publication", 0.4),
        ("issue_date", 0.25),
        ("topic", 0.45),
        ("chunk_type", 0.25),
        ("paragraph_role", 0.25),
        ("stance", 0.2),
        ("register", 0.2),
        ("sentence_pattern", 0.25),
        ("keywords", 0.55),
        ("entities", 0.35),
    )
    boost = _field_overlap(query_set, item.doc_type, 0.5)
    boost += _field_overlap(query_set, item.source, 0.25)
    for field, weight in weighted_fields:
        boost += _field_overlap(query_set, metadata.get(field), weight)

    lowered_question = question.lower()
    title_text = f"{metadata.get('article_title', '')} {metadata.get('section_title', '')}".lower()
    if title_text and any(phrase in title_text for phrase in _query_phrases(lowered_question)):
        boost += 0.45
    if str(metadata.get("chunk_type") or "") in {"rubric", "qa_pair"}:
        boost += _intent_boost(lowered_question, item)
    return min(boost, 3.0)


def _rerank_score(question: str, query_terms: list[str], item: IndexedChunk) -> float:
    query_vec = _token_vector(question)
    text = item.text.lower()
    source = item.source.lower()
    metadata_text = _metadata_text(item.metadata).lower()
    score = 0.55 * _cosine(query_vec, item.tokens)
    score += 0.2 * _phrase_score(question.lower(), text)
    score += 0.15 * _phrase_score(question.lower(), metadata_text)
    score += 0.2 * _proximity_score(query_terms, text)
    if any(term in source for term in set(query_terms)):
        score += 0.08
    score -= _length_penalty(item.text)
    return max(score, 0.0)


def _weighted_final_score(bm25: float, boost: float, rerank: float, mode: str) -> float:
    if mode == "global":
        return 0.5 * bm25 + 0.3 * boost + 0.2 * rerank
    if mode == "local":
        return 0.62 * bm25 + 0.18 * boost + 0.2 * rerank
    return 0.55 * bm25 + 0.25 * boost + 0.2 * rerank


def _dedupe_ranked(items: list[RankedChunk], top_k: int) -> list[RankedChunk]:
    selected: list[RankedChunk] = []
    seen: set[str] = set()
    for ranked in items:
        key = f"{ranked.item.source}:{ranked.item.id}"
        if key in seen:
            continue
        seen.add(key)
        selected.append(ranked)
        if len(selected) >= top_k:
            break
    return selected


def _build_answer(question: str, chunks: list[IndexedChunk]) -> str:
    if not chunks:
        return f"No relevant context found for: {question}"
    lines = []
    for index, item in enumerate(chunks[:4], start=1):
        lines.append(f"[{index}] {item.source}: {item.text[:220]}")
    return "Local RAG support:\n" + "\n".join(lines)


def _doc_length(item: IndexedChunk) -> float:
    return max(sum(item.tokens.values()), 1.0)


def _field_overlap(query_terms: set[str], value: Any, weight: float) -> float:
    if value in (None, "", []):
        return 0.0
    if isinstance(value, list):
        field_text = " ".join(str(item) for item in value)
    else:
        field_text = str(value)
    field_terms = set(_tokenize(field_text))
    if not field_terms:
        return 0.0
    return weight * len(query_terms & field_terms) / max(len(query_terms), 1)


def _metadata_text(metadata: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "article_title",
        "section_title",
        "topic",
        "chunk_type",
        "paragraph_role",
        "stance",
        "register",
        "sentence_pattern",
        "keywords",
        "entities",
    ):
        value = metadata.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value not in (None, ""):
            parts.append(str(value))
    return " ".join(parts)


def _intent_boost(question: str, item: IndexedChunk) -> float:
    boost = 0.0
    if any(token in question for token in ("band", "score", "descriptor", "criterion", "criteria")):
        boost += 0.25 if str(item.metadata.get("register")) == "analytical" else 0.0
        boost += 0.15 if item.doc_type == "writing" else 0.0
    if any(token in question for token in ("question", "task", "sample", "practice")):
        boost += 0.2 if str(item.metadata.get("chunk_type")) == "qa_pair" else 0.0
    if any(token in question for token in ("mistake", "wrong", "error", "review")):
        boost += 0.2 if item.doc_type == "mistakes" else 0.0
    return boost


def _query_phrases(question: str) -> list[str]:
    latin_phrases = re.findall(r"[a-z0-9][a-z0-9' -]{4,}[a-z0-9]", question)
    return [phrase.strip() for phrase in latin_phrases if len(phrase.split()) >= 2][:6]


def _phrase_score(question: str, text: str) -> float:
    score = 0.0
    for phrase in _query_phrases(question):
        if phrase and phrase in text:
            score += min(0.18, 0.03 * len(phrase.split()))
    return min(score, 0.45)


def _proximity_score(query_terms: list[str], text: str) -> float:
    unique_terms = list(dict.fromkeys(query_terms))
    if len(unique_terms) < 2:
        return 0.0
    tokens = _tokenize(text)
    positions: dict[str, list[int]] = {term: [] for term in unique_terms}
    for index, token in enumerate(tokens):
        if token in positions:
            positions[token].append(index)
    active = [values for values in positions.values() if values]
    if len(active) < 2:
        return 0.0

    flattened = sorted((pos, term_index) for term_index, values in enumerate(active) for pos in values)
    counts: dict[int, int] = {}
    left = 0
    best_span: int | None = None
    for right, (pos, term_index) in enumerate(flattened):
        counts[term_index] = counts.get(term_index, 0) + 1
        while len(counts) == len(active):
            span = pos - flattened[left][0]
            best_span = span if best_span is None else min(best_span, span)
            left_term = flattened[left][1]
            counts[left_term] -= 1
            if counts[left_term] == 0:
                del counts[left_term]
            left += 1
    if best_span is None:
        return 0.0
    if best_span <= 12:
        return 0.35
    if best_span <= 35:
        return 0.2
    if best_span <= 80:
        return 0.1
    return 0.0


def _length_penalty(text: str) -> float:
    length = len(text)
    if 180 <= length <= 1500:
        return 0.0
    if length < 80:
        return 0.08
    return min((length - 1500) / 6000.0, 0.18)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


def _token_vector(text: str) -> dict[str, float]:
    tokens = _tokenize(text)
    counts: dict[str, float] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0.0) + 1.0
    return counts


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    overlap = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in overlap)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)
