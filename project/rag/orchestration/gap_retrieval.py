"""Gap-driven multi-round retrieval with novelty-aware ranking."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from project.config import get_config
from project.llm.client import LLMClient
from project.rag.simple_rag import SimpleRAGService
from project.rag.orchestration.novelty_ranker import NoveltyRanker

logger = logging.getLogger(__name__)
RETRIEVAL_MODES = {"local", "global", "mix"}
MAX_FACTS = 12


def build_writing_review_retrieval_state(*, prompt_text: str, essay_text: str, dataset_scope: str = "writing") -> dict[str, Any]:
    config = get_config()
    gaps = _build_writing_gaps(prompt_text, essay_text)
    focus_gap = _pick_focus_gap(gaps) or gaps[0]
    return {
        "task_type": "writing_review",
        "dataset_scope": dataset_scope,
        "prompt_text": prompt_text,
        "essay_text": essay_text,
        "round": 0,
        "max_rounds": config.retrieval_max_rounds,
        "max_no_progress_rounds": config.retrieval_max_no_progress_rounds,
        "top_k_per_round": config.retrieval_top_k_per_round,
        "selected_k": config.retrieval_selected_k,
        "duplicate_rate_threshold": config.retrieval_duplicate_rate_threshold,
        "gap_fill_target": config.retrieval_gap_fill_target,
        "gaps": gaps,
        "known_facts": [],
        "history_doc_ids": [],
        "history_chunk_keys": [],
        "history_queries": [],
        "retrieved_docs": [],
        "support_answer": "",
        "current_query": _build_gap_query(prompt_text, essay_text, focus_gap),
        "current_mode": "mix",
        "current_strategy": "broad_bootstrap",
        "no_progress_rounds": 0,
        "complete": False,
        "stop_reason": "",
        "retrieval_trace": [],
        "metrics": {"rounds": 0, "total_new_facts": 0, "avg_duplicate_rate": 0.0, "latency_ms": 0, "token_estimate": 0},
    }


def is_gap_retrieval_complete(retrieval_state: dict[str, Any]) -> bool:
    return bool(retrieval_state.get("complete"))


def run_gap_retrieval_round(retrieval_state: dict[str, Any]) -> dict[str, Any]:
    if retrieval_state.get("complete"):
        return retrieval_state

    config = get_config()
    service = SimpleRAGService.from_config()
    client = LLMClient.from_config()
    ranker = NoveltyRanker()
    gaps = list(retrieval_state.get("gaps", []))
    focus_gap = _pick_focus_gap(gaps)
    if not focus_gap:
        updated = dict(retrieval_state)
        updated["complete"] = True
        updated["stop_reason"] = "all_gaps_filled"
        return updated

    round_index = int(retrieval_state.get("round", 0)) + 1
    plan = _plan_round(retrieval_state, focus_gap, client)
    started = time.perf_counter()
    result = service.query(
        question=plan["query"],
        dataset_scope=str(retrieval_state.get("dataset_scope") or "writing"),
        top_k=int(retrieval_state.get("top_k_per_round", config.retrieval_top_k_per_round) or config.retrieval_top_k_per_round),
        mode=plan["mode"],
        user_id=str(retrieval_state.get("user_id") or "demo_user"),
        banned_doc_ids=list(retrieval_state.get("history_doc_ids", [])),
        banned_chunk_ids=list(retrieval_state.get("history_chunk_keys", [])),
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    candidates = list(result.get("retrieved_docs", []))
    history_docs = list(retrieval_state.get("retrieved_docs", []))
    ranked = ranker.rank(
        query=plan["query"],
        gap_text=str(focus_gap.get("description") or focus_gap.get("name") or ""),
        candidates=candidates,
        seen_texts=_doc_texts(history_docs),
        top_k=int(retrieval_state.get("selected_k", config.retrieval_selected_k) or config.retrieval_selected_k),
    )
    selected_docs = [item.doc for item in ranked]
    duplicate_rate = _duplicate_rate(selected_docs, list(retrieval_state.get("history_chunk_keys", [])))
    new_facts = _extract_new_facts(focus_gap, selected_docs, str(result.get("answer") or ""), list(retrieval_state.get("known_facts", [])))
    updated_gaps = _update_gap_statuses(gaps, str(focus_gap.get("id") or ""), new_facts, selected_docs, str(result.get("answer") or ""), client, retrieval_state)
    old_fill = _gap_fill_rate(gaps)
    new_fill = _gap_fill_rate(updated_gaps)
    progress = bool(new_facts) or new_fill > old_fill
    no_progress = 0 if progress else int(retrieval_state.get("no_progress_rounds", 0)) + 1

    all_docs = _merge_docs(history_docs, selected_docs)
    history_doc_ids = list(retrieval_state.get("history_doc_ids", []))
    history_chunk_keys = list(retrieval_state.get("history_chunk_keys", []))
    for item in selected_docs:
        doc_id = str(item.get("id") or item.get("source") or "")
        if doc_id and doc_id not in history_doc_ids:
            history_doc_ids.append(doc_id)
        for key in _chunk_keys(item):
            if key not in history_chunk_keys:
                history_chunk_keys.append(key)

    known_facts = list(retrieval_state.get("known_facts", []))
    for fact in new_facts:
        if fact not in known_facts:
            known_facts.append(fact)
    known_facts = known_facts[:MAX_FACTS]
    token_estimate = _estimate_tokens(plan["query"], selected_docs, result)
    trace = list(retrieval_state.get("retrieval_trace", []))
    trace.append({
        "round": round_index,
        "query": plan["query"],
        "mode": plan["mode"],
        "strategy": plan["strategy"],
        "focus_gap": focus_gap.get("id"),
        "enough_context": new_fill >= config.retrieval_gap_fill_target,
        "decision_source": plan["decision_source"],
        "reason": plan["reason"],
        "new_facts": len(new_facts),
        "duplicate_rate": round(duplicate_rate, 3),
        "gap_fill_rate": round(new_fill, 3),
        "selected_docs": len(selected_docs),
        "latency_ms": latency_ms,
        "token_estimate": token_estimate,
    })

    complete = False
    stop_reason = ""
    if new_fill >= config.retrieval_gap_fill_target:
        complete, stop_reason = True, "gap_fill_target_reached"
    elif no_progress >= config.retrieval_max_no_progress_rounds:
        complete, stop_reason = True, "no_progress_limit"
    elif duplicate_rate >= config.retrieval_duplicate_rate_threshold and not progress:
        complete, stop_reason = True, "duplicate_rate_too_high"
    elif round_index >= int(retrieval_state.get("max_rounds", config.retrieval_max_rounds) or config.retrieval_max_rounds):
        complete, stop_reason = True, "budget_exhausted"
    elif all(gap.get("status") == "filled" for gap in updated_gaps):
        complete, stop_reason = True, "all_gaps_filled"

    next_gap = _pick_focus_gap(updated_gaps) or focus_gap
    updated = dict(retrieval_state)
    updated.update({
        "round": round_index,
        "gaps": updated_gaps,
        "known_facts": known_facts,
        "retrieved_docs": all_docs,
        "support_answer": _support_answer(all_docs, result, known_facts),
        "current_query": _build_gap_query(str(retrieval_state.get("prompt_text") or ""), str(retrieval_state.get("essay_text") or ""), next_gap),
        "current_mode": _default_mode(next_gap, round_index + 1),
        "current_strategy": str(next_gap.get("id") or "gap_follow_up"),
        "no_progress_rounds": no_progress,
        "complete": complete,
        "stop_reason": stop_reason,
        "history_doc_ids": history_doc_ids,
        "history_chunk_keys": history_chunk_keys,
        "history_queries": list(retrieval_state.get("history_queries", [])) + [plan["query"]],
        "retrieval_trace": trace,
        "metrics": _update_metrics(retrieval_state.get("metrics", {}), len(new_facts), duplicate_rate, latency_ms, token_estimate),
    })
    return updated


def summarize_gap_retrieval_state(retrieval_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer": str(retrieval_state.get("support_answer") or ""),
        "retrieved_docs": list(retrieval_state.get("retrieved_docs", [])),
        "backend": "simple_rag",
        "query_mode": str(retrieval_state.get("current_mode") or "mix"),
        "retrieval_trace": list(retrieval_state.get("retrieval_trace", [])),
        "known_facts": list(retrieval_state.get("known_facts", [])),
        "gaps": list(retrieval_state.get("gaps", [])),
        "gap_fill_rate": round(_gap_fill_rate(list(retrieval_state.get("gaps", []))), 3),
        "stop_reason": str(retrieval_state.get("stop_reason") or ""),
        "metrics": dict(retrieval_state.get("metrics", {})),
    }


def apply_retrieval_state_to_review_state(review_state: dict[str, Any], retrieval_state: dict[str, Any]) -> dict[str, Any]:
    summary = summarize_gap_retrieval_state(retrieval_state)
    updated = dict(review_state)
    updated["retrieval_state"] = retrieval_state
    updated["retrieval_round"] = int(retrieval_state.get("round", 0) or 0)
    updated["max_rounds"] = int(retrieval_state.get("max_rounds", 0) or 0)
    updated["current_query"] = str(retrieval_state.get("current_query") or "")
    updated["current_mode"] = str(retrieval_state.get("current_mode") or "mix")
    updated["retrieved_docs"] = summary["retrieved_docs"]
    updated["support_answer"] = summary["answer"]
    updated["retrieval_trace"] = summary["retrieval_trace"]
    updated["rag_backend"] = summary["backend"]
    updated["rag_query_mode"] = summary["query_mode"]
    updated["enough_context"] = bool(retrieval_state.get("complete"))
    updated["retrieval_metrics"] = summary["metrics"]
    return updated


def _build_writing_gaps(prompt_text: str, essay_text: str) -> list[dict[str, Any]]:
    word_count = len(re.findall(r"\b[a-zA-Z][a-zA-Z'-]*\b", essay_text))
    return [
        {"id": "task_response", "name": "Task Response", "description": f"Find Task 2 scoring criteria and response expectations for: {prompt_text}", "priority": 1.0, "keywords": ["task response", "position", "addresses", "clear opinion", "support"], "status": "open", "attempts": 0},
        {"id": "revision_guidance", "name": "Revision Guidance", "description": "Find actionable revision guidance and weak-point diagnosis for this essay.", "priority": 0.95 if word_count < 220 else 0.8, "keywords": ["weakness", "improve", "revise", "develop", "examples"], "status": "open", "attempts": 0},
        {"id": "coherence", "name": "Coherence", "description": "Find paragraph structure and cohesion advice for this essay.", "priority": 0.9, "keywords": ["coherence", "cohesion", "paragraph", "logical", "linking"], "status": "open", "attempts": 0},
        {"id": "sample_language", "name": "Sample Language", "description": "Find useful Task 2 language and sample phrasing for this essay.", "priority": 0.75, "keywords": ["sample language", "formal", "vocabulary", "phrasing", "academic"], "status": "open", "attempts": 0},
    ]


def _pick_focus_gap(gaps: list[dict[str, Any]]) -> dict[str, Any] | None:
    open_gaps = [gap for gap in gaps if gap.get("status") != "filled"]
    if not open_gaps:
        return None
    return sorted(
        open_gaps,
        key=lambda item: float(item.get("priority", 0.0)) / (int(item.get("attempts", 0) or 0) + 1),
        reverse=True,
    )[0]


def _plan_round(retrieval_state: dict[str, Any], focus_gap: dict[str, Any], client: LLMClient) -> dict[str, Any]:
    prompt_text = str(retrieval_state.get("prompt_text") or "")
    essay_text = str(retrieval_state.get("essay_text") or "")
    round_index = int(retrieval_state.get("round", 0)) + 1
    heuristics = {"gap_id": str(focus_gap.get("id") or "revision_guidance"), "query": _build_gap_query(prompt_text, essay_text, focus_gap), "mode": _default_mode(focus_gap, round_index), "strategy": f"gap:{focus_gap.get('id', 'unknown')}", "decision_source": "heuristic", "reason": "gap_priority_follow_up"}
    if not client.is_configured:
        return heuristics
    response = client.generate_text(
        system_prompt="You are a retrieval planner. Return JSON only with keys: gap_id, query, mode, strategy, reason. Allowed mode: local, global, mix.",
        user_prompt=f"Round: {round_index}\nPrompt: {prompt_text}\nEssay excerpt: {essay_text[:600]}\nKnown facts: {json.dumps(retrieval_state.get('known_facts', [])[:6], ensure_ascii=False)}\nOpen gaps: {json.dumps(_open_gap_briefs(retrieval_state.get('gaps', [])), ensure_ascii=False)}\nHistory queries: {json.dumps(list(retrieval_state.get('history_queries', []))[-3:], ensure_ascii=False)}",
        temperature=0.0,
        max_tokens=220,
    )
    parsed = _parse_json_object(response)
    if not isinstance(parsed, dict):
        repaired = _repair_plan(response or "", heuristics)
        if repaired:
            repaired["decision_source"] = "llm_repaired"
            return repaired
        return heuristics
    mode = str(parsed.get("mode") or heuristics["mode"]).lower()
    if mode not in RETRIEVAL_MODES:
        mode = heuristics["mode"]
    gap_id = str(parsed.get("gap_id") or heuristics["gap_id"])
    target_gap = next((gap for gap in retrieval_state.get("gaps", []) if str(gap.get("id")) == gap_id), focus_gap)
    return {"gap_id": gap_id, "query": str(parsed.get("query") or _build_gap_query(prompt_text, essay_text, target_gap)).strip() or heuristics["query"], "mode": mode, "strategy": str(parsed.get("strategy") or f"gap:{gap_id}"), "decision_source": "llm", "reason": str(parsed.get("reason") or "llm_gap_plan")}


def _update_gap_statuses(gaps: list[dict[str, Any]], focus_gap_id: str, new_facts: list[str], selected_docs: list[dict[str, Any]], rag_answer: str, client: LLMClient, retrieval_state: dict[str, Any]) -> list[dict[str, Any]]:
    updated = []
    for gap in gaps:
        item = dict(gap)
        if str(gap.get("id")) == focus_gap_id:
            item["attempts"] = int(item.get("attempts", 0) or 0) + 1
            item["status"] = "filled" if _gap_filled(gap, new_facts, selected_docs, rag_answer, client, retrieval_state) else "open"
        updated.append(item)
    return updated


def _gap_filled(gap: dict[str, Any], new_facts: list[str], selected_docs: list[dict[str, Any]], rag_answer: str, client: LLMClient, retrieval_state: dict[str, Any]) -> bool:
    if client.is_configured:
        response = client.generate_text(
            system_prompt="You are a retrieval progress judge. Return JSON only with keys: gap_filled, reason.",
            user_prompt=f"Gap: {json.dumps(gap, ensure_ascii=False)}\nNew facts: {json.dumps(new_facts, ensure_ascii=False)}\nRetrieved snippets: {json.dumps(_doc_briefs(selected_docs), ensure_ascii=False)}\nRAG answer: {rag_answer[:800]}\nKnown facts: {json.dumps(retrieval_state.get('known_facts', [])[:8], ensure_ascii=False)}",
            temperature=0.0,
            max_tokens=120,
        )
        parsed = _parse_json_object(response)
        if isinstance(parsed, dict) and "gap_filled" in parsed:
            return bool(parsed.get("gap_filled"))
    if len(new_facts) >= 2:
        return True
    text = (" ".join(_doc_texts(selected_docs)) + " " + rag_answer).lower()
    hits = sum(1 for keyword in gap.get("keywords", []) if str(keyword).lower() in text)
    return hits >= max(2, min(3, len(gap.get("keywords", []))))


def _extract_new_facts(focus_gap: dict[str, Any], selected_docs: list[dict[str, Any]], rag_answer: str, known_facts: list[str]) -> list[str]:
    keywords = [str(item).lower() for item in focus_gap.get("keywords", [])]
    known_set = {item.strip().lower() for item in known_facts if item.strip()}
    sentences = []
    texts = _doc_texts(selected_docs) + ([rag_answer.strip()] if rag_answer.strip() else [])
    for text in texts:
        for part in re.split(r"(?<=[.!?。！？])\s+|\n+", text):
            item = part.strip()
            if len(item) >= 24:
                sentences.append(item)
    found = []
    for sentence in sentences:
        lowered = sentence.lower()
        if lowered in known_set:
            continue
        if keywords and not any(keyword in lowered for keyword in keywords):
            continue
        found.append(sentence)
        known_set.add(lowered)
        if len(found) >= 4:
            break
    return found


def _default_mode(gap: dict[str, Any], round_index: int) -> str:
    gap_id = str(gap.get("id") or "")
    if round_index <= 1:
        return "mix"
    if gap_id == "task_response":
        return "local"
    if gap_id == "revision_guidance":
        return "global"
    return "mix"


def _build_gap_query(prompt_text: str, essay_text: str, gap: dict[str, Any]) -> str:
    essay_preview = essay_text.strip().replace("\n", " ")[:260]
    return f"{prompt_text}\n\nStudent draft excerpt: {essay_preview}\nCurrent gap: {gap.get('description', gap.get('name', 'support this essay'))}\nRetrieve the most useful IELTS Task 2 evidence for filling this gap."


def _merge_docs(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged, seen = [], set()
    for item in existing + incoming:
        key = _doc_identity(item)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _doc_identity(item: dict[str, Any]) -> str:
    joined = " ".join(chunk.strip() for chunk in item.get("chunks", []) if isinstance(chunk, str) and chunk.strip())
    return f"{item.get('id', '')}|{item.get('source', '')}|{joined[:220]}"


def _chunk_keys(item: dict[str, Any]) -> list[str]:
    keys = [f"{item.get('id', item.get('source', ''))}:{i}:{chunk.strip()[:100]}" for i, chunk in enumerate(item.get("chunks", [])) if isinstance(chunk, str) and chunk.strip()]
    return keys or [_doc_identity(item)]


def _duplicate_rate(selected_docs: list[dict[str, Any]], history_chunk_keys: list[str]) -> float:
    if not selected_docs:
        return 1.0
    history = set(history_chunk_keys)
    total = duplicate = 0
    for item in selected_docs:
        for key in _chunk_keys(item):
            total += 1
            if key in history:
                duplicate += 1
    return duplicate / max(total, 1)


def _gap_fill_rate(gaps: list[dict[str, Any]]) -> float:
    return 1.0 if not gaps else sum(1 for item in gaps if item.get("status") == "filled") / len(gaps)


def _support_answer(docs: list[dict[str, Any]], result: dict[str, Any], known_facts: list[str]) -> str:
    if known_facts:
        return "Known support gathered across rounds:\n" + "\n".join(f"- {fact}" for fact in known_facts[:8])
    answer = str(result.get("answer") or "").strip()
    if answer:
        return answer
    chunks = []
    for item in docs[:5]:
        chunks.extend([chunk.strip() for chunk in item.get("chunks", []) if isinstance(chunk, str) and chunk.strip()])
    return "\n".join(f"- {chunk}" for chunk in chunks[:6])


def _update_metrics(current: dict[str, Any], new_facts: int, duplicate_rate: float, latency_ms: int, token_estimate: int) -> dict[str, Any]:
    rounds = int(current.get("rounds", 0) or 0) + 1
    avg_duplicate = ((float(current.get("avg_duplicate_rate", 0.0) or 0.0) * (rounds - 1)) + duplicate_rate) / rounds
    return {"rounds": rounds, "total_new_facts": int(current.get("total_new_facts", 0) or 0) + new_facts, "avg_duplicate_rate": round(avg_duplicate, 3), "latency_ms": int(current.get("latency_ms", 0) or 0) + latency_ms, "token_estimate": int(current.get("token_estimate", 0) or 0) + token_estimate}


def _estimate_tokens(query: str, selected_docs: list[dict[str, Any]], result: dict[str, Any]) -> int:
    chars = len(query) + len(str(result.get("answer") or ""))
    for item in selected_docs:
        for chunk in item.get("chunks", []):
            if isinstance(chunk, str):
                chars += len(chunk)
    return max(1, chars // 4)


def _open_gap_briefs(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"id": gap.get("id"), "description": gap.get("description"), "priority": gap.get("priority"), "status": gap.get("status")} for gap in gaps if gap.get("status") != "filled"]


def _repair_plan(response: str, heuristics: dict[str, Any]) -> dict[str, Any] | None:
    if not response.strip():
        return None
    repaired = dict(heuristics)
    mode_match = re.search(r"\b(local|global|mix)\b", response, flags=re.IGNORECASE)
    if mode_match:
        repaired["mode"] = mode_match.group(1).lower()
    gap_match = re.search(r"\b(task_response|coherence|revision_guidance|sample_language)\b", response)
    if gap_match:
        repaired["gap_id"] = gap_match.group(1)
        repaired["strategy"] = f"gap:{gap_match.group(1)}"
    query_match = re.search(r"query\s*[:=]\s*(.+)", response, flags=re.IGNORECASE)
    if query_match:
        repaired["query"] = query_match.group(1).strip()
    repaired["reason"] = "heuristic_repair"
    return repaired


def _parse_json_object(response: str | None) -> dict[str, Any] | None:
    if not response:
        return None
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            logger.warning("Failed to parse retrieval JSON response: %s", response[:80])
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("Failed to parse retrieval JSON response: %s", response[:80])
            return None
    return parsed if isinstance(parsed, dict) else None


def _doc_texts(docs: list[dict[str, Any]]) -> list[str]:
    return [" ".join(chunk.strip() for chunk in item.get("chunks", []) if isinstance(chunk, str) and chunk.strip()) for item in docs if item.get("chunks")]


def _doc_briefs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"source": item.get("source"), "preview": " ".join(chunk.strip() for chunk in item.get("chunks", [])[:2] if isinstance(chunk, str) and chunk.strip())[:220]} for item in docs[:4]]
