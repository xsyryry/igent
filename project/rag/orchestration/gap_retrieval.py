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


def build_writing_review_retrieval_state(*, prompt_text: str, essay_text: str, dataset_scope: str = "magazine") -> dict[str, Any]:
    config = get_config()
    writing_diagnosis = _diagnose_writing_issues(prompt_text, essay_text)
    gaps = _build_writing_gaps(prompt_text, essay_text, writing_diagnosis)
    focus_gap = _pick_focus_gap(gaps) or gaps[0]
    return {
        "task_type": "writing_review",
        "dataset_scope": dataset_scope,
        "writing_diagnosis": writing_diagnosis,
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
    if _is_writing_magazine_retrieval(retrieval_state):
        result = _run_writing_style_retrieval(service, retrieval_state, focus_gap, plan, config)
    else:
        result = service.query(
            question=plan["query"],
            dataset_scope=str(retrieval_state.get("dataset_scope") or "magazine"),
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
        "channels": result.get("channel_trace", []),
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
        "writing_diagnosis": dict(retrieval_state.get("writing_diagnosis", {})),
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


def _build_writing_gaps(prompt_text: str, essay_text: str, diagnosis: dict[str, Any]) -> list[dict[str, Any]]:
    word_count = len(re.findall(r"\b[a-zA-Z][a-zA-Z'-]*\b", essay_text))
    defects = set(diagnosis.get("defects", []))
    return [
        {
            "id": "body_development",
            "name": "Body Paragraph Development",
            "description": "Retrieve body paragraph writing patterns with topic sentence, explanation, example, and wrap-up.",
            "priority": 1.15 if "paragraph_development_weak" in defects or word_count < 250 else 0.9,
            "keywords": ["body_argument", "topic_sentence", "explanation", "example", "wrap_up"],
            "target_layers": ["paragraph", "structure_template"],
            "style_query": "body paragraph with clear topic sentence explanation example and wrap up",
            "status": "open",
            "attempts": 0,
        },
        {
            "id": "sentence_variety",
            "name": "Sentence Variety",
            "description": "Retrieve complex sentence patterns for precise logic, concession, relative clauses, and non-finite structures.",
            "priority": 1.1 if "sentence_variety_weak" in defects else 0.75,
            "keywords": ["complex", "concession", "relative_clause", "non_finite", "argument_support"],
            "target_layers": ["sentence"],
            "style_query": "complex sentence with concession relative clause non finite structure and precise logic",
            "status": "open",
            "attempts": 0,
        },
        {
            "id": "concession_rebuttal",
            "name": "Concession and Rebuttal",
            "description": "Retrieve paragraph patterns that concede a point and then rebut or qualify it.",
            "priority": 1.05 if "concession_missing" in defects else 0.7,
            "keywords": ["concession", "rebuttal", "balanced", "however", "although"],
            "target_layers": ["paragraph", "structure_template", "sentence"],
            "style_query": "paragraph showing concession followed by rebuttal balanced argument however although",
            "status": "open",
            "attempts": 0,
        },
        {
            "id": "thesis_statement",
            "name": "Thesis Statement",
            "description": "Retrieve thesis and introduction patterns for clear position in balanced argument essays.",
            "priority": 1.0 if "thesis_weak" in defects else 0.65,
            "keywords": ["thesis", "position", "balanced", "intro_hook", "statement"],
            "target_layers": ["paragraph", "structure_template", "sentence"],
            "style_query": "thesis statement for balanced argument essay clear position introduction",
            "status": "open",
            "attempts": 0,
        },
        {
            "id": "conclusion_closure",
            "name": "Conclusion Closure",
            "description": "Retrieve concise conclusion and wrap-up patterns.",
            "priority": 0.95 if "conclusion_weak" in defects else 0.6,
            "keywords": ["article_conclusion", "summary", "wrap_up", "overall"],
            "target_layers": ["paragraph", "structure_template", "sentence"],
            "style_query": "conclusion paragraph with concise summary wrap up and final judgement",
            "status": "open",
            "attempts": 0,
        },
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


def _diagnose_writing_issues(prompt_text: str, essay_text: str) -> dict[str, Any]:
    del prompt_text
    sentences = _essay_sentences(essay_text)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", essay_text.strip()) if part.strip()]
    word_count = len(re.findall(r"\b[a-zA-Z][a-zA-Z'-]*\b", essay_text))
    connectors = re.findall(
        r"\b(however|although|though|while|whereas|despite|because|therefore|moreover|furthermore|for example|for instance|as a result|in conclusion|overall)\b",
        essay_text.lower(),
    )
    avg_sentence_len = word_count / max(len(sentences), 1)
    complex_count = sum(1 for sentence in sentences if _looks_complex_sentence(sentence))
    example_count = sum(1 for sentence in sentences if re.search(r"\b(for example|for instance|such as)\b", sentence.lower()))
    concession_count = sum(1 for sentence in sentences if re.search(r"\b(although|though|while|whereas|despite|however|nevertheless)\b", sentence.lower()))

    defects: list[str] = []
    if len(sentences) >= 4 and (complex_count / max(len(sentences), 1) < 0.35 or len(set(connectors)) <= 2):
        defects.append("sentence_variety_weak")
    if word_count < 250 or len(paragraphs) < 4 or example_count <= 1:
        defects.append("paragraph_development_weak")
    if concession_count == 0:
        defects.append("concession_missing")
    intro = paragraphs[0].lower() if paragraphs else essay_text[:450].lower()
    if not re.search(r"\b(i believe|i agree|i disagree|this essay|in my view|should|must|partly|extent)\b", intro):
        defects.append("thesis_weak")
    ending = paragraphs[-1].lower() if paragraphs else essay_text[-450:].lower()
    if not re.search(r"\b(in conclusion|overall|to conclude|in short|therefore)\b", ending):
        defects.append("conclusion_weak")
    if not defects:
        defects.append("paragraph_development_weak")
        defects.append("sentence_variety_weak")
    return {
        "defects": defects,
        "word_count": word_count,
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
        "avg_sentence_len": round(avg_sentence_len, 1),
        "complex_sentence_ratio": round(complex_count / max(len(sentences), 1), 3),
        "connector_variety": len(set(connectors)),
        "example_count": example_count,
        "concession_count": concession_count,
    }


def _essay_sentences(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text.strip()) if item.strip()]


def _looks_complex_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    return bool(
        re.search(r"\b(because|although|though|while|whereas|if|unless|which|who|whom|whose|where|that|when|as|since)\b", lowered)
        or re.search(r"[,;:]", sentence)
    )


def _plan_round(retrieval_state: dict[str, Any], focus_gap: dict[str, Any], client: LLMClient) -> dict[str, Any]:
    prompt_text = str(retrieval_state.get("prompt_text") or "")
    essay_text = str(retrieval_state.get("essay_text") or "")
    round_index = int(retrieval_state.get("round", 0)) + 1
    heuristics = {"gap_id": str(focus_gap.get("id") or "body_development"), "query": _build_gap_query(prompt_text, essay_text, focus_gap), "mode": _default_mode(focus_gap, round_index), "strategy": f"style_gap:{focus_gap.get('id', 'unknown')}", "decision_source": "heuristic_style", "reason": "writing_defect_to_style_query"}
    if _is_writing_magazine_retrieval(retrieval_state):
        return heuristics
    if not client.is_configured:
        return heuristics
    response = client.generate_text(
        system_prompt=(
            "You are a retrieval planner. Return exactly one compact JSON object and nothing else. "
            "Do not use markdown. Required keys: gap_id, query, mode, strategy, reason. "
            "Allowed mode values: local, global, mix."
        ),
        user_prompt=f"Round: {round_index}\nPrompt: {prompt_text}\nEssay excerpt: {essay_text[:600]}\nKnown facts: {json.dumps(retrieval_state.get('known_facts', [])[:6], ensure_ascii=False)}\nOpen gaps: {json.dumps(_open_gap_briefs(retrieval_state.get('gaps', [])), ensure_ascii=False)}\nHistory queries: {json.dumps(list(retrieval_state.get('history_queries', []))[-3:], ensure_ascii=False)}",
        temperature=0.0,
        max_tokens=220,
    )
    parsed = _parse_json_object(response, context="retrieval planner")
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
    writing_style_retrieval = _is_writing_magazine_retrieval(retrieval_state)
    for gap in gaps:
        item = dict(gap)
        gap_id = str(gap.get("id") or "")
        if gap_id == focus_gap_id:
            item["attempts"] = int(item.get("attempts", 0) or 0) + 1
            item["status"] = "filled" if _gap_filled(gap, new_facts, selected_docs, rag_answer, client, retrieval_state) else "open"
        elif writing_style_retrieval and item.get("status") != "filled" and _style_docs_match_gap(gap, selected_docs):
            item["status"] = "filled"
        updated.append(item)
    return updated


def _is_writing_magazine_retrieval(retrieval_state: dict[str, Any]) -> bool:
    scope = str(retrieval_state.get("dataset_scope") or "").lower()
    return str(retrieval_state.get("task_type") or "") == "writing_review" and scope in {"magazine", "magazines", "foreign_magazine", "foreign_magazines"}


def _run_writing_style_retrieval(
    service: SimpleRAGService,
    retrieval_state: dict[str, Any],
    focus_gap: dict[str, Any],
    plan: dict[str, Any],
    config: Any,
) -> dict[str, Any]:
    top_k = int(retrieval_state.get("top_k_per_round", config.retrieval_top_k_per_round) or config.retrieval_top_k_per_round)
    channel_plans = _writing_channel_plans(focus_gap, plan, int(retrieval_state.get("round", 0)) + 1)
    all_docs: list[dict[str, Any]] = []
    channel_trace: list[dict[str, Any]] = []
    for channel in channel_plans:
        result = _query_style_channel(
            service,
            retrieval_state,
            channel,
            top_k=max(top_k, 4),
        )
        docs = list(result.get("retrieved_docs", []))
        for doc in docs:
            metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
            metadata["retrieval_channel"] = channel["channel"]
            metadata["style_intent"] = channel["intent"]
            doc["metadata"] = metadata
        all_docs.extend(docs)
        channel_trace.append({
            "channel": channel["channel"],
            "intent": channel["intent"],
            "query": channel["query"],
            "filters": channel.get("filters", {}),
            "docs": len(docs),
            "fallback_used": bool(result.get("fallback_used")),
        })
    merged_docs = _merge_docs([], sorted(all_docs, key=lambda item: float(item.get("score") or 0.0), reverse=True))
    merged_docs = _style_rerank_docs(merged_docs, focus_gap, top_k=max(top_k * 2, 8))
    return {
        "question": plan["query"],
        "answer": _support_answer(merged_docs, {}, []),
        "retrieved_docs": merged_docs,
        "backend": "simple_rag",
        "query_mode": "style_dual_channel",
        "mode": "local",
        "ranker": "style_intent_metadata_rerank",
        "message": "OK",
        "channel_trace": channel_trace,
    }


def _query_style_channel(
    service: SimpleRAGService,
    retrieval_state: dict[str, Any],
    channel: dict[str, Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    common = {
        "question": channel["query"],
        "dataset_scope": str(retrieval_state.get("dataset_scope") or "magazine"),
        "top_k": top_k,
        "mode": channel.get("mode", "mix"),
        "user_id": str(retrieval_state.get("user_id") or "demo_user"),
        "banned_doc_ids": list(retrieval_state.get("history_doc_ids", [])),
        "banned_chunk_ids": list(retrieval_state.get("history_chunk_keys", [])),
    }
    result = service.query(filters=channel.get("filters", {}), **common)
    if result.get("retrieved_docs"):
        return result
    fallback = service.query(**common)
    fallback["fallback_used"] = True
    return fallback


def _writing_channel_plans(focus_gap: dict[str, Any], plan: dict[str, Any], round_index: int) -> list[dict[str, Any]]:
    gap_id = str(focus_gap.get("id") or "")
    base_query = str(focus_gap.get("style_query") or plan.get("query") or "")
    syntax_filters = {"rag_layer": "sentence", "patterns": ["relative_clause", "concession", "non_finite", "condition"]}
    concession_sentence_filters = {"rag_layer": "sentence", "patterns": ["concession"]}
    paragraph_filters = {"rag_layer": "paragraph", "paragraph_role": ["body_argument", "body_argument_example", "topic_sentence_development", "concession_rebuttal"]}
    concession_paragraph_filters = {"rag_layer": "paragraph", "paragraph_role": ["concession_rebuttal", "body_argument_example", "body_argument"]}
    template_filters = {"rag_layer": "structure_template", "paragraph_role": ["body_argument", "body_argument_example", "topic_sentence_development", "concession_rebuttal"]}

    if gap_id == "sentence_variety":
        return [
            {"channel": "syntax_style", "intent": "complex_sentence_pattern", "query": base_query, "filters": syntax_filters, "mode": "local"},
            {"channel": "paragraph_structure", "intent": "sentence_pattern_inside_argument", "query": "argument support paragraph using complex sentence patterns", "filters": paragraph_filters, "mode": "mix"},
        ]
    if gap_id == "concession_rebuttal":
        return [
            {"channel": "paragraph_structure", "intent": "concession_rebuttal_paragraph", "query": base_query, "filters": concession_paragraph_filters, "mode": "mix"},
            {"channel": "syntax_style", "intent": "concession_sentence", "query": "concession sentence although however despite balanced logic", "filters": concession_sentence_filters, "mode": "local"},
            {"channel": "structure_template", "intent": "abstract_concession_rebuttal_template", "query": base_query, "filters": template_filters, "mode": "mix"},
        ][: 2 if round_index <= 1 else 3]
    if gap_id in {"thesis_statement", "conclusion_closure"}:
        return [
            {"channel": "structure_template", "intent": f"{gap_id}_template", "query": base_query, "filters": template_filters, "mode": "mix"},
            {"channel": "syntax_style", "intent": f"{gap_id}_sentence", "query": base_query, "filters": syntax_filters, "mode": "local"},
        ]
    return [
        {"channel": "paragraph_structure", "intent": "body_argument_structure", "query": base_query, "filters": paragraph_filters, "mode": "mix"},
        {"channel": "structure_template", "intent": "abstract_body_argument_template", "query": "topic sentence explanation example wrap up paragraph structure", "filters": template_filters, "mode": "mix"},
        {"channel": "syntax_style", "intent": "argument_support_sentence", "query": "complex sentence for argument support because therefore example precise logic", "filters": syntax_filters, "mode": "local"},
    ][: 2 if round_index <= 1 else 3]


def _style_rerank_docs(docs: list[dict[str, Any]], focus_gap: dict[str, Any], *, top_k: int) -> list[dict[str, Any]]:
    targets = set(str(item) for item in focus_gap.get("target_layers", []))
    keywords = set(str(item).lower() for item in focus_gap.get("keywords", []))
    scored: list[tuple[float, dict[str, Any]]] = []
    for doc in docs:
        metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        text = " ".join(str(chunk) for chunk in doc.get("chunks", []) if isinstance(chunk, str)).lower()
        meta_text = json.dumps(metadata, ensure_ascii=False).lower()
        layer = str(metadata.get("rag_layer") or metadata.get("chunk_type") or "")
        score = float(doc.get("score") or 0.0)
        imitation_score = _imitation_suitability_score(doc, focus_gap)
        if layer in targets:
            score += 1.15
        score += 0.32 * sum(1 for keyword in keywords if keyword and keyword in meta_text)
        score += 0.04 * sum(1 for keyword in keywords if keyword and keyword in text)
        score += 0.95 * imitation_score
        if metadata.get("retrieval_channel") == "structure_template":
            score += 0.2
        if metadata.get("difficulty") in {"medium_high", "high"}:
            score += 0.15
        if _looks_like_noisy_style_doc(doc):
            score -= 0.6
        ranked_doc = dict(doc)
        ranked_doc["style_score"] = round(score, 4)
        ranked_doc["imitation_score"] = round(imitation_score, 4)
        scored.append((score, ranked_doc))
    return [item for _, item in sorted(scored, key=lambda pair: pair[0], reverse=True)[:top_k]]


def _gap_filled(gap: dict[str, Any], new_facts: list[str], selected_docs: list[dict[str, Any]], rag_answer: str, client: LLMClient, retrieval_state: dict[str, Any]) -> bool:
    if _is_writing_magazine_retrieval(retrieval_state):
        return _style_docs_match_gap(gap, selected_docs)
    if client.is_configured:
        response = client.generate_text(
            system_prompt=(
                "You are a retrieval progress judge. Return exactly one compact JSON object and nothing else. "
                "Do not use markdown. Required keys: gap_filled, reason."
            ),
            user_prompt=f"Gap: {json.dumps(gap, ensure_ascii=False)}\nNew facts: {json.dumps(new_facts, ensure_ascii=False)}\nRetrieved snippets: {json.dumps(_doc_briefs(selected_docs), ensure_ascii=False)}\nRAG answer: {rag_answer[:800]}\nKnown facts: {json.dumps(retrieval_state.get('known_facts', [])[:8], ensure_ascii=False)}",
            temperature=0.0,
            max_tokens=120,
        )
        parsed = _parse_json_object(response, context="retrieval progress judge")
        if isinstance(parsed, dict) and "gap_filled" in parsed:
            return bool(parsed.get("gap_filled"))
    if len(new_facts) >= 2:
        return True
    text = (" ".join(_doc_texts(selected_docs)) + " " + rag_answer).lower()
    hits = sum(1 for keyword in gap.get("keywords", []) if str(keyword).lower() in text)
    return hits >= max(2, min(3, len(gap.get("keywords", []))))


def _style_docs_match_gap(gap: dict[str, Any], selected_docs: list[dict[str, Any]]) -> bool:
    targets = set(str(item) for item in gap.get("target_layers", []))
    keywords = {str(item).lower() for item in gap.get("keywords", [])}
    layer_matches = 0
    keyword_hits: set[str] = set()
    high_quality = 0
    for doc in selected_docs:
        metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        layer = str(metadata.get("rag_layer") or metadata.get("chunk_type") or "")
        if layer in targets:
            layer_matches += 1
        doc_text = " ".join(str(chunk) for chunk in doc.get("chunks", []) if isinstance(chunk, str)).lower()
        meta_text = json.dumps(metadata, ensure_ascii=False).lower()
        keyword_hits.update(keyword for keyword in keywords if keyword and (keyword in meta_text or keyword in doc_text))
        if _imitation_suitability_score(doc, gap) >= 0.75:
            high_quality += 1
    required_layers = 1 if len(selected_docs) <= 2 else 2
    required_keywords = 1 if len(keywords) <= 2 else 2
    return layer_matches >= required_layers and len(keyword_hits) >= required_keywords and high_quality >= 1


def _imitation_suitability_score(doc: dict[str, Any], gap: dict[str, Any]) -> float:
    metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    layer = str(metadata.get("rag_layer") or metadata.get("chunk_type") or "")
    gap_id = str(gap.get("id") or "")
    text = " ".join(str(chunk) for chunk in doc.get("chunks", []) if isinstance(chunk, str)).strip()
    text_len = len(text)
    score = 0.0

    if gap_id == "sentence_variety":
        if layer == "sentence":
            score += 0.55
        patterns = {str(item) for item in metadata.get("patterns", [])} if isinstance(metadata.get("patterns"), list) else set()
        if patterns & {"relative_clause", "concession", "non_finite", "condition"}:
            score += 0.45
        if metadata.get("sentence_type") == "complex":
            score += 0.25
    elif gap_id == "body_development":
        if layer in {"paragraph", "structure_template"}:
            score += 0.45
        if str(metadata.get("paragraph_role") or "") in {"body_argument", "body_argument_example", "topic_sentence_development"}:
            score += 0.45
        if metadata.get("structure_pattern"):
            score += 0.25
    elif gap_id == "concession_rebuttal":
        if layer in {"paragraph", "structure_template", "sentence"}:
            score += 0.25
        if "concession" in json.dumps(metadata, ensure_ascii=False).lower():
            score += 0.55
        if any(token in text.lower() for token in ("although", "however", "despite", "nevertheless", "while")):
            score += 0.25
    elif gap_id == "thesis_statement":
        if layer in {"structure_template", "sentence"}:
            score += 0.35
        if any(token in json.dumps(metadata, ensure_ascii=False).lower() for token in ("topic_sentence", "statement", "balanced")):
            score += 0.45
        if any(token in text.lower() for token in ("this", "i argue", "i believe", "should", "must", "whether")):
            score += 0.15
    elif gap_id == "conclusion_closure":
        if layer in {"structure_template", "paragraph", "sentence"}:
            score += 0.3
        if any(token in json.dumps(metadata, ensure_ascii=False).lower() for token in ("summary", "wrap_up", "article_conclusion")):
            score += 0.45
        if any(token in text.lower() for token in ("overall", "in conclusion", "therefore", "then", "in short")):
            score += 0.25

    if layer == "paragraph" and 260 <= text_len <= 1200:
        score += 0.15
    if layer == "sentence" and 80 <= text_len <= 280:
        score += 0.15
    if layer == "structure_template":
        score += 0.15
    if _looks_like_noisy_style_doc(doc):
        score -= 0.35
    return max(0.0, min(score, 1.5))


def _looks_like_noisy_style_doc(doc: dict[str, Any]) -> bool:
    metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    title = f"{metadata.get('article_title', '')} {metadata.get('section_title', '')}".lower()
    text = " ".join(str(chunk) for chunk in doc.get("chunks", []) if isinstance(chunk, str)).lower()
    if any(token in title for token in ("| next |", "section menu", "main menu", "previous |")):
        return True
    if text.count("[entity]") >= 8:
        return True
    if any(token in text for token in ("sign up to", "subscriber-only newsletter", "continued on next page")):
        return True
    return False


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
    if gap_id in {"body_development", "sentence_variety"}:
        return "local"
    if gap_id in {"concession_rebuttal", "thesis_statement", "conclusion_closure"}:
        return "global"
    return "mix"


def _build_gap_query(prompt_text: str, essay_text: str, gap: dict[str, Any]) -> str:
    del prompt_text, essay_text
    style_query = str(gap.get("style_query") or "").strip()
    if style_query:
        return style_query
    return f"{gap.get('description', gap.get('name', 'writing pattern'))} reusable writing structure syntax pattern"


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


def _parse_json_object(response: str | None, *, context: str = "retrieval") -> dict[str, Any] | None:
    if not response:
        return None
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            logger.debug("Failed to parse %s JSON response, falling back: %s", context, response[:120])
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.debug("Failed to parse %s JSON response, falling back: %s", context, response[:120])
            return None
    return parsed if isinstance(parsed, dict) else None


def _doc_texts(docs: list[dict[str, Any]]) -> list[str]:
    return [" ".join(chunk.strip() for chunk in item.get("chunks", []) if isinstance(chunk, str) and chunk.strip()) for item in docs if item.get("chunks")]


def _doc_briefs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"source": item.get("source"), "preview": " ".join(chunk.strip() for chunk in item.get("chunks", [])[:2] if isinstance(chunk, str) and chunk.strip())[:220]} for item in docs[:4]]
