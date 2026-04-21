"""Writing practice and essay review tool."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from project.db.repository import (
    get_random_writing_task2_topic,
    get_writing_task2_topic_by_id,
    list_writing_submissions,
    list_writing_samples,
    list_writing_scoring_descriptors,
    save_mistake_record,
    save_writing_sample,
    save_writing_scoring_descriptor,
    save_writing_submission,
)
from project.llm.client import LLMClient
from project.memory.profile_service import ensure_user_profile, refresh_profile_from_mistakes
from project.rag.orchestration.gap_retrieval import (
    apply_retrieval_state_to_review_state,
    build_writing_review_retrieval_state,
    run_gap_retrieval_round,
    summarize_gap_retrieval_state,
)
from project.tools.db_tool import DEFAULT_USER_ID

logger = logging.getLogger(__name__)
WRITING_RAG_MAX_DOCS = 6


WRITING_REQUEST_KEYWORDS = (
    "给我一题作文",
    "给我一篇作文",
    "来一题作文",
    "来篇作文",
    "来一道作文",
    "抽一题作文",
    "来一道作文题",
    "抽一题作文",
    "作文练习",
    "task 2 题目",
    "大作文题目",
    "我要写作文",
    "给我一个写作题目",
)

WRITING_FEEDBACK_KEYS = ("task_response", "coherence_cohesion", "lexical_resource", "grammar_accuracy")
ESSAY_TYPE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("观点类", ("观点类", "观点", "同意", "不同意", "agree", "disagree", "extent")),
    ("讨论类", ("讨论类", "讨论", "双方观点", "both views", "discuss both", "两种观点")),
    ("优缺点类", ("优缺点类", "优缺点", "利弊", "advantages", "disadvantages", "pros and cons")),
    ("问题解决类", ("问题解决类", "问题解决", "解决", "问题和解决", "problems and solutions", "solutions")),
    ("报告类", ("报告类", "报告", "原因", "影响", "why is this", "causes", "reasons", "effects")),
)


def looks_like_writing_practice_request(user_input: str) -> bool:
    """Detect requests asking for a writing prompt."""

    normalized = user_input.strip().lower()
    if any(keyword in normalized or keyword in user_input for keyword in WRITING_REQUEST_KEYWORDS):
        return True
    requested_type = extract_requested_essay_type(user_input)
    if requested_type and any(token in user_input or token in normalized for token in ("题", "题目", "作文", "task 2", "练习", "essay")):
        return True
    return False


def extract_requested_essay_type(user_input: str) -> str | None:
    """Infer the requested IELTS Task 2 essay type from the user's request."""

    normalized = user_input.strip().lower()
    for essay_type, patterns in ESSAY_TYPE_PATTERNS:
        if any(pattern in user_input or pattern in normalized for pattern in patterns):
            return essay_type
    return None


def looks_like_writing_submission(user_input: str, active_topic_id: str | None = None) -> bool:
    """Detect whether the user is likely submitting an essay draft."""

    if not active_topic_id:
        return False
    stripped = user_input.strip()
    if len(stripped) < 120:
        return False
    english_word_count = len(re.findall(r"\b[a-zA-Z][a-zA-Z'-]*\b", stripped))
    if english_word_count >= 35:
        return True
    if "\n" in stripped:
        return True
    sentence_count = len(re.findall(r"[.!?。！？]", stripped))
    return sentence_count >= 3


def get_random_task2_prompt(essay_type: str | None = None) -> dict[str, Any]:
    """Fetch one random Task 2 prompt from SQLite."""

    topic = get_random_writing_task2_topic(essay_type=essay_type)
    if topic is None:
        return {
            "success": False,
            "message": (
                f"当前题库里还没有可用的{essay_type}大作文题目，请先导入更多题库。"
                if essay_type
                else "当前题库里还没有可用的大作文题目，请先运行 import-task2 导入题库。"
            ),
        }
    return {
        "success": True,
        "topic": topic,
        "requested_essay_type": essay_type,
        "message": (
            f"已为你随机抽取一道{essay_type}大作文题。"
            if essay_type
            else "已为你随机抽取一道大作文题。"
        ),
    }


def review_task2_submission(
    *,
    user_input: str,
    topic_id: str,
    user_id: str = DEFAULT_USER_ID,
) -> dict[str, Any]:
    """Review a Task 2 essay using SQL references plus a retrieve-judge loop."""

    preparation = prepare_task2_review_context(
        user_input=user_input,
        topic_id=topic_id,
        user_id=user_id,
    )
    if not preparation.get("success"):
        return preparation

    review_state = preparation["review_state"]
    while not bool(review_state.get("enough_context", False)) and int(
        review_state.get("retrieval_round", 0)
    ) < int(review_state.get("max_rounds", WRITING_RAG_MAX_ROUNDS)):
        review_state = execute_writing_retrieval_round(review_state)

    return finalize_task2_review(review_state, user_id=user_id)


def prepare_task2_review_context(
    *,
    user_input: str,
    topic_id: str,
    user_id: str = DEFAULT_USER_ID,
) -> dict[str, Any]:
    """Prepare structured review context before the graph enters retrieval loops."""

    topic = get_writing_task2_topic_by_id(topic_id)
    if topic is None:
        return {
            "success": False,
            "message": "当前没有找到对应的大作文题目，请先重新抽题。",
        }

    ensure_user_profile(user_id)
    descriptors = _ensure_default_descriptors()
    samples = _ensure_reference_sample(topic)
    return {
        "success": True,
        "topic": topic,
        "reference_sample": samples[0] if samples else None,
        "descriptors": descriptors,
        "review_state": apply_retrieval_state_to_review_state({
            "active": True,
            "topic": topic,
            "essay_text": user_input,
            "descriptors": descriptors,
            "samples": samples,
        }, build_writing_review_retrieval_state(
            prompt_text=topic["prompt_text"],
            essay_text=user_input,
            dataset_scope="writing",
        )),
    }


def execute_writing_retrieval_round(review_state: dict[str, Any]) -> dict[str, Any]:
    """Execute one retrieval round for writing review."""
    retrieval_state = review_state.get("retrieval_state", {})
    if not isinstance(retrieval_state, dict) or not retrieval_state:
        return review_state
    updated_retrieval_state = run_gap_retrieval_round(retrieval_state)
    return apply_retrieval_state_to_review_state(review_state, updated_retrieval_state)


def finalize_task2_review(
    review_state: dict[str, Any],
    *,
    user_id: str = DEFAULT_USER_ID,
) -> dict[str, Any]:
    """Finalize essay review after retrieval rounds complete."""

    topic = review_state.get("topic", {})
    essay_text = str(review_state.get("essay_text", ""))
    descriptors = list(review_state.get("descriptors", []))
    samples = list(review_state.get("samples", []))
    retrieval_summary = summarize_gap_retrieval_state(dict(review_state.get("retrieval_state", {})))
    rag_result = {
        **retrieval_summary,
        "documents": _to_legacy_documents(list(review_state.get("retrieved_docs", []))),
    }
    evaluation = _evaluate_essay(
        topic=topic,
        essay_text=essay_text,
        descriptors=descriptors,
        samples=samples,
        rag_result=rag_result,
    )
    word_count = _count_words(essay_text)
    saved_submission = save_writing_submission(
        user_id=user_id,
        task2_topic_id=topic["id"],
        essay_text=essay_text,
        word_count=word_count,
        score=evaluation.get("overall_band"),
        feedback_json=evaluation,
        source_of_truth=f"{evaluation.get('evaluation_source', 'heuristic')}+sql+rag",
        metadata={
            "rag_backend": rag_result.get("backend"),
            "query_mode": rag_result.get("query_mode"),
            "sample_count": len(samples),
            "descriptor_count": len(descriptors),
            "retrieval_trace": rag_result.get("retrieval_trace", []),
        },
    )

    top_issue = str(evaluation.get("priority_issue") or "task_response_gap")
    wrong_reason = str(evaluation.get("summary_for_memory") or evaluation.get("overall_comment") or "").strip()
    correction_note = str(evaluation.get("revision_plan", ["建议按评分标准逐段修改。"])[0])
    if wrong_reason:
        save_mistake_record(
            user_id=user_id,
            subject="writing",
            question_type=topic.get("essay_type"),
            question_source=f"{topic.get('exam_date', '')} {topic.get('essay_type', '')}".strip(),
            question_text=topic["prompt_text"],
            user_answer=essay_text[:1500],
            reference_answer=samples[0]["content"] if samples else "",
            is_correct=False,
            score=float(evaluation.get("overall_band", 0.0) or 0.0),
            error_type=top_issue,
            wrong_reason=wrong_reason,
            correction_note=correction_note,
            source_of_truth=f"{evaluation.get('evaluation_source', 'heuristic')}+sql+rag",
            metadata={
                "topic_id": topic["id"],
                "band_breakdown": evaluation.get("band_breakdown", {}),
                "retrieval_trace": rag_result.get("retrieval_trace", []),
            },
        )

    updated_profile = refresh_profile_from_mistakes(user_id=user_id)
    progress_summary = _build_progress_summary(
        user_id=user_id,
        current_submission=saved_submission,
        current_evaluation=evaluation,
        essay_type=topic.get("essay_type"),
    )
    return {
        "success": True,
        "topic": topic,
        "evaluation": evaluation,
        "saved_submission": saved_submission,
        "updated_profile": updated_profile,
        "progress_summary": progress_summary,
        "reference_sample": samples[0] if samples else None,
        "descriptors": descriptors,
        "rag_result": rag_result,
    }


def _ensure_default_descriptors() -> list[dict[str, Any]]:
    descriptors = list_writing_scoring_descriptors(writing_type="task2", limit=20)
    if descriptors:
        return descriptors

    seeds = [
        ("task2", "Task Response", "6-7", "Addresses all parts of the question and develops a clear position with relevant support."),
        ("task2", "Coherence and Cohesion", "6-7", "Organizes ideas logically with clear paragraphing and linking that does not feel mechanical."),
        ("task2", "Lexical Resource", "6-7", "Uses a sufficient range of vocabulary with some flexibility and mostly appropriate word choice."),
        ("task2", "Grammatical Range and Accuracy", "6-7", "Uses a mix of sentence structures and keeps grammar errors limited enough not to block understanding."),
    ]
    created = [
        save_writing_scoring_descriptor(
            writing_type=writing_type,
            criterion_name=criterion_name,
            band_level=band_level,
            descriptor_text=descriptor_text,
            source_label="demo_seed",
            metadata={"seed": True},
        )
        for writing_type, criterion_name, band_level, descriptor_text in seeds
    ]
    return created


def _ensure_reference_sample(topic: dict[str, Any]) -> list[dict[str, Any]]:
    samples = list_writing_samples(task2_topic_id=topic["id"], limit=2)
    if samples:
        return samples

    fallback_outline = (
        "Introduction: paraphrase the topic and present a clear position.\n"
        "Body paragraph 1: explain the main reason with a concrete example.\n"
        "Body paragraph 2: present a supporting or balanced argument and show consequences.\n"
        "Conclusion: restate the position and summarize the key logic."
    )
    created = save_writing_sample(
        task2_topic_id=topic["id"],
        sample_type="reference_outline",
        title=f"{topic.get('essay_type', 'Task 2')} reference outline",
        content=fallback_outline,
        source_label="demo_seed",
        metadata={"seed": True},
    )
    return [created]


def _evaluate_essay(
    *,
    topic: dict[str, Any],
    essay_text: str,
    descriptors: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    rag_result: dict[str, Any],
) -> dict[str, Any]:
    client = LLMClient.from_config()
    if client.is_configured:
        response = client.generate_text(
            system_prompt=_build_evaluation_system_prompt(),
            user_prompt=_build_evaluation_user_prompt(topic, essay_text, descriptors, samples, rag_result),
            temperature=0.1,
            max_tokens=1200,
        )
        parsed = _parse_json_object(response)
        if parsed:
            parsed["evaluation_source"] = "llm"
            return _normalize_evaluation(parsed, essay_text)

    return _build_heuristic_evaluation(topic, essay_text, rag_result)


def _build_evaluation_system_prompt() -> str:
    return (
        "You are an IELTS Writing Task 2 reviewer.\n"
        "Return JSON only.\n"
        "Schema: {\n"
        '  "overall_band": 0.0,\n'
        '  "overall_comment": "...",\n'
        '  "priority_issue": "...",\n'
        '  "summary_for_memory": "...",\n'
        '  "band_breakdown": {"task_response": 0.0, "coherence_cohesion": 0.0, "lexical_resource": 0.0, "grammar_accuracy": 0.0},\n'
        '  "strengths": ["..."],\n'
        '  "issues": ["..."],\n'
        '  "revision_plan": ["...", "..."],\n'
        '  "language_upgrade_notes": ["..."]\n'
        "}\n"
        "Be constructive, concise, and grounded in the provided descriptors."
    )


def _build_evaluation_user_prompt(
    topic: dict[str, Any],
    essay_text: str,
    descriptors: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    rag_result: dict[str, Any],
) -> str:
    descriptor_text = "\n".join(
        f"- {item['criterion_name']} ({item.get('band_level') or 'general'}): {item['descriptor_text']}"
        for item in descriptors
    )
    sample_text = "\n\n".join(
        f"{sample.get('title') or sample['sample_type']}:\n{sample['content']}"
        for sample in samples[:2]
    )
    rag_answer = str(rag_result.get("answer") or "").strip()
    retrieval_trace = rag_result.get("retrieval_trace", [])
    trace_text = "\n".join(
        f"- Round {item.get('round', '?')}: mode={item.get('mode', 'mix')} enough={item.get('enough_context', False)} query={item.get('query', '')}"
        for item in retrieval_trace
    )
    return (
        f"Topic ({topic.get('exam_date', 'unknown')} / {topic.get('essay_type', 'Task 2')}):\n"
        f"{topic['prompt_text']}\n\n"
        f"Student essay:\n{essay_text}\n\n"
        f"Descriptors:\n{descriptor_text}\n\n"
        f"Reference sample(s):\n{sample_text}\n\n"
        f"Additional English support from retrieval:\n{rag_answer}\n\n"
        f"Retrieval trace:\n{trace_text or '- single-pass retrieval'}\n"
    )


def _parse_json_object(response: str | None) -> dict[str, Any] | None:
    if not response:
        return None
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_evaluation(data: dict[str, Any], essay_text: str) -> dict[str, Any]:
    overall_band = float(data.get("overall_band", 5.5) or 5.5)
    breakdown = data.get("band_breakdown", {})
    normalized_breakdown = {
        "task_response": float(breakdown.get("task_response", overall_band)),
        "coherence_cohesion": float(breakdown.get("coherence_cohesion", overall_band)),
        "lexical_resource": float(breakdown.get("lexical_resource", overall_band)),
        "grammar_accuracy": float(breakdown.get("grammar_accuracy", overall_band)),
    }
    return {
        "overall_band": round(overall_band, 1),
        "overall_comment": str(data.get("overall_comment") or "").strip(),
        "priority_issue": str(data.get("priority_issue") or "task_response_gap").strip(),
        "summary_for_memory": str(data.get("summary_for_memory") or "").strip(),
        "band_breakdown": normalized_breakdown,
        "strengths": _normalize_str_list(data.get("strengths")),
        "issues": _normalize_str_list(data.get("issues")),
        "revision_plan": _normalize_str_list(data.get("revision_plan")),
        "language_upgrade_notes": _normalize_str_list(data.get("language_upgrade_notes")),
        "word_count": _count_words(essay_text),
    }


def _build_heuristic_evaluation(topic: dict[str, Any], essay_text: str, rag_result: dict[str, Any]) -> dict[str, Any]:
    del topic
    word_count = _count_words(essay_text)
    paragraph_count = len([part for part in re.split(r"\n\s*\n", essay_text.strip()) if part.strip()])
    overall_band = 5.0
    issues: list[str] = []
    strengths: list[str] = []
    if word_count >= 260:
        overall_band += 0.5
        strengths.append("字数达到了 Task 2 的基本要求。")
    else:
        issues.append("字数偏少，论证展开不足。")
    if paragraph_count >= 3:
        overall_band += 0.5
        strengths.append("段落结构已经基本成形。")
    else:
        issues.append("段落结构不够清晰，建议至少写出引言、主体段和结论。")
    if "because" in essay_text.lower() or "for example" in essay_text.lower():
        overall_band += 0.3
        strengths.append("开始尝试用因果或例子支撑观点。")
    else:
        issues.append("论点后的解释和例子还不够充分。")

    rag_hint = str(rag_result.get("answer") or "").strip()
    language_note = "可增加更自然的连接词和更具体的论证表达。"
    if rag_hint and "No relevant context found" not in rag_hint:
        language_note = "可参考外刊语料中的正式表达，增强论证的学术感。"

    return {
        "overall_band": round(min(overall_band, 6.5), 1),
        "overall_comment": "整体上已经具备基本的 Task 2 回答框架，但论证深度和语言精确度还可以继续提升。",
        "priority_issue": "task_response_gap" if issues else "language_upgrade",
        "summary_for_memory": issues[0] if issues else "继续加强论证展开与正式表达。",
        "band_breakdown": {
            "task_response": round(min(overall_band, 6.0), 1),
            "coherence_cohesion": round(min(overall_band, 6.0), 1),
            "lexical_resource": round(max(5.0, overall_band - 0.3), 1),
            "grammar_accuracy": round(max(5.0, overall_band - 0.3), 1),
        },
        "strengths": strengths or ["已经完成了一篇可批改的完整作文。"],
        "issues": issues or ["可继续加强词汇准确性和句式变化。"],
        "revision_plan": [
            "先补强每个主体段的解释句和例子句。",
            "再通读全文，统一立场并检查连接词是否自然。",
        ],
        "language_upgrade_notes": [language_note],
        "evaluation_source": "heuristic",
        "word_count": word_count,
    }


def _normalize_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))



def _merge_retrieved_docs(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = list(existing)
    seen = {
        (
            str(item.get("id") or ""),
            str(item.get("source") or ""),
            tuple(str(chunk) for chunk in item.get("chunks", []) if isinstance(chunk, str)),
        )
        for item in existing
    }
    for item in incoming:
        key = (
            str(item.get("id") or ""),
            str(item.get("source") or ""),
            tuple(str(chunk) for chunk in item.get("chunks", []) if isinstance(chunk, str)),
        )
        if key in seen:
            continue
        merged.append(item)
        seen.add(key)
    return merged[:WRITING_RAG_MAX_DOCS]


def _build_writing_support_answer(
    collected_docs: list[dict[str, Any]],
    latest_result: dict[str, Any],
) -> str:
    answer = str(latest_result.get("answer") or "").strip()
    if answer and "No relevant context found" not in answer:
        return answer
    snippets: list[str] = []
    for item in collected_docs[:WRITING_RAG_MAX_DOCS]:
        chunks = item.get("chunks", [])
        joined = " ".join(chunk.strip() for chunk in chunks[:2] if isinstance(chunk, str) and chunk.strip())
        if joined:
            snippets.append(f"{item.get('source', 'Reference')}: {joined[:260]}")
    return "\n".join(snippets)


def _to_legacy_documents(retrieved_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for item in retrieved_docs:
        chunks = item.get("chunks", [])
        content = "\n".join(
            chunk for chunk in chunks if isinstance(chunk, str) and chunk.strip()
        ).strip()
        if not content:
            continue
        documents.append(
            {
                "source": str(item.get("source", "")),
                "title": str(item.get("source") or item.get("id") or "Reference"),
                "content": content,
                "score": 1.0,
            }
        )
    return documents


def _build_progress_summary(
    *,
    user_id: str,
    current_submission: dict[str, Any],
    current_evaluation: dict[str, Any],
    essay_type: str | None,
) -> dict[str, Any] | None:
    """Compare the latest submission against the user's previous writing attempt."""

    previous_records = list_writing_submissions(
        user_id=user_id,
        limit=1,
        essay_type=essay_type,
        exclude_submission_id=current_submission["id"],
    )
    if not previous_records:
        return None

    previous = previous_records[0]
    previous_feedback = previous.get("feedback_json", {})
    current_score = float(current_evaluation.get("overall_band") or 0.0)
    previous_score = float(previous.get("score") or previous_feedback.get("overall_band") or 0.0)
    delta = round(current_score - previous_score, 1)

    current_breakdown = current_evaluation.get("band_breakdown", {})
    previous_breakdown = previous_feedback.get("band_breakdown", {})
    criterion_deltas: dict[str, float] = {}
    highlighted_changes: list[str] = []
    label_map = {
        "task_response": "Task Response",
        "coherence_cohesion": "Coherence & Cohesion",
        "lexical_resource": "Lexical Resource",
        "grammar_accuracy": "Grammar Accuracy",
    }
    for key in WRITING_FEEDBACK_KEYS:
        current_value = float(current_breakdown.get(key, current_score) or current_score)
        previous_value = float(previous_breakdown.get(key, previous_score) or previous_score)
        criterion_delta = round(current_value - previous_value, 1)
        criterion_deltas[key] = criterion_delta
        if criterion_delta >= 0.5:
            highlighted_changes.append(f"{label_map[key]} 提升了 {criterion_delta:.1f}")
        elif criterion_delta <= -0.5:
            highlighted_changes.append(f"{label_map[key]} 下降了 {abs(criterion_delta):.1f}")

    if not highlighted_changes:
        if delta > 0:
            highlighted_changes.append("整体表现比上一次更稳定。")
        elif delta < 0:
            highlighted_changes.append("这次整体表现略有回落，建议先回到最核心的评分项。")
        else:
            highlighted_changes.append("整体分数和上一次接近，接下来重点突破单项弱项。")

    return {
        "has_previous": True,
        "previous_submission_id": previous["id"],
        "previous_created_at": previous.get("created_at"),
        "previous_score": round(previous_score, 1),
        "current_score": round(current_score, 1),
        "score_delta": delta,
        "criterion_deltas": criterion_deltas,
        "highlighted_changes": highlighted_changes,
        "same_essay_type": bool(essay_type),
    }
