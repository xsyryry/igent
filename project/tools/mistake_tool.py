"""Submission grading and mistake-notebook utilities."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from project.db.repository import save_mistake_record
from project.llm.client import LLMClient
from project.memory.profile_service import refresh_profile_from_mistakes
from project.tools.db_tool import DEFAULT_USER_ID
from project.tools.rag_tool import retrieve_knowledge
from project.tools.web_search_tool import search_web

logger = logging.getLogger(__name__)


SUBMISSION_FIELD_ALIASES = {
    "subject": ("科目", "学科", "subject"),
    "question_type": ("题型", "question_type", "type"),
    "question_source": ("来源", "题目来源", "source"),
    "question_text": ("题目", "题干", "question", "prompt"),
    "user_answer": ("我的答案", "用户答案", "answer", "user_answer"),
    "reference_answer": ("参考答案", "标准答案", "reference_answer", "correct_answer"),
}


def looks_like_submission_request(user_input: str) -> bool:
    """Heuristic detector for answer-submission / grading requests."""

    normalized = user_input.lower()
    submission_markers = (
        "批改",
        "批阅",
        "帮我改",
        "我的答案",
        "参考答案",
        "标准答案",
        "做完了",
        "grade this",
        "review my answer",
        "correct my answer",
    )
    return any(marker in normalized or marker in user_input for marker in submission_markers)


def grade_submission(
    user_input: str,
    user_id: str = DEFAULT_USER_ID,
) -> dict[str, Any]:
    """Parse a submission, resolve a reference answer, grade it, and update memory."""

    submission = _extract_submission_fields(user_input)
    resolved_reference = _resolve_reference_answer(submission)
    grading = _grade_against_reference(submission, resolved_reference)

    saved_record: dict[str, Any] | None = None
    if grading["should_save_as_mistake"]:
        saved_record = save_mistake_record(
            user_id=user_id,
            subject=submission["subject"],
            question_type=submission.get("question_type"),
            question_source=submission.get("question_source"),
            question_text=submission["question_text"],
            user_answer=submission.get("user_answer"),
            reference_answer=grading.get("reference_answer"),
            is_correct=False,
            score=grading.get("score"),
            error_type=grading.get("error_type"),
            wrong_reason=grading.get("wrong_reason") or "回答与参考答案不一致",
            correction_note=grading.get("correction_note"),
            source_of_truth=grading.get("source_of_truth"),
            metadata={
                "submission": submission,
                "reference_context": grading.get("reference_context", {}),
            },
        )

    refreshed_profile = refresh_profile_from_mistakes(user_id=user_id)
    return {
        "submission": submission,
        "grading": grading,
        "saved_record": saved_record,
        "updated_profile": refreshed_profile,
    }


def _extract_submission_fields(user_input: str) -> dict[str, str]:
    parsed = _parse_labeled_submission(user_input)
    if parsed.get("question_text") and parsed.get("user_answer"):
        return _normalize_submission(parsed)

    llm_submission = _parse_submission_with_llm(user_input)
    if llm_submission.get("question_text") and llm_submission.get("user_answer"):
        return _normalize_submission(llm_submission)

    return _normalize_submission(
        {
            "subject": "reading",
            "question_type": "unknown",
            "question_source": "user_submission",
            "question_text": user_input.strip(),
            "user_answer": "",
            "reference_answer": "",
        }
    )


def _parse_labeled_submission(user_input: str) -> dict[str, str]:
    values: dict[str, str] = {}
    lines = [line.strip() for line in user_input.splitlines() if line.strip()]
    for line in lines:
        for canonical_name, aliases in SUBMISSION_FIELD_ALIASES.items():
            for alias in aliases:
                prefix = f"{alias}:"
                fullwidth_prefix = f"{alias}："
                if line.lower().startswith(prefix.lower()):
                    values[canonical_name] = line[len(prefix) :].strip()
                elif line.startswith(fullwidth_prefix):
                    values[canonical_name] = line[len(fullwidth_prefix) :].strip()
    return values


def _parse_submission_with_llm(user_input: str) -> dict[str, str]:
    client = LLMClient.from_config()
    if not client.is_configured:
        return {}

    response = client.generate_text(
        system_prompt=(
            "You extract IELTS answer submissions. Return one JSON object only with keys: "
            "subject, question_type, question_source, question_text, user_answer, reference_answer."
        ),
        user_prompt=user_input,
        temperature=0.0,
        max_tokens=300,
    )
    if not response:
        return {}

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    return data if isinstance(data, dict) else {}


def _normalize_submission(raw: dict[str, Any]) -> dict[str, str]:
    subject = str(raw.get("subject") or "reading").strip().lower()
    if subject not in {"reading", "writing", "listening", "speaking"}:
        subject = "reading"
    return {
        "subject": subject,
        "question_type": str(raw.get("question_type") or "unknown").strip().lower(),
        "question_source": str(raw.get("question_source") or "user_submission").strip(),
        "question_text": str(raw.get("question_text") or "").strip(),
        "user_answer": str(raw.get("user_answer") or "").strip(),
        "reference_answer": str(raw.get("reference_answer") or "").strip(),
    }


def _resolve_reference_answer(submission: dict[str, str]) -> dict[str, Any]:
    if submission.get("reference_answer"):
        return {
            "reference_answer": submission["reference_answer"],
            "source_of_truth": "user_provided",
            "reference_context": {},
        }

    rag_result = retrieve_knowledge(
        question=submission["question_text"],
        dataset_scope=_infer_dataset_scope(submission["subject"]),
        top_k=3,
    )
    rag_answer = str(rag_result.get("answer") or "").strip()
    if rag_answer and "No relevant context found" not in rag_answer:
        return {
            "reference_answer": _trim_reference_answer(rag_answer),
            "source_of_truth": "rag",
            "reference_context": {
                "backend": rag_result.get("backend"),
                "query_mode": rag_result.get("query_mode"),
            },
        }

    web_results = search_web(query=submission["question_text"])
    if web_results:
        top_item = web_results[0]
        snippet = str(top_item.get("snippet") or top_item.get("title") or "").strip()
        if snippet:
            return {
                "reference_answer": snippet,
                "source_of_truth": "web",
                "reference_context": {"top_result": top_item},
            }

    return {
        "reference_answer": "",
        "source_of_truth": "heuristic",
        "reference_context": {},
    }


def _grade_against_reference(
    submission: dict[str, str],
    resolved_reference: dict[str, Any],
) -> dict[str, Any]:
    reference_answer = str(resolved_reference.get("reference_answer") or "").strip()
    user_answer = submission.get("user_answer", "")
    subject = submission["subject"]

    if subject in {"writing", "speaking"}:
        return _grade_open_response(submission, reference_answer, resolved_reference)
    return _grade_objective_response(submission, reference_answer, resolved_reference)


def _grade_objective_response(
    submission: dict[str, str],
    reference_answer: str,
    resolved_reference: dict[str, Any],
) -> dict[str, Any]:
    user_answer = _normalize_answer_text(submission.get("user_answer", ""))
    reference_norm = _normalize_answer_text(reference_answer)
    is_correct = bool(user_answer and reference_norm and user_answer == reference_norm)
    score = 1.0 if is_correct else 0.0

    if not reference_norm:
        return {
            "is_correct": False,
            "score": 0.0,
            "error_type": "reference_not_found",
            "wrong_reason": "没有找到可靠参考答案，当前无法做严格判定。",
            "correction_note": "建议补充标准答案，或把题目和官方解析一起提交给我。",
            "source_of_truth": resolved_reference.get("source_of_truth", "heuristic"),
            "reference_answer": reference_answer,
            "reference_context": resolved_reference.get("reference_context", {}),
            "should_save_as_mistake": False,
        }

    return {
        "is_correct": is_correct,
        "score": score,
        "error_type": "" if is_correct else _infer_error_type(submission, reference_answer),
        "wrong_reason": "" if is_correct else _build_wrong_reason(submission, reference_answer),
        "correction_note": "" if is_correct else f"参考答案建议写成：{reference_answer}",
        "source_of_truth": resolved_reference.get("source_of_truth", "heuristic"),
        "reference_answer": reference_answer,
        "reference_context": resolved_reference.get("reference_context", {}),
        "should_save_as_mistake": not is_correct,
    }


def _grade_open_response(
    submission: dict[str, str],
    reference_answer: str,
    resolved_reference: dict[str, Any],
) -> dict[str, Any]:
    client = LLMClient.from_config()
    if client.is_configured and reference_answer:
        response = client.generate_text(
            system_prompt=(
                "You are grading an IELTS answer. Return JSON only with keys: "
                "is_correct, score, error_type, wrong_reason, correction_note."
            ),
            user_prompt=(
                f"Subject: {submission['subject']}\n"
                f"Question: {submission['question_text']}\n"
                f"User answer: {submission['user_answer']}\n"
                f"Reference answer or criterion: {reference_answer}\n"
            ),
            temperature=0.1,
            max_tokens=280,
        )
        parsed = _parse_llm_grading(response)
        if parsed:
            return {
                "is_correct": bool(parsed.get("is_correct", False)),
                "score": float(parsed.get("score", 0.0)),
                "error_type": str(parsed.get("error_type") or ""),
                "wrong_reason": str(parsed.get("wrong_reason") or ""),
                "correction_note": str(parsed.get("correction_note") or ""),
                "source_of_truth": resolved_reference.get("source_of_truth", "rag"),
                "reference_answer": reference_answer,
                "reference_context": resolved_reference.get("reference_context", {}),
                "should_save_as_mistake": not bool(parsed.get("is_correct", False)),
            }

    enough_length = len(submission.get("user_answer", "").strip()) >= 50
    return {
        "is_correct": enough_length,
        "score": 0.6 if enough_length else 0.3,
        "error_type": "" if enough_length else "insufficient_development",
        "wrong_reason": "" if enough_length else "回答过短，无法充分展示观点或语言能力。",
        "correction_note": (
            "继续补充观点解释、例子和更具体的表达。"
            if not enough_length
            else "整体可继续优化语言准确性和展开深度。"
        ),
        "source_of_truth": resolved_reference.get("source_of_truth", "heuristic"),
        "reference_answer": reference_answer,
        "reference_context": resolved_reference.get("reference_context", {}),
        "should_save_as_mistake": not enough_length,
    }


def _parse_llm_grading(response: str | None) -> dict[str, Any] | None:
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


def _infer_dataset_scope(subject: str) -> str:
    if subject == "writing":
        return "writing"
    if subject == "speaking":
        return "speaking"
    return "reading"


def _normalize_answer_text(text: str) -> str:
    lowered = text.strip().lower()
    replacements = {
        "true false not given": "",
        "answer": "",
        "the answer is": "",
    }
    for source, target in replacements.items():
        lowered = lowered.replace(source, target)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip(" .,:;")


def _infer_error_type(submission: dict[str, str], reference_answer: str) -> str:
    question_type = submission.get("question_type", "")
    subject = submission.get("subject", "")
    if question_type in {"true_false_not_given", "judgement"}:
        return "logic_mismatch"
    if subject == "reading":
        return "reading_comprehension_error"
    if subject == "listening":
        return "listening_detail_error"
    if subject == "writing":
        return "task_response_gap"
    if subject == "speaking":
        return "insufficient_development"
    return "answer_mismatch"


def _build_wrong_reason(submission: dict[str, str], reference_answer: str) -> str:
    if submission.get("question_type") in {"true_false_not_given", "judgement"}:
        return (
            f"你的答案是 {submission.get('user_answer', '')}，但参考答案更接近 {reference_answer}。"
            " 这通常说明定位后没有继续核对原文逻辑关系。"
        )
    return (
        f"你的答案“{submission.get('user_answer', '')}”与参考答案“{reference_answer}”不一致。"
        " 建议回到题干和依据文本，确认关键词和逻辑关系。"
    )


def _trim_reference_answer(answer: str, limit: int = 220) -> str:
    compact = re.sub(r"\s+", " ", answer).strip()
    return compact[:limit]
