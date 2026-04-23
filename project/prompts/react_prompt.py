"""Prompt helpers for the single-step ReAct controller."""

from __future__ import annotations

import json
from typing import Any


REACT_CONTROLLER_SYSTEM_PROMPT = """
You are the ReAct controller for IELTS Study Assistant.

Your job is not to answer the user. Each turn you make exactly one decision:
1. Decide whether the current evidence is enough.
2. If not enough, choose one skill or tool action.
3. Explain the purpose of this one action.
4. Wait for observation before deciding again.

Rules:
- Choose at most one action.
- Do not create a full multi-step plan.
- Do not answer the user directly.
- If information is insufficient, act before answering.
- Prefer memory, local database, local question bank, and local RAG before web search.
- Stop if evidence is enough, max steps are reached, or repeated actions add no value.
- Output strict JSON only. No markdown, no extra text.
""".strip()


SKILL_CARDS = [
    {
        "name": "study_plan_skill",
        "goal": "Build or explain an IELTS study plan.",
        "use_when": "The user asks for a plan, roadmap, schedule, or study arrangement.",
        "required_inputs": ["target score", "weak skills", "available time"],
        "stop_conditions": ["profile and latest plan are available"],
    },
    {
        "name": "knowledge_retrieval_skill",
        "goal": "Answer IELTS knowledge questions using local RAG.",
        "use_when": "The user asks about IELTS methods, examples, practice materials, or local corpus content.",
        "required_inputs": ["question", "optional dataset scope"],
        "stop_conditions": ["relevant local evidence is retrieved or confirmed unavailable"],
    },
    {
        "name": "writing_prompt_skill",
        "goal": "Fetch a Writing Task 2 prompt from the local bank.",
        "use_when": "The user asks for a writing practice prompt.",
        "required_inputs": ["optional essay type"],
        "stop_conditions": ["one prompt is returned or the bank is empty"],
    },
    {
        "name": "writing_review_skill",
        "goal": "Review a submitted Writing Task 2 essay with skill scoring policy, writing RAG, and mistake/profile updates.",
        "use_when": "The user submits essay text and an active writing topic exists.",
        "skill_path": "project/agent/skills/writing_review_skill.md",
        "required_inputs": ["topic_id", "essay_text"],
        "stop_conditions": ["score, issues, and revision advice are produced"],
    },
    {
        "name": "mistake_review_skill",
        "goal": "Grade a submitted answer or review mistake history.",
        "use_when": "The user asks for correction, grading, or mistake review.",
        "required_inputs": ["submission or user id"],
        "stop_conditions": ["grading or recent mistake records are available"],
    },
    {
        "name": "data_collection_skill",
        "goal": "Collect IELTS materials into the local corpus.",
        "use_when": "The user asks to collect materials, samples, questions, or corpus data.",
        "required_inputs": ["collection request"],
        "stop_conditions": ["collection completes, needs clarification, or fails with reason"],
    },
    {
        "name": "calendar_skill",
        "goal": "Read or create study calendar events.",
        "use_when": "The user asks about reminders, schedules, calendar, or events.",
        "required_inputs": ["date or event description"],
        "stop_conditions": ["schedule is read or event creation result is returned"],
    },
    {
        "name": "web_search_skill",
        "goal": "Search web for time-sensitive information.",
        "use_when": "The user asks for latest, official, policy, news, recent, or date-sensitive facts.",
        "required_inputs": ["query"],
        "stop_conditions": ["current external snippets are available or search fails"],
    },
]


def build_react_reason_prompt(
    *,
    user_query: str,
    intent: str,
    known_facts: dict[str, Any],
    last_observation: str | None,
    action_history: list[dict[str, Any]],
    step_idx: int,
    max_steps: int,
) -> str:
    """Build the per-step controller prompt."""

    payload = {
        "user_query": user_query,
        "intent": intent,
        "known_facts": known_facts,
        "last_observation": last_observation or "",
        "action_history": action_history[-4:],
        "available_skills": SKILL_CARDS,
        "step": f"{step_idx}/{max_steps}",
        "output_schema": {
            "thought": "short Chinese sentence",
            "need_action": True,
            "action_type": "skill or tool or empty",
            "action_name": "one skill/tool name or empty",
            "action_input": {},
            "finish": False,
            "finish_reason": "",
            "answer_ready": False,
            "missing_info": [],
        },
    }
    return (
        "Make only this step decision. If evidence is enough, set finish=true. "
        "Otherwise choose one action. Return strict JSON only.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
