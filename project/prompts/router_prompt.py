"""Prompt templates for the router node."""

from __future__ import annotations

ALLOWED_INTENTS = (
    "study_plan",
    "knowledge_qa",
    "mistake_review",
    "writing_practice",
    "data_collection",
    "calendar_action",
    "general_chat",
)

ROUTER_SYSTEM_PROMPT = """
You are the intent router for an IELTS learning assistant.
Classify the user's request into exactly one intent from this closed set:
- study_plan
- knowledge_qa
- mistake_review
- writing_practice
- data_collection
- calendar_action
- general_chat

Rules:
- Return JSON only.
- Use exactly this format: {"intent": "one_of_the_allowed_labels"}
- The intent value must be copied exactly from the closed set.
- Do not return aliases, trailing commas, markdown, or extra text.
- Do not include explanations.
""".strip()

INTENT_HINTS: dict[str, str] = {
    "study_plan": "Requests about study planning, exam prep strategy, timelines, or daily/weekly study arrangements.",
    "knowledge_qa": "Questions asking for IELTS knowledge, skills, explanations, examples, scoring rules, or learning materials.",
    "mistake_review": "Requests to review mistakes, analyze weak points, summarize wrong answers, or reflect on error patterns.",
    "writing_practice": "Requests to get an IELTS Writing Task 2 prompt, submit an essay draft, or ask for essay feedback and revision advice.",
    "data_collection": "Requests to collect, download, organize, or prepare IELTS/RAG source materials from public web sources.",
    "calendar_action": "Requests to create, update, or check study schedule or calendar events.",
    "general_chat": "Small talk or other requests that do not fit the above categories.",
}


def build_router_user_prompt(user_input: str) -> str:
    """Build the user prompt for LLM-based routing."""

    hint_lines = "\n".join(f"- {key}: {value}" for key, value in INTENT_HINTS.items())
    return (
        "Intent definitions:\n"
        f"{hint_lines}\n\n"
        f"User input:\n{user_input}\n\n"
        "Return JSON only."
    )
