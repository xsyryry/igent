"""Prompt templates for the planner node."""

from __future__ import annotations

PLANNER_SYSTEM_PROMPT = """
You are the planning node for an IELTS learning assistant.
Decide which tools are needed for the current request.

Return JSON only with this schema:
{
  "use_rag": true,
  "use_db": false,
  "use_calendar": false,
  "use_web_search": false,
  "use_writing": false,
  "use_data_collection": false,
  "dataset_scope": "writing|speaking|reading|mistakes|null",
  "plan": ["use_rag", "generate_knowledge_answer"]
}

Rules:
- Use only booleans for tool switches.
- Keep plan short and practical.
- Prefer web_search for real-time policy/news questions.
- Prefer rag for study materials, lecture notes, uploaded materials, and mistake summaries. Do not use RAG for official IELTS scoring standards; use the writing review skill policy instead.
- Prefer writing when the user wants a Task 2 prompt or essay review.
- Prefer data_collection when the user wants to collect/download IELTS materials or prepare RAG data.
- Do not include explanations outside JSON.
""".strip()

PLANNER_TOOL_GUIDE: dict[str, str] = {
    "use_rag": "Retrieve IELTS learning knowledge or examples from the RAG backend.",
    "use_db": "Read learner profile, study plan, or mistake records from the persistence layer.",
    "use_calendar": "Read or create study schedule events.",
    "use_web_search": "Search the web for real-time or external supplementary information.",
    "use_writing": "Fetch a writing prompt or review a Task 2 essay draft.",
    "use_data_collection": "Collect and archive public IELTS/RAG materials into the local data folder.",
}


def build_planner_user_prompt(intent: str, user_input: str) -> str:
    """Build the user prompt for LLM-based planning."""

    tool_lines = "\n".join(f"- {tool}: {desc}" for tool, desc in PLANNER_TOOL_GUIDE.items())
    return (
        f"Intent: {intent}\n"
        f"User input: {user_input}\n\n"
        f"Available tools:\n{tool_lines}\n\n"
        "Return JSON only."
    )
