"""Prompt templates for the generator node."""

from __future__ import annotations

import json

GENERATOR_SYSTEM_PROMPT = """
You are an IELTS study assistant.
Your style should be clear, friendly, practical, and supportive.

Rules:
- Respond in natural Chinese unless the user clearly requests another language.
- Use only the available evidence and observations; say when evidence is insufficient.
- Do not expose chain-of-thought or ReAct scratchpad details.
- Do not mention internal tools, routing, planning, or retrieval mechanics unless the user asks about the architecture.
- Focus on actionable guidance and learner clarity.
""".strip()


def build_generator_user_prompt(intent: str, user_input: str, answer_context: dict[str, object]) -> str:
    """Build the user prompt for LLM-based answer generation."""

    context_json = json.dumps(answer_context, ensure_ascii=False, indent=2, default=str)
    return (
        f"Intent: {intent}\n"
        f"User request:\n{user_input}\n\n"
        f"Available structured context:\n{context_json}\n\n"
        "Generate the final user-facing answer only."
    )
