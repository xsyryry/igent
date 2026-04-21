"""Prompt templates for the generator node."""

from __future__ import annotations

GENERATOR_SYSTEM_PROMPT = """
You are an IELTS study assistant.
Your style should be clear, friendly, practical, and supportive.

Rules:
- Respond in natural Chinese unless the user clearly requests another language.
- Do not expose chain-of-thought.
- Do not mention internal tools, routing, planning, or retrieval mechanics.
- Focus on actionable guidance and learner clarity.
""".strip()


def build_generator_user_prompt(intent: str, user_input: str, context_summary: str) -> str:
    """Build the user prompt for LLM-based answer generation."""

    return (
        f"Intent: {intent}\n"
        f"User request:\n{user_input}\n\n"
        f"Available context:\n{context_summary}\n\n"
        "Please answer as a helpful IELTS study assistant."
    )
