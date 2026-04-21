"""Task 2 topic ingestion utilities for the IELTS writing assistant."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from typing import Any

from project.db.repository import list_writing_task2_topics, upsert_writing_task2_topic
from project.llm.client import LLMClient
from project.rag.chunking_agent import ChunkPlan, export_chunks_jsonl, prepare_chunks
from project.tools.web_search_tool import search_web

logger = logging.getLogger(__name__)


TASK2_TYPES = ("观点类", "讨论类", "优缺点类", "问题解决类", "报告类")
SUPPORTED_IMPORT_SUFFIXES = (".txt", ".md", ".html", ".htm", ".pdf")
JSON_ARRAY_PATTERN = re.compile(r"\[[\s\S]*\]")
TOPIC_BLOCK_PATTERN = re.compile(r"(?=雅思大作文\|)")

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANUAL_TASK2_DIR = WORKSPACE_ROOT / "data" / "真题"
DEFAULT_MANUAL_CORPUS_DIR = WORKSPACE_ROOT / "data" / "外刊"
DEFAULT_STORAGE_ROOT = PROJECT_ROOT / "storage"
DEFAULT_CHUNK_PREVIEW_DIR = DEFAULT_STORAGE_ROOT / "chunk_previews"


@dataclass(slots=True)
class TopicIngestionReport:
    """High-level summary for a Task 2 ingestion run."""

    source_file: str
    plan: ChunkPlan
    chunk_count: int
    topic_count: int
    saved_count: int


def collect_source_files(source_dir: Path) -> list[Path]:
    """Collect supported source files from a manual intake directory."""

    if not source_dir.exists():
        return []
    return sorted(
        [
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMPORT_SUFFIXES
        ]
    )


def import_task2_bank_from_directory(
    source_dir: Path = DEFAULT_MANUAL_TASK2_DIR,
    *,
    llm_model: str | None = None,
    chunk_strategy: str = "llm_auto",
    chunk_size: int = 1800,
    overlap: int = 180,
) -> dict[str, Any]:
    """Parse local Task 2 source documents and store topics in SQLite."""

    reports: list[TopicIngestionReport] = []
    total_saved = 0
    total_topics = 0
    for path in collect_source_files(source_dir):
        plan, chunks = prepare_chunks(
            path,
            strategy=chunk_strategy,
            chunk_size=chunk_size,
            overlap=overlap,
            llm_model=llm_model,
        )
        preview_path = DEFAULT_CHUNK_PREVIEW_DIR / "task2" / f"{path.stem}.{plan.strategy}.jsonl"
        export_chunks_jsonl(chunks, preview_path)

        saved_for_file = 0
        seen_topic_ids: set[str] = set()
        extracted_for_file = 0
        for chunk in chunks:
            topics = extract_task2_topics_from_text(
                chunk.content,
                source_name=path.name,
                llm_model=llm_model,
                source_context={
                    "chunk_id": chunk.chunk_id,
                    "chunk_strategy": chunk.strategy,
                    "source_file": path.name,
                    "source_title": chunk.title,
                },
            )
            extracted_for_file += len(topics)
            for topic in topics:
                saved = upsert_writing_task2_topic(**topic)
                if saved["id"] in seen_topic_ids:
                    continue
                seen_topic_ids.add(saved["id"])
                saved_for_file += 1

        total_saved += saved_for_file
        total_topics += extracted_for_file
        reports.append(
            TopicIngestionReport(
                source_file=path.name,
                plan=plan,
                chunk_count=len(chunks),
                topic_count=extracted_for_file,
                saved_count=saved_for_file,
            )
        )

    return {
        "source_dir": str(source_dir),
        "files": len(reports),
        "reports": reports,
        "topic_count": total_topics,
        "saved_count": total_saved,
    }


def update_task2_bank_from_web(
    *,
    query: str = "latest IELTS writing task 2 questions 2026",
    llm_model: str | None = None,
    result_limit: int = 5,
) -> dict[str, Any]:
    """Search recent web results, parse Task 2 prompts, and store them in SQLite."""

    results = search_web(query)[: max(1, result_limit)]
    source_text = _search_results_to_source_text(results)
    topics = extract_task2_topics_from_text(
        source_text,
        source_name="web_search_results",
        llm_model=llm_model,
        source_context={"query": query, "result_count": len(results)},
    )
    saved_count = 0
    for topic in topics:
        upsert_writing_task2_topic(**topic)
        saved_count += 1

    return {
        "query": query,
        "result_count": len(results),
        "topic_count": len(topics),
        "saved_count": saved_count,
        "results": results,
    }


def prepare_external_corpus_locally(
    source_dir: Path = DEFAULT_MANUAL_CORPUS_DIR,
    *,
    llm_model: str | None = None,
    chunk_strategy: str = "llm_auto",
    chunk_size: int = 1200,
    overlap: int = 150,
    max_chunks_per_file: int = 0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Chunk local corpus files and export previews for the built-in simple RAG."""

    files = collect_source_files(source_dir)
    reports: list[dict[str, Any]] = []

    for path in files:
        plan, chunks = prepare_chunks(
            path,
            strategy=chunk_strategy,
            chunk_size=chunk_size,
            overlap=overlap,
            llm_model=llm_model,
        )
        if max_chunks_per_file > 0:
            chunks = chunks[:max_chunks_per_file]

        preview_path = DEFAULT_CHUNK_PREVIEW_DIR / "local_rag" / f"{path.stem}.{plan.strategy}.jsonl"
        if not dry_run:
            export_chunks_jsonl(chunks, preview_path)

        reports.append(
            {
                "source_file": path.name,
                "plan": plan,
                "chunk_count": len(chunks),
                "prepared": len(chunks),
                "preview_path": str(preview_path),
            }
        )

    return {
        "source_dir": str(source_dir),
        "files": len(files),
        "reports": reports,
        "dry_run": dry_run,
    }


def extract_task2_topics_from_text(
    text: str,
    *,
    source_name: str,
    llm_model: str | None = None,
    source_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Extract structured Task 2 topics from raw source text."""

    raw_entries = _extract_topics_with_llm(text, source_name, llm_model)
    if not raw_entries:
        raw_entries = _fallback_extract_topics(text)

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_entries:
        normalized_item = _normalize_topic_item(item, source_name=source_name, source_context=source_context or {})
        if normalized_item is None:
            continue
        unique_key = "|".join(
            [
                normalized_item["exam_date"],
                normalized_item["essay_type"],
                normalized_item["prompt_text"].lower(),
            ]
        )
        if unique_key in seen:
            continue
        seen.add(unique_key)
        normalized.append(normalized_item)
    return normalized


def _extract_topics_with_llm(text: str, source_name: str, llm_model: str | None) -> list[dict[str, Any]]:
    client = LLMClient.from_config()
    if not client.is_configured:
        return []

    raw_response = client.generate_text(
        _task2_extraction_system_prompt(),
        _task2_extraction_user_prompt(text, source_name),
        model=llm_model or None,
        temperature=0.0,
        max_tokens=1800,
    )
    return _parse_topic_array(raw_response)


def _task2_extraction_system_prompt() -> str:
    return (
        "You extract IELTS Writing Task 2 prompts from source notes.\n"
        "Return JSON only. The root must be an array.\n"
        "Each item schema: "
        '{"exam_date":"...","essay_type":"观点类|讨论类|优缺点类|问题解决类|报告类",'
        '"topic_category":"...","prompt_text":"...","prompt_translation":"...","source_excerpt":"..."}\n'
        "Rules:\n"
        "- Only keep IELTS Writing Task 2 / big essay prompts.\n"
        "- Ignore instructions like 'Write at least 250 words' unless useful in source_excerpt.\n"
        "- prompt_text must be the actual English essay question.\n"
        "- essay_type must use one of the five allowed Chinese labels.\n"
        "- If exam_date is unclear, preserve the source text as-is instead of inventing a precise date.\n"
    )


def _task2_extraction_user_prompt(text: str, source_name: str) -> str:
    return (
        f"Source: {source_name}\n"
        f"Character count: {len(text)}\n\n"
        "Extract all Task 2 prompts from the following text.\n\n"
        f"{text[:10000]}"
    )


def _parse_topic_array(raw_response: str | None) -> list[dict[str, Any]]:
    if not raw_response:
        return []
    candidate = raw_response.strip()
    array_match = JSON_ARRAY_PATTERN.search(candidate)
    if array_match:
        candidate = array_match.group(0)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        logger.warning("Failed to parse Task 2 extraction response: %s", raw_response[:120])
        return []

    if isinstance(parsed, dict):
        parsed = parsed.get("topics", [])
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _fallback_extract_topics(text: str) -> list[dict[str, Any]]:
    blocks = _split_task2_blocks(text)
    topics: list[dict[str, Any]] = []
    for block in blocks:
        header, prompt_lines, translation_lines = _parse_task2_block(block)
        if not prompt_lines:
            continue
        prompt_text = " ".join(prompt_lines).strip()
        topics.append(
            {
                "exam_date": _extract_exam_date(header),
                "essay_type": _extract_essay_type(header, prompt_text),
                "topic_category": _extract_topic_category(header),
                "prompt_text": prompt_text,
                "prompt_translation": " ".join(translation_lines).strip(),
                "source_excerpt": block[:240].strip(),
            }
        )
    return topics


def _split_task2_blocks(text: str) -> list[str]:
    if "雅思大作文|" in text:
        parts = TOPIC_BLOCK_PATTERN.split(text)
        blocks = []
        for part in parts:
            cleaned = part.strip()
            if not cleaned:
                continue
            if not cleaned.startswith("雅思大作文|"):
                cleaned = f"雅思大作文| {cleaned}"
            blocks.append(cleaned)
        return blocks
    return [text]


def _parse_task2_block(block: str) -> tuple[str, list[str], list[str]]:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return "", [], []

    header = lines[0]
    prompt_lines: list[str] = []
    translation_lines: list[str] = []
    for line in lines[1:]:
        if _is_instruction_line(line):
            break
        if _looks_like_english_prompt_line(line):
            prompt_lines.append(line)
        elif prompt_lines:
            translation_lines.append(line)
    return header, prompt_lines, translation_lines


def _is_instruction_line(line: str) -> bool:
    lowered = line.lower()
    return lowered.startswith("give reasons") or lowered.startswith("write at least")


def _looks_like_english_prompt_line(line: str) -> bool:
    ascii_letters = sum(1 for char in line if char.isascii() and char.isalpha())
    return ascii_letters >= 10


def _extract_exam_date(header: str) -> str:
    header = header.strip()
    exact_match = re.search(r"(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?)", header)
    if exact_match:
        return exact_match.group(1).replace("年", "-").replace("月", "-").replace("日", "")

    short_match = re.search(r"(\d{2}年\d{1,2}月\d{1,2}日?)", header)
    if short_match:
        return short_match.group(1)

    year_month_match = re.search(r"(\d{2}年\d{1,2}月)", header)
    if year_month_match:
        return year_month_match.group(1)

    return "unknown"


def _extract_topic_category(header: str) -> str:
    header_parts = [part.strip() for part in re.split(r"[|（）()]", header) if part.strip()]
    for part in header_parts:
        if part in TASK2_TYPES or part.endswith("类") or "-" in part:
            continue
        if "雅思大作文" in part or "大作文" in part:
            continue
        if re.search(r"\d", part):
            continue
        return part
    return ""


def _extract_essay_type(header: str, prompt_text: str) -> str:
    for essay_type in TASK2_TYPES:
        if essay_type in header:
            return essay_type
    inferred = _infer_essay_type(prompt_text)
    return inferred or "观点类"


def _infer_essay_type(prompt_text: str) -> str:
    lowered = prompt_text.lower()
    if "discuss both" in lowered:
        return "讨论类"
    if "advantages" in lowered and "disadvantages" in lowered:
        return "优缺点类"
    if "outweigh the disadvantages" in lowered or "outweigh the advantages" in lowered:
        return "优缺点类"
    if "agree or disagree" in lowered or "to what extent" in lowered:
        return "观点类"
    if "what problems" in lowered or "what solutions" in lowered:
        return "问题解决类"
    if "why is this" in lowered or "what are the effects" in lowered or "what can be done" in lowered:
        return "报告类"
    return "观点类"


def _normalize_topic_item(
    item: dict[str, Any],
    *,
    source_name: str,
    source_context: dict[str, Any],
) -> dict[str, Any] | None:
    prompt_text = " ".join(str(item.get("prompt_text") or "").split()).strip()
    if len(prompt_text) < 20:
        return None

    essay_type = str(item.get("essay_type") or "").strip()
    if essay_type not in TASK2_TYPES:
        essay_type = _infer_essay_type(prompt_text)
    exam_date = str(item.get("exam_date") or "unknown").strip() or "unknown"
    topic_category = str(item.get("topic_category") or "").strip()
    prompt_translation = str(item.get("prompt_translation") or "").strip()
    source_excerpt = str(item.get("source_excerpt") or prompt_text[:240]).strip()

    return {
        "exam_date": exam_date,
        "prompt_text": prompt_text,
        "essay_type": essay_type,
        "topic_category": topic_category,
        "prompt_translation": prompt_translation,
        "source_title": str(source_context.get("source_title") or source_name),
        "source_file": str(source_context.get("source_file") or source_name),
        "source_excerpt": source_excerpt[:500],
        "metadata": {
            "source_context": source_context,
            "import_source": source_name,
        },
    }


def _search_results_to_source_text(results: list[dict[str, Any]]) -> str:
    sections: list[str] = []
    for index, item in enumerate(results, start=1):
        sections.append(
            "\n".join(
                [
                    f"Result {index}",
                    f"Title: {item.get('title', '')}",
                    f"Snippet: {item.get('snippet', '')}",
                    f"URL: {item.get('url', '')}",
                ]
            )
        )
    return "\n\n".join(sections)

def summarize_task2_bank(limit: int = 10) -> list[dict[str, Any]]:
    """Return recently stored Task 2 topics for quick inspection."""

    return list_writing_task2_topics(limit=limit)
