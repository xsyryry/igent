"""LLM-guided chunking utilities for local RAG corpus preparation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from html import unescape
from html.parser import HTMLParser
import json
import logging
from pathlib import Path
import re
from typing import Any, Callable

from project.config import get_config
from project.llm.client import LLMClient

logger = logging.getLogger(__name__)


SUPPORTED_TEXT_SUFFIXES = {".txt", ".md", ".html", ".htm", ".pdf"}
ALLOWED_STRATEGIES = ("auto", "llm_auto", "sliding", "headings", "qa_pairs", "rubric_items", "mistake_rules", "magazine_articles")
STRUCTURED_STRATEGIES = {"headings", "qa_pairs", "rubric_items", "mistake_rules", "magazine_articles"}
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", flags=re.DOTALL)
SKILL_PATH = Path(__file__).resolve().parents[1] / "agent" / "skills" / "chunking_skill.md"

HEADING_LINE_PATTERN = re.compile(
    r"(?m)^("
    r"#{1,4}\s+.+|"
    r"(?:Part|Task|Section|Chapter)\s+\d+.*|"
    r"[一二三四五六七八九十]+、.*|"
    r"（[一二三四五六七八九十]+）.*|"
    r"\d+(?:\.\d+)*\s+.+"
    r")$",
    flags=re.IGNORECASE,
)

QA_TITLE_PATTERN = re.compile(
    r"(?mis)(^|\n)(?P<title>"
    r"(?:Question\s*\d+.*)|"
    r"(?:Q\d+.*)|"
    r"(?:Task\s*[12].*)|"
    r"(?:题目.*)|"
    r"(?:Passage\s*\d+.*)"
    r")(?P<body>.*?)(?=("
    r"\n(?:Question\s*\d+|Q\d+|Task\s*[12]|题目|Passage\s*\d+)"
    r")|\Z)"
)

RUBRIC_ITEMS = [
    "Task Achievement",
    "Task Response",
    "Coherence and Cohesion",
    "Lexical Resource",
    "Grammatical Range and Accuracy",
    "Fluency and Coherence",
    "Pronunciation",
    "任务回应",
    "任务完成情况",
    "连贯与衔接",
    "词汇资源",
    "语法多样性与准确性",
    "流利度与连贯性",
    "发音",
]

MAGAZINE_SECTIONS = {
    "the world this week",
    "leaders",
    "letters",
    "by invitation",
    "briefing",
    "united states",
    "the americas",
    "asia",
    "china",
    "middle east & africa",
    "europe",
    "britain",
    "international",
    "business",
    "finance & economics",
    "science & technology",
    "culture",
    "economic & financial indicators",
    "obituary",
}


@dataclass(slots=True)
class ChunkRecord:
    """Normalized chunk object used for preview and later ingestion."""

    chunk_id: str
    source_file: str
    strategy: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChunkPlan:
    """Chunking decision produced by heuristics or LLM guidance."""

    requested_strategy: str
    strategy: str
    chunk_size: int
    overlap: int
    decision_source: str
    model_name: str
    reason: str


class _HTMLTextExtractor(HTMLParser):
    """Very small HTML to text extractor using stdlib only."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style"}:
            self._skip_depth += 1
            return
        if self._skip_depth == 0 and tag in {"p", "br", "div", "li", "section", "article", "h1", "h2", "h3", "h4"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if self._skip_depth == 0 and tag in {"p", "div", "li", "section", "article"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def load_document_text(path: Path) -> str:
    """Load text from a local document path."""

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_TEXT_SUFFIXES:
        raise ValueError(f"Unsupported file type for chunking: {suffix}")

    if suffix == ".pdf":
        return _read_pdf_text(path)
    if suffix in {".html", ".htm"}:
        return _read_html_text(path)
    return path.read_text(encoding="utf-8-sig", errors="ignore")


def prepare_chunks(
    path: Path,
    *,
    strategy: str = "auto",
    chunk_size: int = 1200,
    overlap: int = 150,
    llm_model: str | None = None,
) -> tuple[ChunkPlan, list[ChunkRecord]]:
    """Prepare chunks, optionally using an LLM to decide the chunking plan."""

    normalized_text = _normalize_text(load_document_text(path))
    plan = build_chunk_plan(
        path,
        normalized_text,
        requested_strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
        llm_model=llm_model,
    )
    chunks = _chunk_text(
        path,
        normalized_text,
        strategy=plan.strategy,
        chunk_size=plan.chunk_size,
        overlap=plan.overlap,
    )
    if chunks:
        executed_strategy = chunks[0].strategy
        if executed_strategy != plan.strategy:
            plan.reason = f"{plan.reason} Execution fell back to {executed_strategy} based on document structure."
            plan.strategy = executed_strategy
    return plan, chunks


def build_chunk_plan(
    path: Path,
    text: str,
    *,
    requested_strategy: str = "auto",
    chunk_size: int = 1200,
    overlap: int = 150,
    llm_model: str | None = None,
) -> ChunkPlan:
    """Resolve a final chunking plan from user intent and document content."""

    normalized_text = _normalize_text(text)
    config = get_config()
    effective_model = llm_model or config.chunk_llm_model or config.llm_model
    heuristic_strategy = resolve_chunk_strategy(path, normalized_text, "auto")
    if requested_strategy not in {"llm_auto", "auto"}:
        return ChunkPlan(
            requested_strategy=requested_strategy,
            strategy=requested_strategy,
            chunk_size=_sanitize_chunk_size(chunk_size),
            overlap=_sanitize_overlap(overlap, chunk_size),
            decision_source="manual",
            model_name=effective_model if requested_strategy == "llm_auto" else "",
            reason="User-specified chunking strategy.",
        )

    if requested_strategy == "auto":
        return ChunkPlan(
            requested_strategy=requested_strategy,
            strategy=heuristic_strategy,
            chunk_size=_sanitize_chunk_size(chunk_size),
            overlap=_sanitize_overlap(overlap, chunk_size),
            decision_source="heuristic",
            model_name="",
            reason="Heuristic strategy selection based on filename and text structure.",
        )

    client = LLMClient.from_config()
    if not client.is_configured:
        return ChunkPlan(
            requested_strategy=requested_strategy,
            strategy=heuristic_strategy,
            chunk_size=_sanitize_chunk_size(chunk_size),
            overlap=_sanitize_overlap(overlap, chunk_size),
            decision_source="fallback_heuristic",
            model_name=effective_model,
            reason="LLM unavailable, fell back to heuristic chunk planning.",
        )

    skill_prompt = _load_skill_prompt()
    raw_response = client.generate_text(
        skill_prompt,
        _build_chunking_user_prompt(path, normalized_text, heuristic_strategy, chunk_size, overlap),
        model=effective_model or None,
        temperature=0.0,
        max_tokens=350,
    )
    parsed = _parse_llm_plan(raw_response)
    if parsed is None and raw_response:
        retry_response = client.generate_text(
            "You convert chunk planning output into valid JSON. Return one JSON object only.",
            _build_chunking_repair_prompt(raw_response),
            model=effective_model or None,
            temperature=0.0,
            max_tokens=220,
        )
        parsed = _parse_llm_plan(retry_response)
        if parsed is not None:
            parsed["decision_source"] = "llm_retry"
    if parsed is None:
        return ChunkPlan(
            requested_strategy=requested_strategy,
            strategy=heuristic_strategy,
            chunk_size=_sanitize_chunk_size(chunk_size),
            overlap=_sanitize_overlap(overlap, chunk_size),
            decision_source="fallback_heuristic",
            model_name=effective_model or client.model,
            reason="LLM chunk planner returned invalid output, fell back to heuristic chunk planning.",
        )

    decision_source = str(parsed.get("decision_source", "llm")).strip() or "llm"
    chosen_strategy = parsed.get("strategy", heuristic_strategy)
    if chosen_strategy not in STRUCTURED_STRATEGIES and chosen_strategy != "sliding":
        chosen_strategy = heuristic_strategy
    guarded_strategy = _apply_strategy_guardrails(path, normalized_text, chosen_strategy, heuristic_strategy)
    if guarded_strategy != chosen_strategy:
        chosen_strategy = guarded_strategy
        decision_source = "llm_guardrailed" if decision_source == "llm" else f"{decision_source}_guardrailed"

    chosen_chunk_size = _sanitize_chunk_size(_safe_int(parsed.get("chunk_size"), chunk_size))
    chosen_overlap = _sanitize_overlap(_safe_int(parsed.get("overlap"), overlap), chosen_chunk_size)
    if chosen_strategy in STRUCTURED_STRATEGIES:
        chosen_overlap = _sanitize_overlap(min(chosen_overlap, 120), chosen_chunk_size)

    return ChunkPlan(
        requested_strategy=requested_strategy,
        strategy=chosen_strategy,
        chunk_size=chosen_chunk_size,
        overlap=chosen_overlap,
        decision_source=decision_source,
        model_name=effective_model or client.model,
        reason=str(parsed.get("reason", "LLM selected the chunking plan.")).strip() or "LLM selected the chunking plan.",
    )


def export_chunks_jsonl(chunks: list[ChunkRecord], output_path: Path) -> None:
    """Write chunk records to JSONL for later inspection."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        for chunk in chunks:
            file_obj.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def resolve_chunk_strategy(path: Path, text: str, requested: str) -> str:
    """Resolve the final strategy, using heuristics when `auto` is requested."""

    if requested != "auto":
        return requested

    filename = path.name.lower()
    lower_text = text.lower()
    if path.suffix.lower() == ".pdf" and "awesome-english-ebooks" in {part.lower() for part in path.parts}:
        return "magazine_articles"
    if "band descriptor" in filename or "评分标准" in path.name:
        return "rubric_items"
    if "真题" in path.name or "question" in filename or "task 2" in lower_text:
        return "qa_pairs"
    if "错题" in path.name or "wrong" in filename or "mistake" in filename:
        return "mistake_rules"
    if _count_heading_markers(text) >= 3:
        return "headings"
    return "sliding"


def _read_html_text(path: Path) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(path.read_text(encoding="utf-8-sig", errors="ignore"))
    return unescape(parser.get_text())


def _read_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "PDF chunking requires pypdf. Install it with: pip install -r project/requirements.txt"
        ) from exc

    try:
        reader = PdfReader(str(path))
        pages: list[str] = []
        for page_index, page in enumerate(reader.pages, start=1):
            extracted = (page.extract_text() or "").strip()
            if extracted:
                pages.append(f"[Page {page_index}]\n{extracted}")
    except Exception as exc:  # pragma: no cover - parser errors depend on source files
        raise RuntimeError(
            f"PDF text extraction failed for {path.name}: {exc}"
        ) from exc

    if not pages:
        raise RuntimeError(f"PDF text extraction produced no text for {path.name}")
    return "\n\n".join(pages)


def _normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    collapsed = "\n".join(lines)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
    return collapsed.strip()


def _count_heading_markers(text: str) -> int:
    patterns = (
        r"(?m)^#{1,4}\s+",
        r"(?m)^(Part|Task|Section|Chapter)\s+\d+",
        r"(?m)^[一二三四五六七八九十]+、",
        r"(?m)^（[一二三四五六七八九十]+）",
        r"(?m)^\d+(?:\.\d+)*\s+",
    )
    return sum(len(re.findall(pattern, text, flags=re.IGNORECASE)) for pattern in patterns)


def _chunk_text(
    path: Path,
    text: str,
    *,
    strategy: str,
    chunk_size: int,
    overlap: int,
) -> list[ChunkRecord]:
    builders: dict[str, Callable[[Path, str, int, int], list[ChunkRecord]]] = {
        "sliding": _chunk_by_sliding_window,
        "headings": _chunk_by_headings,
        "qa_pairs": _chunk_by_qa_pairs,
        "rubric_items": _chunk_by_rubric_items,
        "mistake_rules": _chunk_by_mistake_rules,
        "magazine_articles": _chunk_by_magazine_articles,
    }
    chunks = builders[strategy](path, text, chunk_size, overlap)
    return [chunk for chunk in chunks if chunk.content.strip()]


def _chunk_by_sliding_window(path: Path, text: str, chunk_size: int, overlap: int) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    step = max(1, chunk_size - overlap)
    cursor = 0
    index = 1

    while cursor < len(text):
        chunk_text = text[cursor : cursor + chunk_size].strip()
        if chunk_text:
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{path.stem}-sliding-{index:03d}",
                    source_file=path.name,
                    strategy="sliding",
                    title=f"{path.stem} chunk {index}",
                    content=chunk_text,
                    metadata={"start": cursor, "end": cursor + len(chunk_text)},
                )
            )
            index += 1
        cursor += step

    return chunks


def _chunk_by_headings(path: Path, text: str, chunk_size: int, overlap: int) -> list[ChunkRecord]:
    del chunk_size, overlap
    matches = list(HEADING_LINE_PATTERN.finditer(text))
    if not matches:
        return _chunk_by_sliding_window(path, text, 1200, 150)

    chunks: list[ChunkRecord] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        title = match.group(0).strip()
        content = block[len(title) :].strip() if block.startswith(title) else block
        chunks.append(
            ChunkRecord(
                chunk_id=f"{path.stem}-headings-{index + 1:03d}",
                source_file=path.name,
                strategy="headings",
                title=title,
                content=content or block,
                metadata={"heading": title},
            )
        )
    return chunks


def _chunk_by_qa_pairs(path: Path, text: str, chunk_size: int, overlap: int) -> list[ChunkRecord]:
    del chunk_size, overlap
    matches = list(QA_TITLE_PATTERN.finditer(text))
    if not matches:
        return _chunk_by_headings(path, text, 1200, 150)

    chunks: list[ChunkRecord] = []
    for index, match in enumerate(matches, start=1):
        title = match.group("title").strip()
        body = match.group("body").strip()
        chunks.append(
            ChunkRecord(
                chunk_id=f"{path.stem}-qa-{index:03d}",
                source_file=path.name,
                strategy="qa_pairs",
                title=title,
                content=body,
                metadata={"pair_type": "question_answer"},
            )
        )
    return chunks


def _chunk_by_rubric_items(path: Path, text: str, chunk_size: int, overlap: int) -> list[ChunkRecord]:
    del chunk_size, overlap
    title_pattern = "|".join(re.escape(item) for item in RUBRIC_ITEMS)
    item_regex = re.compile(
        rf"(?mis)(^|\n)(?P<title>{title_pattern})(?P<body>.*?)(?=(\n(?:{title_pattern}))|\Z)"
    )
    matches = list(item_regex.finditer(text))
    if not matches:
        return _chunk_by_headings(path, text, 1200, 150)

    chunks: list[ChunkRecord] = []
    for index, match in enumerate(matches, start=1):
        title = match.group("title").strip()
        body = match.group("body").strip()
        chunks.append(
            ChunkRecord(
                chunk_id=f"{path.stem}-rubric-{index:03d}",
                source_file=path.name,
                strategy="rubric_items",
                title=title,
                content=body,
                metadata={"rubric_item": title},
            )
        )
    return chunks


def _chunk_by_mistake_rules(path: Path, text: str, chunk_size: int, overlap: int) -> list[ChunkRecord]:
    del chunk_size, overlap
    rule_regex = re.compile(
        r"(?m)^(?:[-*]\s+|\d+[.)]\s+|[一二三四五六七八九十]+、|（[一二三四五六七八九十]+）)(.+)$"
    )
    matches = list(rule_regex.finditer(text))
    if not matches:
        return _chunk_by_sliding_window(path, text, 800, 100)

    chunks: list[ChunkRecord] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        title = match.group(1).strip()
        chunks.append(
            ChunkRecord(
                chunk_id=f"{path.stem}-rule-{index + 1:03d}",
                source_file=path.name,
                strategy="mistake_rules",
                title=title[:80],
                content=block,
                metadata={"rule_title": title[:80]},
            )
        )
    return chunks


def _chunk_by_magazine_articles(path: Path, text: str, chunk_size: int, overlap: int) -> list[ChunkRecord]:
    del overlap
    pages = _split_pdf_pages(text)
    if not pages:
        return _chunk_by_headings(path, text, chunk_size, 0)

    sections = _collect_magazine_sections(pages)
    articles: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_section = ""

    for page_number, page_text in pages:
        header = _detect_magazine_header(page_text, current_section, sections)
        if header["section"]:
            current_section = str(header["section"])
        title = str(header["title"])
        if title:
            if current is not None:
                articles.append(current)
            current = {"section": current_section, "title": title, "pages": []}
        elif current is None:
            if page_number < 6 or _looks_like_contents_page(page_text, sections):
                continue
            current = {"section": current_section or "magazine", "title": f"{path.stem} page {page_number}", "pages": []}

        body = str(header["body"]).strip()
        if body and not _is_ad_page(body):
            current["pages"].append((page_number, body))

    if current is not None:
        articles.append(current)

    chunks: list[ChunkRecord] = []
    for article in articles:
        chunks.extend(_split_magazine_article(path, article, chunk_size, len(chunks) + 1))
    return chunks or _chunk_by_sliding_window(path, text, chunk_size, 0)


def _split_pdf_pages(text: str) -> list[tuple[int, str]]:
    matches = list(re.finditer(r"(?m)^\[Page\s+(\d+)\]\s*$", text))
    pages: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        page_text = text[start:end].strip()
        if page_text:
            pages.append((int(match.group(1)), page_text))
    return pages


def _collect_magazine_sections(pages: list[tuple[int, str]]) -> set[str]:
    sections = set(MAGAZINE_SECTIONS)
    for _, page_text in pages[:4]:
        for line in _clean_lines(page_text):
            if 2 <= len(line) <= 40 and not _looks_like_date(line):
                sections.add(line.lower())
    return sections


def _detect_magazine_header(page_text: str, current_section: str, sections: set[str]) -> dict[str, str]:
    lines = _clean_lines(page_text)
    section = current_section
    title = ""
    consumed = 0
    for index, line in enumerate(lines[:10]):
        lowered = line.lower()
        if _is_noise_line(line):
            consumed = index + 1
            continue
        if lowered in sections:
            section = line
            consumed = index + 1
            if index + 1 < len(lines) and lines[index + 1].lower() == lowered:
                title = lines[index + 1]
                consumed = index + 2
                break
            continue
        if _looks_like_magazine_title(line):
            title = line
            consumed = index + 1
            break
        break

    body_lines = lines[consumed:]
    while body_lines and (_is_noise_line(body_lines[0]) or body_lines[0] == title):
        body_lines = body_lines[1:]
    return {"section": section, "title": title, "body": "\n".join(body_lines)}


def _split_magazine_article(path: Path, article: dict[str, Any], chunk_size: int, start_index: int) -> list[ChunkRecord]:
    title = str(article["title"])
    section = str(article["section"] or "magazine")
    max_chars = max(chunk_size, 900)
    chunks: list[ChunkRecord] = []
    buffer: list[str] = []
    buffer_pages: list[int] = []
    index = start_index

    for page_number, body in article["pages"]:
        page_block = f"[Page {page_number}]\n{body}".strip()
        if buffer and sum(len(part) for part in buffer) + len(page_block) > max_chars:
            chunks.append(_magazine_chunk(path, title, section, buffer, buffer_pages, index))
            index += 1
            buffer = []
            buffer_pages = []
        buffer.append(page_block)
        buffer_pages.append(int(page_number))

    if buffer:
        chunks.append(_magazine_chunk(path, title, section, buffer, buffer_pages, index))
    return chunks


def _magazine_chunk(path: Path, title: str, section: str, parts: list[str], pages: list[int], index: int) -> ChunkRecord:
    page_range = _page_range(pages)
    content = f"{section}\n{title}\n\n" + "\n\n".join(parts)
    return ChunkRecord(
        chunk_id=f"{path.stem}-magazine-{index:03d}",
        source_file=path.name,
        strategy="magazine_articles",
        title=title,
        content=content.strip(),
        metadata={
            "article_title": title,
            "section": section,
            "section_title": section,
            "page_range": page_range,
            "page_start": min(pages),
            "page_end": max(pages),
            "chunk_type": "article",
        },
    )


def _clean_lines(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]


def _looks_like_magazine_title(line: str) -> bool:
    if not 4 <= len(line) <= 90 or _looks_like_date(line):
        return False
    if line.endswith(".") and len(line.split()) > 5:
        return False
    if len(line.split()) > 12:
        return False
    return bool(re.search(r"[A-Za-z]", line))


def _is_noise_line(line: str) -> bool:
    lowered = line.lower()
    return (
        _looks_like_date(line)
        or lowered.startswith("vol ")
        or lowered == "contents"
        or "点击" in line
        or ("app" in lowered and re.search(r"[\u4e00-\u9fff]", line) is not None)
    )


def _is_ad_page(text: str) -> bool:
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_count = len(re.findall(r"[A-Za-z]", text))
    return cjk_count > latin_count and ("点击" in text or "App" in text)


def _looks_like_contents_page(text: str, sections: set[str]) -> bool:
    lines = _clean_lines(text)
    if len(lines) < 6:
        return False
    section_hits = sum(1 for line in lines if line.lower() in sections)
    return section_hits >= max(5, len(lines) // 2)


def _looks_like_date(line: str) -> bool:
    return bool(re.search(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?\s+\d{4}\b", line))


def _page_range(pages: list[int]) -> str:
    if not pages:
        return ""
    start, end = min(pages), max(pages)
    return str(start) if start == end else f"{start}-{end}"


def _load_skill_prompt() -> str:
    if not SKILL_PATH.exists():
        return (
            "You are a chunking planner. Return JSON only with strategy, chunk_size, overlap, and reason. "
            "Allowed strategies: sliding, headings, qa_pairs, rubric_items, mistake_rules, magazine_articles."
        )
    return SKILL_PATH.read_text(encoding="utf-8")


def _build_chunking_user_prompt(
    path: Path,
    text: str,
    heuristic_strategy: str,
    chunk_size: int,
    overlap: int,
) -> str:
    snapshot = text[:5000]
    structure_summary = _summarize_structure_features(text)
    return (
        "You are choosing a chunking plan for local RAG ingestion.\n"
        f"Filename: {path.name}\n"
        f"Suffix: {path.suffix.lower()}\n"
        f"Character count: {len(text)}\n"
        f"Heuristic fallback strategy: {heuristic_strategy}\n"
        f"Observed structure: {structure_summary}\n"
        f"Current defaults: chunk_size={chunk_size}, overlap={overlap}\n\n"
        "Document preview:\n"
        f"{snapshot}\n\n"
        "Return one JSON object only. Do not add markdown, explanation, or code fences. "
        "If uncertain, still return valid JSON with one allowed strategy.\n"
        "Schema: "
        '{"strategy": "sliding|headings|qa_pairs|rubric_items|mistake_rules|magazine_articles", '
        '"chunk_size": 1200, "overlap": 120, "reason": "..."}'
    )


def _build_chunking_repair_prompt(raw_response: str) -> str:
    return (
        "Rewrite the following chunk-plan answer into valid JSON only.\n"
        "Allowed strategies: sliding, headings, qa_pairs, rubric_items, mistake_rules, magazine_articles.\n"
        "Schema: "
        '{"strategy": "sliding|headings|qa_pairs|rubric_items|mistake_rules|magazine_articles", '
        '"chunk_size": 1200, "overlap": 120, "reason": "..."}\n\n'
        "Original answer:\n"
        f"{raw_response}\n"
    )


def _parse_llm_plan(raw_response: str | None) -> dict[str, Any] | None:
    if not raw_response:
        return None

    candidate = _strip_code_fences(raw_response.strip())
    parsed = _try_parse_json_object(candidate)
    if parsed is not None:
        return parsed

    repaired = _repair_plan_from_text(candidate)
    if repaired is not None:
        logger.warning("Recovered non-JSON LLM chunking response via heuristic repair: %s", raw_response)
        repaired["decision_source"] = "llm_repaired"
        return repaired

    logger.warning("Failed to parse LLM chunking response: %s", raw_response)
    return None


def _strip_code_fences(text: str) -> str:
    fenced = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    fenced = re.sub(r"\s*```$", "", fenced.strip())
    return fenced.strip()


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    match = JSON_OBJECT_PATTERN.search(text)
    if match:
        text = match.group(0)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _repair_plan_from_text(text: str) -> dict[str, Any] | None:
    lowered = text.lower()
    strategy = next((item for item in STRUCTURED_STRATEGIES.union({"sliding"}) if item in lowered), None)
    if strategy is None:
        return None

    size_match = re.search(r"(?:chunk_size|size)\D{0,8}(\d{3,4})", lowered)
    overlap_match = re.search(r"overlap\D{0,8}(\d{1,3})", lowered)
    bare_numbers = [int(value) for value in re.findall(r"\b(\d{2,4})\b", lowered)]

    chunk_size = int(size_match.group(1)) if size_match else None
    overlap = int(overlap_match.group(1)) if overlap_match else None
    if chunk_size is None:
        chunk_size = next((value for value in bare_numbers if 400 <= value <= 2000), 1200)
    if overlap is None:
        overlap = next((value for value in bare_numbers if 0 <= value <= 300 and value != chunk_size), 120)

    cleaned_reason = re.sub(r"\s+", " ", text).strip()
    return {
        "strategy": strategy,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "reason": cleaned_reason[:240] or "Recovered from non-JSON LLM output.",
    }


def _apply_strategy_guardrails(
    path: Path,
    text: str,
    chosen_strategy: str,
    heuristic_strategy: str,
) -> str:
    filename = path.name.lower()
    lowered_text = text.lower()

    if "band descriptor" in filename or "assessment-criteria" in filename:
        allowed = {"rubric_items", "headings"}
        if chosen_strategy not in allowed:
            return heuristic_strategy if heuristic_strategy in allowed else "rubric_items"

    if any(marker in filename for marker in ("sample", "practice", "question")) or "listening and speaking tests are the same" in lowered_text:
        if chosen_strategy == "mistake_rules":
            return "qa_pairs"

    if "format-reading" in filename or ("format" in filename and "reading" in filename):
        if chosen_strategy in {"mistake_rules", "qa_pairs"}:
            return "headings"

    return chosen_strategy


def _summarize_structure_features(text: str) -> str:
    heading_count = _count_heading_markers(text)
    qa_count = len(QA_TITLE_PATTERN.findall(text))
    rubric_count = sum(1 for item in RUBRIC_ITEMS if item.lower() in text.lower())
    line_count = len(text.splitlines())
    return (
        f"headings={heading_count}, "
        f"qa_markers={qa_count}, "
        f"rubric_terms={rubric_count}, "
        f"lines={line_count}"
    )


def _sanitize_chunk_size(value: int) -> int:
    return max(400, min(2000, int(value)))


def _sanitize_overlap(value: int, chunk_size: int) -> int:
    bounded_chunk_size = _sanitize_chunk_size(chunk_size)
    max_overlap = max(0, min(300, bounded_chunk_size // 3))
    return max(0, min(max_overlap, int(value)))


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
