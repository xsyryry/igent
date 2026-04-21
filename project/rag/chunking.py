"""Document chunking utilities for local RAG preprocessing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any, Callable


SUPPORTED_TEXT_SUFFIXES = {".txt", ".md", ".html", ".htm", ".pdf"}


@dataclass(slots=True)
class ChunkRecord:
    """Normalized chunk object used for preview and later ingestion."""

    chunk_id: str
    source_file: str
    strategy: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class _HTMLTextExtractor(HTMLParser):
    """Very small HTML to text extractor using stdlib only."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
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


def chunk_document(
    path: Path,
    *,
    strategy: str = "auto",
    chunk_size: int = 1200,
    overlap: int = 150,
) -> list[ChunkRecord]:
    """Chunk a local document with the requested strategy."""

    text = _normalize_text(load_document_text(path))
    chosen_strategy = resolve_chunk_strategy(path, text, strategy)
    builders: dict[str, Callable[[Path, str, int, int], list[ChunkRecord]]] = {
        "sliding": _chunk_by_sliding_window,
        "headings": _chunk_by_headings,
        "qa_pairs": _chunk_by_qa_pairs,
        "rubric_items": _chunk_by_rubric_items,
        "mistake_rules": _chunk_by_mistake_rules,
    }
    chunks = builders[chosen_strategy](path, text, chunk_size, overlap)
    return [chunk for chunk in chunks if chunk.content.strip()]


def resolve_chunk_strategy(path: Path, text: str, requested: str) -> str:
    """Resolve the final strategy, using heuristics when `auto` is requested."""

    if requested != "auto":
        return requested

    filename = path.name.lower()
    if "band descriptor" in filename or "评分标准" in filename:
        return "rubric_items"
    if "真题" in path.name or "question" in filename or "task 2" in text.lower():
        return "qa_pairs"
    if "错题" in path.name or "wrong" in filename or "mistake" in filename:
        return "mistake_rules"
    if _count_heading_markers(text) >= 3:
        return "headings"
    return "sliding"


def export_chunks_jsonl(chunks: list[ChunkRecord], output_path: Path) -> None:
    """Write chunk records to JSONL for later inspection."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        for chunk in chunks:
            file_obj.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


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

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page_index, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        extracted = extracted.strip()
        if extracted:
            pages.append(f"[Page {page_index}]\n{extracted}")
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
        r"(?m)^\d+(\.\d+)*\s+",
    )
    return sum(len(re.findall(pattern, text, flags=re.IGNORECASE)) for pattern in patterns)


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
    heading_regex = re.compile(
        r"(?m)^(#{1,4}\s+.+|(?:Part|Task|Section|Chapter)\s+\d+.*|[一二三四五六七八九十]+、.*|（[一二三四五六七八九十]+）.*|\d+(?:\.\d+)*\s+.+)$",
        flags=re.IGNORECASE,
    )

    matches = list(heading_regex.finditer(text))
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
    pair_regex = re.compile(
        r"(?mis)(^|\n)(?P<title>(Question\s*\d+.*|Q\d+.*|Task\s*[12].*|题目.*|Passage\s*\d+.*))(?P<body>.*?)(?=(\n(?:Question\s*\d+|Q\d+|Task\s*[12]|题目|Passage\s*\d+))|\Z)"
    )
    matches = list(pair_regex.finditer(text))
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
    rubric_patterns = [
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
    item_regex = re.compile(
        rf"(?mis)(^|\n)(?P<title>{'|'.join(re.escape(item) for item in rubric_patterns)})(?P<body>.*?)(?=(\n(?:{'|'.join(re.escape(item) for item in rubric_patterns)}))|\Z)"
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
