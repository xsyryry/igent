"""Data collection tool guided by get_data_skill."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
import json
import logging
import os
import re
from pathlib import Path
import shutil
import time
import textwrap
from typing import Any
from urllib.parse import urlparse

import requests
from requests import Response, Session
from requests.exceptions import ConnectTimeout, ConnectionError, HTTPError, ReadTimeout, Timeout

from project.tools.web_search_tool import search


DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
MANIFEST_PATH = DATA_ROOT / "manifests" / "data_manifest.json"
RAW_DIR = DATA_ROOT / "raw"
EXPORT_DIR = DATA_ROOT / "exports"
PARSER_VERSION = "v1"
DOWNLOAD_CHUNK_SIZE = 8192
DEFAULT_CONNECT_TIMEOUT = 5
DEFAULT_READ_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = (2, 4, 8)
SLOW_DOMAIN_TIMEOUTS = {
    "takeielts.britishcouncil.org": (5, 45),
    "britishcouncil.org": (5, 45),
}
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) IELTS-Agent/1.0",
    "Accept": "text/html,application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
}

logger = logging.getLogger(__name__)

CATEGORY_DIRS = {
    "official_rubrics": "official_rubrics",
    "official_questions": "official_questions",
    "lecture_notes": "lecture_notes",
    "news_corpus": "news_corpus",
}


@dataclass(slots=True)
class DataSource:
    category: str
    title: str
    url: str
    file_name: str
    source: str
    notes: str
    module: str = "general"
    task_type: str = "general"
    source_type: str = "official"
    language: str = "en"
    tags: tuple[str, ...] = ()


CURATED_SOURCES: tuple[DataSource, ...] = (
    DataSource(
        "official_rubrics",
        "IELTS Writing band descriptors",
        "https://cdn.ielts.org/Guides/ielts-writing-band-descriptors.pdf",
        "official_writing_band_descriptors.pdf",
        "IELTS",
        "Official writing scoring descriptors.",
        module="writing",
        task_type="band_descriptors",
        tags=("writing", "rubric"),
    ),
    DataSource(
        "official_rubrics",
        "IELTS Speaking band descriptors",
        "https://ielts.org/cdn/ielts-guides/ielts-speaking-band-descriptors.pdf",
        "official_speaking_band_descriptors.pdf",
        "IELTS",
        "Official speaking scoring descriptors.",
        module="speaking",
        task_type="band_descriptors",
        tags=("speaking", "rubric"),
    ),
    DataSource(
        "official_rubrics",
        "IELTS writing assessment criteria",
        "https://ielts.org/en-us/news/2023/ielts-writing-band-descriptors-and-key-assessment-criteria",
        "official_writing_assessment_criteria.md",
        "IELTS",
        "Official explanation page for writing assessment criteria.",
        module="writing",
        task_type="assessment_criteria",
        tags=("writing", "rubric"),
    ),
    DataSource(
        "official_questions",
        "British Council free IELTS practice tests",
        "https://takeielts.britishcouncil.org/take-ielts/prepare/free-ielts-english-practice-tests",
        "british_council_free_ielts_practice_tests.md",
        "British Council",
        "Official practice test landing page.",
        module="general",
        task_type="practice_index",
        tags=("official", "practice"),
    ),
    DataSource(
        "official_questions",
        "IELTS Academic Writing Task 1 practice question",
        "https://takeielts.britishcouncil.org/take-ielts/prepare/free-ielts-english-practice-tests/writing/academic/task-1",
        "official_academic_writing_task1_practice.md",
        "British Council",
        "Official academic writing Task 1 practice page.",
        module="writing",
        task_type="task1",
        tags=("academic", "writing", "task1", "official"),
    ),
    DataSource(
        "official_questions",
        "IELTS Academic Writing Task 2 practice question",
        "https://takeielts.britishcouncil.org/take-ielts/prepare/free-ielts-english-practice-tests/writing/academic/task-2",
        "official_academic_writing_task2_practice.md",
        "British Council",
        "Official academic writing Task 2 practice page.",
        module="writing",
        task_type="task2",
        tags=("academic", "writing", "task2", "official"),
    ),
    DataSource(
        "official_questions",
        "IELTS Academic Reading practice questions",
        "https://takeielts.britishcouncil.org/sites/default/files/2018-01/Reading_Practice_1_IELTS_Academic_Questions.pdf",
        "official_academic_reading_practice_questions_01.pdf",
        "British Council",
        "Official academic reading practice PDF.",
        module="reading",
        task_type="mixed",
        tags=("academic", "reading", "official"),
    ),
    DataSource(
        "official_questions",
        "IELTS Academic Reading format",
        "https://ielts.org/take-a-test/test-types/ielts-academic-test/ielts-academic-format-reading",
        "official_academic_reading_format.md",
        "IELTS",
        "Official academic reading format page.",
        module="reading",
        task_type="format_overview",
        tags=("academic", "reading", "official"),
    ),
    DataSource(
        "lecture_notes",
        "British Council IELTS preparation resources",
        "https://takeielts.britishcouncil.org/take-ielts/prepare",
        "british_council_ielts_preparation_resources.md",
        "British Council",
        "Public IELTS preparation resource page.",
        source_type="teaching",
        tags=("preparation", "teaching"),
    ),
    DataSource(
        "news_corpus",
        "BBC Future education articles",
        "https://www.bbc.com/future/tags/education",
        "bbc_future_education_articles.md",
        "BBC",
        "Public article index for education-related reading materials.",
        source_type="practice",
        tags=("education", "reading"),
    ),
    DataSource(
        "news_corpus",
        "Reuters environment news",
        "https://www.reuters.com/business/environment/",
        "reuters_environment_news.md",
        "Reuters",
        "Public environment news page for reading corpus discovery.",
        source_type="practice",
        tags=("environment", "reading"),
    ),
)


class DataCollectionError(Exception):
    """Structured failure for data collection."""

    def __init__(self, status: str, reason: str, *, attempts: int = 0, elapsed_seconds: float = 0.0) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.attempts = attempts
        self.elapsed_seconds = elapsed_seconds

    def to_failure(self, source: DataSource) -> dict[str, Any]:
        return {
            "title": source.title,
            "url": source.url,
            "category": source.category,
            "status": self.status,
            "reason": self.reason,
            "retry_count": max(self.attempts - 1, 0),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style", "nav", "footer", "aside"}:
            self.skip_depth += 1
        elif self.skip_depth == 0 and tag in {"h1", "h2", "h3", "p", "li", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "footer", "aside"} and self.skip_depth > 0:
            self.skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.skip_depth == 0 and data.strip():
            self.parts.append(data.strip())

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "\n".join(self.parts)).strip()


def collect_data(user_input: str, *, category: str | None = None, limit: int = 4) -> dict[str, Any]:
    """Collect public IELTS/RAG data into raw + normalized local files."""

    categories = _infer_categories(user_input, category)
    request_profile = _extract_request_profile(user_input)
    requested_count = _extract_request_count(user_input)
    candidate_limit = min(max(limit, requested_count * 2), 30)
    search_query = _build_data_search_query(user_input, request_profile)
    discovered_sources = _discover_web_sources(search_query, categories, request_profile)
    curated_sources = _select_sources(categories, request_profile, candidate_limit)
    selected = _dedupe_sources([*discovered_sources, *curated_sources])[:candidate_limit]
    requested_format = _requested_export_format(user_input)
    request_meta = _extract_request_date(user_input)
    _ensure_data_dirs()

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    if not selected:
        failures.append(
            {
                "title": "No matching curated data source",
                "url": "",
                "category": ",".join(categories),
                "status": "failed_no_matching_source",
                "reason": (
                    "No curated source matched the requested module/task_type. "
                    f"request={request_profile}"
                ),
            }
        )

    with requests.Session() as session:
        session.headers.update(REQUEST_HEADERS)
        total_sources = len(selected)
        for index, source in enumerate(selected, start=1):
            try:
                saved = _download_source(
                    source,
                    index=index,
                    total_sources=total_sources,
                    session=session,
                    requested_format=requested_format if requested_count <= 1 else None,
                    request_meta=request_meta,
                    request_profile=request_profile,
                )
                successes.append(saved)
                if not _is_successful_collection(saved):
                    failures.append(
                        {
                            "title": source.title,
                            "url": source.url,
                            "category": source.category,
                            "status": "failed_parse",
                            "reason": (
                                "Saved raw/normalized files, but did not extract a complete "
                                f"structured question. status={saved.get('status')}"
                            ),
                            "saved_path": saved.get("saved_path", ""),
                            "export_path": saved.get("export_path", ""),
                        }
                    )
            except DataCollectionError as exc:
                failures.append(exc.to_failure(source))
            except Exception as exc:
                failures.append(
                    {
                        "title": source.title,
                        "url": source.url,
                        "category": source.category,
                        "status": "failed_unknown",
                        "reason": str(exc),
                    }
                )

    if successes:
        _append_manifest(successes)

    parsed_successes = _dedupe_successes_by_question([item for item in successes if _is_successful_collection(item)])
    selected_successes = parsed_successes[:requested_count]
    collection_export_path = (
        _export_collection(
            selected_successes,
            requested_format,
            request_profile=request_profile,
            request_meta=request_meta,
        )
        if requested_count > 1
        else None
    )
    if requested_count > len(parsed_successes):
        failures.append(
            {
                "title": "Requested question count not satisfied",
                "url": "",
                "category": ",".join(categories),
                "status": "partial_count_not_met",
                "reason": f"Requested {requested_count}, parsed {len(parsed_successes)} unique item(s).",
            }
        )

    return {
        "skill": "get_data_skill",
        "categories": categories,
        "success_count": len(selected_successes),
        "requested_count": requested_count,
        "matched_count": len(parsed_successes),
        "collection_status": "complete" if len(parsed_successes) >= requested_count else "partial",
        "saved_count": len(successes),
        "parsed_count": sum(1 for item in successes if item.get("status") == "parsed"),
        "partial_count": sum(1 for item in successes if item.get("status") == "partial"),
        "raw_only_count": sum(1 for item in successes if item.get("status") == "raw_only"),
        "downloaded_files": sum(1 for item in successes if item.get("raw_file_type") == "pdf"),
        "extracted_pages": sum(1 for item in successes if item.get("raw_file_type") == "html"),
        "requested_format": requested_format,
        "request_profile": request_profile,
        "pipeline": {
            "structured_request": {**request_profile, **request_meta, "format": requested_format, "count": requested_count},
            "search_query": search_query,
            "web_candidates": len(discovered_sources),
            "selected_sources": [
                {
                    "title": source.title,
                    "url": source.url,
                    "module": source.module,
                    "task_type": source.task_type,
                }
                for source in selected
            ],
            "storage_format": "standard_json",
            "export_format": requested_format,
        },
        "save_root": str(DATA_ROOT),
        "fail_count": len(failures),
        "files": selected_successes,
        "saved_files": successes,
        "collection_export_path": str(collection_export_path) if collection_export_path else "",
        "failures": failures,
        "next_step": "Run python -m project.app.rag_indexer build --publication economist --date-from 2026.01.01 --date-to 2026.01.31 to build the local RAG index.",
    }


def _download_source(
    source: DataSource,
    *,
    index: int,
    total_sources: int,
    session: Session,
    requested_format: str | None,
    request_meta: dict[str, int | None],
    request_profile: dict[str, str | None],
) -> dict[str, Any]:
    started = time.perf_counter()
    requested_module = request_profile.get("module")
    requested_task_type = request_profile.get("task_type")
    if requested_module and source.module != requested_module:
        raise DataCollectionError(
            "failed_module_mismatch",
            f"Requested module={requested_module}, but source module={source.module}",
            attempts=0,
            elapsed_seconds=0.0,
        )
    if requested_task_type and source.task_type != requested_task_type:
        raise DataCollectionError(
            "failed_task_type_mismatch",
            f"Requested task_type={requested_task_type}, but source task_type={source.task_type}",
            attempts=0,
            elapsed_seconds=0.0,
        )
    content, content_type, final_url, attempts = _download_bytes_with_progress(
        source,
        index=index,
        total_sources=total_sources,
        session=session,
    )
    is_pdf = _is_pdf_response(content, content_type)

    raw_path = _save_raw_source(source, content, is_pdf=is_pdf)
    extracted_text = _extract_text(content, is_pdf=is_pdf, source=source)
    normalized = _build_standard_record(
        source=source,
        text=extracted_text,
        raw_path=raw_path,
        final_url=final_url,
        is_pdf=is_pdf,
        request_meta=request_meta,
    )
    normalized_path = _save_standard_record(normalized, source)
    export_path = _export_requested_format(normalized, raw_path, requested_format, is_pdf=is_pdf)

    status = str(normalized["status"])
    elapsed = time.perf_counter() - started
    logger.info(
        "Collected data source url=%s final_url=%s type=%s status=%s attempts=%s elapsed=%.2fs",
        source.url,
        final_url,
        content_type,
        status,
        attempts,
        elapsed,
    )
    return {
        "file_name": normalized_path.name,
        "category": source.category,
        "module": source.module,
        "status": status,
        "source": source.source,
        "title": source.title,
        "url": source.url,
        "final_url": final_url,
        "saved_path": str(normalized_path),
        "raw_path": str(raw_path),
        "export_path": str(export_path) if export_path else "",
        "collected_at": normalized["collected_at"],
        "language": source.language,
        "notes": source.notes,
        "file_type": "json",
        "raw_file_type": "pdf" if is_pdf else "html",
        "retry_count": max(attempts - 1, 0),
        "elapsed_seconds": round(elapsed, 2),
    }


def _build_standard_record(
    *,
    source: DataSource,
    text: str,
    raw_path: Path,
    final_url: str,
    is_pdf: bool,
    request_meta: dict[str, int | None],
) -> dict[str, Any]:
    year = request_meta.get("year")
    month = request_meta.get("month")
    collected_at = datetime.utcnow().replace(microsecond=0).isoformat()
    status = _parse_status(source, text)
    base = {
        "id": _record_id(source, year, month),
        "source": source.source,
        "source_type": source.source_type,
        "exam": "IELTS",
        "module": source.module,
        "task_type": source.task_type,
        "title": source.title,
        "year": year,
        "month": month,
        "set_name": _set_name(source, year, month),
        "language": source.language,
        "url": source.url,
        "final_url": final_url,
        "local_raw_path": str(raw_path),
        "collected_at": collected_at,
        "parser_version": PARSER_VERSION,
        "status": status,
        "tags": list(source.tags),
        "notes": source.notes,
    }
    if source.category == "official_questions":
        return _question_record(base, source, text, is_pdf=is_pdf)
    if source.category == "official_rubrics":
        return {**base, "rubric_text": text, "raw_format": "pdf" if is_pdf else "html"}
    return {**base, "content": text, "raw_format": "pdf" if is_pdf else "html"}


def _question_record(base: dict[str, Any], source: DataSource, text: str, *, is_pdf: bool) -> dict[str, Any]:
    module = source.module
    if module == "reading":
        groups = _extract_question_groups(text)
        return {
            **base,
            "status": "partial" if groups else "raw_only",
            "passage": {"title": source.title, "text": text},
            "question_groups": groups,
        }
    if module == "listening":
        return {
            **base,
            "status": "raw_only",
            "audio_url": None,
            "transcript": text if not is_pdf else "",
            "question_groups": _extract_question_groups(text),
        }
    if module == "writing":
        prompt = _extract_writing_prompt(text, source.task_type)
        return {
            **base,
            "status": "parsed" if prompt else "raw_only",
            "prompt": prompt,
            "essay_type": _infer_essay_type(prompt),
            "topic_tags": list(source.tags),
            "word_limit": 250 if source.task_type == "task2" else 150,
            "sample_answer": None,
            "band_descriptors_refs": [],
            "raw_text_excerpt": text[:1000] if not prompt else "",
        }
    if module == "speaking":
        return {
            **base,
            "status": "raw_only",
            "parts": {"part1": {"topic": None, "questions": []}, "part2": {}, "part3": {"questions": []}},
            "raw_text": text,
        }
    return {**base, "status": "raw_only", "raw_text": text}


def _download_bytes_with_progress(
    source: DataSource,
    *,
    index: int,
    total_sources: int,
    session: Session,
) -> tuple[bytes, str, str, int]:
    print(f"Downloading {index}/{total_sources}: {source.file_name}")
    started = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(1, DEFAULT_MAX_RETRIES + 2):
        downloaded = 0
        chunks: list[bytes] = []
        try:
            timeout = _timeout_for_url(source.url)
            logger.info("Downloading url=%s attempt=%s timeout=%s", source.url, attempt, timeout)
            with session.get(source.url, timeout=timeout, stream=True) as response:
                _raise_for_retriable_status(response)
                response.raise_for_status()
                total = _safe_content_length(response.headers.get("content-length"))
                content_type = response.headers.get("content-type", "").lower()
                final_url = response.url
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    downloaded += len(chunk)
                    _print_download_progress(source.file_name, downloaded, total)
            _print_download_progress(source.file_name, downloaded, downloaded, done=True)
            return b"".join(chunks), content_type, final_url, attempt
        except (ReadTimeout, ConnectTimeout, Timeout) as exc:
            last_error = exc
            status = "failed_timeout"
        except (ConnectionError, HTTPError) as exc:
            last_error = exc
            status = "failed_http"
        if attempt > DEFAULT_MAX_RETRIES or not _should_retry(last_error):
            elapsed = time.perf_counter() - started
            raise DataCollectionError(status, str(last_error), attempts=attempt, elapsed_seconds=elapsed) from last_error
        delay = RETRY_BACKOFF_SECONDS[min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)]
        print(f"\n  retrying in {delay}s after: {last_error}")
        time.sleep(delay)
    elapsed = time.perf_counter() - started
    raise DataCollectionError("failed_unknown", "download failed", attempts=DEFAULT_MAX_RETRIES + 1, elapsed_seconds=elapsed)


def _infer_categories(user_input: str, category: str | None) -> list[str]:
    if category and category in CATEGORY_DIRS:
        return [category]
    text = user_input.lower()
    categories: list[str] = []
    if any(token in user_input for token in ("评分标准", "评分", "band descriptor")) or "rubric" in text:
        categories.append("official_rubrics")
    if any(token in user_input for token in ("样题", "真题", "练习题", "题目")) or "sample" in text or "practice" in text:
        categories.append("official_questions")
    if any(token in user_input for token in ("讲义", "教学", "备考资料")) or "lecture" in text or "notes" in text:
        categories.append("lecture_notes")
    if any(token in user_input for token in ("外刊", "语料", "阅读素材")) or "news" in text or "corpus" in text:
        categories.append("news_corpus")
    if any(token in user_input for token in ("知识库", "rag", "资料库", "全部", "数据")):
        categories = list(CATEGORY_DIRS)
    request_profile = _extract_request_profile(user_input)
    if request_profile.get("task_type") and "official_questions" not in categories:
        categories.append("official_questions")
    if request_profile.get("task_type") and "official_questions" in categories and "rubric" not in text:
        categories = [item for item in categories if item == "official_questions"]
    return categories or ["official_rubrics", "official_questions"]


def _extract_request_profile(user_input: str) -> dict[str, str | None]:
    """Extract module/task hints so source selection cannot drift to the wrong IELTS module."""

    text = user_input.lower()
    module = None
    if "reading" in text or "阅读" in user_input:
        module = "reading"
    elif "listening" in text or "听力" in user_input:
        module = "listening"
    elif "writing" in text or "写作" in user_input:
        module = "writing"
    elif "speaking" in text or "口语" in user_input:
        module = "speaking"

    task_type = None
    markers = {
        "task1": ("task 1", "task1", "小作文"),
        "task2": ("task 2", "task2", "大作文"),
        "true_false_not_given": ("true false not given", "tfng", "判断题"),
        "matching_headings": ("matching headings", "标题匹配"),
        "part1": ("part 1", "part1"),
        "part2": ("part 2", "part2", "cue card"),
        "part3": ("part 3", "part3"),
    }
    for label, aliases in markers.items():
        if any(alias in text or alias in user_input for alias in aliases):
            task_type = label
            break

    if task_type in {"task1", "task2"} and module is None:
        module = "writing"
    if task_type in {"part1", "part2", "part3"} and module is None:
        module = "speaking"

    return {"module": module, "task_type": task_type}


def _extract_request_count(user_input: str) -> int:
    count_match = re.search(r"(?<!\d)(\d{1,2})\s*(?:份|道|个|套|篇|items?)", user_input, flags=re.IGNORECASE)
    if not count_match:
        count_match = re.search(r"\bcount\s*[:=]\s*(\d{1,2})\b", user_input, flags=re.IGNORECASE)
    count = int(count_match.group(1)) if count_match else 1
    return max(1, min(count, 30))


def _select_sources(categories: list[str], request_profile: dict[str, str | None], limit: int) -> list[DataSource]:
    selected = [source for source in CURATED_SOURCES if source.category in categories]
    requested_module = request_profile.get("module")
    requested_task_type = request_profile.get("task_type")

    if requested_task_type and "official_questions" in categories:
        selected = [source for source in selected if source.category == "official_questions"]
    if "official_questions" in categories and requested_module:
        selected = [
            source
            for source in selected
            if source.category != "official_questions" or source.module == requested_module
        ]
    if "official_questions" in categories and requested_task_type:
        selected = [
            source
            for source in selected
            if source.category != "official_questions" or source.task_type == requested_task_type
        ]

    return selected[:limit]


def _build_data_search_query(user_input: str, request_profile: dict[str, str | None]) -> str:
    parts = ["IELTS official public practice material"]
    if request_profile.get("module"):
        parts.append(str(request_profile["module"]))
    if request_profile.get("task_type"):
        parts.append(str(request_profile["task_type"]))
    parts.append(user_input)
    return " ".join(part for part in parts if part)


def _discover_web_sources(
    query: str,
    categories: list[str],
    request_profile: dict[str, str | None],
) -> list[DataSource]:
    """Turn real web-search results into crawlable sources.

    The current mock search backend returns placeholder URLs, so those are ignored.
    Once a real backend is configured, matching results can enter the same crawl ->
    normalize -> export pipeline as curated sources.
    """

    if "official_questions" not in categories:
        return []
    try:
        search_result = search(query=query, max_results=10, need_extract=False)
        results = search_result.get("results", [])
    except Exception as exc:
        logger.warning("Data collection web search failed: %s", exc)
        return []

    sources: list[DataSource] = []
    module = request_profile.get("module") or "general"
    task_type = request_profile.get("task_type") or "general"
    for index, result in enumerate(results[:5], start=1):
        url = str(result.get("url") or "").strip()
        if not url or "example.com" in url or "tavily.com" in url:
            continue
        title = str(result.get("title") or f"Web result {index}").strip()
        source = str(result.get("source") or urlparse(url).netloc or "web").strip()
        file_name = f"web_{module}_{task_type}_{index}.md"
        sources.append(
            DataSource(
                "official_questions",
                title,
                url,
                file_name,
                source,
                "Discovered by web search for the user's data collection request.",
                module=module,
                task_type=task_type,
                source_type="practice",
                tags=("web_search", module, task_type),
            )
        )
    return sources


def _dedupe_sources(sources: list[DataSource]) -> list[DataSource]:
    seen: set[str] = set()
    deduped: list[DataSource] = []
    for source in sources:
        key = source.url.rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


def _ensure_data_dirs() -> None:
    for path in (
        DATA_ROOT,
        RAW_DIR,
        EXPORT_DIR,
        MANIFEST_PATH.parent,
        DATA_ROOT / "official_rubrics",
        DATA_ROOT / "lecture_notes",
        DATA_ROOT / "news_corpus",
        DATA_ROOT / "official_questions" / "reading",
        DATA_ROOT / "official_questions" / "listening",
        DATA_ROOT / "official_questions" / "writing",
        DATA_ROOT / "official_questions" / "speaking",
    ):
        path.mkdir(parents=True, exist_ok=True)


def _save_raw_source(source: DataSource, content: bytes, *, is_pdf: bool) -> Path:
    suffix = ".pdf" if is_pdf else ".html"
    raw_path = RAW_DIR / f"{Path(source.file_name).stem}{suffix}"
    if is_pdf:
        raw_path.write_bytes(content)
    else:
        raw_path.write_text(content.decode("utf-8", errors="ignore"), encoding="utf-8")
    return raw_path


def _save_standard_record(record: dict[str, Any], source: DataSource) -> Path:
    if source.category == "official_questions":
        target_dir = DATA_ROOT / "official_questions" / source.module
        file_name = f"{record['id']}.json"
    else:
        target_dir = DATA_ROOT / CATEGORY_DIRS[source.category]
        file_name = f"{Path(source.file_name).stem}.json"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / file_name
    target_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return target_path


def _export_requested_format(record: dict[str, Any], raw_path: Path, requested_format: str | None, *, is_pdf: bool) -> Path | None:
    if not requested_format:
        return None
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stem = str(record["id"])
    if requested_format == "json":
        target = EXPORT_DIR / f"{stem}.json"
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return target
    if requested_format == "txt":
        target = EXPORT_DIR / f"{stem}.txt"
        target.write_text(_record_to_text(record), encoding="utf-8")
        return target
    if requested_format == "pdf":
        target = EXPORT_DIR / f"{stem}.pdf"
        _write_simple_pdf(target, title=str(record.get("title", stem)), text=_record_to_text(record))
        return target
    return None


def _export_collection(
    items: list[dict[str, Any]],
    requested_format: str | None,
    *,
    request_profile: dict[str, str | None],
    request_meta: dict[str, int | None],
) -> Path | None:
    if not requested_format or not items:
        return None
    records = []
    for item in items:
        try:
            records.append(json.loads(Path(str(item.get("saved_path", ""))).read_text(encoding="utf-8")))
        except Exception:
            continue
    if not records:
        return None

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stem = _collection_export_stem(request_profile, request_meta, len(records))
    if requested_format == "json":
        target = EXPORT_DIR / f"{stem}.json"
        target.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        return target
    text = _records_to_collection_text(records)
    if requested_format == "txt":
        target = EXPORT_DIR / f"{stem}.txt"
        target.write_text(text, encoding="utf-8")
        return target
    if requested_format == "pdf":
        target = EXPORT_DIR / f"{stem}.pdf"
        _write_simple_pdf(target, title="IELTS Question Collection", text=text)
        return target
    return None


def _collection_export_stem(request_profile: dict[str, str | None], request_meta: dict[str, int | None], count: int) -> str:
    module = request_profile.get("module") or "general"
    task_type = request_profile.get("task_type") or "mixed"
    year = request_meta.get("year") or "recent"
    month = f"{int(request_meta['month']):02d}" if request_meta.get("month") else "any"
    return re.sub(r"[^a-z0-9_]+", "_", f"ielts_{module}_{year}_{month}_{task_type}_{count}_items".lower())


def _records_to_collection_text(records: list[dict[str, Any]]) -> str:
    parts = []
    for index, record in enumerate(records, start=1):
        parts.append("=" * 72)
        parts.append(f"Question {index}")
        parts.append("=" * 72)
        parts.append(_record_to_text(record))
        parts.append("")
    return "\n".join(parts).strip()


def _write_simple_pdf(path: Path, *, title: str, text: str) -> None:
    """Write a dependency-free PDF export from normalized structured text."""

    font_name = _pdf_font_name()
    font_size = _pdf_font_size()
    line_height = max(font_size + 4, 12)
    max_lines = max(1, int(720 / line_height))
    lines = _wrap_pdf_text(f"{title}\n\n{text}")
    pages = [lines[index : index + max_lines] for index in range(0, len(lines), max_lines)] or [[""]]

    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"__PAGES__")
    objects.append(f"<< /Type /Font /Subtype /Type1 /BaseFont /{font_name} >>".encode("ascii"))

    page_ids: list[int] = []
    for page_lines in pages:
        page_id = len(objects) + 1
        content_id = page_id + 1
        page_ids.append(page_id)
        stream = _pdf_content_stream(page_lines, font_size=font_size, line_height=line_height)
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode("ascii")
        )
        objects.append(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")

    offsets: list[int] = []
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref_pos = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(bytes(output))


def _wrap_pdf_text(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            lines.append("")
            continue
        safe_line = raw_line.encode("latin-1", errors="replace").decode("latin-1")
        lines.extend(textwrap.wrap(safe_line, width=_pdf_wrap_width(), replace_whitespace=False) or [""])
    return lines


def _pdf_content_stream(lines: list[str], *, font_size: int, line_height: int) -> bytes:
    parts = ["BT", f"/F1 {font_size} Tf", "50 790 Td", f"{line_height} TL"]
    for line in lines:
        parts.append(f"({_pdf_escape(line)}) Tj")
        parts.append("T*")
    parts.append("ET")
    return "\n".join(parts).encode("latin-1", errors="replace")


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_font_name() -> str:
    allowed = {"Helvetica", "Times-Roman", "Courier"}
    value = os.getenv("DATA_EXPORT_PDF_FONT", "Helvetica").strip() or "Helvetica"
    return value if value in allowed else "Helvetica"


def _pdf_font_size() -> int:
    try:
        return max(8, min(int(os.getenv("DATA_EXPORT_PDF_FONT_SIZE", "10")), 16))
    except ValueError:
        return 10


def _pdf_wrap_width() -> int:
    font = _pdf_font_name()
    if font == "Courier":
        return 82
    return 92


def _extract_text(content: bytes, *, is_pdf: bool, source: DataSource) -> str:
    if not is_pdf:
        return _html_to_text(content.decode("utf-8", errors="ignore"))
    try:
        from pypdf import PdfReader
        import io

        reader = PdfReader(io.BytesIO(content))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        text = "\n\n".join(page for page in pages if page)
        return text or source.title
    except Exception:
        return source.title


def _html_to_text(html: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(html)
    return extractor.text()


def _is_pdf_response(content: bytes, content_type: str) -> bool:
    return "application/pdf" in content_type.lower() or content.startswith(b"%PDF")


def _parse_status(source: DataSource, text: str) -> str:
    if source.category != "official_questions":
        return "parsed" if text.strip() else "raw_only"
    if _extract_question_groups(text):
        return "partial"
    return "raw_only"


def _is_successful_collection(item: dict[str, Any]) -> bool:
    """Only fully parsed question records count as successful collection."""

    if item.get("category") == "official_questions":
        return item.get("status") == "parsed"
    return item.get("status") == "parsed"


def _dedupe_successes_by_question(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = _question_identity(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _question_identity(item: dict[str, Any]) -> str:
    try:
        record = json.loads(Path(str(item.get("saved_path", ""))).read_text(encoding="utf-8"))
    except Exception:
        return str(item.get("url") or item.get("saved_path") or item.get("title"))
    if record.get("module") == "writing":
        return re.sub(r"\W+", " ", str(record.get("prompt", "")).lower()).strip()
    if record.get("module") == "reading":
        passage = record.get("passage", {})
        return re.sub(r"\W+", " ", str(passage.get("text", "")).lower()).strip()[:500]
    return str(record.get("id") or item.get("saved_path"))


def _extract_writing_prompt(text: str, task_type: str) -> str:
    cleaned = _remove_noise_lines(text)
    if task_type == "task2":
        return _extract_task2_prompt(cleaned)
    if task_type == "task1":
        return _extract_task1_prompt(cleaned)
    return ""


def _remove_noise_lines(text: str) -> str:
    noise_markers = (
        "Skip to main content",
        "Menu",
        "Home",
        "In this section",
        "How to approach",
        "What should I do next?",
        "Once you",
        "Remember,",
        "See also",
        "Share this",
        "email",
        "facebook",
        "twitter",
        "linkedin",
    )
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(marker.lower() in stripped.lower() for marker in noise_markers):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def _extract_task2_prompt(text: str) -> str:
    marker_match = re.search(r"Task\s*2\s*-\s*Write about the following topic:\s*", text, flags=re.IGNORECASE)
    if marker_match:
        body = text[marker_match.end() :]
    else:
        body = text

    stop_match = re.search(
        r"\n(?:What should I do next\?|Once you|Remember,|See also|Share this|If you want to see)",
        body,
        flags=re.IGNORECASE,
    )
    if stop_match:
        body = body[: stop_match.start()]

    reason_match = re.search(
        r"You should give reasons for your answer.*?(?:experience or knowledge to support your response\.)",
        body,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if reason_match:
        body = body[: reason_match.end()]

    prompt = _normalize_prompt_text(body)
    if not _looks_like_writing_task2_prompt(prompt):
        return ""
    return prompt


def _extract_task1_prompt(text: str) -> str:
    marker_match = re.search(r"Task\s*1\s*-\s*Write about the following topic:\s*", text, flags=re.IGNORECASE)
    body = text[marker_match.end() :] if marker_match else text
    stop_match = re.search(r"\n(?:What should I do next\?|See also|Share this)", body, flags=re.IGNORECASE)
    if stop_match:
        body = body[: stop_match.start()]
    prompt = _normalize_prompt_text(body)
    if "summarise" not in prompt.lower() and "chart" not in prompt.lower() and "graph" not in prompt.lower():
        return ""
    return prompt


def _normalize_prompt_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    prompt = "\n".join(lines)
    prompt = re.sub(r"\n{3,}", "\n\n", prompt)
    return prompt.strip()


def _looks_like_writing_task2_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    if len(prompt) < 80:
        return False
    return any(
        marker in lowered
        for marker in (
            "in your opinion",
            "to what extent",
            "agree or disagree",
            "discuss both",
            "advantages",
            "disadvantages",
            "give reasons for your answer",
        )
    )


def _extract_question_groups(text: str) -> list[dict[str, Any]]:
    questions = []
    for match in re.finditer(r"(?m)^\s*(\d{1,2})[\).]?\s+(.{8,220})$", text):
        number = int(match.group(1))
        questions.append(
            {
                "number": number,
                "question_text": match.group(2).strip(),
                "options": [],
                "answer": None,
                "explanation": None,
            }
        )
    if not questions:
        return []
    qtype = "true_false_not_given" if "TRUE" in text and "FALSE" in text and "NOT GIVEN" in text else "mixed"
    return [
        {
            "group_id": "g1",
            "question_type": qtype,
            "instruction": "",
            "question_range": [questions[0]["number"], questions[-1]["number"]],
            "questions": questions[:40],
        }
    ]


def _record_to_text(record: dict[str, Any]) -> str:
    lines = [
        f"title: {record.get('title', '')}",
        f"source: {record.get('source', '')}",
        f"url: {record.get('url', '')}",
        f"module: {record.get('module', '')}",
        f"task_type: {record.get('task_type', '')}",
        f"status: {record.get('status', '')}",
        "",
    ]
    if record.get("module") == "reading":
        passage = record.get("passage", {})
        lines.extend([f"passage_title: {passage.get('title', '')}", "", str(passage.get("text", ""))])
        for group in record.get("question_groups", []):
            lines.extend(["", f"[{group.get('group_id')}] {group.get('question_type')}"])
            for question in group.get("questions", []):
                lines.append(f"{question.get('number')}. {question.get('question_text')}")
    elif record.get("module") == "writing":
        lines.extend(
            [
                f"essay_type: {record.get('essay_type', '')}",
                f"word_limit: {record.get('word_limit', '')}",
                "",
                str(record.get("prompt", "")),
            ]
        )
    elif record.get("module") == "listening":
        lines.extend(["transcript:", str(record.get("transcript", ""))])
    elif record.get("module") == "speaking":
        lines.append(json.dumps(record.get("parts", {}), ensure_ascii=False, indent=2))
    elif "content" in record:
        lines.append(str(record.get("content", "")))
    elif "rubric_text" in record:
        lines.append(str(record.get("rubric_text", "")))
    else:
        lines.append(json.dumps(record, ensure_ascii=False, indent=2))
    return "\n".join(lines)


def _record_id(source: DataSource, year: int | None, month: int | None) -> str:
    year_part = str(year or "unknown")
    month_part = f"{month:02d}" if month else "unknown"
    base = f"ielts_{source.module}_{year_part}_{month_part}_{source.task_type}_{Path(source.file_name).stem}"
    return re.sub(r"[^a-z0-9_]+", "_", base.lower()).strip("_")


def _set_name(source: DataSource, year: int | None, month: int | None) -> str:
    if year and month:
        return f"{source.source.lower().replace(' ', '_')}_practice_{year}_{month:02d}"
    return f"{source.source.lower().replace(' ', '_')}_practice"


def _extract_request_date(user_input: str) -> dict[str, int | None]:
    year_match = re.search(r"(20\d{2})", user_input)
    month_match = re.search(r"(?:20\d{2}\s*年)?\s*(1[0-2]|0?[1-9])\s*月", user_input)
    return {
        "year": int(year_match.group(1)) if year_match else None,
        "month": int(month_match.group(1)) if month_match else None,
    }


def _extract_request_date(user_input: str) -> dict[str, int | None]:
    """Extract year/month from user wording such as 2025年12月."""

    year_match = re.search(r"(20\d{2})", user_input)
    month_match = re.search(r"(?:20\d{2}\s*年)?\s*(1[0-2]|0?[1-9])\s*月", user_input)
    return {
        "year": int(year_match.group(1)) if year_match else None,
        "month": int(month_match.group(1)) if month_match else None,
    }


def _extract_request_date(user_input: str) -> dict[str, int | None]:
    """Extract year/month from user wording such as 2025年12月 or 25年12月."""

    year_match = re.search(r"(20\d{2}|(?<!\d)\d{2}(?=年))", user_input)
    month_match = re.search(r"(?:20?\d{2}\s*年)?\s*(1[0-2]|0?[1-9])\s*月", user_input)
    year = int(year_match.group(1)) if year_match else None
    if year is not None and year < 100:
        year += 2000
    return {
        "year": year,
        "month": int(month_match.group(1)) if month_match else None,
    }


def _requested_export_format(user_input: str) -> str | None:
    lowered = user_input.lower()
    if "pdf" in lowered:
        return "pdf"
    if "txt" in lowered or "文本" in user_input:
        return "txt"
    if "json" in lowered:
        return "json"
    return None


def _infer_essay_type(text: str) -> str | None:
    lowered = text.lower()
    if "to what extent" in lowered or "agree or disagree" in lowered or "in your opinion" in lowered:
        return "opinion"
    if "discuss both" in lowered:
        return "discussion"
    if "advantages" in lowered and "disadvantages" in lowered:
        return "advantages_disadvantages"
    if "problem" in lowered and "solution" in lowered:
        return "problem_solution"
    return None


def _timeout_for_url(url: str) -> tuple[int, int]:
    host = urlparse(url).netloc.lower()
    return SLOW_DOMAIN_TIMEOUTS.get(host, (DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT))


def _raise_for_retriable_status(response: Response) -> None:
    if response.status_code == 429 or 500 <= response.status_code < 600:
        raise HTTPError(f"retriable HTTP status {response.status_code}", response=response)


def _should_retry(exc: Exception | None) -> bool:
    if isinstance(exc, (ReadTimeout, ConnectTimeout, Timeout, ConnectionError)):
        return True
    if isinstance(exc, HTTPError):
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", 0)
        return status_code == 429 or 500 <= status_code < 600 or "retriable HTTP status" in str(exc)
    return False


def _safe_content_length(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


def _print_download_progress(file_name: str, downloaded: int, total: int, *, done: bool = False) -> None:
    if total > 0:
        percent = min(downloaded / total, 1.0)
        filled = int(percent * 24)
        bar = "#" * filled + "-" * (24 - filled)
        message = f"\r  [{bar}] {percent * 100:6.2f}% {_format_size(downloaded)} / {_format_size(total)} {file_name}"
    else:
        message = f"\r  downloaded {_format_size(downloaded)} {file_name}"
    print(message, end="\n" if done else "", flush=True)


def _format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}GB"


def _append_manifest(records: list[dict[str, Any]]) -> None:
    existing: list[dict[str, Any]] = []
    if MANIFEST_PATH.exists():
        try:
            loaded = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing = loaded
        except json.JSONDecodeError:
            existing = []
    by_path = {item.get("saved_path"): item for item in existing}
    for record in records:
        by_path[record["saved_path"]] = record
    MANIFEST_PATH.write_text(json.dumps(list(by_path.values()), ensure_ascii=False, indent=2), encoding="utf-8")
