"""Export local Cambridge IELTS writing questions to a printable PDF."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from project.tools.question_bank_paths import CAMBRIDGE_EXPORT_DIR, CAMBRIDGE_RECORD_DIR

QUESTION_DIR = CAMBRIDGE_RECORD_DIR
EXPORT_DIR = CAMBRIDGE_EXPORT_DIR
DEFAULT_COUNT = 5
PAGE_SIZE = (1240, 1754)  # A4 at roughly 150 DPI.
MARGIN = 90
PARSER_VERSION = "export_question_pdf_v1"


def export_question_pdf(
    *,
    user_input: str = "",
    count: int | None = None,
    cambridge_book: int | None = None,
    task_no: int | None = None,
    part_no: int | None = None,
    include_images: bool | None = None,
    output_filename: str | None = None,
) -> dict[str, Any]:
    """Read local structured writing questions and export a clean PDF."""

    extracted = _extract_export_request(user_input)
    requested_count = int(count or extracted.get("count") or DEFAULT_COUNT)
    filters = {
        "cambridge_book": cambridge_book or extracted.get("cambridge_book"),
        "task_no": task_no or extracted.get("task_no"),
        "part_no": part_no or extracted.get("part_no"),
        "include_images": include_images if include_images is not None else extracted.get("include_images", True),
    }
    requested_count = max(1, min(requested_count, 100))

    records, load_failures = _load_question_records()
    eligible, excluded = _filter_records(records, filters)
    selected = _deduplicate_records(eligible)[:requested_count]

    if not selected:
        reasons = dict(excluded)
        if load_failures:
            reasons["load_failed"] = len(load_failures)
        return {
            "success": False,
            "error": "no_matching_questions",
            "message": "本地题库中没有符合条件的已解析写作题。",
            "export_path": "",
            "requested_count": requested_count,
            "exported_count": 0,
            "completion_status": "failed",
            "filters": filters,
            "excluded_count": sum(reasons.values()),
            "excluded_reasons": reasons,
            "parser_version": PARSER_VERSION,
        }

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    export_path = EXPORT_DIR / _resolve_output_filename(
        output_filename or extracted.get("output_filename"),
        filters,
        len(selected),
    )
    image_warnings: list[str] = []

    try:
        _write_pdf(
            export_path=export_path,
            records=selected,
            requested_count=requested_count,
            filters=filters,
            include_images=bool(filters["include_images"]),
            image_warnings=image_warnings,
        )
    except Exception as exc:  # pragma: no cover - defensive failure reporting.
        return {
            "success": False,
            "error": "pdf_generation_failed",
            "message": str(exc),
            "export_path": "",
            "requested_count": requested_count,
            "exported_count": 0,
            "completion_status": "failed",
            "filters": filters,
            "excluded_count": sum(excluded.values()),
            "excluded_reasons": dict(excluded),
            "parser_version": PARSER_VERSION,
        }

    completion_status = "complete" if len(selected) >= requested_count else "partial"
    excluded_reasons = dict(excluded)
    if load_failures:
        excluded_reasons["load_failed"] = len(load_failures)
    if image_warnings:
        excluded_reasons["image_not_rendered"] = len(image_warnings)

    return {
        "success": True,
        "export_path": str(export_path),
        "requested_count": requested_count,
        "exported_count": len(selected),
        "completion_status": completion_status,
        "filters": filters,
        "excluded_count": sum(excluded_reasons.values()),
        "excluded_reasons": excluded_reasons,
        "image_warnings": image_warnings[:10],
        "parser_version": PARSER_VERSION,
    }


def _extract_export_request(user_input: str) -> dict[str, Any]:
    text = user_input or ""
    normalized = text.lower()

    count_match = re.search(r"(?<!\d)(\d{1,3})\s*(?:道|份|篇|个|套|items?)", text, re.I)
    book_match = re.search(r"(?:剑雅|剑桥|cambridge)\s*(\d{1,2})", text, re.I)
    task_match = re.search(r"task\s*([12])|task([12])|大作文|小作文", text, re.I)
    part_match = re.search(r"part\s*(\d{1,2})|part(\d{1,2})", text, re.I)

    task_no = None
    if task_match:
        if "大作文" in task_match.group(0):
            task_no = 2
        elif "小作文" in task_match.group(0):
            task_no = 1
        else:
            task_no = int(task_match.group(1) or task_match.group(2))

    include_images = not any(token in normalized for token in ("不要图", "不带图", "no image", "without image"))
    output_name = _extract_output_filename(text)

    return {
        "count": int(count_match.group(1)) if count_match else None,
        "cambridge_book": int(book_match.group(1)) if book_match else None,
        "task_no": task_no,
        "part_no": int(part_match.group(1) or part_match.group(2)) if part_match else None,
        "include_images": include_images,
        "output_filename": output_name,
    }


def _extract_output_filename(text: str) -> str | None:
    match = re.search(r"(?:文件名|命名为|保存为)\s*[:：]?\s*([^\s，。]+\.pdf)", text, re.I)
    if not match:
        return None
    return Path(match.group(1)).name


def _load_question_records() -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    if not QUESTION_DIR.exists():
        return records, ["question_dir_missing"]

    for path in sorted(QUESTION_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            failures.append(str(path))
            continue
        if isinstance(data, dict):
            data["_json_path"] = str(path)
            records.append(data)
    return records, failures


def _filter_records(records: list[dict[str, Any]], filters: dict[str, Any]) -> tuple[list[dict[str, Any]], Counter[str]]:
    eligible: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()

    for record in records:
        if str(record.get("module", "")).lower() != "writing":
            excluded["not_writing"] += 1
            continue
        if str(record.get("question_type", "")).lower() != "cambridge":
            excluded["not_cambridge"] += 1
            continue
        if str(record.get("parse_status", "")).lower() != "parsed":
            excluded["not_parsed"] += 1
            continue
        if not str(record.get("prompt_text", "")).strip():
            excluded["missing_prompt_text"] += 1
            continue
        if filters.get("cambridge_book") and _to_int(record.get("cambridge_book")) != int(filters["cambridge_book"]):
            excluded["book_filter_mismatch"] += 1
            continue
        if filters.get("task_no") and _to_int(record.get("task_no")) != int(filters["task_no"]):
            excluded["task_filter_mismatch"] += 1
            continue
        if filters.get("part_no") and _to_int(record.get("part_no")) != int(filters["part_no"]):
            excluded["part_filter_mismatch"] += 1
            continue
        eligible.append(record)

    eligible.sort(key=lambda item: (
        _to_int(item.get("cambridge_book")) or 0,
        _to_int(item.get("part_no")) or 0,
        _to_int(item.get("task_no")) or 0,
        str(item.get("id", "")),
    ))
    return eligible, excluded


def _deduplicate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for record in records:
        key = str(record.get("id") or "").strip()
        if not key:
            key = "|".join(
                [
                    str(record.get("cambridge_book", "")),
                    str(record.get("part_no", "")),
                    str(record.get("task_no", "")),
                    _normalize_prompt(str(record.get("prompt_text", ""))),
                ]
            )
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def _write_pdf(
    *,
    export_path: Path,
    records: list[dict[str, Any]],
    requested_count: int,
    filters: dict[str, Any],
    include_images: bool,
    image_warnings: list[str],
) -> None:
    fonts = _load_fonts()
    pages = [
        _build_cover_page(
            requested_count=requested_count,
            exported_count=len(records),
            filters=filters,
            fonts=fonts,
        )
    ]

    for index, record in enumerate(records, start=1):
        pages.append(
            _build_question_page(
                index=index,
                record=record,
                include_images=include_images,
                image_warnings=image_warnings,
                fonts=fonts,
            )
        )

    first, rest = pages[0], pages[1:]
    first.save(export_path, "PDF", resolution=150.0, save_all=True, append_images=rest)


def _build_cover_page(
    *,
    requested_count: int,
    exported_count: int,
    filters: dict[str, Any],
    fonts: dict[str, ImageFont.ImageFont],
) -> Image.Image:
    page = _new_page()
    draw = ImageDraw.Draw(page)
    y = 160
    draw.text((MARGIN, y), "Cambridge IELTS Writing Question Collection", fill=(22, 42, 54), font=fonts["title"])
    y += 90
    draw.line((MARGIN, y, PAGE_SIZE[0] - MARGIN, y), fill=(40, 90, 110), width=3)
    y += 70

    status = "complete" if exported_count >= requested_count else "partial"
    lines = [
        f"Requested Count: {requested_count}",
        f"Exported Count: {exported_count}",
        f"Completion Status: {status}",
        f"Cambridge Book: {filters.get('cambridge_book') or 'Any'}",
        f"Part: {filters.get('part_no') or 'Any'}",
        f"Task: {filters.get('task_no') or 'Any'}",
        f"Include Images: {'Yes' if filters.get('include_images') else 'No'}",
        f"Exported At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    for line in lines:
        draw.text((MARGIN, y), line, fill=(40, 45, 48), font=fonts["body"])
        y += 48
    return page


def _build_question_page(
    *,
    index: int,
    record: dict[str, Any],
    include_images: bool,
    image_warnings: list[str],
    fonts: dict[str, ImageFont.ImageFont],
) -> Image.Image:
    page = _new_page()
    draw = ImageDraw.Draw(page)
    y = 80

    draw.text((MARGIN, y), f"题目 {index}", fill=(18, 44, 62), font=fonts["heading"])
    y += 68
    draw.rounded_rectangle((MARGIN, y, PAGE_SIZE[0] - MARGIN, y + 155), radius=18, fill=(235, 244, 246))
    y += 25
    info_lines = [
        f"剑雅：{record.get('cambridge_book', '未标注')}",
        f"Part：{record.get('part_no', '未标注')}",
        f"Task：{record.get('task_no', '未标注')}",
        f"字数要求：{_extract_word_limit(record)}",
    ]
    for line in info_lines:
        draw.text((MARGIN + 28, y), line, fill=(38, 55, 64), font=fonts["meta"])
        y += 31

    y += 55
    draw.text((MARGIN, y), "题目：", fill=(18, 44, 62), font=fonts["subheading"])
    y += 50
    y = _draw_wrapped_text(
        draw,
        text=str(record.get("prompt_text", "")).strip(),
        x=MARGIN,
        y=y,
        max_width=PAGE_SIZE[0] - 2 * MARGIN,
        font=fonts["body"],
        fill=(32, 32, 32),
        line_spacing=13,
    )

    if include_images:
        image_path = str(record.get("image_local_path") or "").strip()
        if image_path:
            y += 45
            try:
                question_image = Image.open(image_path).convert("RGB")
                question_image.thumbnail((PAGE_SIZE[0] - 2 * MARGIN, max(260, PAGE_SIZE[1] - y - 100)))
                page.paste(question_image, (MARGIN, y))
            except Exception:
                image_warnings.append(str(record.get("id") or image_path))
    return page


def _new_page() -> Image.Image:
    return Image.new("RGB", PAGE_SIZE, (255, 255, 252))


def _load_fonts() -> dict[str, ImageFont.ImageFont]:
    font_path = _find_font_path()
    if font_path:
        return {
            "title": ImageFont.truetype(font_path, 44),
            "heading": ImageFont.truetype(font_path, 42),
            "subheading": ImageFont.truetype(font_path, 31),
            "meta": ImageFont.truetype(font_path, 25),
            "body": ImageFont.truetype(font_path, 28),
        }
    default = ImageFont.load_default()
    return {"title": default, "heading": default, "subheading": default, "meta": default, "body": default}


def _find_font_path() -> str | None:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    x: int,
    y: int,
    max_width: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    line_spacing: int,
) -> int:
    for paragraph in text.splitlines() or [text]:
        words = paragraph.split()
        if not words:
            y += _line_height(draw, font) + line_spacing
            continue
        line = ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if _text_width(draw, candidate, font) <= max_width or not line:
                line = candidate
                continue
            draw.text((x, y), line, fill=fill, font=font)
            y += _line_height(draw, font) + line_spacing
            line = word
        if line:
            draw.text((x, y), line, fill=fill, font=font)
            y += _line_height(draw, font) + line_spacing
        y += 10
    return y


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _line_height(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), "Ag", font=font)
    return bbox[3] - bbox[1]


def _extract_word_limit(record: dict[str, Any]) -> str:
    explicit = record.get("word_limit")
    if explicit:
        return f"至少 {explicit} words"
    prompt = str(record.get("prompt_text", ""))
    match = re.search(r"write\s+at\s+least\s+(\d{2,3})\s+words", prompt, re.I)
    if match:
        return f"至少 {match.group(1)} words"
    return "未明确标注"


def _resolve_output_filename(output_filename: str | None, filters: dict[str, Any], exported_count: int) -> str:
    if output_filename:
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", Path(output_filename).name).strip("._")
        return safe if safe.lower().endswith(".pdf") else f"{safe}.pdf"

    parts = ["cambridge"]
    if filters.get("cambridge_book"):
        parts.append(str(filters["cambridge_book"]))
    parts.append("writing")
    if filters.get("task_no"):
        parts.append(f"task{filters['task_no']}")
    if filters.get("part_no"):
        parts.append(f"part{filters['part_no']}")
    parts.append(f"{exported_count}_items")
    return "_".join(parts) + ".pdf"


def _normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip().lower())


def _to_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
