"""Utilities for building a local RAG corpus plan from data files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


SECTION_TO_KEY = {
    "Tier 1: Must Ingest First": "tier1",
    "Tier 2: Ingest After Tier 1 Is Stable": "tier2",
    "Tier 3: Add Selectively": "tier3",
    "Convert Before Ingesting": "convert",
    "Do Not Ingest Directly": "skip",
}

OFFICIAL_TIER1_NAMES = {
    "ielts-academic-format-reading.html",
    "academic-test-sample-questions.html",
}


@dataclass(slots=True)
class IngestionPlan:
    """Structured ingestion plan grouped by priority tier."""

    tier1: list[Path]
    tier2: list[Path]
    tier3: list[Path]
    convert: list[Path]
    skip: list[Path]
    unresolved_from_guide: list[str]


def build_ingestion_plan(data_dir: Path, guide_path: Path) -> IngestionPlan:
    """Create an ingestion plan using the guide first, then safe local heuristics."""

    parsed_sections = _parse_guide(guide_path)
    files = [path for path in data_dir.iterdir() if path.is_file()]
    indexed_files = {path.name: path for path in files}

    tier1, unresolved_tier1 = _resolve_paths(parsed_sections.get("tier1", []), indexed_files)
    tier2, unresolved_tier2 = _resolve_paths(parsed_sections.get("tier2", []), indexed_files)
    tier3, unresolved_tier3 = _resolve_paths(parsed_sections.get("tier3", []), indexed_files)
    convert, unresolved_convert = _resolve_paths(parsed_sections.get("convert", []), indexed_files)
    skip, unresolved_skip = _resolve_paths(parsed_sections.get("skip", []), indexed_files)

    # Fallback heuristics keep the tool usable even if the guide cannot exactly match local names.
    for path in files:
        if path.name == guide_path.name:
            continue
        if path.suffix.lower() == ".epub":
            convert.append(path)
            continue
        if path.suffix.lower() in {".url", ".txt"}:
            skip.append(path)
            continue
        if path.name in OFFICIAL_TIER1_NAMES:
            tier1.append(path)
            continue
        if _looks_like_cambridge_test(path):
            tier2.append(path)
            continue
        if _looks_like_guide_pdf(path):
            tier3.append(path)

    return IngestionPlan(
        tier1=_dedupe_paths(tier1),
        tier2=_dedupe_paths(tier2),
        tier3=_dedupe_paths(tier3),
        convert=_dedupe_paths(convert),
        skip=_dedupe_paths(skip),
        unresolved_from_guide=sorted(
            set(unresolved_tier1 + unresolved_tier2 + unresolved_tier3 + unresolved_convert + unresolved_skip)
        ),
    )


def _parse_guide(guide_path: Path) -> dict[str, list[str]]:
    """Parse bullet-point filenames grouped by markdown section title."""

    content = guide_path.read_text(encoding="utf-8-sig")
    sections: dict[str, list[str]] = {key: [] for key in SECTION_TO_KEY.values()}
    current_section: str | None = None

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            title = line.removeprefix("## ").strip()
            current_section = SECTION_TO_KEY.get(title)
            continue
        if line in {"Why:", "Notes:", "## Recommended Ingestion Order"}:
            current_section = None
            continue
        if not current_section or not line.startswith("- "):
            continue

        match = re.search(r"`([^`]+)`", line)
        if match:
            sections[current_section].append(match.group(1))

    return sections


def _resolve_paths(
    filenames: list[str],
    indexed_files: dict[str, Path],
) -> tuple[list[Path], list[str]]:
    """Resolve exact guide filenames to local files."""

    matched: list[Path] = []
    unresolved: list[str] = []

    for name in filenames:
        path = indexed_files.get(name)
        if path is None:
            unresolved.append(name)
            continue
        matched.append(path)

    return matched, unresolved


def _looks_like_cambridge_test(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() == ".pdf" and any(
        keyword in name
        for keyword in (
            "ielts academic with answer",
            "剑桥雅思",
            "官方真题集",
            "真题19",
            "真题15",
            "真题18",
        )
    ) and "精讲" not in path.name


def _looks_like_guide_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf" and "精讲" in path.name


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(path.resolve())] = path
    return sorted(unique.values(), key=lambda item: item.name.lower())
