"""Shared local paths for question-bank assets."""

from __future__ import annotations

import os
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name, "").strip()
    return Path(value).expanduser() if value else default


WRITING_QUESTION_BANK_ROOT = _env_path("WRITING_QUESTION_BANK_DIR", DATA_ROOT / "writing_questions")
CAMBRIDGE_WRITING_ROOT = WRITING_QUESTION_BANK_ROOT / "cambridge"
CAMBRIDGE_RAW_DIR = CAMBRIDGE_WRITING_ROOT / "raw"
CAMBRIDGE_IMAGE_DIR = CAMBRIDGE_WRITING_ROOT / "images"
CAMBRIDGE_RECORD_DIR = CAMBRIDGE_WRITING_ROOT / "records"
CAMBRIDGE_EXPORT_DIR = CAMBRIDGE_WRITING_ROOT / "exports"
