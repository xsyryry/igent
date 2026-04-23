"""Persistent local RAG index.

The index follows the whitepaper shape at a small local scale:
documents -> paragraph chunks -> high-value sentence chunks, all with metadata.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import hashlib
import json
import re
import shutil
from typing import Any

from project.rag.chunking_agent import ChunkRecord, prepare_chunks
from project.rag.simple_rag import (
    IndexedChunk,
    SUPPORTED_SUFFIXES,
    _build_chunk_metadata,
    _infer_doc_type,
    _token_vector,
)

INDEX_VERSION = 1
CHUNKER_VERSION = 3
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "awesome-english-ebooks"
DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "rag_index"
MANIFEST_PATH = DEFAULT_INDEX_DIR / "manifest.json"
DOCUMENTS_PATH = DEFAULT_INDEX_DIR / "documents.jsonl"
CHUNKS_PATH = DEFAULT_INDEX_DIR / "chunks.jsonl"


def build_persistent_index(
    *,
    data_dir: Path | None = None,
    index_dir: Path = DEFAULT_INDEX_DIR,
    strategy: str = "auto",
    chunk_size: int = 1200,
    overlap: int = 150,
    max_files: int = 0,
    include_sentence_chunks: bool = True,
    publication: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """Build and save a local JSONL index for RAG retrieval."""

    data_dir = data_dir or _default_data_dir()
    normalized_date_from = _normalize_date_filter(date_from)
    normalized_date_to = _normalize_date_filter(date_to)
    if normalized_date_from and normalized_date_to and normalized_date_from > normalized_date_to:
        raise ValueError("--date-from must be earlier than or equal to --date-to")
    files = _discover_files(
        data_dir,
        publication=publication,
        date_from=normalized_date_from,
        date_to=normalized_date_to,
    )
    if max_files > 0:
        files = files[:max_files]

    index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = index_dir / "manifest.json"
    documents_path = index_dir / "documents.jsonl"
    chunks_path = index_dir / "chunks.jsonl"
    previous_docs, previous_chunks = _load_previous_index(documents_path, chunks_path)
    document_count = 0
    chunk_count = 0
    reused_count = 0
    rebuilt_count = 0
    skipped: list[dict[str, str]] = []

    with documents_path.open("w", encoding="utf-8") as doc_file, chunks_path.open("w", encoding="utf-8") as chunk_file:
        for path in files:
            source = _source_path(path)
            content_hash = _file_hash(path)
            stat = path.stat()
            previous_doc = previous_docs.get(source)
            if _can_reuse_document(
                previous_doc,
                content_hash=content_hash,
                mtime=stat.st_mtime,
                strategy=strategy,
                chunk_size=chunk_size,
                overlap=overlap,
                include_sentence_chunks=include_sentence_chunks,
            ):
                doc_file.write(json.dumps(previous_doc, ensure_ascii=False) + "\n")
                reused_chunks = previous_chunks.get(source, [])
                for record in reused_chunks:
                    chunk_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                document_count += 1
                chunk_count += len(reused_chunks)
                reused_count += 1
                continue

            try:
                plan, chunks = prepare_chunks(
                    path,
                    strategy=strategy,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            except Exception as exc:
                skipped.append({"source": str(path), "reason": str(exc)})
                continue

            document_count += 1
            rebuilt_count += 1
            doc_record = _document_record(
                path,
                data_dir,
                plan.strategy,
                chunk_size,
                overlap,
                len(chunks),
                content_hash=content_hash,
                requested_strategy=strategy,
                include_sentence_chunks=include_sentence_chunks,
            )
            doc_file.write(json.dumps(doc_record, ensure_ascii=False) + "\n")

            for chunk in chunks:
                record = _chunk_record(path, chunk)
                chunk_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunk_count += 1

                if include_sentence_chunks and chunk.strategy != "magazine_articles" and chunk.metadata.get("rag_layer") != "sentence":
                    for sentence_chunk in _derive_sentence_chunks(chunk):
                        sentence_record = _chunk_record(path, sentence_chunk)
                        chunk_file.write(json.dumps(sentence_record, ensure_ascii=False) + "\n")
                        chunk_count += 1

    manifest = {
        "version": INDEX_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "built_at": datetime.utcnow().replace(microsecond=0).isoformat(),
        "data_dir": str(data_dir),
        "strategy": strategy,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "include_sentence_chunks": include_sentence_chunks,
        "publication": publication or "",
        "date_from": normalized_date_from or "",
        "date_to": normalized_date_to or "",
        "file_count": len(files),
        "document_count": document_count,
        "chunk_count": chunk_count,
        "reused_count": reused_count,
        "rebuilt_count": rebuilt_count,
        "skipped": skipped,
        "files": {
            "documents": str(documents_path),
            "chunks": str(chunks_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_persistent_index(index_dir: Path = DEFAULT_INDEX_DIR) -> tuple[IndexedChunk, ...]:
    """Load indexed chunks from JSONL if the persistent index exists."""

    chunks_path = index_dir / "chunks.jsonl"
    if not chunks_path.exists():
        return ()

    indexed: list[IndexedChunk] = []
    with chunks_path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            indexed.append(
                IndexedChunk(
                    id=str(record.get("id") or record.get("metadata", {}).get("chunk_id") or ""),
                    source=str(record.get("source") or ""),
                    text=str(record.get("text") or ""),
                    doc_type=str(record.get("doc_type") or "general"),
                    tokens={str(key): float(value) for key, value in dict(record.get("tokens") or {}).items()},
                    metadata=dict(record.get("metadata") or {}),
                )
            )
    return tuple(item for item in indexed if item.id and item.text)


def get_index_status(index_dir: Path = DEFAULT_INDEX_DIR) -> dict[str, Any]:
    """Return a lightweight status summary for the persistent index."""

    manifest_path = index_dir / "manifest.json"
    chunks_path = index_dir / "chunks.jsonl"
    documents_path = index_dir / "documents.jsonl"
    if not manifest_path.exists():
        return {"exists": False, "index_dir": str(index_dir)}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("cleared"):
        return {"exists": False, "index_dir": str(index_dir), "cleared_at": manifest.get("cleared_at")}
    return {
        "exists": True,
        "index_dir": str(index_dir),
        "manifest": manifest,
        "documents_file_exists": documents_path.exists(),
        "chunks_file_exists": chunks_path.exists(),
        "chunks_file_size": chunks_path.stat().st_size if chunks_path.exists() else 0,
    }


def clear_persistent_index(index_dir: Path = DEFAULT_INDEX_DIR) -> None:
    """Remove the local persistent index directory."""

    if not index_dir.exists():
        return
    try:
        shutil.rmtree(index_dir)
        return
    except PermissionError:
        pass

    index_dir.mkdir(parents=True, exist_ok=True)
    for name in ("documents.jsonl", "chunks.jsonl"):
        (index_dir / name).write_text("", encoding="utf-8")
    (index_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": INDEX_VERSION,
                "cleared": True,
                "cleared_at": datetime.utcnow().replace(microsecond=0).isoformat(),
                "document_count": 0,
                "chunk_count": 0,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _default_data_dir() -> Path:
    return DEFAULT_DATA_DIR if DEFAULT_DATA_DIR.exists() else Path.cwd() / "data"


def _load_previous_index(documents_path: Path, chunks_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    docs: dict[str, dict[str, Any]] = {}
    chunks: dict[str, list[dict[str, Any]]] = {}
    if documents_path.exists():
        with documents_path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                source = str(record.get("source") or "")
                if source:
                    docs[source] = record
    if chunks_path.exists():
        with chunks_path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                source = str(record.get("source") or "")
                if source:
                    chunks.setdefault(source, []).append(record)
    return docs, chunks


def _can_reuse_document(
    record: dict[str, Any] | None,
    *,
    content_hash: str,
    mtime: float,
    strategy: str,
    chunk_size: int,
    overlap: int,
    include_sentence_chunks: bool,
) -> bool:
    if not record:
        return False
    if str(record.get("content_hash") or "") != content_hash:
        return False
    if abs(float(record.get("mtime") or 0.0) - float(mtime)) > 1e-6:
        return False
    if str(record.get("requested_strategy") or record.get("chunk_strategy") or "") != strategy:
        return False
    if int(record.get("chunker_version") or 0) != CHUNKER_VERSION:
        return False
    if int(record.get("chunk_size") or 0) != chunk_size or int(record.get("overlap") or 0) != overlap:
        return False
    return bool(record.get("include_sentence_chunks", True)) == include_sentence_chunks


def _discover_files(
    data_dir: Path,
    *,
    publication: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[Path]:
    if not data_dir.exists():
        return []
    ignored_parts = {"rag_index", "chunk_previews", "chunk_eval_reports"}
    is_ebook_root = data_dir.name == "awesome-english-ebooks"
    publication_filter = _normalize_publication(publication or "")
    date_from = _normalize_date_filter(date_from)
    date_to = _normalize_date_filter(date_to)
    files = []
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        if is_ebook_root and path.suffix.lower() != ".pdf":
            continue
        if not is_ebook_root and path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if path.name.startswith("_"):
            continue
        if any(part in ignored_parts for part in path.parts):
            continue
        issue = _publication_issue(path, data_dir)
        if publication_filter and _normalize_publication(issue["publication"]) != publication_filter:
            continue
        issue_date = issue["issue_date"]
        if date_from and (not issue_date or issue_date < date_from):
            continue
        if date_to and (not issue_date or issue_date > date_to):
            continue
        files.append(path)
    return sorted(files)


def _document_record(
    path: Path,
    data_dir: Path,
    strategy: str,
    chunk_size: int,
    overlap: int,
    chunk_count: int,
    *,
    content_hash: str | None = None,
    requested_strategy: str = "",
    include_sentence_chunks: bool = True,
) -> dict[str, Any]:
    stat = path.stat()
    issue = _publication_issue(path, data_dir)
    return {
        "doc_id": path.stem,
        "source": _source_path(path),
        "article_title": path.stem,
        "publication": issue["publication"],
        "issue_date": issue["issue_date"],
        "issue_folder": issue["issue_folder"],
        "suffix": path.suffix.lower(),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "content_hash": content_hash or _file_hash(path),
        "chunker_version": CHUNKER_VERSION,
        "requested_strategy": requested_strategy or strategy,
        "chunk_strategy": strategy,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "include_sentence_chunks": include_sentence_chunks,
        "chunk_count": chunk_count,
    }


def _chunk_record(path: Path, chunk: ChunkRecord) -> dict[str, Any]:
    text = chunk.content.strip()
    metadata = _build_chunk_metadata(path, chunk, text)
    metadata["raw_chunk"] = asdict(chunk)
    return {
        "id": chunk.chunk_id,
        "source": _source_path(path),
        "text": text,
        "doc_type": _infer_doc_type(path, text),
        "tokens": _token_vector(text),
        "metadata": metadata,
    }


def _derive_sentence_chunks(chunk: ChunkRecord) -> list[ChunkRecord]:
    sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", chunk.content)
    derived: list[ChunkRecord] = []
    for index, sentence in enumerate(sentences, start=1):
        sentence = sentence.strip()
        if not _is_high_value_sentence(sentence):
            continue
        derived.append(
            ChunkRecord(
                chunk_id=f"{chunk.chunk_id}-sent-{index:02d}",
                source_file=chunk.source_file,
                strategy="sentence",
                title=f"{chunk.title} sentence {index}",
                content=sentence,
                metadata={
                    **chunk.metadata,
                    "parent_chunk_id": chunk.chunk_id,
                    "chunk_type": "sentence",
                    "sentence_pattern": _sentence_pattern(sentence),
                },
            )
        )
    return derived[:5]


def _is_high_value_sentence(sentence: str) -> bool:
    if not 50 <= len(sentence) <= 360:
        return False
    lowered = sentence.lower()
    markers = (
        "because", "therefore", "however", "although", "for example",
        "as a result", "in conclusion", "whereas", "compared with",
    )
    return any(marker in lowered for marker in markers)


def _sentence_pattern(sentence: str) -> str:
    lowered = sentence.lower()
    if any(item in lowered for item in ("because", "therefore", "as a result")):
        return "cause-effect"
    if any(item in lowered for item in ("however", "although", "despite")):
        return "concession"
    if any(item in lowered for item in ("for example", "for instance", "such as")):
        return "example"
    if any(item in lowered for item in ("whereas", "compared with")):
        return "comparison"
    if any(item in lowered for item in ("in conclusion", "overall", "to sum up")):
        return "summary"
    return "general"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _publication_issue(path: Path, data_dir: Path) -> dict[str, str]:
    try:
        relative_parts = path.relative_to(data_dir).parts
    except ValueError:
        marker = "awesome-english-ebooks"
        parts = path.parts
        if marker in parts:
            start = parts.index(marker) + 1
            relative_parts = parts[start:]
        else:
            relative_parts = path.parts

    publication_folder = relative_parts[0] if len(relative_parts) >= 1 else ""
    issue_folder = relative_parts[1] if len(relative_parts) >= 2 else ""
    return {
        "publication": _display_publication(publication_folder),
        "issue_date": _extract_issue_date(issue_folder) or _extract_issue_date(path.name),
        "issue_folder": issue_folder,
    }


def _display_publication(folder_name: str) -> str:
    return re.sub(r"^\d+[_-]*", "", folder_name).strip("_-").lower()


def _normalize_publication(value: str) -> str:
    value = re.sub(r"^\d+[_-]*", "", value.strip().lower())
    value = re.sub(r"^the[\s_-]+", "", value)
    return re.sub(r"[^a-z0-9]+", "", value)


def _normalize_date_filter(value: str | None) -> str | None:
    if not value:
        return None
    normalized = _extract_issue_date(value)
    if not normalized:
        raise ValueError(f"Invalid date: {value}. Use YYYY.MM.DD or YYYY-MM-DD.")
    return normalized


def _extract_issue_date(value: str) -> str:
    match = re.search(r"(20\d{2})[.-](\d{1,2})[.-](\d{1,2})", value)
    if not match:
        return ""
    year, month, day = match.groups()
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
