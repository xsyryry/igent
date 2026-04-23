"""CLI for building the persistent local RAG index."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from project.config import get_config
from project.logging_config import setup_logging
from project.rag.local_index import (
    DEFAULT_DATA_DIR,
    DEFAULT_INDEX_DIR,
    build_persistent_index,
    clear_persistent_index,
    get_index_status,
)
from project.rag.simple_rag import _load_index


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/status/clear the local persistent RAG index.")
    parser.add_argument(
        "action",
        choices=("build", "status", "clear"),
        help="build creates JSONL index, status shows current index, clear removes it.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help=f"Default: {DEFAULT_DATA_DIR}")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR, help=f"Default: {DEFAULT_INDEX_DIR}")
    parser.add_argument(
        "--strategy",
        choices=("auto", "llm_auto", "sliding", "headings", "qa_pairs", "mistake_rules", "magazine_articles"),
        default="auto",
        help="Chunking strategy for index build.",
    )
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--max-files", type=int, default=0, help="0 means all supported files.")
    parser.add_argument(
        "--publication",
        "--magazine",
        dest="publication",
        default="",
        help="Filter by publication, e.g. economist, new_yorker, atlantic, wired.",
    )
    parser.add_argument("--date-from", default="", help="Issue start date, YYYY.MM.DD or YYYY-MM-DD.")
    parser.add_argument("--date-to", default="", help="Issue end date, YYYY.MM.DD or YYYY-MM-DD.")
    parser.add_argument(
        "--no-sentence-chunks",
        action="store_true",
        help="Disable high-value sentence sub-chunks.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    config = get_config()
    setup_logging(config.log_level)
    args = _parse_args()

    if args.action == "clear":
        clear_persistent_index(args.index_dir)
        _load_index.cache_clear()
        print(f"Cleared local RAG index: {args.index_dir}")
        return 0

    if args.action == "status":
        status = get_index_status(args.index_dir)
        print("Local RAG index status")
        if not status["exists"]:
            print(f"- exists: false\n- index_dir: {status['index_dir']}")
            return 0
        manifest = status["manifest"]
        print(f"- exists       : true")
        print(f"- built_at     : {manifest.get('built_at')}")
        print(f"- data_dir     : {manifest.get('data_dir')}")
        print(f"- documents    : {manifest.get('document_count')}")
        print(f"- chunks       : {manifest.get('chunk_count')}")
        print(f"- strategy     : {manifest.get('strategy')}")
        print(f"- sentence_idx : {manifest.get('include_sentence_chunks')}")
        print(f"- skipped      : {len(manifest.get('skipped', []))}")
        return 0

    if not args.data_dir.exists():
        print(f"Data directory not found: {args.data_dir}")
        return 1
    try:
        manifest = build_persistent_index(
            data_dir=args.data_dir,
            index_dir=args.index_dir,
            strategy=args.strategy,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            max_files=args.max_files,
            include_sentence_chunks=not args.no_sentence_chunks,
            publication=args.publication or None,
            date_from=args.date_from or None,
            date_to=args.date_to or None,
        )
    except ValueError as exc:
        print(str(exc))
        return 1
    _load_index.cache_clear()
    print("Built local RAG index")
    print(f"- documents: {manifest['document_count']}")
    print(f"- chunks   : {manifest['chunk_count']}")
    print(f"- selected : {manifest['file_count']} pdf file(s)")
    print(f"- reused   : {manifest.get('reused_count', 0)}")
    print(f"- rebuilt  : {manifest.get('rebuilt_count', 0)}")
    if manifest.get("publication"):
        print(f"- magazine : {manifest['publication']}")
    if manifest.get("date_from") or manifest.get("date_to"):
        print(f"- dates    : {manifest.get('date_from') or '*'} ~ {manifest.get('date_to') or '*'}")
    print(f"- skipped  : {len(manifest['skipped'])}")
    print(f"- index    : {args.index_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
