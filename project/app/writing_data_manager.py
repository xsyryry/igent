"""CLI for managing IELTS writing data sources and corpora."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from project.config import get_config
from project.logging_config import setup_logging
from project.tools.db_tool import get_writing_task2_bank
from project.writing.task2_bank import (
    DEFAULT_MANUAL_CORPUS_DIR,
    DEFAULT_MANUAL_TASK2_DIR,
    prepare_external_corpus_locally,
    import_task2_bank_from_directory,
    update_task2_bank_from_web,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage IELTS writing Task 2 bank and local corpus preparation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_task2 = subparsers.add_parser(
        "import-task2",
        help="Parse local Task 2 source documents and store prompts in SQLite.",
    )
    import_task2.add_argument("--source-dir", type=Path, default=DEFAULT_MANUAL_TASK2_DIR)
    import_task2.add_argument("--strategy", default="llm_auto")
    import_task2.add_argument("--chunk-size", type=int, default=1800)
    import_task2.add_argument("--overlap", type=int, default=180)
    import_task2.add_argument("--model", default="")

    update_task2 = subparsers.add_parser(
        "update-task2-bank",
        help="Search recent web results, parse Task 2 prompts, and store them in SQLite.",
    )
    update_task2.add_argument("--query", default="latest IELTS writing task 2 questions 2026")
    update_task2.add_argument("--limit", type=int, default=5)
    update_task2.add_argument("--model", default="")

    import_corpus = subparsers.add_parser(
        "import-corpus",
        help="Chunk local external corpus files and prepare them for the built-in local RAG.",
    )
    import_corpus.add_argument("--source-dir", type=Path, default=DEFAULT_MANUAL_CORPUS_DIR)
    import_corpus.add_argument("--strategy", default="llm_auto")
    import_corpus.add_argument("--chunk-size", type=int, default=1200)
    import_corpus.add_argument("--overlap", type=int, default=150)
    import_corpus.add_argument("--max-chunks-per-file", type=int, default=0)
    import_corpus.add_argument("--model", default="")
    import_corpus.add_argument("--dry-run", action="store_true")

    list_task2 = subparsers.add_parser(
        "list-task2",
        help="Inspect recently stored Writing Task 2 prompts from SQLite.",
    )
    list_task2.add_argument("--limit", type=int, default=10)
    list_task2.add_argument("--essay-type", default="")
    return parser


def main() -> int:
    load_dotenv()
    config = get_config()
    setup_logging(config.log_level)
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "import-task2":
        result = import_task2_bank_from_directory(
            args.source_dir,
            llm_model=args.model or None,
            chunk_strategy=args.strategy,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        print("Task 2 import completed")
        print(f"- Source dir : {result['source_dir']}")
        print(f"- Files      : {result['files']}")
        print(f"- Topics seen: {result['topic_count']}")
        print(f"- Topics saved: {result['saved_count']}")
        for report in result["reports"]:
            print(
                f"- {report.source_file}: plan={report.plan.strategy} "
                f"source={report.plan.decision_source} chunks={report.chunk_count} "
                f"topics={report.topic_count} saved={report.saved_count}"
            )
        return 0

    if args.command == "update-task2-bank":
        result = update_task2_bank_from_web(
            query=args.query,
            llm_model=args.model or None,
            result_limit=args.limit,
        )
        print("Task 2 web update completed")
        print(f"- Query       : {result['query']}")
        print(f"- Search hits : {result['result_count']}")
        print(f"- Topics found: {result['topic_count']}")
        print(f"- Topics saved: {result['saved_count']}")
        return 0

    if args.command == "import-corpus":
        result = prepare_external_corpus_locally(
            args.source_dir,
            llm_model=args.model or None,
            chunk_strategy=args.strategy,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            max_chunks_per_file=args.max_chunks_per_file,
            dry_run=args.dry_run,
        )
        print("External corpus preparation completed")
        print(f"- Source dir : {result['source_dir']}")
        print(f"- Files      : {result['files']}")
        print(f"- Dry run    : {result['dry_run']}")
        for report in result["reports"]:
            print(
                f"- {report['source_file']}: plan={report['plan'].strategy} "
                f"source={report['plan'].decision_source} chunks={report['chunk_count']} "
                f"prepared={report['prepared']}"
            )
        return 0

    if args.command == "list-task2":
        topics = get_writing_task2_bank(limit=args.limit, essay_type=args.essay_type or None)
        print("Stored Task 2 topics")
        for item in topics:
            print(
                f"- [{item['exam_date']}] {item['essay_type']} / {item.get('topic_category') or '未分类'} :: "
                f"{item['prompt_text'][:120]}"
            )
        print(f"Total shown: {len(topics)}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
