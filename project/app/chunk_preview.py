"""CLI utility to preview chunking strategies on local IELTS materials."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from project.config import get_config
from project.logging_config import setup_logging
from project.rag.chunking_agent import export_chunks_jsonl, prepare_chunks


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "chunk_previews"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview local document chunking before preparing files for local RAG.",
    )
    parser.add_argument("--file", required=True, help="Filename under data/ or an absolute path.")
    parser.add_argument(
        "--strategy",
        choices=("auto", "llm_auto", "sliding", "headings", "qa_pairs", "mistake_rules"),
        default="auto",
        help="Chunking strategy. Default: auto.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Optional LLM model override when --strategy llm_auto is used.",
    )
    parser.add_argument("--chunk-size", type=int, default=1200, help="Sliding chunk size.")
    parser.add_argument("--overlap", type=int, default=150, help="Sliding overlap size.")
    parser.add_argument("--preview-count", type=int, default=5, help="How many chunks to print.")
    parser.add_argument("--preview-chars", type=int, default=200, help="Chars shown per chunk preview.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSONL export path.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help=f"Default: {DEFAULT_DATA_DIR}")
    return parser.parse_args()


def _resolve_input_path(raw_path: str, data_dir: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return data_dir / raw_path


def _print_preview(chunks: list, preview_count: int, preview_chars: int) -> None:
    print(f"\nGenerated {len(chunks)} chunk(s).")
    for index, chunk in enumerate(chunks[:preview_count], start=1):
        snippet = chunk.content[:preview_chars].replace("\n", " ")
        print("-" * 72)
        print(f"Chunk {index}")
        print(f"id       : {chunk.chunk_id}")
        print(f"strategy : {chunk.strategy}")
        print(f"title    : {chunk.title}")
        print(f"length   : {len(chunk.content)}")
        print(f"preview  : {snippet}")
    print("-" * 72)


def main() -> int:
    load_dotenv()
    config = get_config()
    setup_logging(config.log_level)
    args = _parse_args()

    source_path = _resolve_input_path(args.file, args.data_dir)
    if not source_path.exists():
        print(f"File not found: {source_path}")
        return 1

    plan, chunks = prepare_chunks(
        source_path,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        llm_model=args.model or None,
    )
    print(
        "\nChunk plan"
        f"\n- requested      : {plan.requested_strategy}"
        f"\n- resolved       : {plan.strategy}"
        f"\n- chunk_size     : {plan.chunk_size}"
        f"\n- overlap        : {plan.overlap}"
        f"\n- decision_source: {plan.decision_source}"
        f"\n- model          : {plan.model_name or 'default/not-used'}"
        f"\n- reason         : {plan.reason}"
    )
    _print_preview(chunks, args.preview_count, args.preview_chars)

    output_path = args.output or (
        DEFAULT_OUTPUT_DIR / f"{source_path.stem}.{args.strategy}.{plan.strategy}.chunks.jsonl"
    )
    export_chunks_jsonl(chunks, output_path)
    print(f"Saved chunk preview to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
