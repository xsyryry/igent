"""CLI utility for preparing local IELTS resources for the simple local RAG."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from project.config import get_config
from project.logging_config import setup_logging
from project.rag.chunking_agent import ChunkPlan, ChunkRecord, export_chunks_jsonl, prepare_chunks
from project.rag.ingestion_plan import IngestionPlan, build_ingestion_plan


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_GUIDE_PATH = DEFAULT_DATA_DIR / "RAG_COLLECTION_GUIDE.md"
DEFAULT_CHUNK_OUTPUT_DIR = DEFAULT_DATA_DIR / "chunk_previews"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare local IELTS resources for the built-in simple RAG according to RAG_COLLECTION_GUIDE.md.",
    )
    parser.add_argument(
        "--tier",
        choices=("tier1", "tier2", "tier3", "all"),
        default="tier1",
        help="Which collection tier to prepare. Default: tier1.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing local RAG files. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--guide-path",
        type=Path,
        default=DEFAULT_GUIDE_PATH,
        help=f"Guide markdown path. Default: {DEFAULT_GUIDE_PATH}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the preparation plan without changing anything.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of files to prepare from the selected tier.",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Optional specific filename under data/ or absolute path. Can be used multiple times.",
    )
    parser.add_argument(
        "--use-chunks",
        action="store_true",
        help="Chunk selected files first, then export chunk previews for local RAG inspection.",
    )
    parser.add_argument(
        "--chunk-strategy",
        choices=("auto", "llm_auto", "sliding", "headings", "qa_pairs", "mistake_rules"),
        default="auto",
        help="Chunking strategy when --use-chunks is enabled.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Optional LLM model override when --chunk-strategy llm_auto is used.",
    )
    parser.add_argument("--chunk-size", type=int, default=1200, help="Sliding chunk size when chunking.")
    parser.add_argument("--overlap", type=int, default=150, help="Sliding overlap when chunking.")
    parser.add_argument(
        "--max-chunks-per-file",
        type=int,
        default=0,
        help="Optional limit for how many chunks per file to upload.",
    )
    return parser.parse_args()


def _select_files(plan: IngestionPlan, tier: str) -> list[Path]:
    if tier == "tier1":
        return plan.tier1
    if tier == "tier2":
        return plan.tier2
    if tier == "tier3":
        return plan.tier3
    return plan.tier1 + plan.tier2 + plan.tier3


def _print_plan(plan: IngestionPlan) -> None:
    print("Local RAG preparation plan")
    print(f"- Tier 1 : {len(plan.tier1)} file(s)")
    print(f"- Tier 2 : {len(plan.tier2)} file(s)")
    print(f"- Tier 3 : {len(plan.tier3)} file(s)")
    print(f"- Convert: {len(plan.convert)} file(s)")
    print(f"- Skip   : {len(plan.skip)} file(s)")
    if plan.unresolved_from_guide:
        print("- Unresolved guide entries:")
        for name in plan.unresolved_from_guide:
            print(f"  - {name}")


def _resolve_files(file_args: list[str], data_dir: Path) -> list[Path]:
    resolved: list[Path] = []
    for raw_path in file_args:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = data_dir / raw_path
        resolved.append(candidate)
    return resolved


def _prepare_chunk_job(path: Path, args: argparse.Namespace) -> tuple[ChunkPlan, list[ChunkRecord]]:
    plan, chunks = prepare_chunks(
        path,
        strategy=args.chunk_strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        llm_model=args.model or None,
    )
    if args.max_chunks_per_file > 0:
        chunks = chunks[: args.max_chunks_per_file]
    return plan, chunks


def _prepare_one(path: Path, args: argparse.Namespace, data_dir: Path) -> dict[str, str]:
    if not args.use_chunks:
        inside_data_dir = data_dir.resolve() in path.resolve().parents or path.resolve() == data_dir.resolve()
        message = "file is already available for local simple RAG indexing"
        if not inside_data_dir:
            message = "file validated, but simple_rag only auto-indexes files under data/ and project/data/"
        return {"status": "ok", "message": message}

    plan, chunks = _prepare_chunk_job(path, args)
    output_path = DEFAULT_CHUNK_OUTPUT_DIR / f"{path.stem}.{args.chunk_strategy}.{plan.strategy}.chunks.jsonl"
    export_chunks_jsonl(chunks, output_path)
    return {
        "status": "ok",
        "message": (
            f"{len(chunks)} chunk(s) prepared locally; "
            f"plan={plan.strategy} source={plan.decision_source}; preview saved to {output_path}"
        ),
    }


def main() -> int:
    load_dotenv()
    config = get_config()
    setup_logging(config.log_level)
    args = _parse_args()

    if not args.data_dir.exists():
        print(f"Data directory not found: {args.data_dir}")
        return 1
    if not args.guide_path.exists():
        print(f"Guide file not found: {args.guide_path}")
        return 1

    plan = build_ingestion_plan(args.data_dir, args.guide_path)
    _print_plan(plan)

    if plan.convert:
        print("\nFiles that should be converted before preparation:")
        for path in plan.convert:
            print(f"- {path.name}")

    selected = _resolve_files(args.file, args.data_dir) if args.file else _select_files(plan, args.tier)
    if args.limit > 0:
        selected = selected[: args.limit]

    print(f"\nSelected source: {'manual file list' if args.file else args.tier}")
    for path in selected:
        print(f"- {path.name}")

    missing_files = [path for path in selected if not path.exists()]
    if missing_files:
        print("\nMissing files:")
        for path in missing_files:
            print(f"- {path}")
        return 1

    if args.use_chunks:
        print(
            f"\nChunk mode enabled: strategy={args.chunk_strategy}, chunk_size={args.chunk_size}, overlap={args.overlap}, "
            f"max_chunks_per_file={'all' if args.max_chunks_per_file <= 0 else args.max_chunks_per_file}, "
            f"model={args.model or 'default/not-used'}"
        )

    if args.dry_run:
        if args.use_chunks:
            print("\nChunk estimation:")
            for path in selected:
                try:
                    plan, chunks = _prepare_chunk_job(path, args)
                    print(
                        f"- {path.name}: {len(chunks)} chunk(s) "
                        f"[resolved_strategy={plan.strategy}, source={plan.decision_source}, model={plan.model_name or 'default/not-used'}]"
                    )
                except Exception as exc:  # pragma: no cover - dry-run reporting only
                    print(f"- {path.name}: failed to estimate chunks ({exc})")
            print("\nDry run only. No files changed.")
        return 0

    print("\nPreparing files...")
    failures = 0
    for path in selected:
        try:
            result = _prepare_one(path, args, args.data_dir)
            print(f"[OK] {path.name} :: {result['message']}")
        except Exception as exc:
            failures += 1
            print(f"[FAILED] {path.name} :: {exc}")

    if failures:
        print(f"\nFinished with {failures} failure(s).")
        return 1

    print("\nLocal RAG preparation completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
