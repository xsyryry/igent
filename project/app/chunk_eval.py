"""CLI utility for evaluating chunking quality on a minimal IELTS benchmark."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from project.config import get_config
from project.logging_config import setup_logging
from project.rag.chunk_eval import (
    DEFAULT_CASES_PATH,
    DEFAULT_DATA_DIR,
    StrategyEvalResult,
    evaluate_strategies,
    export_eval_report,
    load_eval_cases,
)


DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "chunk_eval_reports"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate local chunking strategies on a minimal IELTS benchmark.",
    )
    parser.add_argument(
        "--strategies",
        default="sliding,headings,llm_auto",
        help="Comma-separated strategies to evaluate. Default: sliding,headings,llm_auto",
    )
    parser.add_argument(
        "--cases-path",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help=f"Default: {DEFAULT_CASES_PATH}",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Optional LLM model override for llm_auto.",
    )
    parser.add_argument("--chunk-size", type=int, default=1200, help="Fallback sliding chunk size.")
    parser.add_argument("--overlap", type=int, default=150, help="Fallback sliding overlap.")
    parser.add_argument(
        "--show-cases",
        action="store_true",
        help="Print per-case matched ranks for each strategy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report output path.",
    )
    return parser.parse_args()


def _print_summary(results: list[StrategyEvalResult]) -> None:
    print("\nChunk evaluation summary")
    print("-" * 110)
    print(
        f"{'strategy':<14}"
        f"{'hit@1':>10}"
        f"{'hit@3':>10}"
        f"{'mrr':>10}"
        f"{'cases':>10}"
        f"{'chunks':>12}"
        f"{'avg/file':>12}"
        f"{'avg chars':>12}"
    )
    print("-" * 110)
    for result in results:
        print(
            f"{result.strategy:<14}"
            f"{result.hit_at_1:>10.2%}"
            f"{result.hit_at_3:>10.2%}"
            f"{result.mrr:>10.3f}"
            f"{result.evaluated_case_count:>10}"
            f"{result.chunk_count:>12}"
            f"{result.avg_chunks_per_file:>12.2f}"
            f"{result.avg_chunk_chars:>12.1f}"
        )
    print("-" * 110)


def _print_file_plans(results: list[StrategyEvalResult]) -> None:
    print("\nResolved file plans")
    for result in results:
        print(f"\n[{result.strategy}]")
        for plan in result.file_plans:
            print(
                f"- {plan['source_file']}: resolved={plan['resolved_strategy']}, "
                f"source={plan['decision_source']}, chunks={plan['chunk_count']}, "
                f"model={plan['model_name'] or 'default/not-used'}"
            )
        if result.skipped_files:
            print(f"  skipped files: {', '.join(result.skipped_files)}")


def _print_case_details(results: list[StrategyEvalResult]) -> None:
    print("\nCase-level results")
    for result in results:
        print(f"\n[{result.strategy}]")
        for case_result in result.case_results:
            print(
                f"- {case_result.case_id}: rank={case_result.matched_at or 'miss'}, "
                f"top_source={case_result.top_result_source or 'none'}, "
                f"top_score={case_result.top_result_score:.2f}"
            )


def main() -> int:
    load_dotenv()
    config = get_config()
    setup_logging(config.log_level)
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    args = _parse_args()

    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    cases = load_eval_cases(args.cases_path)
    results = evaluate_strategies(
        strategies=strategies,
        cases=cases,
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        llm_model=args.model or None,
    )

    _print_summary(results)
    _print_file_plans(results)
    if args.show_cases:
        _print_case_details(results)

    output_path = args.output or (DEFAULT_OUTPUT_DIR / "chunk_eval_report.json")
    export_eval_report(results, output_path)
    print(f"\nSaved evaluation report to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
