"""CLI entrypoint for gap-driven retrieval evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from project.rag.orchestration.eval import evaluate_gap_retrieval_cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate gap-driven retrieval.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N cases.")
    args = parser.parse_args()

    result = evaluate_gap_retrieval_cases(limit=args.limit)
    summary = result["summary"]
    print("\nGap retrieval evaluation")
    print("-" * 108)
    print(
        "cases={case_count}  dup={avg_cross_round_duplicate_rate:.3f}  new_facts={avg_new_facts:.2f}  "
        "gap_fill={avg_gap_fill_rate:.3f}  acc={avg_final_answer_accuracy:.3f}  rounds={avg_rounds:.2f}  "
        "latency_ms={avg_latency_ms:.1f}  tokens={avg_token_estimate:.1f}".format(**summary)
    )
    print("-" * 108)
    for item in result["cases"]:
        print(
            "{case_id}: rounds={rounds} dup={avg_duplicate_rate:.3f} new={total_new_facts} "
            "gap_fill={gap_fill_rate:.3f} acc={final_answer_accuracy:.3f} stop={stop_reason}".format(**item)
        )

    output_dir = Path("data") / "retrieval_eval_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "retrieval_eval_report.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved evaluation report to: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
