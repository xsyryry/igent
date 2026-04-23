"""Run IELTS Task 2 writing-review RAG evaluation.

Full mode output contract per case:
- review_result.json: machine-readable review result only
- monitor.json: module / skill / prompt / RAG call-chain only
- review.txt: concise human-readable summary

Light mode output contract per case:
- retrieval_report.json: machine-readable retrieval result only
- monitor.json: module / RAG / blocked LLM call-chain only
- retrieval_report.txt: concise human-readable summary
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.db.repository import save_writing_sample, upsert_writing_task2_topic
from project.llm.client import LLMClient
from project.tools import writing_tool
from project.rag.orchestration.gap_retrieval import summarize_gap_retrieval_state
from project.tools.writing_tool import (
    execute_writing_retrieval_round,
    finalize_task2_review,
    prepare_task2_review_context,
)


DEFAULT_SAMPLE_FILE = ROOT / "ielts_task2_test_samples.txt"
DEFAULT_OUTPUT_ROOT = ROOT / "rag_performance_tests" / "outputs"
BAND_ORDER = ("9", "7", "5")

LBR = chr(0x3010)
RBR = chr(0x3011)
WORD_SAMPLE = "\u6837\u4f8b"
WORD_TITLE = "\u9898\u76ee"
WORD_EXPECTED = "\u9884\u671f"
WORD_DEDUCTION = "\u6263\u5206\u70b9"
_PROBE_LOCAL = threading.local()
_PROGRESS_LOCK = threading.Lock()


@dataclass
class Task2Sample:
    sample_no: int
    category: str
    question: str
    essays: dict[str, str]
    expected_deductions: dict[str, list[str]]


class GlobalProbePatch(AbstractContextManager["GlobalProbePatch"]):
    """Patch project call sites once; each worker records into thread-local probe."""

    def __init__(self, *, disable_llm_network: bool = False) -> None:
        self._orig_skill = None
        self._orig_prompt = None
        self._orig_generate = None
        self.disable_llm_network = disable_llm_network

    def __enter__(self) -> "GlobalProbePatch":
        self._orig_skill = writing_tool._load_writing_review_skill_instructions
        self._orig_prompt = writing_tool._load_official_prompt_file
        self._orig_generate = LLMClient.generate_text

        def skill_wrapper() -> str:
            probe = getattr(_PROBE_LOCAL, "probe", None)
            if probe is not None:
                probe.skill_loaded = True
            return self._orig_skill()

        def prompt_wrapper(prompt_name: str) -> str:
            probe = getattr(_PROBE_LOCAL, "probe", None)
            if probe is not None:
                probe.prompt_files_loaded.append(prompt_name)
            return self._orig_prompt(prompt_name)

        def generate_wrapper(client: LLMClient, system_prompt: str, user_prompt: str, **kwargs: Any) -> str | None:
            probe = getattr(_PROBE_LOCAL, "probe", None)
            if probe is not None:
                call = {
                    "system_len": len(system_prompt or ""),
                    "user_len": len(user_prompt or ""),
                    "is_writing_review_prompt": "You are an IELTS Writing reviewer" in (system_prompt or ""),
                    "has_writing_review_skill": "writing_review_skill" in (system_prompt or ""),
                    "has_task1_prompt": "official_ielts_writing_task1_scoring_prompt" in (system_prompt or ""),
                    "has_task2_prompt": "official_ielts_writing_task2_scoring_prompt" in (system_prompt or ""),
                    "has_key_criteria_prompt": "official_ielts_writing_key_assessment_criteria_prompt" in (system_prompt or ""),
                    "has_magazine_rag_support": "External magazine RAG support" in (user_prompt or ""),
                    "max_tokens": kwargs.get("max_tokens"),
                    "temperature": kwargs.get("temperature"),
                }
                probe.llm_calls.append(call)
                if self.disable_llm_network:
                    probe.llm_network_blocked_count += 1
                    return None
            if self.disable_llm_network:
                return None
            return self._orig_generate(client, system_prompt, user_prompt, **kwargs)

        writing_tool._load_writing_review_skill_instructions = skill_wrapper
        writing_tool._load_official_prompt_file = prompt_wrapper
        LLMClient.generate_text = generate_wrapper
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._orig_skill is not None:
            writing_tool._load_writing_review_skill_instructions = self._orig_skill
        if self._orig_prompt is not None:
            writing_tool._load_official_prompt_file = self._orig_prompt
        if self._orig_generate is not None:
            LLMClient.generate_text = self._orig_generate


class ReviewProbe:
    """Per-thread probe data."""

    def __init__(self) -> None:
        self.skill_loaded = False
        self.prompt_files_loaded: list[str] = []
        self.llm_calls: list[dict[str, Any]] = []
        self.llm_network_blocked_count = 0

    def __enter__(self) -> "ReviewProbe":
        _PROBE_LOCAL.probe = self
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if hasattr(_PROBE_LOCAL, "probe"):
            delattr(_PROBE_LOCAL, "probe")

    def snapshot(self) -> dict[str, Any]:
        writing_calls = [item for item in self.llm_calls if item.get("is_writing_review_prompt")]
        latest = writing_calls[-1] if writing_calls else {}
        return {
            "skill_loaded": self.skill_loaded,
            "prompt_files_loaded": self.prompt_files_loaded,
            "llm_call_count": len(self.llm_calls),
            "llm_network_blocked_count": self.llm_network_blocked_count,
            "writing_review_llm_call_count": len(writing_calls),
            "evaluation_prompt_has_skill": bool(latest.get("has_writing_review_skill")),
            "evaluation_prompt_has_task2_prompt": bool(latest.get("has_task2_prompt")),
            "evaluation_prompt_has_key_criteria_prompt": bool(latest.get("has_key_criteria_prompt")),
            "evaluation_prompt_has_task1_prompt": bool(latest.get("has_task1_prompt")),
            "evaluation_user_prompt_has_magazine_rag_support": bool(latest.get("has_magazine_rag_support")),
        }


def main() -> int:
    args = parse_args()
    load_project_env()

    samples = parse_samples(Path(args.sample_file))
    if args.limit_topics:
        samples = samples[: args.limit_topics]
    bands = parse_bands(args.bands)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_report = build_mapping_report(samples)
    write_json(output_dir / "mapping_report.json", mapping_report)
    (output_dir / "mapping_report.txt").write_text(format_mapping_report(mapping_report), encoding="utf-8")

    total = len(samples) * len(bands)
    unit_label = "reviews" if args.mode == "full" else "retrieval_cases"
    print(f"loaded samples={len(samples)}, bands={','.join(bands)}, mode={args.mode}, total_{unit_label}={total}")
    print(f"output_dir={output_dir}")
    if mapping_report["warning_count"]:
        print(f"mapping warnings={mapping_report['warning_count']} (see mapping_report.txt)")

    if args.dry_run:
        print("dry_run=true, evaluation calls skipped")
        return 0

    records: list[dict[str, Any]] = []
    with GlobalProbePatch(disable_llm_network=args.mode == "light"):
        max_workers = max(1, min(args.workers, len(samples) or 1))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_sample_type, sample, bands, output_dir, args.user_id, args.seed_band9_reference, args.mode)
                for sample in samples
            ]
            for future in as_completed(futures):
                records.extend(future.result())

    records.sort(key=lambda item: (int(item["sample_no"]), float(item["expected_band"])))
    index_path = output_dir / "run_index.jsonl"
    with index_path.open("w", encoding="utf-8") as index_file:
        for record in records:
            index_file.write(json.dumps({key: value for key, value in record.items() if key != "monitor"}, ensure_ascii=False) + "\n")

    summary = build_full_summary(records, output_dir, mapping_report) if args.mode == "full" else build_light_summary(records, output_dir, mapping_report)
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print()
    print(f"summary written: {output_dir / 'summary.txt'}")
    return 0


def process_sample_type(
    sample: Task2Sample,
    bands: list[str],
    output_dir: Path,
    base_user_id: str,
    seed_band9_reference: bool,
    mode: str,
) -> list[dict[str, Any]]:
    """Run one topic/category in one worker thread."""

    topic = ensure_topic(sample)
    if seed_band9_reference and sample.essays.get("9"):
        save_writing_sample(
            task2_topic_id=topic["id"],
            sample_type="reference_band9",
            title=f"sample_{sample.sample_no:02d}_band9_reference",
            content=sample.essays["9"],
            source_label="rag_performance_tests",
            metadata={"sample_no": sample.sample_no, "category": sample.category, "expected_band": 9},
        )

    records: list[dict[str, Any]] = []
    for index, band in enumerate(bands, start=1):
        print_progress(sample.category, index - 1, len(bands), f"sample={sample.sample_no:02d} current_band={band}")
        case_user_id = f"{base_user_id}_s{sample.sample_no:02d}_b{band}"
        record = (
            run_one_review(sample=sample, band=band, topic_id=topic["id"], user_id=case_user_id)
            if mode == "full"
            else run_one_light_retrieval(sample=sample, band=band, topic_id=topic["id"], user_id=case_user_id)
        )
        case_dir = case_output_dir(output_dir, sample, band)
        if mode == "full":
            write_case_outputs(case_dir, record)
            records.append(
                {
                    "sample_no": sample.sample_no,
                    "category": sample.category,
                    "expected_band": float(band),
                    "case_dir": str(case_dir),
                    "overall_band": record["review_result"].get("evaluation", {}).get("overall_band"),
                    "score_error_vs_expected": record["review_result"].get("score_error_vs_expected"),
                    "monitor": record["monitor"],
                }
            )
        else:
            write_light_case_outputs(case_dir, record)
            records.append(
                {
                    "sample_no": sample.sample_no,
                    "category": sample.category,
                    "expected_band": float(band),
                    "case_dir": str(case_dir),
                    "retrieved_doc_count": record["retrieval_report"].get("retrieved_doc_count"),
                    "unique_source_count": record["retrieval_report"].get("unique_source_count"),
                    "avg_top_score": record["retrieval_report"].get("avg_top_score"),
                    "monitor": record["monitor"],
                }
            )
        print_progress(sample.category, index, len(bands), f"sample={sample.sample_no:02d} done_band={band}", newline=index >= len(bands))
    return records


def load_project_env() -> None:
    project_env = ROOT / "project" / ".env"
    if project_env.exists():
        load_dotenv(project_env, override=True)
    else:
        load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate writing-review RAG behavior on IELTS Task 2 samples.")
    parser.add_argument("--mode", choices=("full", "light"), default="full", help="full=complete review, light=RAG retrieval only without LLM judge.")
    parser.add_argument("--sample-file", default=str(DEFAULT_SAMPLE_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--user-id", default="rag_eval_user")
    parser.add_argument("--limit-topics", type=int, default=0)
    parser.add_argument("--bands", default="9,7,5")
    parser.add_argument("--workers", type=int, default=3, help="Topic-level worker threads. Default: 3.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-seed-band9-reference", action="store_false", dest="seed_band9_reference")
    parser.set_defaults(seed_band9_reference=True)
    return parser.parse_args()


def parse_bands(value: str) -> list[str]:
    requested = {item.strip() for item in value.split(",") if item.strip()}
    bands = [item for item in BAND_ORDER if item in requested]
    if not bands:
        raise SystemExit("No valid bands selected. Use any of: 5,7,9")
    return bands


def parse_samples(path: Path) -> list[Task2Sample]:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8-sig")
    blocks = [block for block in text.split("=" * 60) if f"{LBR}{WORD_SAMPLE}" in block]
    samples: list[Task2Sample] = []
    for block in blocks:
        headings = collect_sections(block)
        sample_heading = next((head for head in headings if head.startswith(f"{LBR}{WORD_SAMPLE}")), "")
        if not sample_heading:
            continue
        sample_no = int(sample_heading.split(WORD_SAMPLE, 1)[1].split(RBR, 1)[0].strip())
        category = sample_heading.split(RBR, 1)[1].strip()
        question = ""
        essays: dict[str, str] = {}
        deductions: dict[str, list[str]] = {}
        for heading, body in headings.items():
            if WORD_TITLE in heading:
                question = body.strip()
            elif "Band 9" in heading and WORD_EXPECTED not in heading:
                essays["9"] = body.strip()
            elif "Band 7" in heading and WORD_EXPECTED not in heading:
                essays["7"] = body.strip()
            elif "Band 5" in heading and WORD_EXPECTED not in heading:
                essays["5"] = body.strip()
            elif "Band 7" in heading and WORD_DEDUCTION in heading:
                deductions["7"] = bullet_lines(body)
            elif "Band 5" in heading and WORD_DEDUCTION in heading:
                deductions["5"] = bullet_lines(body)
        if question and essays:
            samples.append(Task2Sample(sample_no, category, question, essays, deductions))
    return samples


def collect_sections(block: str) -> dict[str, str]:
    lines = block.splitlines()
    starts: list[tuple[str, int]] = []
    cursor = 0
    for line in lines:
        idx = block.find(line, cursor)
        if idx >= 0:
            cursor = idx + len(line)
        if line.startswith(LBR):
            starts.append((line.strip(), idx))
    sections: dict[str, str] = {}
    for index, (heading, start) in enumerate(starts):
        end = starts[index + 1][1] if index + 1 < len(starts) else len(block)
        sections[heading] = block[start + len(heading):end].strip()
    return sections


def bullet_lines(text: str) -> list[str]:
    return [line.strip("- ").strip() for line in text.splitlines() if line.strip().startswith("-")]


def build_mapping_report(samples: list[Task2Sample]) -> dict[str, Any]:
    items = []
    warning_count = 0
    for sample in samples:
        warnings = []
        missing_bands = [band for band in BAND_ORDER if not sample.essays.get(band)]
        if missing_bands:
            warnings.append(f"missing bands: {','.join(missing_bands)}")
        if not sample.question.strip():
            warnings.append("empty question")
        band_hashes = {band: short_hash(text) for band, text in sample.essays.items()}
        if len(set(band_hashes.values())) != len(band_hashes):
            warnings.append("duplicate essay text across bands")
        band9 = sample.essays.get("9", "")
        if not band9:
            warnings.append("missing Band 9 essay")
        if band9 and len(band9.split()) < 220:
            warnings.append(f"Band 9 word count looks low: {len(band9.split())}")
        if any(marker in band9 for marker in (f"{LBR}Band 7", f"{LBR}Band 5", f"{LBR}{WORD_TITLE}")):
            warnings.append("Band 9 essay contains another heading marker")
        if sample.expected_deductions.get("9"):
            warnings.append("Band 9 unexpectedly has deduction labels")
        warning_count += len(warnings)
        items.append(
            {
                "sample_no": sample.sample_no,
                "category": sample.category,
                "question_hash": short_hash(sample.question),
                "question_preview": one_line(sample.question, 140),
                "band_word_counts": {band: len(sample.essays.get(band, "").split()) for band in BAND_ORDER},
                "band_hashes": band_hashes,
                "deduction_counts": {band: len(sample.expected_deductions.get(band, [])) for band in ("7", "5")},
                "warnings": warnings,
                "band9_mapping_ok": not warnings,
            }
        )
    return {"sample_count": len(samples), "warning_count": warning_count, "items": items}


def format_mapping_report(report: dict[str, Any]) -> str:
    lines = [f"sample_count: {report['sample_count']}", f"warning_count: {report['warning_count']}", ""]
    for item in report["items"]:
        lines.append(f"sample {item['sample_no']:02d} | {item['category']} | band9_ok={item['band9_mapping_ok']}")
        lines.append(f"question: {item['question_preview']}")
        lines.append(f"word_counts: {item['band_word_counts']}")
        if item["warnings"]:
            lines.extend(f"warning: {warning}" for warning in item["warnings"])
        lines.append("")
    return "\n".join(lines)


def ensure_topic(sample: Task2Sample) -> dict[str, Any]:
    return upsert_writing_task2_topic(
        exam_date=f"rag-eval-sample-{sample.sample_no:02d}",
        prompt_text=sample.question,
        essay_type="Task 2",
        topic_category=sample.category,
        source_title="ielts_task2_test_samples",
        source_file=str(DEFAULT_SAMPLE_FILE),
        metadata={"rag_eval": True, "sample_no": sample.sample_no, "category": sample.category},
    )


def run_one_review(*, sample: Task2Sample, band: str, topic_id: str, user_id: str) -> dict[str, Any]:
    essay = sample.essays[band]
    started = time.perf_counter()
    with ReviewProbe() as probe:
        preparation = prepare_task2_review_context(user_input=essay, topic_id=topic_id, user_id=user_id)
        if not preparation.get("success"):
            monitor = failure_monitor(sample, band, preparation, probe, started)
            return {"review_result": failure_review_result(sample, band, essay, preparation), "monitor": monitor}

        review_state = preparation["review_state"]
        while not bool(review_state.get("enough_context", False)) and int(review_state.get("retrieval_round", 0) or 0) < int(review_state.get("max_rounds", 3) or 3):
            review_state = execute_writing_retrieval_round(review_state)

        result = finalize_task2_review(review_state, user_id=user_id)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        monitor = build_monitor(probe=probe, result=result, review_state=review_state, expected_band=band, elapsed_ms=elapsed_ms)
        review_result = build_review_result(sample=sample, band=band, essay=essay, result=result, monitor=monitor)
        return {"review_result": review_result, "monitor": monitor}


def run_one_light_retrieval(*, sample: Task2Sample, band: str, topic_id: str, user_id: str) -> dict[str, Any]:
    """Run the formal retrieval loop only; LLM network calls are blocked by GlobalProbePatch."""

    essay = sample.essays[band]
    started = time.perf_counter()
    with ReviewProbe() as probe:
        preparation = prepare_task2_review_context(user_input=essay, topic_id=topic_id, user_id=user_id)
        if not preparation.get("success"):
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            monitor = build_light_monitor(
                probe=probe,
                retrieval_summary={},
                review_state={},
                success=False,
                elapsed_ms=elapsed_ms,
                error=preparation.get("message") or "prepare_task2_review_context failed",
            )
            return {
                "retrieval_report": failure_retrieval_report(sample, band, essay, preparation),
                "monitor": monitor,
            }

        review_state = preparation["review_state"]
        while not bool(review_state.get("enough_context", False)) and int(review_state.get("retrieval_round", 0) or 0) < int(review_state.get("max_rounds", 3) or 3):
            review_state = execute_writing_retrieval_round(review_state)

        retrieval_summary = summarize_gap_retrieval_state(dict(review_state.get("retrieval_state", {})))
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        monitor = build_light_monitor(
            probe=probe,
            retrieval_summary=retrieval_summary,
            review_state=review_state,
            success=True,
            elapsed_ms=elapsed_ms,
        )
        report = build_light_retrieval_report(
            sample=sample,
            band=band,
            essay=essay,
            retrieval_summary=retrieval_summary,
            review_state=review_state,
            monitor=monitor,
        )
        return {"retrieval_report": report, "monitor": monitor}


def failure_retrieval_report(sample: Task2Sample, band: str, essay: str, preparation: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample": sample_meta(sample, band, essay),
        "question": sample.question,
        "success": False,
        "error": preparation.get("message") or "prepare_task2_review_context failed",
        "retrieved_doc_count": 0,
        "top_docs": [],
    }


def failure_review_result(sample: Task2Sample, band: str, essay: str, preparation: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample": sample_meta(sample, band, essay),
        "question": sample.question,
        "expected_deductions": sample.expected_deductions.get(band, []),
        "success": False,
        "error": preparation.get("message") or "prepare_task2_review_context failed",
        "evaluation": {},
    }


def failure_monitor(sample: Task2Sample, band: str, preparation: dict[str, Any], probe: ReviewProbe, started: float) -> dict[str, Any]:
    del sample, band
    return {
        **probe.snapshot(),
        **inspect_prompt_config(),
        "llm_configured": LLMClient.from_config().is_configured,
        "success": False,
        "writing_review_module_called": False,
        "error": preparation.get("message") or "prepare_task2_review_context failed",
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


def build_review_result(*, sample: Task2Sample, band: str, essay: str, result: dict[str, Any], monitor: dict[str, Any]) -> dict[str, Any]:
    del monitor
    evaluation = canonical_evaluation(result.get("evaluation", {}) if isinstance(result, dict) else {})
    try:
        score_error = round(float(evaluation.get("overall_band")) - float(band), 2)
    except (TypeError, ValueError):
        score_error = None
    return {
        "sample": sample_meta(sample, band, essay),
        "question": sample.question,
        "expected_deductions": sample.expected_deductions.get(band, []),
        "success": bool(result.get("success")),
        "score_error_vs_expected": score_error,
        "evaluation": evaluation,
    }


def sample_meta(sample: Task2Sample, band: str, essay: str) -> dict[str, Any]:
    return {
        "sample_no": sample.sample_no,
        "category": sample.category,
        "expected_band": float(band),
        "essay_word_count": len(essay.split()),
        "essay_hash": short_hash(essay),
        "question_hash": short_hash(sample.question),
    }


def canonical_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    breakdown = evaluation.get("band_breakdown", {}) if isinstance(evaluation.get("band_breakdown"), dict) else {}
    return {
        "task_type": evaluation.get("task_type") or "task2",
        "overall_band": evaluation.get("overall_band"),
        "band_breakdown": {
            "task_response": first_present(breakdown, ("task_response", "task_response_or_achievement", "task_achievement")),
            "coherence_and_cohesion": first_present(breakdown, ("coherence_and_cohesion", "coherence_cohesion")),
            "lexical_resource": first_present(breakdown, ("lexical_resource",)),
            "grammatical_range_and_accuracy": first_present(breakdown, ("grammatical_range_and_accuracy", "grammar_accuracy")),
        },
        "evidence_based_comment": evaluation.get("evidence_based_comment") or evaluation.get("overall_comment", ""),
        "strengths": evaluation.get("strengths", []),
        "issues": evaluation.get("issues", []),
        "priority_issue": evaluation.get("priority_issue", {}),
        "revision_plan": evaluation.get("revision_plan", []),
        "language_upgrade_notes": evaluation.get("language_upgrade_notes", []),
        "score_evidence": evaluation.get("score_evidence", {}),
        "confidence": evaluation.get("confidence", ""),
        "limitations": evaluation.get("limitations", []),
        "summary_for_memory": evaluation.get("summary_for_memory", ""),
        "evaluation_source": evaluation.get("evaluation_source", ""),
    }


def first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data:
            return data.get(key)
    return None


def build_monitor(*, probe: ReviewProbe, result: dict[str, Any], review_state: dict[str, Any], expected_band: str, elapsed_ms: int) -> dict[str, Any]:
    rag_result = result.get("rag_result", {}) if isinstance(result, dict) else {}
    evaluation = result.get("evaluation", {}) if isinstance(result, dict) else {}
    trace = rag_result.get("retrieval_trace", [])
    docs = rag_result.get("retrieved_docs", [])
    try:
        score_error = round(float(evaluation.get("overall_band")) - float(expected_band), 2)
    except (TypeError, ValueError):
        score_error = None
    return {
        "module_chain": {
            "writing_review_module_called": True,
            "success": bool(result.get("success")),
            "elapsed_ms": elapsed_ms,
        },
        "llm": {
            **probe.snapshot(),
            "llm_configured": LLMClient.from_config().is_configured,
        },
        "prompt_config": inspect_prompt_config(),
        "rag": {
            "rag_called": bool(trace or docs),
            "backend": rag_result.get("backend"),
            "query_mode": rag_result.get("query_mode"),
            "retrieval_rounds": review_state.get("retrieval_round", 0),
            "retrieved_doc_count": len(docs) if isinstance(docs, list) else 0,
            "gap_fill_rate": rag_result.get("gap_fill_rate"),
            "stop_reason": rag_result.get("stop_reason"),
            "result": compact_rag_result(rag_result),
        },
    }


def build_light_monitor(
    *,
    probe: ReviewProbe,
    retrieval_summary: dict[str, Any],
    review_state: dict[str, Any],
    success: bool,
    elapsed_ms: int,
    error: str = "",
) -> dict[str, Any]:
    docs = retrieval_summary.get("retrieved_docs", []) if isinstance(retrieval_summary, dict) else []
    trace = retrieval_summary.get("retrieval_trace", []) if isinstance(retrieval_summary, dict) else []
    return {
        "module_chain": {
            "mode": "light",
            "writing_review_module_called": False,
            "retrieval_module_called": True,
            "success": success,
            "elapsed_ms": elapsed_ms,
            "error": error,
        },
        "llm": {
            **probe.snapshot(),
            "llm_configured": LLMClient.from_config().is_configured,
            "llm_network_disabled_by_light_mode": True,
        },
        "rag": {
            "rag_called": bool(trace or docs),
            "backend": retrieval_summary.get("backend"),
            "query_mode": retrieval_summary.get("query_mode"),
            "retrieval_rounds": review_state.get("retrieval_round", 0),
            "retrieved_doc_count": len(docs) if isinstance(docs, list) else 0,
            "gap_fill_rate": retrieval_summary.get("gap_fill_rate"),
            "stop_reason": retrieval_summary.get("stop_reason"),
            "metrics": retrieval_summary.get("metrics", {}),
            "result": compact_rag_result(retrieval_summary),
        },
    }


def build_light_retrieval_report(
    *,
    sample: Task2Sample,
    band: str,
    essay: str,
    retrieval_summary: dict[str, Any],
    review_state: dict[str, Any],
    monitor: dict[str, Any],
) -> dict[str, Any]:
    del monitor
    docs = retrieval_summary.get("retrieved_docs", [])
    docs = docs if isinstance(docs, list) else []
    top_docs = [light_doc_brief(item, index + 1) for index, item in enumerate(docs[:10]) if isinstance(item, dict)]
    scores = [float(item.get("score")) for item in docs if isinstance(item, dict) and isinstance(item.get("score"), (int, float))]
    sources = sorted({str(item.get("source") or "") for item in docs if isinstance(item, dict) and item.get("source")})
    publications = sorted({
        str((item.get("metadata") or {}).get("publication") or "")
        for item in docs
        if isinstance(item, dict) and (item.get("metadata") or {}).get("publication")
    })
    return {
        "sample": sample_meta(sample, band, essay),
        "question": sample.question,
        "success": True,
        "mode": "light_retrieval_only",
        "llm_judge_enabled": False,
        "retrieved_doc_count": len(docs),
        "unique_source_count": len(sources),
        "unique_sources": sources[:20],
        "publications": publications,
        "avg_top_score": round(sum(scores[:5]) / len(scores[:5]), 4) if scores[:5] else 0.0,
        "top_docs": top_docs,
        "retrieval_trace": retrieval_summary.get("retrieval_trace", []),
        "known_facts": retrieval_summary.get("known_facts", []),
        "gaps": retrieval_summary.get("gaps", []),
        "support_answer_preview": one_line(retrieval_summary.get("answer", ""), 1200),
        "stop_reason": retrieval_summary.get("stop_reason", ""),
        "gap_fill_rate": retrieval_summary.get("gap_fill_rate"),
        "metrics": retrieval_summary.get("metrics", {}),
        "review_state": {
            "retrieval_round": review_state.get("retrieval_round", 0),
            "current_query": review_state.get("current_query", ""),
            "current_mode": review_state.get("current_mode", ""),
            "enough_context": review_state.get("enough_context", False),
        },
    }


def light_doc_brief(item: dict[str, Any], rank: int) -> dict[str, Any]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    chunks = item.get("chunks") if isinstance(item.get("chunks"), list) else []
    text = next((chunk for chunk in chunks if isinstance(chunk, str) and chunk.strip()), "")
    return {
        "rank": rank,
        "id": item.get("id"),
        "source": item.get("source"),
        "score": item.get("score"),
        "ranking": item.get("ranking", {}),
        "metadata": {
            "publication": metadata.get("publication"),
            "issue_date": metadata.get("issue_date"),
            "article_title": metadata.get("article_title"),
            "section_title": metadata.get("section_title"),
            "page_range": metadata.get("page_range"),
            "topic": metadata.get("topic"),
            "chunk_type": metadata.get("chunk_type"),
            "strategy": metadata.get("strategy"),
        },
        "text_preview": one_line(text, 500),
    }


def inspect_prompt_config() -> dict[str, Any]:
    skill_path = writing_tool.WRITING_REVIEW_SKILL_PATH
    prompt_dir = writing_tool.WRITING_OFFICIAL_PROMPTS_DIR
    task2_names = writing_tool._official_prompt_names_for_task("task2")
    task1_names = writing_tool._official_prompt_names_for_task("task1_academic")
    task2_files = [writing_tool.WRITING_OFFICIAL_PROMPT_FILES.get(name, "") for name in task2_names]
    task1_files = [writing_tool.WRITING_OFFICIAL_PROMPT_FILES.get(name, "") for name in task1_names]
    skill_text = skill_path.read_text(encoding="utf-8") if skill_path.exists() else ""
    return {
        "skill_file_exists": skill_path.exists(),
        "skill_mentions_task2_rule": "task2" in skill_text and "official_ielts_writing_task2_scoring_prompt.md" in skill_text,
        "task2_prompt_file_exists": any("task2" in name and (prompt_dir / name).exists() for name in task2_files),
        "key_criteria_file_exists": any("key_assessment" in name and (prompt_dir / name).exists() for name in task2_files),
        "task2_mapping": task2_names,
        "task1_mapping": task1_names,
        "task2_files": task2_files,
        "task1_files": task1_files,
        "task2_excludes_task1_scoring_file": not any("task1_scoring" in name for name in task2_files),
    }


def compact_rag_result(rag_result: dict[str, Any]) -> dict[str, Any]:
    docs = rag_result.get("retrieved_docs", [])
    doc_briefs = []
    if isinstance(docs, list):
        for item in docs[:5]:
            if isinstance(item, dict):
                doc_briefs.append({"source": item.get("source"), "score": item.get("score"), "metadata": item.get("metadata", {})})
    return {
        "backend": rag_result.get("backend"),
        "query_mode": rag_result.get("query_mode"),
        "gap_fill_rate": rag_result.get("gap_fill_rate"),
        "stop_reason": rag_result.get("stop_reason"),
        "retrieval_trace": rag_result.get("retrieval_trace", []),
        "doc_briefs": doc_briefs,
    }


def write_case_outputs(case_dir: Path, record: dict[str, Any]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    review_result = record["review_result"]
    monitor = record["monitor"]
    write_json(case_dir / "review_result.json", review_result)
    write_json(case_dir / "monitor.json", monitor)
    (case_dir / "review.txt").write_text(format_human_review(review_result, monitor), encoding="utf-8")


def write_light_case_outputs(case_dir: Path, record: dict[str, Any]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    retrieval_report = record["retrieval_report"]
    monitor = record["monitor"]
    write_json(case_dir / "retrieval_report.json", retrieval_report)
    write_json(case_dir / "monitor.json", monitor)
    (case_dir / "retrieval_report.txt").write_text(format_human_retrieval_report(retrieval_report, monitor), encoding="utf-8")


def format_human_review(review_result: dict[str, Any], monitor: dict[str, Any]) -> str:
    sample = review_result["sample"]
    evaluation = review_result.get("evaluation", {})
    breakdown = evaluation.get("band_breakdown", {})
    issues = evaluation.get("issues", [])
    priority = evaluation.get("priority_issue", {})
    if isinstance(priority, dict):
        priority_text = priority.get("problem") or priority.get("improvement_goal") or ""
    else:
        priority_text = str(priority)
    return "\n".join(
        [
            "IELTS Task 2 Review",
            f"sample: {sample['sample_no']:02d}",
            f"category: {sample['category']}",
            f"expected_band: {sample['expected_band']}",
            f"actual_band: {evaluation.get('overall_band')}",
            f"score_error: {review_result.get('score_error_vs_expected')}",
            f"TR: {breakdown.get('task_response')}",
            f"CC: {breakdown.get('coherence_and_cohesion')}",
            f"LR: {breakdown.get('lexical_resource')}",
            f"GRA: {breakdown.get('grammatical_range_and_accuracy')}",
            "",
            "priority_issue:",
            one_line(priority_text, 240),
            "",
            "top_issues:",
            "\n".join(f"- {one_line(issue.get('problem', issue) if isinstance(issue, dict) else issue, 220)}" for issue in issues[:3]),
            "",
            "monitor:",
            f"- llm_configured: {monitor.get('llm', {}).get('llm_configured')}",
            f"- skill_loaded: {monitor.get('llm', {}).get('skill_loaded')}",
            f"- task2_prompt_injected: {monitor.get('llm', {}).get('evaluation_prompt_has_task2_prompt')}",
            f"- key_prompt_injected: {monitor.get('llm', {}).get('evaluation_prompt_has_key_criteria_prompt')}",
            f"- rag_called: {monitor.get('rag', {}).get('rag_called')}",
            f"- retrieved_docs: {monitor.get('rag', {}).get('retrieved_doc_count')}",
            f"- retrieval_rounds: {monitor.get('rag', {}).get('retrieval_rounds')}",
        ]
    )


def format_human_retrieval_report(retrieval_report: dict[str, Any], monitor: dict[str, Any]) -> str:
    sample = retrieval_report["sample"]
    docs = retrieval_report.get("top_docs", [])
    lines = [
        "IELTS Task 2 RAG Light Retrieval Report",
        f"sample: {sample['sample_no']:02d}",
        f"category: {sample['category']}",
        f"expected_band: {sample['expected_band']}",
        f"success: {retrieval_report.get('success')}",
        f"retrieved_docs: {retrieval_report.get('retrieved_doc_count')}",
        f"unique_sources: {retrieval_report.get('unique_source_count')}",
        f"avg_top_score: {retrieval_report.get('avg_top_score')}",
        f"rounds: {monitor.get('rag', {}).get('retrieval_rounds')}",
        f"gap_fill_rate: {monitor.get('rag', {}).get('gap_fill_rate')}",
        f"stop_reason: {monitor.get('rag', {}).get('stop_reason')}",
        f"llm_network_blocked: {monitor.get('llm', {}).get('llm_network_blocked_count')}",
        "",
        "top_docs:",
    ]
    for doc in docs[:5]:
        meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}
        lines.append(
            f"- #{doc.get('rank')} score={doc.get('score')} pub={meta.get('publication') or ''} "
            f"date={meta.get('issue_date') or ''} source={doc.get('source')}"
        )
        lines.append(f"  {one_line(doc.get('text_preview', ''), 220)}")
    lines.extend(["", "retrieval_trace:"])
    for item in retrieval_report.get("retrieval_trace", []):
        if isinstance(item, dict):
            lines.append(
                f"- round={item.get('round')} mode={item.get('mode')} "
                f"docs={item.get('selected_docs')} fill={item.get('gap_fill_rate')} query={one_line(item.get('query', ''), 180)}"
            )
    return "\n".join(lines) + "\n"


def build_full_summary(records: list[dict[str, Any]], output_dir: Path, mapping_report: dict[str, Any]) -> str:
    total = len(records)
    monitors = [item.get("monitor", {}) for item in records]

    def pct(path: tuple[str, ...]) -> str:
        passed = sum(1 for item in monitors if nested_get(item, path))
        return f"{passed}/{total} ({passed / total:.1%})" if total else "0/0"

    score_errors = [
        float(item["score_error_vs_expected"])
        for item in records
        if isinstance(item.get("score_error_vs_expected"), (int, float))
    ]
    avg_abs_error = sum(abs(value) for value in score_errors) / len(score_errors) if score_errors else 0.0
    avg_latency = (
        sum(int(nested_get(item, ("monitor", "module_chain", "elapsed_ms")) or 0) for item in records) / total
        if total
        else 0
    )
    lines = [
        "RAG Writing Review A/B Summary",
        f"output_dir: {output_dir}",
        f"total_reviews: {total}",
        f"mapping_warning_count: {mapping_report['warning_count']}",
        "",
        "monitor_pass_rates:",
        f"- writing_review_module_called: {pct(('module_chain', 'writing_review_module_called'))}",
        f"- llm_configured: {pct(('llm', 'llm_configured'))}",
        f"- skill_loaded: {pct(('llm', 'skill_loaded'))}",
        f"- task2_prompt_injected: {pct(('llm', 'evaluation_prompt_has_task2_prompt'))}",
        f"- key_prompt_injected: {pct(('llm', 'evaluation_prompt_has_key_criteria_prompt'))}",
        f"- magazine_rag_support_in_prompt: {pct(('llm', 'evaluation_user_prompt_has_magazine_rag_support'))}",
        f"- rag_called: {pct(('rag', 'rag_called'))}",
        "",
        f"avg_abs_score_error_vs_expected: {avg_abs_error:.2f}",
        f"avg_latency_ms: {avg_latency:.0f}",
        "",
        "per_review:",
    ]
    for item in records:
        m = item["monitor"]
        lines.append(
            f"- sample={item['sample_no']:02d} band={int(item['expected_band'])} "
            f"score={item.get('overall_band')} "
            f"err={item.get('score_error_vs_expected')} "
            f"docs={nested_get(m, ('rag', 'retrieved_doc_count'))} "
            f"rounds={nested_get(m, ('rag', 'retrieval_rounds'))} "
            f"case={item['case_dir']}"
        )
    return "\n".join(lines) + "\n"


def build_light_summary(records: list[dict[str, Any]], output_dir: Path, mapping_report: dict[str, Any]) -> str:
    total = len(records)
    monitors = [item.get("monitor", {}) for item in records]

    def pct(path: tuple[str, ...]) -> str:
        passed = sum(1 for item in monitors if nested_get(item, path))
        return f"{passed}/{total} ({passed / total:.1%})" if total else "0/0"

    doc_counts = [int(item.get("retrieved_doc_count") or 0) for item in records]
    source_counts = [int(item.get("unique_source_count") or 0) for item in records]
    avg_scores = [float(item.get("avg_top_score") or 0.0) for item in records]
    avg_latency = (
        sum(int(nested_get(item, ("monitor", "module_chain", "elapsed_ms")) or 0) for item in records) / total
        if total
        else 0
    )
    total_blocked = sum(int(nested_get(item, ("monitor", "llm", "llm_network_blocked_count")) or 0) for item in records)
    lines = [
        "RAG Light Retrieval A/B Summary",
        f"output_dir: {output_dir}",
        f"total_retrieval_cases: {total}",
        f"mapping_warning_count: {mapping_report['warning_count']}",
        "",
        "monitor_pass_rates:",
        f"- retrieval_module_called: {pct(('module_chain', 'retrieval_module_called'))}",
        f"- rag_called: {pct(('rag', 'rag_called'))}",
        f"- llm_network_disabled_by_light_mode: {pct(('llm', 'llm_network_disabled_by_light_mode'))}",
        "",
        f"avg_retrieved_docs: {(sum(doc_counts) / len(doc_counts)) if doc_counts else 0:.2f}",
        f"avg_unique_sources: {(sum(source_counts) / len(source_counts)) if source_counts else 0:.2f}",
        f"avg_top_score: {(sum(avg_scores) / len(avg_scores)) if avg_scores else 0:.4f}",
        f"avg_latency_ms: {avg_latency:.0f}",
        f"llm_network_calls_blocked: {total_blocked}",
        "",
        "per_case:",
    ]
    for item in records:
        m = item["monitor"]
        lines.append(
            f"- sample={item['sample_no']:02d} band={int(item['expected_band'])} "
            f"docs={item.get('retrieved_doc_count')} "
            f"sources={item.get('unique_source_count')} "
            f"avg_top_score={item.get('avg_top_score')} "
            f"rounds={nested_get(m, ('rag', 'retrieval_rounds'))} "
            f"case={item['case_dir']}"
        )
    return "\n".join(lines) + "\n"


def nested_get(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def case_output_dir(output_dir: Path, sample: Task2Sample, band: str) -> Path:
    return output_dir / f"sample{sample.sample_no:02d}_band{band}"


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def one_line(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def print_progress(label: str, current: int, total: int, suffix: str = "", *, newline: bool = False) -> None:
    width = 28
    ratio = current / total if total else 1
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    del newline
    with _PROGRESS_LOCK:
        print(f"{label} [{bar}] {current}/{total} {ratio:.0%} {suffix}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
