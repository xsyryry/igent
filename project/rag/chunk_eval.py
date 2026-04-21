"""Minimal chunking evaluation utilities for the IELTS RAG demo."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any

from project.rag.chunking_agent import ChunkPlan, ChunkRecord, prepare_chunks


DEFAULT_CASES_PATH = Path(__file__).resolve().parents[1] / "data" / "chunk_eval_cases.json"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_BENCHMARK_FILES = (
    "academic-test-sample-questions.html",
    "ielts-academic-format-reading.html",
    "ielts-writing-band-descriptors-and-key-assessment-criteria.html",
    "ielts-writing-band-descriptors.pdf",
    "ielts-speaking-band-descriptors.pdf",
)


@dataclass(slots=True)
class EvalCase:
    """Single evaluation item with expected source and anchor keywords."""

    case_id: str
    question: str
    source_file: str
    keywords: list[str]
    min_keyword_matches: int = 1


@dataclass(slots=True)
class CaseEvalResult:
    """Per-case retrieval outcome."""

    case_id: str
    question: str
    expected_source: str
    matched_at: int | None
    hit_at_1: bool
    hit_at_3: bool
    reciprocal_rank: float
    top_result_source: str
    top_result_title: str
    top_result_score: float


@dataclass(slots=True)
class StrategyEvalResult:
    """Aggregated metrics for one chunking strategy."""

    strategy: str
    requested_strategy: str
    file_count: int
    chunk_count: int
    avg_chunks_per_file: float
    avg_chunk_chars: float
    hit_at_1: float
    hit_at_3: float
    mrr: float
    evaluated_case_count: int
    skipped_files: list[str]
    file_plans: list[dict[str, Any]]
    case_results: list[CaseEvalResult]


def load_eval_cases(cases_path: Path = DEFAULT_CASES_PATH) -> list[EvalCase]:
    """Load evaluation cases from JSON."""

    raw_cases = json.loads(cases_path.read_text(encoding="utf-8"))
    cases: list[EvalCase] = []
    for item in raw_cases:
        cases.append(
            EvalCase(
                case_id=str(item["id"]),
                question=str(item["question"]),
                source_file=str(item["source_file"]),
                keywords=[str(keyword) for keyword in item["keywords"]],
                min_keyword_matches=int(item.get("min_keyword_matches", 1)),
            )
        )
    return cases


def evaluate_strategies(
    *,
    strategies: list[str],
    cases: list[EvalCase],
    data_dir: Path = DEFAULT_DATA_DIR,
    chunk_size: int = 1200,
    overlap: int = 150,
    llm_model: str | None = None,
) -> list[StrategyEvalResult]:
    """Evaluate one or more chunking strategies against the local benchmark."""

    files = _resolve_eval_files(cases, data_dir)
    results: list[StrategyEvalResult] = []
    for strategy in strategies:
        file_plan_records: list[dict[str, Any]] = []
        all_chunks: list[ChunkRecord] = []
        per_file_chunk_counts: list[int] = []
        skipped_files: list[str] = []

        for path in files:
            try:
                plan, chunks = prepare_chunks(
                    path,
                    strategy=strategy,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    llm_model=llm_model,
                )
            except RuntimeError as exc:
                skipped_files.append(path.name)
                file_plan_records.append(
                    {
                        "source_file": path.name,
                        "requested_strategy": strategy,
                        "resolved_strategy": "skipped",
                        "decision_source": "unavailable_dependency",
                        "model_name": llm_model or "",
                        "chunk_size": chunk_size,
                        "overlap": overlap,
                        "reason": str(exc),
                        "chunk_count": 0,
                    }
                )
                continue
            file_plan_records.append(_plan_to_record(path, plan, len(chunks)))
            all_chunks.extend(chunks)
            per_file_chunk_counts.append(len(chunks))

        active_cases = [case for case in cases if case.source_file not in skipped_files]
        case_results = [_evaluate_case(case, all_chunks) for case in active_cases]
        results.append(
            StrategyEvalResult(
                strategy=strategy,
                requested_strategy=strategy,
                file_count=len(files),
                chunk_count=len(all_chunks),
                avg_chunks_per_file=_safe_mean(per_file_chunk_counts),
                avg_chunk_chars=_safe_mean([len(chunk.content) for chunk in all_chunks]),
                hit_at_1=_safe_mean([1.0 if result.hit_at_1 else 0.0 for result in case_results]),
                hit_at_3=_safe_mean([1.0 if result.hit_at_3 else 0.0 for result in case_results]),
                mrr=_safe_mean([result.reciprocal_rank for result in case_results]),
                evaluated_case_count=len(case_results),
                skipped_files=skipped_files,
                file_plans=file_plan_records,
                case_results=case_results,
            )
        )
    return results


def export_eval_report(results: list[StrategyEvalResult], output_path: Path) -> None:
    """Write an evaluation report to JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for result in results:
        serializable.append(
            {
                "strategy": result.strategy,
                "requested_strategy": result.requested_strategy,
                "file_count": result.file_count,
                "chunk_count": result.chunk_count,
                "avg_chunks_per_file": result.avg_chunks_per_file,
                "avg_chunk_chars": result.avg_chunk_chars,
                "hit_at_1": result.hit_at_1,
                "hit_at_3": result.hit_at_3,
                "mrr": result.mrr,
                "evaluated_case_count": result.evaluated_case_count,
                "skipped_files": result.skipped_files,
                "file_plans": result.file_plans,
                "case_results": [asdict(case_result) for case_result in result.case_results],
            }
        )
    output_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_eval_files(cases: list[EvalCase], data_dir: Path) -> list[Path]:
    unique_names = sorted({case.source_file for case in cases} | set(DEFAULT_BENCHMARK_FILES))
    files = [data_dir / name for name in unique_names]
    missing = [path for path in files if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing evaluation source files: {missing_text}")
    return files


def _plan_to_record(path: Path, plan: ChunkPlan, chunk_count: int) -> dict[str, Any]:
    return {
        "source_file": path.name,
        "requested_strategy": plan.requested_strategy,
        "resolved_strategy": plan.strategy,
        "decision_source": plan.decision_source,
        "model_name": plan.model_name,
        "chunk_size": plan.chunk_size,
        "overlap": plan.overlap,
        "reason": plan.reason,
        "chunk_count": chunk_count,
    }


def _evaluate_case(case: EvalCase, chunks: list[ChunkRecord]) -> CaseEvalResult:
    ranked = sorted(
        (
            (
                chunk,
                _score_chunk(case, chunk),
            )
            for chunk in chunks
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    relevant_rank: int | None = None
    for index, (chunk, _) in enumerate(ranked, start=1):
        if _is_relevant(case, chunk):
            relevant_rank = index
            break

    top_chunk, top_score = ranked[0] if ranked else (None, 0.0)
    return CaseEvalResult(
        case_id=case.case_id,
        question=case.question,
        expected_source=case.source_file,
        matched_at=relevant_rank,
        hit_at_1=relevant_rank == 1,
        hit_at_3=relevant_rank is not None and relevant_rank <= 3,
        reciprocal_rank=0.0 if relevant_rank is None else 1.0 / relevant_rank,
        top_result_source="" if top_chunk is None else top_chunk.source_file,
        top_result_title="" if top_chunk is None else top_chunk.title,
        top_result_score=top_score,
    )


def _score_chunk(case: EvalCase, chunk: ChunkRecord) -> float:
    query_text = f"{case.question} {' '.join(case.keywords)}"
    chunk_text = f"{chunk.title}\n{chunk.content}"
    query_counter = Counter(_tokenize(query_text))
    chunk_counter = Counter(_tokenize(chunk_text))
    overlap = sum(min(query_counter[token], chunk_counter[token]) for token in query_counter)

    lowered_chunk = chunk_text.lower()
    keyword_hits = sum(1 for keyword in case.keywords if keyword.lower() in lowered_chunk)
    title_bonus = 0.5 if any(keyword.lower() in chunk.title.lower() for keyword in case.keywords) else 0.0
    length_penalty = min(len(chunk.content) / 2400.0, 0.35)
    return float(overlap) + keyword_hits * 3.0 + title_bonus - length_penalty


def _is_relevant(case: EvalCase, chunk: ChunkRecord) -> bool:
    if chunk.source_file != case.source_file:
        return False
    lowered_chunk = f"{chunk.title}\n{chunk.content}".lower()
    keyword_hits = sum(1 for keyword in case.keywords if keyword.lower() in lowered_chunk)
    return keyword_hits >= case.min_keyword_matches


def _tokenize(text: str) -> list[str]:
    lowered = text.lower()
    latin_tokens = re.findall(r"[a-z0-9]+", lowered)
    cjk_tokens = re.findall(r"[\u4e00-\u9fff]", lowered)
    return latin_tokens + cjk_tokens


def _safe_mean(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    return float(mean(values))
