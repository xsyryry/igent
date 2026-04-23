# -*- coding: utf-8 -*-
"""Run ReAct checkpoint-chain checks for the current runtime.

The test vectors mirror writing_prompt_chain_case1_3_real_runtime.txt and
react_runtime_chain_case4_10_stability.txt. The graph and ReAct LLM calls are
real; selected external tools are mocked so the chain checks stay repeatable.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import os
import sys
import time
from typing import Any
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "project"
CASE1_3_FILE = ROOT / "writing_prompt_chain_case1_3_real_runtime.txt"
CASE4_10_FILE = ROOT / "react_runtime_chain_case4_10_stability.txt"
CASE_FILES = (CASE1_3_FILE, CASE4_10_FILE)
REPORT_FILE = Path(__file__).resolve().parent / "react_runtime_case_report.md"
CHAIN_ONE_TOOL = (
    "metrics_init",
    "router",
    "react_init",
    "reason",
    "action_selector",
    "tool_executor",
    "observation_compressor",
    "react_control",
    "reason",
    "action_selector",
    "react_control",
    "context_builder",
    "generator",
    "metrics_finalize",
    "memory_writer",
)
CHAIN_WRITING_AGENT = (
    "metrics_init",
    "router",
    "react_init",
    "reason",
    "action_selector",
    "writing_agent",
    "data_agent",
    "observation_compressor",
    "react_control",
    "reason",
    "action_selector",
    "react_control",
    "context_builder",
    "generator",
    "metrics_finalize",
    "memory_writer",
)
CHAIN_DATA_AGENT = (
    "metrics_init",
    "router",
    "react_init",
    "reason",
    "action_selector",
    "writing_agent",
    "data_agent",
    "observation_compressor",
    "react_control",
    "reason",
    "action_selector",
    "react_control",
    "context_builder",
    "generator",
    "metrics_finalize",
    "memory_writer",
)
CHAIN_NO_REACT = (
    "metrics_init",
    "router",
    "react_init",
    "context_builder",
    "generator",
    "metrics_finalize",
    "memory_writer",
)
CHAIN_TWO_TOOLS = (
    "metrics_init",
    "router",
    "react_init",
    "reason",
    "action_selector",
    "tool_executor",
    "observation_compressor",
    "react_control",
    "reason",
    "action_selector",
    "tool_executor",
    "observation_compressor",
    "react_control",
    "reason",
    "action_selector",
    "react_control",
    "context_builder",
    "generator",
    "metrics_finalize",
    "memory_writer",
)
CHAIN_FALLBACK = (
    "metrics_init",
    "router",
    "react_init",
    "reason",
    "action_selector",
    "tool_executor",
    "observation_compressor",
    "react_control",
    "reason",
    "action_selector",
    "tool_executor",
    "observation_compressor",
    "react_control",
    "reason",
    "action_selector",
    "react_control",
    "context_builder",
    "generator",
    "metrics_finalize",
    "memory_writer",
)


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    title: str
    source_file: Path
    user_input: str
    expected_intent: str
    expected_checkpoint_chain: tuple[str, ...]
    pre_study_context: dict[str, Any]
    required_tools: tuple[str, ...]
    forbidden_tools: tuple[str, ...]
    mock_results: dict[str, Any]
    expected_react_max_steps: int
    expected_observations: int
    expected_selected_tool_calls: int
    expected_finish_reason: str | None = "enough_information"
    expected_topic_id: str | None = None
    expected_prompt_text: str | None = None
    expected_active_topic_id: str | None = None
    expected_final_contains: tuple[str, ...] = ()
    expected_fallback_success_rate: float = 0.0
    expected_failure_recovery_rate: float = 0.0
    expected_llm_calls_min: int = 1
    allow_retry: bool = False


CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        case_id="writing_prompt_001_real",
        title="基础抽题成功链",
        source_file=CASE1_3_FILE,
        user_input="给我一道雅思大作文题",
        expected_intent="writing_practice",
        expected_checkpoint_chain=CHAIN_WRITING_AGENT,
        pre_study_context={"total_turns": 0},
        required_tools=("writing.get_random_task2_prompt",),
        forbidden_tools=(
            "rag.retrieve_knowledge",
            "web_search.search_web",
            "data.collect_data",
            "db.get_mistake_records",
        ),
        mock_results={
            "writing.get_random_task2_prompt": {
                "success": True,
                "topic": {
                    "id": "task2_001",
                    "essay_type": "观点类",
                    "exam_date": "unknown",
                    "prompt_text": (
                        "Some people think that universities should only offer subjects "
                        "that are useful for employment. To what extent do you agree or disagree?"
                    ),
                },
                "requested_essay_type": None,
                "message": "已为你随机抽取一道大作文题。",
            }
        },
        expected_react_max_steps=3,
        expected_observations=1,
        expected_selected_tool_calls=1,
        expected_topic_id="task2_001",
        expected_prompt_text=(
            "Some people think that universities should only offer subjects "
            "that are useful for employment. To what extent do you agree or disagree?"
        ),
        expected_active_topic_id="task2_001",
    ),
    CaseSpec(
        case_id="writing_prompt_002_real",
        title="IELTS Task 2 题目同义表达",
        source_file=CASE1_3_FILE,
        user_input="帮我随机抽一道 IELTS Task 2 题目",
        expected_intent="writing_practice",
        expected_checkpoint_chain=CHAIN_WRITING_AGENT,
        pre_study_context={"total_turns": 0},
        required_tools=("writing.get_random_task2_prompt",),
        forbidden_tools=(
            "rag.retrieve_knowledge",
            "web_search.search_web",
            "data.collect_data",
        ),
        mock_results={
            "writing.get_random_task2_prompt": {
                "success": True,
                "topic": {
                    "id": "task2_002",
                    "essay_type": "观点类",
                    "exam_date": "unknown",
                    "prompt_text": (
                        "Some people believe that unpaid community service should be a "
                        "compulsory part of high school programmes. To what extent do you agree or disagree?"
                    ),
                },
                "requested_essay_type": None,
                "message": "已为你随机抽取一道大作文题。",
            }
        },
        expected_react_max_steps=3,
        expected_observations=1,
        expected_selected_tool_calls=1,
        expected_topic_id="task2_002",
        expected_prompt_text=(
            "Some people believe that unpaid community service should be a "
            "compulsory part of high school programmes. To what extent do you agree or disagree?"
        ),
        expected_active_topic_id="task2_002",
    ),
    CaseSpec(
        case_id="writing_prompt_003_real",
        title="已有 active topic 时抽新题",
        source_file=CASE1_3_FILE,
        user_input="再给我一道新的雅思大作文题",
        expected_intent="writing_practice",
        expected_checkpoint_chain=CHAIN_WRITING_AGENT,
        pre_study_context={
            "total_turns": 0,
            "active_writing_topic_id": "task2_old_001",
            "active_writing_prompt": "An increasing number of people choose to work from home...",
        },
        required_tools=("writing.get_random_task2_prompt",),
        forbidden_tools=(
            "writing.prepare_task2_review_context",
            "writing.review_task2_submission",
            "rag.retrieve_knowledge",
            "web_search.search_web",
        ),
        mock_results={
            "writing.get_random_task2_prompt": {
                "success": True,
                "topic": {
                    "id": "task2_003",
                    "essay_type": "双边讨论",
                    "exam_date": "unknown",
                    "prompt_text": (
                        "Some people think that children should start school as early as possible, "
                        "while others believe they should start later. Discuss both views and give your own opinion."
                    ),
                },
                "requested_essay_type": None,
                "message": "已为你随机抽取一道大作文题。",
            }
        },
        expected_react_max_steps=3,
        expected_observations=1,
        expected_selected_tool_calls=1,
        expected_topic_id="task2_003",
        expected_prompt_text=(
            "Some people think that children should start school as early as possible, "
            "while others believe they should start later. Discuss both views and give your own opinion."
        ),
        expected_active_topic_id="task2_003",
    ),
    CaseSpec(
        case_id="react_general_004_real",
        title="普通 greeting 不应进入工具链",
        source_file=CASE4_10_FILE,
        user_input="hello",
        expected_intent="general_chat",
        expected_checkpoint_chain=CHAIN_NO_REACT,
        pre_study_context={"total_turns": 0},
        required_tools=(),
        forbidden_tools=(
            "writing.get_random_task2_prompt",
            "rag.retrieve_knowledge",
            "web_search.search_web",
            "db.get_user_profile",
            "data.collect_data",
        ),
        mock_results={},
        expected_react_max_steps=0,
        expected_observations=0,
        expected_selected_tool_calls=0,
        expected_finish_reason=None,
    ),
    CaseSpec(
        case_id="react_knowledge_rag_005_real",
        title="非时效性知识问题走 RAG，不走 web",
        source_file=CASE4_10_FILE,
        user_input="雅思写作 Task 2 主体段怎么展开？",
        expected_intent="knowledge_qa",
        expected_checkpoint_chain=CHAIN_ONE_TOOL,
        pre_study_context={"total_turns": 0},
        required_tools=("rag.retrieve_knowledge",),
        forbidden_tools=("web_search.search_web", "writing.get_random_task2_prompt", "data.collect_data"),
        mock_results={
            "rag.retrieve_knowledge": {
                "answer": "主体段应包含 topic sentence、explanation、example、link back。",
                "documents": [
                    {
                        "source": "writing_guide",
                        "title": "Task 2 body paragraph",
                        "content": "A body paragraph usually develops one controlling idea with explanation and evidence.",
                        "score": 0.92,
                    }
                ],
                "retrieved_docs": [
                    {
                        "id": "writing_guide_001",
                        "source": "writing_guide",
                        "chunks": [
                            "A body paragraph usually develops one controlling idea with explanation and evidence."
                        ],
                    }
                ],
                "query_mode": "mix",
            }
        },
        expected_react_max_steps=4,
        expected_observations=1,
        expected_selected_tool_calls=1,
        expected_finish_reason="*",
        expected_final_contains=("主体段",),
    ),
    CaseSpec(
        case_id="react_knowledge_fallback_006_real",
        title="最新政策类问题先 web，web 失败后 RAG fallback",
        source_file=CASE4_10_FILE,
        user_input="最新 IELTS Writing Task 2 评分政策是什么？",
        expected_intent="knowledge_qa",
        expected_checkpoint_chain=CHAIN_FALLBACK,
        pre_study_context={"total_turns": 0},
        required_tools=("web_search.search_web", "rag.retrieve_knowledge"),
        forbidden_tools=("writing.get_random_task2_prompt", "data.collect_data"),
        mock_results={
            "web_search.search_web": {
                "success": False,
                "error": "service unavailable",
            },
            "rag.retrieve_knowledge": {
                "answer": "本地资料显示 IELTS Writing Task 2 仍按 TR/CC/LR/GRA 四项评分。",
                "documents": [
                    {
                        "source": "local_band_descriptor",
                        "title": "IELTS Writing Band Descriptors",
                        "content": (
                            "Task 2 is assessed by Task Response, Coherence and Cohesion, "
                            "Lexical Resource, Grammatical Range and Accuracy."
                        ),
                        "score": 0.88,
                    }
                ],
                "retrieved_docs": [
                    {
                        "id": "band_descriptor_001",
                        "source": "local_band_descriptor",
                        "chunks": [
                            "Task 2 is assessed by Task Response, Coherence and Cohesion, Lexical Resource, Grammatical Range and Accuracy."
                        ],
                    }
                ],
                "query_mode": "mix",
            },
        },
        expected_react_max_steps=4,
        expected_observations=2,
        expected_selected_tool_calls=2,
        expected_fallback_success_rate=1.0,
        expected_failure_recovery_rate=1.0,
        allow_retry=True,
    ),
    CaseSpec(
        case_id="react_study_plan_007_real",
        title="学习计划先取画像，再取计划",
        source_file=CASE4_10_FILE,
        user_input="帮我制定一个雅思学习计划",
        expected_intent="study_plan",
        expected_checkpoint_chain=CHAIN_TWO_TOOLS,
        pre_study_context={"total_turns": 0},
        required_tools=("db.get_user_profile", "db.get_study_plan"),
        forbidden_tools=("rag.retrieve_knowledge", "web_search.search_web", "writing.get_random_task2_prompt"),
        mock_results={
            "db.get_user_profile": {
                "user_id": "demo_user",
                "current_level": "6.0",
                "target_score": "7.0",
                "preferred_focus": "writing",
                "available_hours_per_week": 8,
            },
            "db.get_study_plan": {
                "weekly_goals": ["完成 2 次 Task 2 结构训练", "完成 2 次听力精听"],
                "daily_tasks": ["背 5 个写作观点句", "做 1 组听力 Section 2"],
            },
        },
        expected_react_max_steps=4,
        expected_observations=2,
        expected_selected_tool_calls=2,
        expected_final_contains=("计划",),
    ),
    CaseSpec(
        case_id="react_mistake_review_008_real",
        title="非提交类错题复盘读取 profile 和 mistake records",
        source_file=CASE4_10_FILE,
        user_input="帮我复盘一下最近的错题",
        expected_intent="mistake_review",
        expected_checkpoint_chain=CHAIN_TWO_TOOLS,
        pre_study_context={"total_turns": 0},
        required_tools=("db.get_user_profile", "db.get_mistake_records"),
        forbidden_tools=("mistake.grade_submission", "writing.review_task2_submission", "rag.retrieve_knowledge"),
        mock_results={
            "db.get_user_profile": {
                "user_id": "demo_user",
                "current_level": "6.0",
                "weak_skills": ["writing", "listening"],
            },
            "db.get_mistake_records": [
                {
                    "subject": "writing",
                    "error_type": "观点展开不充分",
                    "correction_note": "主体段补 explanation 和 example",
                },
                {
                    "subject": "listening",
                    "error_type": "单复数漏听",
                    "correction_note": "注意题目前后名词形式",
                },
            ],
        },
        expected_react_max_steps=4,
        expected_observations=2,
        expected_selected_tool_calls=2,
        expected_final_contains=("错题",),
    ),
    CaseSpec(
        case_id="react_writing_review_009_real",
        title="已有 active topic 时，长作文输入进入批改工具",
        source_file=CASE4_10_FILE,
        user_input=(
            "In recent years, many people have argued that children should start school as early as possible. "
            "I partly disagree with this view because young children need time to develop emotionally before formal education. "
            "For example, if a four-year-old is forced to sit in a classroom for many hours, he may lose curiosity and confidence. "
            "However, early education can also help children build basic social skills when it is playful and age-appropriate. "
            "Therefore, I believe school should not start too early, but families can provide informal learning at home."
        ),
        expected_intent="writing_practice",
        expected_checkpoint_chain=CHAIN_WRITING_AGENT,
        pre_study_context={
            "total_turns": 0,
            "active_writing_topic_id": "task2_active_001",
            "active_writing_prompt": "Some people think that children should start school as early as possible...",
        },
        required_tools=("writing.review_task2_submission",),
        forbidden_tools=(
            "writing.get_random_task2_prompt",
            "rag.retrieve_knowledge",
            "web_search.search_web",
            "db.get_mistake_records",
        ),
        mock_results={
            "writing.review_task2_submission": {
                "success": True,
                "topic": {
                    "id": "task2_active_001",
                    "prompt_text": "Some people think that children should start school as early as possible...",
                },
                "evaluation": {
                    "overall_band": 6.0,
                    "priority_issue": "task_response_gap",
                    "overall_comment": "Position is clear but examples need more detail.",
                    "band_breakdown": {
                        "task_response": 6.0,
                        "coherence_cohesion": 6.0,
                        "lexical_resource": 6.0,
                        "grammar_accuracy": 6.0,
                    },
                    "strengths": ["position is clear"],
                    "issues": ["examples need more detail"],
                    "revision_plan": ["expand each body paragraph with a concrete example"],
                },
            }
        },
        expected_react_max_steps=3,
        expected_observations=1,
        expected_selected_tool_calls=1,
        expected_active_topic_id="task2_active_001",
        expected_final_contains=("6.0",),
    ),
    CaseSpec(
        case_id="react_data_pdf_010_real",
        title="Cambridge Writing PDF 请求走 question_pdf.export_question_pdf",
        source_file=CASE4_10_FILE,
        user_input="帮我导出 5 道剑桥雅思 Writing Task 2 真题 PDF",
        expected_intent="data_collection",
        expected_checkpoint_chain=CHAIN_DATA_AGENT,
        pre_study_context={"total_turns": 0},
        required_tools=("question_pdf.export_question_pdf",),
        forbidden_tools=(
            "data.collect_data",
            "cambridge_crawler.crawl_writing_questions",
            "rag.retrieve_knowledge",
            "web_search.search_web",
            "writing.get_random_task2_prompt",
        ),
        mock_results={
            "question_pdf.export_question_pdf": {
                "success": True,
                "export_path": r"D:\Afile\igent\project\data\exports\cambridge_writing_task2.pdf",
                "requested_count": 5,
                "exported_count": 5,
                "completion_status": "complete",
            }
        },
        expected_react_max_steps=3,
        expected_observations=1,
        expected_selected_tool_calls=1,
        expected_final_contains=("cambridge_writing_task2.pdf",),
        expected_llm_calls_min=0,
    ),
)
CASE_BY_ID = {case.case_id: case for case in CASES}


def render_progress(index: int, total: int, case_id: str, status: str, *, started_at: float) -> None:
    width = 28
    completed = max(0, min(index, total))
    filled = int(width * completed / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.monotonic() - started_at
    line = f"\r[{bar}] {completed:>2}/{total:<2} {status:<8} {case_id:<36} elapsed={elapsed:6.1f}s"
    print(line[:140].ljust(140), end="", flush=True)


def run_cases_with_progress(cases: tuple[CaseSpec, ...]) -> list[dict[str, Any]]:
    total = len(cases)
    started_at = time.monotonic()
    results: list[dict[str, Any]] = []
    print(f"Running {total} ReAct runtime cases with checkpoint monitoring...")
    for index, case in enumerate(cases, start=1):
        render_progress(index - 1, total, case.case_id, "RUNNING", started_at=started_at)
        try:
            result = run_case(case)
        except Exception as exc:
            result = {
                "case_id": case.case_id,
                "title": case.title,
                "source_file": str(case.source_file),
                "passed": False,
                "deviations": [f"case raised {type(exc).__name__}: {exc}"],
                "actual": {
                    "intent": "n/a",
                    "react_step": "n/a",
                    "react_max_steps": "n/a",
                    "react_finish_reason": "n/a",
                    "tools": [],
                    "observations_length": 0,
                    "last_observation": {},
                    "active_writing_topic_id": None,
                    "active_writing_prompt": None,
                    "final_answer_preview": "",
                    "metrics": {},
                    "checkpoint_trace": {"available": False, "reason": str(exc), "full_chain": [], "snapshots": []},
                },
            }
        results.append(result)
        render_progress(index, total, case.case_id, "PASS" if result["passed"] else "FAIL", started_at=started_at)
        if not result["passed"]:
            print(f"\n  deviation: {result['deviations'][0] if result['deviations'] else 'unknown failure'}")
    print()
    return results


def configure_runtime() -> None:
    """Load real LLM config while keeping checkpoints deterministic for tests."""

    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except Exception:
        pass

    for key in (
        "LANGGRAPH_CHECKPOINT_BACKEND",
    ):
        os.environ[key] = ""
    os.environ["LANGGRAPH_CHECKPOINT_BACKEND"] = "memory"
    os.environ["MEMORY_EXTRACTION_SYNC"] = "0"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def require_real_llm() -> str:
    """Fail fast when tests would silently fall back to rule-based reasoning."""

    from project.llm.client import LLMClient

    client = LLMClient.from_config()
    if not client.is_configured:
        raise RuntimeError(
            "Real LLM is not configured. Set LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL "
            "in project/.env or the process environment before running this test."
        )
    return client.model


def tool_label(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    return f"{action.get('tool_name')}.{action.get('action')}"


def collect_tool_ids(state: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for item in state.get("observations", []):
        if isinstance(item, dict):
            label = tool_label(item.get("action"))
            if label:
                ids.append(label)
    for item in state.get("tool_results", {}).get("_tool_policy_trace", []):
        if isinstance(item, dict) and item.get("tool"):
            ids.append(str(item["tool"]))
    return list(dict.fromkeys(ids))


def add_check(deviations: list[str], ok: bool, label: str, actual: Any, expected: Any) -> None:
    if not ok:
        deviations.append(f"{label}: actual={actual!r}, expected={expected!r}")


def _snapshot_attr(snapshot: Any, key: str, default: Any = None) -> Any:
    if isinstance(snapshot, dict):
        return snapshot.get(key, default)
    return getattr(snapshot, key, default)


def _snapshot_step(snapshot: Any, fallback: int) -> int:
    metadata = _snapshot_attr(snapshot, "metadata", {}) or {}
    if isinstance(metadata, dict):
        try:
            return int(metadata.get("step", fallback))
        except (TypeError, ValueError):
            return fallback
    return fallback


def _snapshot_nodes(snapshot: Any) -> list[str]:
    tasks = _snapshot_attr(snapshot, "tasks", ()) or ()
    if isinstance(tasks, (list, tuple)):
        task_names = [
            str(getattr(task, "name", "") or task.get("name", "") if isinstance(task, dict) else getattr(task, "name", ""))
            for task in tasks
        ]
        task_names = [name for name in task_names if name and name not in {"__start__", "input"}]
        if task_names:
            return task_names

    metadata = _snapshot_attr(snapshot, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        return []
    writes = metadata.get("writes")
    if isinstance(writes, dict):
        return [str(key) for key in writes.keys() if str(key) not in {"__start__", "input"}]
    if isinstance(writes, list):
        return [str(item) for item in writes if str(item) not in {"__start__", "input"}]
    source = metadata.get("source")
    if source in {"input", "loop"}:
        return []
    return [str(source)] if source else []


def _checkpoint_snapshot_summary(snapshot: Any, fallback_index: int) -> dict[str, Any]:
    metadata = _snapshot_attr(snapshot, "metadata", {}) or {}
    values = _snapshot_attr(snapshot, "values", {}) or {}
    next_nodes = _snapshot_attr(snapshot, "next", ()) or ()
    tasks = _snapshot_attr(snapshot, "tasks", ()) or ()
    if not isinstance(values, dict):
        values = {}
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "step": _snapshot_step(snapshot, fallback_index),
        "nodes": _snapshot_nodes(snapshot),
        "source": metadata.get("source"),
        "next": list(next_nodes) if isinstance(next_nodes, (list, tuple)) else [str(next_nodes)],
        "intent": values.get("intent"),
        "react_step": values.get("react_step"),
        "react_finish_reason": values.get("react_finish_reason"),
        "selected_tool": tool_label(values.get("selected_tool_call")),
        "tool_result_keys": sorted(list(values.get("tool_results", {}).keys()))
        if isinstance(values.get("tool_results"), dict)
        else [],
        "observation_count": len(values.get("observations", [])) if isinstance(values.get("observations"), list) else 0,
        "task_count": len(tasks) if isinstance(tasks, (list, tuple)) else 0,
    }


def collect_checkpoint_trace(graph: Any, config: dict[str, Any]) -> dict[str, Any]:
    if not hasattr(graph, "get_state_history"):
        return {
            "available": False,
            "reason": "graph does not expose get_state_history; LangGraph checkpoint is unavailable",
            "full_chain": [],
            "snapshots": [],
        }
    try:
        raw_snapshots = list(graph.get_state_history(config))
    except Exception as exc:
        return {
            "available": False,
            "reason": f"get_state_history failed: {exc}",
            "full_chain": [],
            "snapshots": [],
        }

    snapshots = [
        _checkpoint_snapshot_summary(snapshot, index)
        for index, snapshot in enumerate(raw_snapshots)
    ]
    snapshots.sort(key=lambda item: item["step"])
    full_chain: list[str] = []
    for snapshot in snapshots:
        full_chain.extend(snapshot["nodes"])
    return {
        "available": bool(raw_snapshots),
        "reason": "" if raw_snapshots else "checkpoint history is empty",
        "full_chain": full_chain,
        "snapshots": snapshots,
    }


def evaluate_case(case: CaseSpec, state: dict[str, Any], checkpoint_trace: dict[str, Any]) -> dict[str, Any]:
    deviations: list[str] = []
    tool_results = state.get("tool_results", {})
    observations = state.get("observations", [])
    last_observation = state.get("last_observation", {})
    metrics = state.get("runtime_metrics", {}).get("current", {})
    counts = metrics.get("counts", {}) if isinstance(metrics, dict) else {}
    llm_usage = metrics.get("llm_usage", {}) if isinstance(metrics, dict) else {}
    study_context = state.get("study_context", {})
    tool_ids = collect_tool_ids(state)
    trace = tool_results.get("_tool_policy_trace", [])
    trace = trace if isinstance(trace, list) else []
    checkpoint_chain = checkpoint_trace.get("full_chain", [])

    add_check(deviations, checkpoint_trace.get("available") is True, "checkpoint_history.available", checkpoint_trace.get("reason"), True)
    add_check(
        deviations,
        tuple(checkpoint_chain) == case.expected_checkpoint_chain,
        "checkpoint_full_chain",
        checkpoint_chain,
        list(case.expected_checkpoint_chain),
    )
    add_check(deviations, state.get("intent") == case.expected_intent, "intent", state.get("intent"), case.expected_intent)
    add_check(
        deviations,
        state.get("react_max_steps") == case.expected_react_max_steps,
        "react_max_steps",
        state.get("react_max_steps"),
        case.expected_react_max_steps,
    )
    if case.expected_finish_reason is not None:
        if case.expected_finish_reason == "*":
            add_check(
                deviations,
                bool(state.get("react_finish_reason")),
                "react_finish_reason",
                state.get("react_finish_reason"),
                "any non-empty success reason",
            )
        else:
            add_check(
                deviations,
                state.get("react_finish_reason") == case.expected_finish_reason,
                "react_finish_reason",
                state.get("react_finish_reason"),
                case.expected_finish_reason,
            )
    else:
        add_check(deviations, state.get("react_finish_reason") is None, "react_finish_reason", state.get("react_finish_reason"), None)
    add_check(
        deviations,
        int(state.get("react_step", 0) or 0) <= max(case.expected_react_max_steps, 0),
        "react_step<=max",
        state.get("react_step"),
        f"<={case.expected_react_max_steps}",
    )
    add_check(deviations, len(observations) == case.expected_observations, "observations.length", len(observations), case.expected_observations)
    if case.expected_observations:
        add_check(deviations, bool(last_observation.get("success")) is True, "last_observation.success", last_observation.get("success"), True)
    if case.expected_topic_id:
        prompt_result = tool_results.get("get_random_task2_prompt", {})
        add_check(
            deviations,
            isinstance(prompt_result, dict) and prompt_result.get("success") is True,
            "tool_results.get_random_task2_prompt.success",
            prompt_result.get("success") if isinstance(prompt_result, dict) else None,
            True,
        )
        add_check(
            deviations,
            isinstance(prompt_result, dict) and prompt_result.get("topic", {}).get("id") == case.expected_topic_id,
            "tool_results.get_random_task2_prompt.topic.id",
            prompt_result.get("topic", {}).get("id") if isinstance(prompt_result, dict) else None,
            case.expected_topic_id,
        )
    add_check(deviations, bool(state.get("answer_context")), "answer_context exists", bool(state.get("answer_context")), True)
    add_check(deviations, "context_summary" not in state, "context_summary absent", "context_summary" in state, False)
    final_answer = str(state.get("final_answer", ""))
    if case.expected_prompt_text:
        add_check(
            deviations,
            case.expected_prompt_text in final_answer,
            "final_answer contains prompt_text",
            final_answer[:120],
            case.expected_prompt_text,
        )
        add_check(
            deviations,
            study_context.get("active_writing_prompt") == case.expected_prompt_text,
            "study_context.active_writing_prompt",
            study_context.get("active_writing_prompt"),
            case.expected_prompt_text,
        )
    for expected_text in case.expected_final_contains:
        add_check(
            deviations,
            expected_text in final_answer,
            f"final_answer contains {expected_text}",
            final_answer[:180],
            expected_text,
        )
    if case.expected_active_topic_id:
        add_check(
            deviations,
            study_context.get("active_writing_topic_id") == case.expected_active_topic_id,
            "study_context.active_writing_topic_id",
            study_context.get("active_writing_topic_id"),
            case.expected_active_topic_id,
        )
    for required_tool in case.required_tools:
        add_check(deviations, required_tool in tool_ids, f"required tool {required_tool}", tool_ids, f"contains {required_tool}")
    for forbidden_tool in case.forbidden_tools:
        add_check(deviations, forbidden_tool not in tool_ids, f"forbidden tool {forbidden_tool}", tool_ids, f"not contains {forbidden_tool}")
    add_check(deviations, bool(metrics.get("task_success")) is True, "metrics.task_success", metrics.get("task_success"), True)
    add_check(
        deviations,
        int(llm_usage.get("call_count", 0) or 0) >= case.expected_llm_calls_min,
        f"metrics.llm_usage.call_count>={case.expected_llm_calls_min}",
        llm_usage.get("call_count"),
        f">={case.expected_llm_calls_min}",
    )
    add_check(deviations, counts.get("steps") == case.expected_observations, "metrics.counts.steps", counts.get("steps"), case.expected_observations)
    add_check(
        deviations,
        counts.get("selected_tool_calls") == case.expected_selected_tool_calls,
        "metrics.counts.selected_tool_calls",
        counts.get("selected_tool_calls"),
        case.expected_selected_tool_calls,
    )
    add_check(deviations, metrics.get("duplicate_tool_call_rate") == 0.0, "metrics.duplicate_tool_call_rate", metrics.get("duplicate_tool_call_rate"), 0.0)
    add_check(
        deviations,
        metrics.get("fallback_success_rate") == case.expected_fallback_success_rate,
        "metrics.fallback_success_rate",
        metrics.get("fallback_success_rate"),
        case.expected_fallback_success_rate,
    )
    add_check(
        deviations,
        metrics.get("tool_failure_recovery_rate") == case.expected_failure_recovery_rate,
        "metrics.tool_failure_recovery_rate",
        metrics.get("tool_failure_recovery_rate"),
        case.expected_failure_recovery_rate,
    )
    if case.expected_fallback_success_rate == 0.0:
        add_check(deviations, not tool_results.get("_tool_fallbacks"), "fallback_used", tool_results.get("_tool_fallbacks"), False)
    if not case.allow_retry:
        add_check(
            deviations,
            all(int(item.get("attempt", 1) or 1) == 1 for item in trace if isinstance(item, dict)),
            "retry_count",
            trace,
            "no attempt > 1",
        )

    return {
        "case_id": case.case_id,
        "title": case.title,
        "source_file": str(case.source_file),
        "passed": not deviations,
        "deviations": deviations,
        "actual": {
            "intent": state.get("intent"),
            "react_step": state.get("react_step"),
            "react_max_steps": state.get("react_max_steps"),
            "react_finish_reason": state.get("react_finish_reason"),
            "tools": tool_ids,
            "observations_length": len(observations),
            "last_observation": last_observation,
            "active_writing_topic_id": study_context.get("active_writing_topic_id"),
            "active_writing_prompt": study_context.get("active_writing_prompt"),
            "final_answer_preview": str(state.get("final_answer", ""))[:260],
            "metrics": metrics,
            "checkpoint_trace": checkpoint_trace,
        },
    }


def run_case(case: CaseSpec) -> dict[str, Any]:
    from project.agent.graph import build_graph
    from project.agent.state import build_initial_state
    from project.agent.checkpointing import checkpoint_config

    initial_state = build_initial_state(
        user_input=case.user_input,
        study_context=dict(case.pre_study_context),
    )

    def cloned_result(tool_id: str, default: Any) -> Any:
        return json.loads(json.dumps(case.mock_results.get(tool_id, default), ensure_ascii=False))

    def fake_random_prompt(essay_type: str | None = None) -> dict[str, Any]:
        result = cloned_result(
            "writing.get_random_task2_prompt",
            {"success": False, "message": "unexpected get_random_task2_prompt call"},
        )
        result["requested_essay_type"] = essay_type
        return result

    def fake_review_task2_submission(*, user_input: str, topic_id: str, user_id: str = "demo_user") -> dict[str, Any]:
        del user_input, topic_id, user_id
        return cloned_result("writing.review_task2_submission", {"success": False, "message": "unexpected review call"})

    def fake_retrieve_knowledge(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return cloned_result(
            "rag.retrieve_knowledge",
            {"answer": "", "documents": [], "retrieved_docs": [], "query_mode": "mix"},
        )

    def fake_search(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return cloned_result("web_search.search_web", {"success": True, "results": []})

    def fake_get_user_profile() -> dict[str, Any]:
        return cloned_result("db.get_user_profile", {})

    def fake_get_study_plan() -> dict[str, Any]:
        return cloned_result("db.get_study_plan", {})

    def fake_get_mistake_records() -> list[dict[str, Any]]:
        return cloned_result("db.get_mistake_records", [])

    def fake_grade_submission(user_input: str) -> dict[str, Any]:
        del user_input
        return cloned_result("mistake.grade_submission", {"success": False, "message": "unexpected grade call"})

    def fake_export_question_pdf(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return cloned_result("question_pdf.export_question_pdf", {"success": False, "message": "unexpected PDF export"})

    def fake_collect_data(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return cloned_result("data.collect_data", {"success": False, "message": "unexpected collect_data call"})

    def fake_collect_cambridge_writing_questions(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return cloned_result(
            "data.collect_cambridge_writing_questions",
            {"success": False, "message": "unexpected collect_cambridge_writing_questions call"},
        )

    def fake_crawl_writing_questions(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return cloned_result("cambridge_crawler.crawl_writing_questions", {"success": False, "message": "unexpected crawler call"})

    with ExitStack() as stack:
        stack.enter_context(patch("project.agent.nodes.tool_executor.get_random_task2_prompt", fake_random_prompt))
        stack.enter_context(patch("project.agent.nodes.tool_executor.review_task2_submission", fake_review_task2_submission))
        stack.enter_context(patch("project.agent.nodes.tool_executor.retrieve_knowledge", fake_retrieve_knowledge))
        stack.enter_context(patch("project.agent.nodes.tool_executor.search", fake_search))
        stack.enter_context(patch("project.agent.nodes.tool_executor.get_user_profile", fake_get_user_profile))
        stack.enter_context(patch("project.agent.nodes.tool_executor.get_study_plan", fake_get_study_plan))
        stack.enter_context(patch("project.agent.nodes.tool_executor.get_mistake_records", fake_get_mistake_records))
        stack.enter_context(patch("project.agent.nodes.tool_executor.grade_submission", fake_grade_submission))
        stack.enter_context(patch("project.agent.nodes.tool_executor.export_question_pdf", fake_export_question_pdf))
        stack.enter_context(patch("project.agent.nodes.tool_executor.collect_data", fake_collect_data))
        stack.enter_context(
            patch("project.agent.nodes.tool_executor.collect_cambridge_writing_questions", fake_collect_cambridge_writing_questions)
        )
        stack.enter_context(patch("project.agent.nodes.tool_executor.crawl_writing_questions", fake_crawl_writing_questions))
        stack.enter_context(patch("project.agent.nodes.context_builder.retrieve_relevant_memories", return_value={"available": True, "items": []}))
        stack.enter_context(patch("project.agent.nodes.context_builder.build_memory_snapshot", return_value={"items": []}))
        stack.enter_context(patch("project.agent.nodes.memory_writer.request_memory_extraction_async", return_value=None))
        stack.enter_context(patch("project.agent.nodes.memory_writer.request_memory_extraction", return_value={}))
        graph = build_graph()
        config = checkpoint_config(case.case_id)
        final_state = graph.invoke(initial_state, config=config)
        checkpoint_trace = collect_checkpoint_trace(graph, config)

    return evaluate_case(case, dict(final_state), checkpoint_trace)


def write_report(results: list[dict[str, Any]], *, llm_model: str) -> None:
    passed = sum(1 for item in results if item["passed"])
    lines: list[str] = [
        "# ReAct runtime chain test report",
        "",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- source_case_files: `{'; '.join(str(path) for path in CASE_FILES)}`",
        f"- source_case_files_exist: `{all(path.exists() for path in CASE_FILES)}`",
        f"- tested_graph_root: `{PROJECT_ROOT}`",
        f"- total: {len(results)}",
        f"- passed: {passed}",
        f"- failed: {len(results) - passed}",
        f"- llm_mode: `real`",
        f"- llm_model: `{llm_model}`",
        "",
        "## Results",
        "",
    ]

    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        actual = item["actual"]
        lines.extend(
            [
                f"### {item['case_id']} - {status}",
                "",
                f"- title: {item['title']}",
                f"- source_file: `{item['source_file']}`",
                f"- intent: `{actual['intent']}`",
                f"- expected_chain: `{' -> '.join(CASE_BY_ID[item['case_id']].expected_checkpoint_chain)}`",
                f"- react: step=`{actual['react_step']}`, max=`{actual['react_max_steps']}`, finish=`{actual['react_finish_reason']}`",
                f"- tools: `{', '.join(actual['tools']) or 'none'}`",
                f"- checkpoint.available: `{actual['checkpoint_trace'].get('available')}`",
                f"- checkpoint.full_chain: `{' -> '.join(actual['checkpoint_trace'].get('full_chain', [])) or 'none'}`",
                f"- observations.length: `{actual['observations_length']}`",
                f"- llm.call_count: `{actual['metrics'].get('llm_usage', {}).get('call_count') if isinstance(actual['metrics'], dict) else 'n/a'}`",
                f"- active_writing_topic_id: `{actual['active_writing_topic_id']}`",
                f"- final_answer_preview: {actual['final_answer_preview']}",
                "",
            ]
        )
        if item["deviations"]:
            lines.append("Deviations:")
            for deviation in item["deviations"]:
                lines.append(f"- {deviation}")
        else:
            lines.append("Deviations: none")
        lines.extend(
            [
                "",
                "Checkpoint snapshots:",
                "",
                "```json",
                json.dumps(actual["checkpoint_trace"].get("snapshots", []), ensure_ascii=False, indent=2, default=str),
                "```",
                "",
                "Metrics:",
                "",
                "```json",
                json.dumps(actual["metrics"], ensure_ascii=False, indent=2, default=str),
                "```",
                "",
            ]
        )

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> int:
    configure_runtime()
    from project.config import get_config

    get_config.cache_clear()
    llm_model = require_real_llm()
    results = run_cases_with_progress(CASES)
    write_report(results, llm_model=llm_model)
    failed = [item["case_id"] for item in results if not item["passed"]]
    print(f"Report written: {REPORT_FILE}")
    if failed:
        print(f"Failed cases: {', '.join(failed)}")
        return 1
    print(f"All cases passed with real LLM model: {llm_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
