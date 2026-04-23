"""Evaluate the project memory module with scripted conversations.

Default mode exercises the memory writer node directly. This keeps the test
focused on memory extraction/retrieval and avoids mixing generator quality into
the score. Use ``--mode agent`` for a full graph run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COVERAGE_FILE = Path(
    ROOT / "memory_tests" / "data" / "memory_coverage_testset_100_facts_cuhk_cs_first_person.txt"
)
DEFAULT_RETRIEVAL_FILE = Path(
    ROOT / "memory_tests" / "data" / "memory_retrieval_queries_50_with_answers.txt"
)
DEFAULT_DB_PATH = ROOT / "memory_tests" / "memory_test.db"
DEFAULT_REPORT_PATH = ROOT / "memory_tests" / "memory_test_report.txt"
USER_ID = "memory_eval_user"


@dataclass(frozen=True)
class CoverageCase:
    id: int
    category: str
    priority: str
    should_remember: bool
    text: str


@dataclass(frozen=True)
class RetrievalCase:
    id: int
    query: str
    answer: str


@dataclass(frozen=True)
class MemoryDoc:
    id: str
    source: str
    pool: str
    text: str
    is_long_term: bool
    field: str = ""
    value: str = ""
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class QueryRoute:
    should_search: bool
    query_type: str
    reason: str = ""


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return raw.decode(encoding)
        except UnicodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_coverage_cases(path: Path) -> list[CoverageCase]:
    cases: list[CoverageCase] = []
    for line in read_text(path).splitlines():
        match = re.match(
            r"^\s*(\d{3})\s*\|\s*类别:(.*?)\s*\|\s*优先级:(.*?)\s*\|\s*应记:(.*?)\s*\|\s*(.+?)\s*$",
            line,
        )
        if not match:
            continue
        case_id, category, priority, should_text, text = match.groups()
        case_no = int(case_id)
        cases.append(
            CoverageCase(
                id=case_no,
                category=category.strip(),
                priority=priority.strip(),
                should_remember=case_no <= 90 and "否" not in should_text,
                text=text.strip(),
            )
        )
    return cases


def parse_retrieval_cases(path: Path) -> list[RetrievalCase]:
    text = read_text(path)
    pattern = re.compile(
        r"(?s)(\d{3})\.\s*Query[：:]\s*(.*?)\s*标准答案[：:]\s*(.*?)(?=\n\s*\d{3}\.\s*Query[：:]|\Z)"
    )
    cases = []
    for match in pattern.finditer(text):
        case_id, query, answer = match.groups()
        clean_answer = re.split(r"\n=+\n", answer.strip(), maxsplit=1)[0].strip()
        cases.append(RetrievalCase(id=int(case_id), query=query.strip(), answer=clean_answer))
    return cases


def normalize_text(text: Any) -> str:
    raw = json.dumps(text, ensure_ascii=False) if not isinstance(text, str) else text
    return re.sub(r"[\s\W_]+", "", raw.lower(), flags=re.UNICODE)


def char_tokens(text: str) -> list[str]:
    compact = normalize_text(text)
    return [ch for ch in compact if ch]


def coverage_score(needle: str, haystack: str) -> float:
    needle_norm = normalize_text(needle)
    haystack_norm = normalize_text(haystack)
    if not needle_norm or not haystack_norm:
        return 0.0
    if needle_norm in haystack_norm:
        return 1.0
    needle_chars = set(needle_norm)
    haystack_chars = set(haystack_norm)
    return len(needle_chars & haystack_chars) / max(len(needle_chars), 1)


def text_matches(needle: str, haystack: str, threshold: float = 0.58) -> bool:
    if not needle.strip() or not haystack.strip():
        return False
    if normalize_text(needle) in normalize_text(haystack):
        return True
    return coverage_score(needle, haystack) >= threshold


def answer_in_doc(answer: str, doc_text: str) -> bool:
    answer_norm = normalize_text(answer)
    doc_norm = normalize_text(doc_text)
    if not answer_norm:
        return False
    if answer_norm in doc_norm:
        return True
    return coverage_score(answer, doc_text) >= 0.72


def setup_environment(db_path: Path, keep_db: bool, disable_llm: bool, llm_timeout: int | None) -> None:
    for env_file in (ROOT / ".env", ROOT / "project" / ".env"):
        if env_file.exists():
            try:
                from dotenv import load_dotenv

                load_dotenv(env_file, override=False)
            except Exception:
                pass
    if not keep_db and db_path.exists():
        try:
            db_path.unlink()
        except PermissionError as exc:
            raise SystemExit(
                f"Cannot reset test DB because it is in use: {db_path}\n"
                "Close the previous test process or rerun after a few seconds. "
                "Use --keep-db only if you intentionally want to reuse it."
            ) from exc
    db_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["IELTS_DB_PATH"] = str(db_path)
    os.environ.setdefault("IELTS_ENABLE_DB_MIGRATIONS", "1")
    os.environ.setdefault("MEMORY_EXTRACTION_SYNC", "1")
    if disable_llm:
        os.environ["LLM_API_KEY"] = ""
        os.environ["LLM_BASE_URL"] = ""
        os.environ["LLM_MODEL"] = ""
        os.environ["MEMORY_LLM_API_KEY"] = ""
        os.environ["MEMORY_LLM_BASE_URL"] = ""
        os.environ["MEMORY_LLM_MODEL"] = ""
    if llm_timeout is not None:
        os.environ["LLM_TIMEOUT"] = str(llm_timeout)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def reset_extractor_watermark() -> None:
    try:
        import project.memory.extractor as extractor

        with extractor._lock:  # type: ignore[attr-defined]
            extractor._running = False  # type: ignore[attr-defined]
            extractor._dirty = False  # type: ignore[attr-defined]
            extractor._watermark = 0  # type: ignore[attr-defined]
            extractor._pending = None  # type: ignore[attr-defined]
            extractor._async_results = {}  # type: ignore[attr-defined]
    except Exception:
        return


def render_progress(label: str, index: int, total: int, sample_id: int | str, status: str) -> None:
    width = 28
    ratio = index / max(total, 1)
    filled = min(width, int(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    sample = f"{sample_id:03d}" if isinstance(sample_id, int) else str(sample_id)
    print(
        f"\r{label} [{bar}] {index}/{total} {ratio:6.2%} current={sample} {status:<7}",
        end="",
        flush=True,
    )


def invoke_memory_turn(
    user_input: str,
    messages: list[dict[str, str]],
    user_profile: dict[str, Any],
    study_context: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    from project.agent.nodes.memory_writer import write_memory_node
    from project.agent.state import AgentState

    state: AgentState = {
        "messages": messages,
        "user_input": user_input,
        "intent": "general_chat",
        "plan": [],
        "tool_calls": [],
        "tool_results": {},
        "retrieved_docs": [],
        "user_profile": user_profile,
        "study_context": study_context,
        "writing_review_state": {},
        "context_summary": "",
        "final_answer": "已收到。",
    }
    result = write_memory_node(state)
    return (
        list(result.get("messages", messages)),
        dict(result.get("user_profile", user_profile)),
        dict(result.get("study_context", study_context)),
    )


def invoke_agent_turn(
    graph: Any,
    user_input: str,
    messages: list[dict[str, str]],
    user_profile: dict[str, Any],
    study_context: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    from project.agent.state import build_initial_state

    result = graph.invoke(
        build_initial_state(
            user_input=user_input,
            messages=messages,
            user_profile=user_profile,
            study_context=study_context,
        )
    )
    return (
        list(result.get("messages", messages)),
        dict(result.get("user_profile", user_profile)),
        dict(result.get("study_context", study_context)),
    )


def run_scripted_conversation(
    inputs: list[str],
    *,
    mode: str,
    user_profile: dict[str, Any] | None = None,
    study_context: dict[str, Any] | None = None,
    progress_label: str = "",
    sample_ids: list[int] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    from project.db.models import init_db
    from project.memory.profile_service import ensure_user_profile

    init_db()
    ensure_user_profile(USER_ID, defaults={"name": "Memory Eval User"})
    messages: list[dict[str, str]] = []
    profile = dict(user_profile or {"user_id": USER_ID, "id": USER_ID})
    context = dict(study_context or {"total_turns": 0})

    graph = None
    if mode == "agent":
        from project.agent.graph import build_graph

        graph = build_graph()

    total = len(inputs)
    for index, user_input in enumerate(inputs, start=1):
        sample_id = sample_ids[index - 1] if sample_ids and index - 1 < len(sample_ids) else index
        if progress_label:
            render_progress(progress_label, index, total, sample_id, "running")
        if mode == "agent":
            messages, profile, context = invoke_agent_turn(graph, user_input, messages, profile, context)
        else:
            messages, profile, context = invoke_memory_turn(user_input, messages, profile, context)
        if progress_label:
            render_progress(progress_label, index, total, sample_id, "done")
    if progress_label:
        print()
    return messages, profile, context


def serialize_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def deserialize_value(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def add_profile_doc(docs: list[MemoryDoc], field: str, value: Any, updated_at: str = "") -> None:
    if value in (None, "", [], {}):
        return
    serialized = serialize_value(value)
    text = f"pool=user_profile\nfield={field}\nvalue={serialized}\nsummary={field}: {serialized}\nupdated_at={updated_at}"
    docs.append(
        MemoryDoc(
            id=f"profile:{field}",
            source="user_profile",
            pool="user_profile",
            field=field,
            value=serialized,
            text=text,
            is_long_term=True,
            metadata={"updated_at": updated_at},
        )
    )


def flatten_preferences(preferences: Any) -> list[tuple[str, Any]]:
    if not isinstance(preferences, dict):
        return []
    flattened: list[tuple[str, Any]] = []
    for key, value in preferences.items():
        if key == "memory_highlights":
            continue
        flattened.append((str(key), value))
    return flattened


def collect_memory_docs(study_context: dict[str, Any]) -> list[MemoryDoc]:
    from project.memory.retriever import collect_memory_docs as collect_project_memory_docs

    return collect_project_memory_docs(USER_ID, study_context)


def evaluate_write_coverage(cases: list[CoverageCase], docs: list[MemoryDoc]) -> dict[str, Any]:
    remembered = [case for case in cases if case.should_remember]
    noise = [case for case in cases if not case.should_remember]
    all_text = "\n".join(doc.text for doc in docs)
    long_term_text = "\n".join(doc.text for doc in docs if doc.is_long_term)
    long_term_docs = [doc.text for doc in docs if doc.is_long_term]

    covered = []
    missed = []
    for case in remembered:
        if text_matches(case.text, all_text):
            covered.append(case)
        else:
            missed.append(case)

    false_positive_noise = []
    for case in noise:
        if any(text_matches(case.text, doc_text, threshold=0.64) for doc_text in long_term_docs):
            false_positive_noise.append(case)

    return {
        "remembered_total": len(remembered),
        "covered": covered,
        "missed": missed,
        "coverage_rate": len(covered) / max(len(remembered), 1),
        "noise_total": len(noise),
        "false_positive_noise": false_positive_noise,
        "noise_control_rate": 1.0 if not noise else (len(noise) - len(false_positive_noise)) / len(noise),
    }


QUERY_TYPES = {"profile_query", "update_query", "recent_query", "mixed_query"}
POOL_WEIGHTS = {
    "profile_query": {"user_profile": 1.0, "memory_events": 0.6, "short_term_memory": 0.2},
    "update_query": {"memory_events": 1.0, "user_profile": 0.5, "short_term_memory": 0.2},
    "recent_query": {"short_term_memory": 1.0, "user_profile": 0.4, "memory_events": 0.3},
    "mixed_query": {"user_profile": 0.8, "short_term_memory": 0.8, "memory_events": 0.5},
}
FIELD_ALIASES = {
    "target_school": ("学校", "院校", "志愿", "港中文", "cuhk", "大学"),
    "major": ("专业", "本科方向"),
    "target_program_type": ("授课型", "研究型", "项目类型", "硕士类型"),
    "career_goal": ("工作", "职业", "研发", "算法", "大厂", "发展"),
    "target_score": ("目标分", "雅思目标", "总分", "分数", "target score"),
    "exam_date": ("考试时间", "考试日期", "几月", "日期"),
    "exam_type": ("雅思", "托福", "考试类型"),
    "subscore_requirement": ("单项", "小分", "不要低于"),
    "weak_skills": ("弱项", "薄弱", "拖后腿"),
    "budget_constraint": ("预算", "费用", "经济", "性价比", "付费"),
    "response_preference": ("喜欢", "风格", "建议", "解释", "输出"),
    "language_preference": ("中文", "英文", "交流", "语言"),
    "preferred_focus": ("重点", "优先", "主攻", "先补", "focus"),
    "available_hours_per_week": ("每周", "时间", "小时", "学习几天"),
    "available_days_per_week": ("每周", "几天", "学习天数"),
    "current_stage_goal": ("当前阶段", "核心任务", "这阶段", "申请港中文", "雅思备考计划"),
    "current_task_mode": ("每天", "小任务", "当前任务", "这轮"),
    "retrieval_source_preference": ("题库", "本地题库", "本地库", "本地 JSON", "公开资料"),
    "progress_review_cycle": ("按周", "按天", "进展", "评估"),
    "export_format_requirements": ("序号", "明确序号", "题目之间", "间隔"),
    "export_content_constraints": ("图片", "网页导航", "评论区", "网页噪音", "保留图片"),
    "export_layout_preferences": ("分页", "单独分页", "同页分节", "布局"),
    "study_strategy": ("弱项优先", "四项并行", "先补口语", "同步推进写作", "策略"),
    "weak_skill_details": ("最担心", "没信心", "阅读基础", "口语", "写作"),
    "writing_difficulty": ("写作困难", "展开论证", "限时写作", "超时"),
    "listening_difficulty": ("听力", "抓细节", "细节"),
    "learning_style": ("模板", "死记硬背", "高频反馈", "小步快跑", "备考节奏"),
    "system_workflow_preference": ("系统命令", "普通用户", "改成系统", "工作流"),
    "data_source_policy": ("本地数据库", "本地库", "本地 JSON", "直接查"),
    "tool_usage_policy": ("普通用户", "联网抓取", "临时调用", "系统命令"),
}
FIELD_INTENT_RULES = {
    "target_school": ("学校", "院校", "志愿", "大学", "cuhk", "港中文"),
    "major": ("专业", "本科", "方向"),
    "target_program_type": ("授课型", "研究型", "项目类型", "硕士类型"),
    "career_goal": ("工作", "职业", "研发", "算法", "大厂"),
    "target_score": ("目标分", "总分", "分数", "雅思目标"),
    "exam_date": ("考试时间", "考试日期", "几月"),
    "exam_type": ("雅思", "托福", "考试类型"),
    "subscore_requirement": ("单项", "小分", "不要低于"),
    "weak_skills": ("弱项", "薄弱", "拖后腿", "担心"),
    "budget_constraint": ("预算", "费用", "经济", "性价比", "付费", "免费", "课程"),
    "response_preference": ("偏好", "风格", "喜欢", "建议", "解释", "输出", "表格", "清单"),
    "language_preference": ("中文", "英文", "语言", "交流"),
    "preferred_focus": ("优先", "主攻", "先补", "重点"),
    "available_hours_per_week": ("每周", "几天", "小时", "学习时间"),
    "available_days_per_week": ("每周", "几天", "学习天数"),
    "current_stage_goal": ("核心任务", "当前阶段", "这阶段", "申请港中文", "备考计划"),
    "current_task_mode": ("每天", "小任务", "当前任务", "这轮"),
    "retrieval_source_preference": ("从哪里", "题库", "本地题库", "本地库", "本地 JSON", "公开资料", "读取"),
    "progress_review_cycle": ("按天", "按周", "进展", "评估"),
    "export_format_requirements": ("序号", "明显间隔", "明确序号"),
    "export_content_constraints": ("图片", "网页导航", "评论区", "混进", "保留图片"),
    "export_layout_preferences": ("分页", "单独分页", "同页分节", "导出格式"),
    "study_strategy": ("备考策略", "弱项优先", "四项并行", "先补口语", "同步推进写作"),
    "weak_skill_details": ("最担心", "没信心", "阅读基础", "口语相比", "哪个基础"),
    "writing_difficulty": ("写作", "展开论证", "限时写作", "超时"),
    "listening_difficulty": ("听力", "抓细节"),
    "learning_style": ("模板", "死记硬背", "高频反馈", "小步快跑", "备考节奏"),
    "system_workflow_preference": ("系统命令", "普通用户", "改成系统"),
    "data_source_policy": ("本地数据库", "本地库", "本地 JSON", "直接查"),
    "tool_usage_policy": ("普通用户", "联网", "临时调用", "爬取"),
}
CURRENT_MARKERS = ("现在", "当前", "最新", "后来", "改成", "最终")
HISTORY_MARKERS = ("原来", "最开始", "最初", "之前", "一开始", "以前")
RECENT_MARKERS = ("这轮", "这次", "刚刚", "当前阶段", "当前任务", "本周", "现在导出", "当前在")
MIXED_MARKERS = ("结合", "根据", "按我的", "同时", "综合")


def route_memory_query(query: str) -> QueryRoute:
    llm_route = route_memory_query_with_llm(query)
    if llm_route is not None:
        return llm_route
    return route_memory_query_with_rules(query)


def route_memory_query_with_llm(query: str) -> QueryRoute | None:
    try:
        from project.llm.client import LLMClient

        client = LLMClient.from_memory_config()
        if not client.is_configured:
            return None
        raw = client.generate_text(
            system_prompt=(
                "You are a memory query router. Decide if the user question needs memory search. "
                "If yes, classify it as profile_query, update_query, recent_query, or mixed_query. "
                "Return JSON only: {\"should_search\":true,\"query_type\":\"profile_query\",\"reason\":\"...\"}."
            ),
            user_prompt=f"User question:\n{query}",
            temperature=0.0,
            max_tokens=120,
        )
        parsed = parse_json_object(raw)
        if not isinstance(parsed, dict):
            return None
        query_type = str(parsed.get("query_type") or "mixed_query")
        if query_type not in QUERY_TYPES:
            query_type = "mixed_query"
        return QueryRoute(
            should_search=bool(parsed.get("should_search", True)),
            query_type=query_type,
            reason=str(parsed.get("reason") or "llm_router"),
        )
    except Exception:
        return None


def parse_json_object(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def route_memory_query_with_rules(query: str) -> QueryRoute:
    normalized = query.lower()
    if not any(marker in query for marker in ("我", "我的", "记得", "现在", "当前", "之前", "这轮", "这次")):
        return QueryRoute(False, "mixed_query", "no_memory_marker")
    if any(marker in query for marker in MIXED_MARKERS):
        return QueryRoute(True, "mixed_query", "mixed_marker")
    if any(marker in query for marker in HISTORY_MARKERS) or "最新" in query or "改" in query or "后来" in query:
        return QueryRoute(True, "update_query", "update_marker")
    if any(marker in query for marker in RECENT_MARKERS):
        return QueryRoute(True, "recent_query", "recent_marker")
    if any(marker in normalized for marker in ("目标", "学校", "专业", "弱项", "预算", "喜欢", "风格", "本科", "雅思")):
        return QueryRoute(True, "profile_query", "profile_marker")
    return QueryRoute(True, "mixed_query", "default")


def query_field_intents(query: str) -> set[str]:
    fields = {
        field
        for field, markers in FIELD_INTENT_RULES.items()
        if any(marker.lower() in query.lower() or marker in query for marker in markers)
    }
    if "target_school" in fields and "target_score" in fields and "分" not in query and "总分" not in query:
        fields.discard("target_score")
    if "target_score" in fields and "学校" not in query and "院校" not in query and "志愿" not in query:
        fields.discard("target_school")
    if "writing_difficulty" in fields:
        fields.discard("weak_skills")
        fields.discard("response_preference")
    if "listening_difficulty" in fields:
        fields.discard("weak_skills")
    if "export_layout_preferences" in fields or "export_format_requirements" in fields or "export_content_constraints" in fields:
        fields.discard("response_preference")
    if "retrieval_source_preference" in fields or "data_source_policy" in fields:
        fields.discard("budget_constraint")
    return fields


def bm25_rank(query: str, docs: list[MemoryDoc], k: int = 5) -> list[tuple[MemoryDoc, float]]:
    tokenized_docs = [char_tokens(doc.text) for doc in docs]
    query_tokens = char_tokens(query)
    if not query_tokens or not docs:
        return []

    doc_freq: Counter[str] = Counter()
    for tokens in tokenized_docs:
        for token in set(tokens):
            doc_freq[token] += 1

    avgdl = sum(len(tokens) for tokens in tokenized_docs) / max(len(tokenized_docs), 1)
    total_docs = len(docs)
    k1 = 1.5
    b = 0.75
    query_counts = Counter(query_tokens)
    scored: list[tuple[MemoryDoc, float]] = []

    for doc, tokens in zip(docs, tokenized_docs):
        token_counts = Counter(tokens)
        doc_len = len(tokens) or 1
        score = 0.0
        for token, query_weight in query_counts.items():
            freq = token_counts.get(token, 0)
            if not freq:
                continue
            idf = math.log(1 + (total_docs - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5))
            denom = freq + k1 * (1 - b + b * doc_len / max(avgdl, 1))
            score += idf * (freq * (k1 + 1) / denom) * min(query_weight, 3)
        scored.append((doc, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]


def metadata_boost(query: str, doc: MemoryDoc, route: QueryRoute) -> float:
    boost = 0.0
    intended_fields = query_field_intents(query)
    query_norm = normalize_text(query)
    field_norm = normalize_text(doc.field)
    value_norm = normalize_text(doc.value)
    if intended_fields:
        if doc.field in intended_fields:
            boost += 3.0
        elif doc.pool in {"user_profile", "memory_events"}:
            boost -= 20.0
    elif field_norm and field_norm in query_norm:
        boost += 0.25
    if value_norm and value_norm in query_norm:
        boost += 0.25
    aliases = FIELD_ALIASES.get(doc.field, ())
    if doc.field in intended_fields and any(alias.lower() in query.lower() or alias in query for alias in aliases):
        boost += 0.4

    metadata = doc.metadata or {}
    if doc.pool == "memory_events":
        if metadata.get("memory_type") != "core_memory":
            boost -= 1.0
        if any(marker in query for marker in CURRENT_MARKERS) and metadata.get("is_latest"):
            boost += 0.45
        if any(marker in query for marker in HISTORY_MARKERS) and metadata.get("old_value") not in (None, "", []):
            boost += 0.45
        if "改" in query or "后来" in query:
            boost += 0.25
    elif doc.pool == "short_term_memory":
        if any(marker in query for marker in RECENT_MARKERS):
            boost += 0.45
        if metadata.get("importance") == "focus":
            boost += 0.15
    elif doc.pool == "user_profile":
        if route.query_type == "profile_query":
            boost += 0.2
        if any(marker in query for marker in CURRENT_MARKERS):
            boost += 0.2
    return boost


def weighted_memory_retrieve(query: str, docs: list[MemoryDoc], k: int = 5) -> tuple[QueryRoute, list[tuple[MemoryDoc, float]]]:
    from project.memory.retriever import weighted_memory_retrieve as retrieve_project_memory

    return retrieve_project_memory(query, docs, k=k)


def resolve_conflicts(
    query: str,
    route: QueryRoute,
    ranked: list[tuple[MemoryDoc, float]],
    *,
    k: int,
) -> list[tuple[MemoryDoc, float]]:
    selected: list[tuple[MemoryDoc, float]] = []
    seen_text: set[str] = set()
    selected_fields: set[str] = set()
    wants_current = any(marker in query for marker in CURRENT_MARKERS)
    wants_history = any(marker in query for marker in HISTORY_MARKERS)

    for doc, score in ranked:
        text_key = normalize_text(doc.text)
        if text_key in seen_text:
            continue
        metadata = doc.metadata or {}
        if wants_current and doc.pool == "memory_events" and not metadata.get("is_latest") and doc.field in selected_fields:
            continue
        if wants_history and doc.pool == "user_profile" and doc.field in selected_fields:
            continue
        selected.append((doc, score))
        seen_text.add(text_key)
        if doc.field:
            selected_fields.add(doc.field)
        if len(selected) >= k:
            break
    return selected


STALE_MARKERS = {
    41: ["6.5"],
    43: ["6.0"],
    45: ["6月", "6 月"],
    47: ["主攻写作"],
    48: ["5天", "5 天"],
    49: ["完全免费"],
    50: ["同页分节"],
}


def evaluate_retrieval(cases: list[RetrievalCase], docs: list[MemoryDoc]) -> dict[str, Any]:
    results = []
    hit1 = hit3 = recall5 = 0
    reciprocal_sum = 0.0
    update_latest_ok = 0
    update_total = 0

    total = len(cases)
    for index, case in enumerate(cases, start=1):
        render_progress("Test 2 retrieve", index, total, case.id, "routing")
        route, ranked = weighted_memory_retrieve(case.query, docs, k=5)
        match_rank = None
        for rank, (doc, score) in enumerate(ranked, start=1):
            if answer_in_doc(case.answer, doc.text):
                match_rank = rank
                break

        if match_rank == 1:
            hit1 += 1
        if match_rank is not None and match_rank <= 3:
            hit3 += 1
        if match_rank is not None and match_rank <= 5:
            recall5 += 1
            reciprocal_sum += 1 / match_rank

        stale_detected = False
        if 41 <= case.id <= 50:
            update_total += 1
            joined_top = normalize_text("\n".join(doc.text for doc, _score in ranked))
            for marker in STALE_MARKERS.get(case.id, []):
                marker_norm = normalize_text(marker)
                if marker_norm and marker_norm in joined_top and match_rank is None:
                    stale_detected = True
                    break
            if match_rank is not None and not stale_detected:
                update_latest_ok += 1

        results.append(
            {
                "case": case,
                "route": route,
                "rank": match_rank,
                "stale_detected": stale_detected,
                "top5": [
                    {
                        "rank": rank,
                        "score": score,
                        "id": doc.id,
                        "source": doc.source,
                        "text": doc.text,
                    }
                    for rank, (doc, score) in enumerate(ranked, start=1)
                ],
            }
        )
        render_progress("Test 2 retrieve", index, total, case.id, "done")

    print()
    return {
        "total": total,
        "hit1": hit1 / max(total, 1),
        "hit3": hit3 / max(total, 1),
        "recall5": recall5 / max(total, 1),
        "mrr": reciprocal_sum / max(total, 1),
        "update_latest_accuracy": update_latest_ok / max(update_total, 1),
        "route_counts": dict(Counter(item["route"].query_type for item in results)),
        "skipped_memory_count": sum(1 for item in results if not item["route"].should_search),
        "results": results,
    }


def short(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def build_report(
    *,
    args: argparse.Namespace,
    coverage_cases: list[CoverageCase],
    retrieval_cases: list[RetrievalCase],
    docs: list[MemoryDoc],
    write_eval: dict[str, Any],
    retrieval_eval: dict[str, Any],
) -> str:
    lines = [
        "Memory Module Evaluation Report",
        f"generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"mode: {args.mode}",
        f"db_path: {args.db_path}",
        f"coverage_file: {args.coverage_file}",
        f"retrieval_file: {args.retrieval_file}",
        "",
        "Corpus",
        f"- memory_docs: {len(docs)}",
        f"- long_term_docs: {sum(1 for doc in docs if doc.is_long_term)}",
        f"- short_term_docs: {sum(1 for doc in docs if not doc.is_long_term)}",
        "",
        "Test 1: Memory Write",
        f"- cases_loaded: {len(coverage_cases)}",
        f"- write_coverage: {len(write_eval['covered'])}/{write_eval['remembered_total']} = {write_eval['coverage_rate']:.2%}",
        f"- noise_control: {write_eval['noise_total'] - len(write_eval['false_positive_noise'])}/{write_eval['noise_total']} = {write_eval['noise_control_rate']:.2%}",
        "",
        "Test 2: Retrieval Top5",
        f"- cases_loaded: {len(retrieval_cases)}",
        f"- Hit@1: {retrieval_eval['hit1']:.2%}",
        f"- Hit@3: {retrieval_eval['hit3']:.2%}",
        f"- Recall@5: {retrieval_eval['recall5']:.2%}",
        f"- MRR: {retrieval_eval['mrr']:.4f}",
        f"- update_latest_accuracy(41-50): {retrieval_eval['update_latest_accuracy']:.2%}",
        f"- route_counts: {retrieval_eval.get('route_counts', {})}",
        f"- skipped_memory_count: {retrieval_eval.get('skipped_memory_count', 0)}",
        "",
        "Missed Write Cases (first 30)",
    ]
    for case in write_eval["missed"][:30]:
        lines.append(f"- {case.id:03d}: {short(case.text)}")

    lines.extend(["", "Noise False Positives"])
    for case in write_eval["false_positive_noise"]:
        lines.append(f"- {case.id:03d}: {short(case.text)}")
    if not write_eval["false_positive_noise"]:
        lines.append("- none")

    failed = [item for item in retrieval_eval["results"] if item["rank"] is None or item["stale_detected"]]
    lines.extend(["", "Failed Retrieval Cases"])
    if not failed:
        lines.append("- none")
    for item in failed[:30]:
        case = item["case"]
        route = item.get("route")
        route_label = route.query_type if isinstance(route, QueryRoute) else "unknown"
        reason = "stale" if item["stale_detected"] else "miss"
        lines.append(f"- {case.id:03d} [{reason}/{route_label}] Q={short(case.query, 80)} A={short(case.answer, 80)}")
        for top in item["top5"][:5]:
            lines.append(
                f"  top{top['rank']} score={top['score']:.3f} {top['id']} {top['source']}: {short(top['text'], 120)}"
            )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run memory write/retrieval evaluation.")
    parser.add_argument("--coverage-file", type=Path, default=DEFAULT_COVERAGE_FILE)
    parser.add_argument("--retrieval-file", type=Path, default=DEFAULT_RETRIEVAL_FILE)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--mode", choices=("memory", "agent"), default="memory")
    parser.add_argument("--keep-db", action="store_true")
    parser.add_argument("--disable-llm", action="store_true", help="Use rule fallback for smoke/debug runs.")
    parser.add_argument("--llm-timeout", type=int, default=None)
    parser.add_argument("--limit-coverage", type=int, default=0)
    parser.add_argument("--limit-retrieval", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_environment(args.db_path, args.keep_db, args.disable_llm, args.llm_timeout)
    reset_extractor_watermark()

    coverage_cases = parse_coverage_cases(args.coverage_file)
    retrieval_cases = parse_retrieval_cases(args.retrieval_file)
    if args.limit_coverage:
        coverage_cases = coverage_cases[: args.limit_coverage]
    if args.limit_retrieval:
        retrieval_cases = retrieval_cases[: args.limit_retrieval]
    if not coverage_cases:
        raise SystemExit(f"No coverage cases parsed from {args.coverage_file}")
    if not retrieval_cases:
        raise SystemExit(f"No retrieval cases parsed from {args.retrieval_file}")

    print(f"loaded coverage={len(coverage_cases)}, retrieval={len(retrieval_cases)}")
    _messages, profile, context = run_scripted_conversation(
        [case.text for case in coverage_cases],
        mode=args.mode,
        progress_label="Test 1 write",
        sample_ids=[case.id for case in coverage_cases],
    )
    docs_after_write = collect_memory_docs(context)
    write_eval = evaluate_write_coverage(coverage_cases, docs_after_write)

    _messages, profile, context = run_scripted_conversation(
        [case.query for case in retrieval_cases],
        mode=args.mode,
        user_profile=profile,
        study_context=context,
        progress_label="Test 2 query",
        sample_ids=[case.id for case in retrieval_cases],
    )
    retrieval_eval = evaluate_retrieval(retrieval_cases, docs_after_write)

    report = build_report(
        args=args,
        coverage_cases=coverage_cases,
        retrieval_cases=retrieval_cases,
        docs=docs_after_write,
        write_eval=write_eval,
        retrieval_eval=retrieval_eval,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8-sig")
    print(f"report written: {args.output}")
    print(f"write_coverage={write_eval['coverage_rate']:.2%}")
    print(f"noise_control={write_eval['noise_control_rate']:.2%}")
    print(f"Recall@5={retrieval_eval['recall5']:.2%}, MRR={retrieval_eval['mrr']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
