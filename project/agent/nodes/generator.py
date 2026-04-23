"""Response generator node with LLM-first and template fallback."""

from __future__ import annotations

import logging

from project.agent.state import AgentState, RetrievedDoc
from project.agent.nodes.tracing import trace_node
from project.llm.client import LLMClient
from project.prompts.generator_prompt import GENERATOR_SYSTEM_PROMPT, build_generator_user_prompt

logger = logging.getLogger(__name__)


def _format_doc_points(docs: list[RetrievedDoc]) -> str:
    if not docs:
        return "当前没有检索到额外资料，我先基于已有信息给你一个可执行回答。"

    points = [f"{index}. {doc['title']}：{doc['content']}" for index, doc in enumerate(docs[:3], start=1)]
    return "\n".join(points)


def _generate_study_plan_answer(state: AgentState) -> str:
    profile = state.get("user_profile", {})
    plan_data = state.get("tool_results", {}).get("get_study_plan", {})
    daily_tasks = plan_data.get("daily_tasks", [])
    weekly_goals = plan_data.get("weekly_goals", [])

    lines = [
        "这是一个适合当前阶段的雅思备考起步计划：",
        f"- 当前水平：{profile.get('current_level', '待评估')}",
        f"- 目标分数：{profile.get('target_score', '待确认')}",
        f"- 每周可投入时间：{profile.get('available_hours_per_week', '待确认')} 小时",
    ]
    if weekly_goals:
        lines.append(f"- 本周重点：{'；'.join(weekly_goals)}")
    if daily_tasks:
        lines.append(f"- 今日建议：{'；'.join(daily_tasks)}")
    lines.append("如果你愿意，下一阶段我们可以把这个计划细化成按天执行的任务清单。")
    return "\n".join(lines)


def _generate_knowledge_answer(state: AgentState) -> str:
    docs = state.get("retrieved_docs", [])
    web_results = state.get("tool_results", {}).get("search_web", [])
    if isinstance(web_results, dict):
        web_results = web_results.get("results", [])
    if not docs and web_results:
        top_web = web_results[:2]
        web_points = "\n".join(
            f"{index}. {item.get('title', 'Search Result')}：{item.get('snippet', '')}"
            for index, item in enumerate(top_web, start=1)
        )
        return (
            "我先结合当前可用的外部信息给你一个简明回答：\n"
            f"{web_points}\n"
            "这类问题通常会变化较快，正式决定前建议再核对一次官方渠道。"
        )

    return (
        "我先结合已有资料给你一个简明回答：\n"
        f"{_format_doc_points(docs)}\n"
        "建议你先抓住核心方法，再配一到两个针对性练习，这样更容易把知识点转化成分数。"
    )


def _generate_mistake_review_answer(state: AgentState) -> str:
    grading_result = state.get("tool_results", {}).get("grade_submission")
    if isinstance(grading_result, dict):
        grading = grading_result.get("grading", {})
        submission = grading_result.get("submission", {})
        updated_profile = grading_result.get("updated_profile", {})
        is_correct = bool(grading.get("is_correct", False))
        focus_recommendations = (
            updated_profile.get("preferences", {}).get("focus_recommendations", [])
            if isinstance(updated_profile, dict)
            else []
        )
        if is_correct:
            return (
                "这道题整体判定为正确。\n"
                f"- 科目：{submission.get('subject', 'unknown')}\n"
                f"- 得分：{grading.get('score', 1.0):.2f}\n"
                f"- 依据来源：{grading.get('source_of_truth', 'heuristic')}\n"
                "建议你继续保持当前做题方法，并把这类题型当成巩固项。"
            )

        lines = [
            "这道题我已经帮你批改并记入错题本了：",
            f"- 科目：{submission.get('subject', 'unknown')}",
            f"- 题型：{submission.get('question_type', 'unknown')}",
            f"- 你的答案：{submission.get('user_answer', '') or '未提供'}",
            f"- 参考答案：{grading.get('reference_answer', '') or '未找到可靠参考答案'}",
            f"- 错误类型：{grading.get('error_type', '未分类')}",
            f"- 错误原因：{grading.get('wrong_reason', '待补充')}",
            f"- 改正建议：{grading.get('correction_note', '建议重新核对题干与依据文本。')}",
            f"- 依据来源：{grading.get('source_of_truth', 'heuristic')}",
        ]
        if focus_recommendations:
            lines.append(f"- 当前优先提升方向：{focus_recommendations[0]}")
        return "\n".join(lines)

    records = state.get("tool_results", {}).get("get_mistake_records", [])
    if not records:
        return "当前还没有错题记录。后续接入更多练习数据后，我们可以按题型、错误原因和复习次数做系统复盘。"

    top_errors = [record["error_type"] for record in records[:3]]
    return (
        "根据当前错题记录，你最值得优先处理的薄弱点是："
        f"{'、'.join(top_errors)}。\n"
        "建议做法：\n"
        "1. 先把每类错误对应的判断标准写成 1 句话。\n"
        "2. 每类错误各做 2 道小练习，确认自己能改对。\n"
        "3. 明天再做一次回顾，检查是否还会重复犯错。"
    )


def _generate_writing_feedback_answer(state: AgentState) -> str:
    prompt_result = state.get("tool_results", {}).get("get_random_task2_prompt")
    if isinstance(prompt_result, dict) and not prompt_result.get("success"):
        return str(prompt_result.get("message") or "当前还没有可用的大作文题目。")
    if isinstance(prompt_result, dict) and prompt_result.get("success"):
        topic = prompt_result.get("topic", {})
        requested_essay_type = str(prompt_result.get("requested_essay_type") or "").strip()
        lines = [
            "这是你这次的 IELTS Writing Task 2 练习题：",
            f"- 考试时间：{topic.get('exam_date', 'unknown')}",
            f"- 题型：{topic.get('essay_type', '观点类')}",
            f"- 题目：{topic.get('prompt_text', '')}",
        ]
        if requested_essay_type:
            lines.insert(1, f"- 你的指定方向：{requested_essay_type}")
        lines.extend(
            [
                "",
                "请直接把你的完整作文发给我，我会按评分维度给你批改、指出优点和重点修改方向。",
                "建议至少写 250 词，尽量包含清晰立场、主体段展开和结论。",
                "如果你需要，我也可以再给你补一版中文题意说明。",
            ]
        )
        return "\n".join(lines)

    review_result = state.get("tool_results", {}).get("review_task2_submission")
    if isinstance(review_result, dict) and not review_result.get("success"):
        return str(review_result.get("message") or "当前还没有找到可批改的作文题目，请先让我给你抽一题。")
    if isinstance(review_result, dict) and review_result.get("success"):
        topic = review_result.get("topic", {})
        evaluation = review_result.get("evaluation", {})
        progress_summary = review_result.get("progress_summary") or {}
        breakdown = evaluation.get("band_breakdown", {})
        strengths = evaluation.get("strengths", [])
        issues = evaluation.get("issues", [])
        revision_plan = evaluation.get("revision_plan", [])
        language_notes = evaluation.get("language_upgrade_notes", [])
        lines = [
            "这篇作文我已经帮你批改并归纳到记忆里了：",
            f"- 题目：{topic.get('prompt_text', '')}",
            f"- 估计总分：{evaluation.get('overall_band', 'N/A')}",
            f"- Task Response：{breakdown.get('task_response', 'N/A')}",
            f"- Coherence & Cohesion：{breakdown.get('coherence_cohesion', 'N/A')}",
            f"- Lexical Resource：{breakdown.get('lexical_resource', 'N/A')}",
            f"- Grammar Accuracy：{breakdown.get('grammar_accuracy', 'N/A')}",
            f"- 总评：{evaluation.get('overall_comment', '')}",
            f"- 亮点：{'；'.join(strengths) if strengths else '需要继续积累。'}",
            f"- 主要问题：{'；'.join(issues) if issues else evaluation.get('priority_issue', '待进一步观察')}",
            f"- 修改优先级：{'；'.join(revision_plan) if revision_plan else '先加强论证展开。'}",
            f"- 表达升级建议：{'；'.join(language_notes) if language_notes else '继续积累更正式的连接和论证表达。'}",
        ]
        if progress_summary:
            score_delta = float(progress_summary.get("score_delta", 0.0) or 0.0)
            if score_delta > 0:
                score_line = f"- 和上一次相比：总分提升了 {score_delta:.1f}"
            elif score_delta < 0:
                score_line = f"- 和上一次相比：总分下降了 {abs(score_delta):.1f}"
            else:
                score_line = "- 和上一次相比：总分基本持平"
            lines.append(score_line)
            changes = progress_summary.get("highlighted_changes", [])
            if changes:
                lines.append(f"- 进步/波动概览：{'；'.join(changes)}")
        return "\n".join(lines)

    return "我已经准备好给你抽一题 IELTS Writing Task 2，或者继续批改你刚写完的作文。"


def _generate_calendar_answer(state: AgentState) -> str:
    results = state.get("tool_results", {})
    if "create_study_event" in results:
        event = results["create_study_event"]
        time_range = f"{event.get('start_time', 'TBD')} -> {event.get('end_time', 'TBD')}"
        return (
            "学习日程已经创建好了：\n"
            f"- 标题：{event['title']}\n"
            f"- 时间：{time_range}\n"
            f"- 状态：{event.get('status', 'unknown')}\n"
            "后续接入真实日历服务后，这里会展示真实写入结果。"
        )

    schedule = results.get("get_schedule", [])
    if not schedule:
        return "当前没有已安排的学习日程。你可以让我继续帮你创建一个雅思学习时段。"

    items = [f"- {item['date']} {item['title']}（{item['duration_minutes']} 分钟）" for item in schedule]
    return "这是你当前的学习安排：\n" + "\n".join(items)


def _generate_data_collection_answer(state: AgentState) -> str:
    plan = state.get("plan", [])
    pdf_result = state.get("tool_results", {}).get("export_question_pdf")
    if isinstance(pdf_result, dict):
        if not pdf_result.get("success"):
            return (
                "PDF 导出失败：\n"
                f"- 原因：{pdf_result.get('message') or pdf_result.get('error')}\n"
                f"- 请求数量：{pdf_result.get('requested_count', 0)}\n"
                f"- 已导出：{pdf_result.get('exported_count', 0)}"
            )
        return (
            "本地题库 PDF 已导出：\n"
            f"- 文件：{pdf_result.get('export_path')}\n"
            f"- 请求数量：{pdf_result.get('requested_count')}\n"
            f"- 实际导出：{pdf_result.get('exported_count')}\n"
            f"- 完成状态：{pdf_result.get('completion_status')}"
        )

    crawler_result = state.get("tool_results", {}).get("crawl_writing_questions")
    if isinstance(crawler_result, dict):
        return (
            "剑桥雅思 Writing 真题爬取结果：\n"
            f"- 请求 Task：{crawler_result.get('task_no') or 'Task 1 + Task 2'}\n"
            f"- 解析数量：{crawler_result.get('parsed_count', 0)}\n"
            f"- 写入数据库：{crawler_result.get('saved_count', 0)}\n"
            f"- 题库根目录：{crawler_result.get('question_bank_root', '')}\n"
            f"- 结构化 JSON：{crawler_result.get('json_dir', '')}\n"
            f"- 图片目录：{crawler_result.get('image_dir', '')}\n"
            f"- 原始快照：{crawler_result.get('raw_dir', '')}\n"
            f"- 失败数：{crawler_result.get('fail_count', 0)}"
        )

    if "data_collection_error_report" in plan:
        pending = state.get("study_context", {}).get("data_collection_request", {})
        return (
            "资料采集已连续 5 轮没有拿到满意真题，已上报为 data_collection_failed。\n"
            f"- 原始需求：{pending.get('original_request', state.get('user_input', '')) if isinstance(pending, dict) else state.get('user_input', '')}\n"
            "- 建议下一步：请提供可访问的官方链接，或改为收集官方公开样题。"
        )

    if "clarify_data_collection_request" in plan:
        pending = state.get("study_context", {}).get("data_collection_request", {})
        attempts = int(pending.get("attempts", 0) or 0) + 1 if isinstance(pending, dict) else 1
        details = state.get("data_collection_details", {})
        details = details if isinstance(details, dict) else {}
        missing = state.get("data_collection_missing", [])
        missing = missing if isinstance(missing, list) else []

        labels = {
            "module": "模块：reading / listening / writing / speaking",
            "question_type": "题型：Writing 请说明 Task 1 还是 Task 2；Reading 可说明 True/False/Not Given、Matching Headings 等",
            "format": "格式：pdf / txt / json",
        }
        missing_lines = [f"- {labels[item]}" for item in missing if item in labels]
        known_parts = []
        if details.get("module"):
            known_parts.append(f"模块={details['module']}")
        if details.get("task_type"):
            known_parts.append(f"题型={details['task_type']}")
        if details.get("year"):
            date_value = f"{details['year']}年"
            if details.get("month"):
                date_value += f"{int(details['month']):02d}月"
            known_parts.append(f"日期={date_value}")
        if details.get("format"):
            known_parts.append(f"格式={details['format']}")
        if details.get("count"):
            known_parts.append(f"数量={details['count']}份")

        lines = ["我还缺少这些真题细节："]
        lines.extend(missing_lines or ["- 请补充更具体的资料要求。"])
        if known_parts:
            lines.append(f"已识别：{'，'.join(known_parts)}。")
        lines.append(f"当前尝试轮数：{attempts}/5。")
        return "\n".join(lines)

    result = state.get("tool_results", {}).get("collect_data", {})
    if not isinstance(result, dict):
        return "资料收集工具没有返回有效结果，请稍后重试。"

    lines = [
        "本次资料收集结果：",
        f"- 收集类别：{', '.join(result.get('categories', []))}",
        f"- 请求数量：{result.get('requested_count', 1)}",
        f"- 匹配数量：{result.get('matched_count', result.get('success_count', 0))}",
        f"- 完成状态：{result.get('collection_status', 'complete')}",
        f"- 结构化成功数：{result.get('parsed_count', result.get('success_count', 0))}",
        f"- 部分提取数：{result.get('partial_count', 0)}",
        f"- 原始保留数：{result.get('raw_only_count', 0)}",
        f"- 已保存文件数：{result.get('saved_count', result.get('success_count', 0))}",
        f"- 保存目录：{result.get('save_root', '')}",
        f"- 失败/未满足数：{result.get('fail_count', 0)}",
    ]
    files = result.get("files", [])
    collection_export_path = result.get("collection_export_path")
    if collection_export_path:
        lines.append(f"合并导出文件：{collection_export_path}")
    if files:
        lines.append("文件清单：")
        for index, item in enumerate(files[:8], start=1):
            export_path = item.get("export_path")
            if export_path:
                lines.append(
                    f"{index}. {item.get('file_name')} | {item.get('source')} | "
                    f"规范化：{item.get('saved_path')} | 导出：{export_path}"
                )
            else:
                lines.append(f"{index}. {item.get('file_name')} | {item.get('source')} | {item.get('saved_path')}")
    failures = result.get("failures", [])
    if failures:
        lines.append("失败概述：")
        for item in failures[:5]:
            lines.append(f"- {item.get('title')}: {item.get('status', 'failed')} | {item.get('reason')}")
    if result.get("success_count", 0) == 0 and failures:
        lines.append("这轮没有拿到满意真题。请补充模块、题型、日期或提供官方链接，我会继续尝试。")
    lines.append(str(result.get("next_step", "")))
    return "\n".join(lines)


def _generate_general_chat_answer(state: AgentState) -> str:
    return (
        "我已经准备好作为你的雅思学习助手。你可以直接告诉我想做哪一类事，比如：\n"
        "1. 制定学习计划\n"
        "2. 抽一道大作文题并让我批改\n"
        "3. 讲解某个雅思知识点\n"
        "4. 复盘错题\n"
        "5. 安排学习日程"
    )


def _generate_with_fallback(state: AgentState) -> str:
    intent = state["intent"]
    if intent == "study_plan":
        return _generate_study_plan_answer(state)
    if intent == "knowledge_qa":
        return _generate_knowledge_answer(state)
    if intent == "mistake_review":
        return _generate_mistake_review_answer(state)
    if intent == "writing_practice":
        return _generate_writing_feedback_answer(state)
    if intent == "data_collection":
        return _generate_data_collection_answer(state)
    if intent == "calendar_action":
        return _generate_calendar_answer(state)
    return _generate_general_chat_answer(state)


def _try_llm_generate(state: AgentState) -> str | None:
    """Generate the final answer with the configured LLM."""

    client = LLMClient.from_config()
    response_text = client.generate_text(
        system_prompt=GENERATOR_SYSTEM_PROMPT,
        user_prompt=build_generator_user_prompt(
            intent=state["intent"],
            user_input=state["user_input"],
            answer_context=state.get("answer_context", {}),
        ),
        temperature=0.4,
        max_tokens=900,
    )
    if not response_text:
        return None

    cleaned = response_text.strip()
    return cleaned or None


@trace_node("generator")
def generate_node(state: AgentState) -> dict[str, str]:
    """Generate a final response from the current state."""

    if state["intent"] in {"writing_practice", "data_collection"}:
        final_answer = _generate_with_fallback(state)
    else:
        final_answer = _try_llm_generate(state) or _generate_with_fallback(state)
    logger.info("Final answer generated for intent: %s", state["intent"])
    return {"final_answer": final_answer}
