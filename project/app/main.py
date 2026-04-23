"""CLI entrypoint for the IELTS Study Assistant local demo."""

from __future__ import annotations

import logging

from dotenv import load_dotenv

from project.agent.checkpointing import checkpoint_config
from project.agent.graph import build_graph
from project.agent.state import Message, StudyContext, UserProfile, build_initial_state
from project.config import get_config
from project.logging_config import setup_logging

logger = logging.getLogger(__name__)


WELCOME_TEXT = """
IELTS Study Assistant 本地 Demo 已启动。
你可以直接输入学习问题、抽一道大作文题、提交作文让系统批改、错题复盘需求、日程安排需求，或输入 exit 退出。
""".strip()


def _print_separator() -> None:
    print("-" * 72)


def run_cli() -> None:
    """Start the interactive CLI loop."""

    load_dotenv()
    config = get_config()
    setup_logging(config.log_level)

    graph = build_graph()
    messages: list[Message] = []
    user_profile: UserProfile = {}
    study_context: StudyContext = {"total_turns": 0}

    _print_separator()
    print(f"{config.app_name}")
    print(WELCOME_TEXT)
    _print_separator()

    while True:
        try:
            user_input = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSession ended. Bye.")
            break

        if user_input.lower() == "exit":
            print("\nSession ended. Bye.")
            break

        if not user_input:
            print("\nAssistant> 请输入你的雅思学习问题或需求。")
            continue

        try:
            initial_state = build_initial_state(
                user_input=user_input,
                messages=messages,
                user_profile=user_profile,
                study_context=study_context,
            )
            result = graph.invoke(initial_state, config=checkpoint_config("cli-demo-user"))

            messages = result.get("messages", messages)
            user_profile = result.get("user_profile", user_profile)
            study_context = result.get("study_context", study_context)

            _print_separator()
            print(f"Intent     : {result.get('intent', 'unknown')}")
            print(f"Plan       : {', '.join(result.get('plan', [])) or 'none'}")
            print(f"Assistant> {result.get('final_answer', '').strip()}")
            _print_separator()
        except Exception as exc:  # pragma: no cover - user-facing runtime guard
            logger.exception("CLI turn failed")
            print("\nAssistant> 这一轮处理时遇到了一个异常，但程序没有中断。")
            print(f"Assistant> 错误信息：{exc}")
            print("Assistant> 你可以继续输入下一个问题。")


if __name__ == "__main__":
    run_cli()
