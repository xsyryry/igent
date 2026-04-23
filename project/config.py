"""Application-level configuration for the IELTS study assistant demo."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os


@dataclass(slots=True)
class AppConfig:
    """Centralized runtime settings."""

    app_name: str = "IELTS Study Assistant"
    log_level: str = "INFO"
    default_top_k: int = 5
    default_dataset_scope: str = "ielts_core"
    rag_backend: str = "simple_local"
    db_backend: str = "mock_sqlite"
    calendar_backend: str = "mock_calendar"
    checkpoint_backend: str = "sqlite"
    checkpoint_sqlite_path: str = "data/langgraph_checkpoints.sqlite"
    retrieval_max_rounds: int = 3
    retrieval_max_no_progress_rounds: int = 2
    retrieval_top_k_per_round: int = 5
    retrieval_selected_k: int = 3
    retrieval_duplicate_rate_threshold: float = 0.75
    retrieval_gap_fill_target: float = 0.75
    retrieval_novelty_weight: float = 0.35
    retrieval_history_penalty: float = 0.4
    retrieval_min_new_facts: int = 1
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    chunk_llm_model: str = ""
    memory_llm_api_key: str = ""
    memory_llm_base_url: str = ""
    memory_llm_model: str = ""
    memory_llm_timeout: int = 30
    llm_timeout: int = 30


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Build a config object from environment variables."""

    return AppConfig(
        app_name=os.getenv("APP_NAME", "IELTS Study Assistant"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        default_top_k=int(os.getenv("DEFAULT_TOP_K", "5")),
        default_dataset_scope=os.getenv("DEFAULT_DATASET_SCOPE", "ielts_core"),
        rag_backend=os.getenv("RAG_BACKEND", "simple_local"),
        db_backend=os.getenv("DB_BACKEND", "mock_sqlite"),
        calendar_backend=os.getenv("CALENDAR_BACKEND", "mock_calendar"),
        checkpoint_backend=os.getenv("LANGGRAPH_CHECKPOINT_BACKEND", "sqlite"),
        checkpoint_sqlite_path=os.getenv("LANGGRAPH_CHECKPOINT_SQLITE_PATH", "data/langgraph_checkpoints.sqlite"),
        retrieval_max_rounds=int(os.getenv("RETRIEVAL_MAX_ROUNDS", "3")),
        retrieval_max_no_progress_rounds=int(os.getenv("RETRIEVAL_MAX_NO_PROGRESS_ROUNDS", "2")),
        retrieval_top_k_per_round=int(os.getenv("RETRIEVAL_TOP_K_PER_ROUND", "5")),
        retrieval_selected_k=int(os.getenv("RETRIEVAL_SELECTED_K", "3")),
        retrieval_duplicate_rate_threshold=float(os.getenv("RETRIEVAL_DUPLICATE_RATE_THRESHOLD", "0.75")),
        retrieval_gap_fill_target=float(os.getenv("RETRIEVAL_GAP_FILL_TARGET", "0.75")),
        retrieval_novelty_weight=float(os.getenv("RETRIEVAL_NOVELTY_WEIGHT", "0.35")),
        retrieval_history_penalty=float(os.getenv("RETRIEVAL_HISTORY_PENALTY", "0.4")),
        retrieval_min_new_facts=int(os.getenv("RETRIEVAL_MIN_NEW_FACTS", "1")),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_base_url=os.getenv("LLM_BASE_URL", "").rstrip("/"),
        llm_model=os.getenv("LLM_MODEL", ""),
        chunk_llm_model=os.getenv("CHUNK_LLM_MODEL", os.getenv("LLM_MODEL", "")),
        memory_llm_api_key=os.getenv("MEMORY_LLM_API_KEY", os.getenv("LLM_API_KEY", "")),
        memory_llm_base_url=os.getenv("MEMORY_LLM_BASE_URL", os.getenv("LLM_BASE_URL", "")).rstrip("/"),
        memory_llm_model=os.getenv("MEMORY_LLM_MODEL", os.getenv("LLM_MODEL", "")),
        memory_llm_timeout=int(os.getenv("MEMORY_LLM_TIMEOUT", os.getenv("LLM_TIMEOUT", "30"))),
        llm_timeout=int(os.getenv("LLM_TIMEOUT", "30")),
    )
