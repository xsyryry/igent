"""Formal RAG tool entrypoint for the agent layer."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from project.rag.simple_rag import SimpleRAGService


@lru_cache(maxsize=1)
def _get_rag_service() -> SimpleRAGService:
    """Build the shared local RAG service once per process."""

    return SimpleRAGService.from_config()


def _to_legacy_documents(retrieved_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map retrieved references into the legacy document shape used by nodes."""

    documents: list[dict[str, Any]] = []
    for item in retrieved_docs:
        chunks = item.get("chunks", [])
        joined_content = "\n".join(
            chunk for chunk in chunks if isinstance(chunk, str) and chunk.strip()
        ).strip()
        if not joined_content:
            continue
        documents.append(
            {
                "source": str(item.get("source", "")),
                "title": str(item.get("source") or item.get("id") or "RAG Reference"),
                "content": joined_content,
                "score": float(item.get("score") or 0.0),
            }
        )
    return documents


def retrieve_knowledge(
    question: str,
    dataset_scope: str | None = None,
    top_k: int = 5,
    mode: str = "mix",
    filters: dict[str, Any] | None = None,
    user_id: str | None = None,
    banned_doc_ids: list[str] | None = None,
    banned_chunk_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Retrieve structured knowledge from the local simple RAG service."""

    service = _get_rag_service()
    retrieval_result = service.query(
        question=question,
        top_k=top_k,
        dataset_scope=dataset_scope,
        mode=mode,
        filters=filters,
        user_id=user_id,
        banned_doc_ids=banned_doc_ids,
        banned_chunk_ids=banned_chunk_ids,
    )
    retrieved_docs = retrieval_result.get("retrieved_docs", [])
    return {
        "question": question,
        "answer": retrieval_result.get("answer", ""),
        "retrieved_docs": retrieved_docs,
        "backend": retrieval_result.get("backend", "simple_rag"),
        "query_mode": retrieval_result.get("query_mode", mode),
        "documents": _to_legacy_documents(retrieved_docs),
        "mode": retrieval_result.get("mode", "local"),
        "message": retrieval_result.get("message", ""),
    }
