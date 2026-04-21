"""Deprecated compatibility shim for the old ranker path."""

from project.rag.orchestration.novelty_ranker import NoveltyRanker, RankedCandidate

QdrantNoveltyRanker = NoveltyRanker
