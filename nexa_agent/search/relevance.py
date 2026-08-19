"""Deterministic search-result relevance gate for noisy free providers."""

from __future__ import annotations

import re

from .base import SearchResult


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9.-]*|[\u4e00-\u9fff]{2,}", re.I)
_QUOTE_RE = re.compile(r'["“”\']([^"“”\']{2,})["“”\']')
_GENERIC_TERMS = {
    "company",
    "startup",
    "official",
    "website",
    "careers",
    "career",
    "jobs",
    "hiring",
    "review",
    "scam",
    "fraud",
    "funding",
    "engineer",
    "2025",
    "2026",
}


def _tokens(text: str) -> set[str]:
    return {
        token.lower().strip(".-")
        for token in _TOKEN_RE.findall(text)
        if token.strip(".-")
    }


def filter_relevant_results(
    query: str,
    results: list[SearchResult],
) -> list[SearchResult]:
    """Drop obvious lexical mismatches without claiming semantic relevance.

    Quoted entities are treated as required anchors. Otherwise at least one
    non-generic query token must appear in title, URL, or snippet. When the
    query has no discriminative token, results are left unchanged.
    """

    quoted = [_tokens(item) for item in _QUOTE_RE.findall(query)]
    quoted = [tokens for tokens in quoted if tokens]
    query_tokens = _tokens(query)
    specific = {
        token
        for token in query_tokens
        if token not in _GENERIC_TERMS and len(token) >= 3
    }
    if not quoted and not specific:
        return results

    kept: list[SearchResult] = []
    for result in results:
        haystack = _tokens(f"{result.title} {result.url} {result.snippet}")
        quoted_ok = all(anchor.issubset(haystack) for anchor in quoted)
        specific_ok = bool(specific.intersection(haystack)) if specific else True
        if quoted_ok and specific_ok:
            kept.append(result)
    return kept
