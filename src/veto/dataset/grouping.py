"""Duplicate / near-duplicate families, used only for grouping, never as model input."""

from __future__ import annotations

import re

import pandas as pd

_URL = re.compile(r"(https?://|www\.)\S+|\b[\w-]+(\.[\w-]+)+/\S*", re.I)
_PHONE = re.compile(r"\+?\d[\d\-\s().]{6,}\d")
_NUM = re.compile(r"\d+")
_TOKEN = re.compile(r"[a-z]+")
_WS = re.compile(r"\s+")
_PLACEHOLDERS = {"URL", "PHONE", "NUM"}


def dedup_key_text(text: str) -> str:
    """Aggressive normalization for duplicate detection (URLs, phones, numbers, case, spaces)."""
    s = _URL.sub(" URL ", text)
    s = _PHONE.sub(" PHONE ", s)
    s = _NUM.sub(" NUM ", s)
    return _WS.sub(" ", s).strip().lower()


def _shingles(key: str, n: int = 3) -> frozenset[str]:
    toks = [t for t in _TOKEN.findall(key)]
    if len(toks) < n:
        return frozenset([" ".join(toks)])
    return frozenset(" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def assign_groups(texts: pd.Series, near_threshold: float = 0.5) -> pd.Series:
    """Return an int group id per row.

    Rows are joined when their aggressively-normalized text is identical, or when
    word-3-shingle Jaccard similarity >= ``near_threshold`` (texts with fewer than
    5 words only join on exact normalized equality). Groups are connected components,
    numbered deterministically by first appearance.
    """
    keys = [dedup_key_text(t) for t in texts]
    shingles = [_shingles(k) for k in keys]
    n_words = [len(_TOKEN.findall(k)) for k in keys]
    parent = list(range(len(keys)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            same = keys[i] == keys[j]
            near = n_words[i] >= 5 and n_words[j] >= 5 and (
                jaccard(shingles[i], shingles[j]) >= near_threshold
            )
            if same or near:
                parent[find(j)] = find(i)
    roots: dict[int, int] = {}
    return pd.Series(
        [roots.setdefault(find(i), len(roots)) for i in range(len(keys))], index=texts.index
    )
