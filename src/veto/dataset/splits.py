"""Deterministic grouped, stratified splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _group_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("group_id").agg(n=("label", "size"), pos=("label", "sum")).reset_index()
    return g


def grouped_stratified_folds(df: pd.DataFrame, k: int, seed: int) -> pd.Series:
    """Assign each row to one of ``k`` folds; groups never straddle folds.

    Positive-bearing groups are distributed first (round-robin over a shuffled order) so each
    fold gets positives whenever ``positives >= k``; remaining groups go to the currently
    smallest fold. Deterministic for a given ``seed``.
    """
    rng = np.random.default_rng(seed)
    g = _group_table(df)
    pos_groups = g[g["pos"] > 0].sample(frac=1, random_state=int(rng.integers(1 << 31)))
    neg_groups = g[g["pos"] == 0].sample(frac=1, random_state=int(rng.integers(1 << 31)))
    assign: dict[int, int] = {}
    sizes = [0] * k
    pos_counts = [0] * k
    for row in pos_groups.itertuples():
        f = min(range(k), key=lambda i: (pos_counts[i], sizes[i], i))
        assign[row.group_id] = f
        sizes[f] += row.n
        pos_counts[f] += int(row.pos)
    for row in neg_groups.itertuples():
        f = min(range(k), key=lambda i: (sizes[i], i))
        assign[row.group_id] = f
        sizes[f] += row.n
    return df["group_id"].map(assign)


def train_val_test_split(df: pd.DataFrame, seed: int) -> pd.Series:
    """Label each row 'train' / 'validation' / 'test' (~60/20/20), grouped and stratified.

    Implemented as 5 grouped stratified folds. Positive groups are dealt out first in fold
    order, so validation then test receive positives before train does (it matters when
    positives are scarce).
    """
    folds = grouped_stratified_folds(df, 5, seed)
    names = {0: "validation", 1: "test", 2: "train", 3: "train", 4: "train"}
    return folds.map(names)
