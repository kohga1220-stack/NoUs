"""
Inter-rater reliability — Intraclass Correlation Coefficients (Shrout & Fleiss, 1979).

Rows = targets (hypotheses), columns = raters (LLM judges or humans).
"""
from __future__ import annotations
import numpy as np


def icc(ratings) -> dict[str, float]:
    """
    Compute the six Shrout & Fleiss ICCs for a complete n×k matrix.

    Returns keys: ICC1, ICC2, ICC3 (single rater) and ICC1k, ICC2k, ICC3k (mean of k raters).
      ICC1 : one-way random
      ICC2 : two-way random, absolute agreement
      ICC3 : two-way mixed, consistency
    """
    x = np.asarray(ratings, dtype=float)
    if x.ndim != 2:
        raise ValueError("ratings must be a 2-D targets × raters matrix")
    if np.isnan(x).any():
        raise ValueError("ratings contain missing values; drop incomplete rows first")
    n, k = x.shape
    if n < 2 or k < 2:
        raise ValueError("need at least 2 targets and 2 raters")

    gm = x.mean()
    row_m = x.mean(axis=1)
    col_m = x.mean(axis=0)

    ss_total = ((x - gm) ** 2).sum()
    ss_rows  = k * ((row_m - gm) ** 2).sum()
    ss_cols  = n * ((col_m - gm) ** 2).sum()
    ss_err   = ss_total - ss_rows - ss_cols

    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    msw = (ss_total - ss_rows) / (n * (k - 1))

    def _safe(num, den):
        return float(num / den) if den != 0 else float("nan")

    return {
        "ICC1":  _safe(msr - msw, msr + (k - 1) * msw),
        "ICC2":  _safe(msr - mse, msr + (k - 1) * mse + k * (msc - mse) / n),
        "ICC3":  _safe(msr - mse, msr + (k - 1) * mse),
        "ICC1k": _safe(msr - msw, msr),
        "ICC2k": _safe(msr - mse, msr + (msc - mse) / n),
        "ICC3k": _safe(msr - mse, msr),
        "n_targets": n,
        "n_raters":  k,
    }


def interpret(value: float) -> str:
    """Koo & Li (2016) guideline for ICC magnitude."""
    if value != value:  # NaN
        return "undefined"
    if value < 0.5:
        return "poor"
    if value < 0.75:
        return "moderate"
    if value < 0.9:
        return "good"
    return "excellent"
