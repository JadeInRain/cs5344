"""
滚动窗口特征、直方图重心与分箱标签。
所有计算按 Asset_ID 分组，保证变长序列下无跨资产干扰。
"""
from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np
import pandas as pd

from adapter import ASSET_ID, TIME

DatasetName = Literal["scania", "backblaze"]

# SCANIA 直方图列 167_0 .. 167_9
SCANIA_HIST_COLS = [f"167_{i}" for i in range(10)]


def add_rolling_stats(
    df: pd.DataFrame,
    columns: List[str],
    window: int = 5,
    time_col: str = TIME,
) -> pd.DataFrame:
    """
    按 Asset_ID 分组，在时间维上对指定列计算滚动 mean、std 与 diff（当前值减前值，表趋势）。
    不产生跨资产干扰；组内按 time_col 排序后再 rolling。
    """
    out = df.copy()
    if ASSET_ID not in out.columns:
        raise ValueError(f"add_rolling_stats: 需要列 {ASSET_ID!r}")
    sort_cols = [ASSET_ID, time_col] if time_col in out.columns else [ASSET_ID]
    out = out.sort_values(sort_cols).reset_index(drop=True)

    for col in columns:
        if col not in out.columns:
            continue
        g = out.groupby(ASSET_ID)[col]
        out[f"{col}_rolling_mean"] = g.transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        out[f"{col}_rolling_std"] = g.transform(
            lambda s: s.rolling(window, min_periods=1).std()
        )
        out[f"{col}_diff"] = g.diff()

    return out


def add_histogram_center(df: pd.DataFrame) -> pd.DataFrame:
    """
    SCANIA 直方图特性：对列 167_0 到 167_9 计算加权重心。
    桶索引 0..9 作为位置，列值作为权重： COM = sum(i * v_i) / sum(v_i)，空和时返回 NaN。
    """
    out = df.copy()
    present = [c for c in SCANIA_HIST_COLS if c in out.columns]
    if not present:
        return out

    vals = out[present].to_numpy(dtype=float)
    weights = np.nansum(vals, axis=1, keepdims=True)
    np.putmask(vals, np.isnan(vals), 0.0)
    positions = np.arange(len(present), dtype=float)
    com = np.dot(vals, positions) / np.where(weights.ravel() > 0, weights.ravel(), np.nan)
    out["167_hist_com"] = com
    return out


def get_binned_labels(
    df: pd.DataFrame,
    dataset_name: str,
    *,
    rul_col: Optional[str] = None,
    time_col: str = TIME,
) -> pd.Series:
    """
    生成分箱序数标签。
    - SCANIA: RUL <= 6 -> 4, <= 12 -> 3, <= 24 -> 2, <= 48 -> 1；其余及右删失为 0。
    - Backblaze: RUL <= 10 -> 2, <= 20 -> 1；其余为 0。

    RUL 来源（二选一）：
    - rul_col 指定列名时，直接使用该列作为 RUL（或 Backblaze 的 days_to_failure）。
    - 未指定时，SCANIA 需有 length_of_study_time_step、in_study_repair 与 time 列，据此计算 RUL；Backblaze 需有 days_to_failure 列。
    """
    name = dataset_name.strip().lower()
    if name not in ("scania", "backblaze"):
        raise ValueError(f"dataset_name 应为 'scania' 或 'backblaze'，得到 {dataset_name!r}")

    if rul_col is not None and rul_col in df.columns:
        rul = np.asarray(df[rul_col], dtype=float)
    elif name == "scania":
        rul = _rul_scania(df, time_col)
    elif name == "backblaze":
        rul = _rul_backblaze(df)
    else:
        raise ValueError("未提供 rul_col 且无法从 df 推断 RUL 列")

    if name == "scania":
        y = np.zeros(len(rul), dtype=int)
        finite = np.isfinite(rul)
        r = rul.copy()
        np.putmask(r, ~finite, np.nan)
        y[(r <= 48) & finite] = 1
        y[(r <= 24) & finite] = 2
        y[(r <= 12) & finite] = 3
        y[(r <= 6) & finite] = 4
        return pd.Series(y, index=df.index)
    else:
        y = np.zeros(len(rul), dtype=int)
        finite = np.isfinite(rul)
        r = rul.copy()
        np.putmask(r, ~finite, np.nan)
        y[(r <= 20) & finite] = 1
        y[(r <= 10) & finite] = 2
        return pd.Series(y, index=df.index)


def _rul_scania(df: pd.DataFrame, time_col: str) -> np.ndarray:
    """从 length_of_study_time_step、in_study_repair 与 time 列计算 RUL；右删失为 NaN。"""
    for c in ("length_of_study_time_step", "in_study_repair", time_col):
        if c not in df.columns:
            raise ValueError(f"get_binned_labels(SCANIA): 需要列 {c!r}")
    T = np.asarray(df["length_of_study_time_step"], dtype=float)
    t = np.asarray(df[time_col], dtype=float)
    rep = np.asarray(df["in_study_repair"], dtype=float)
    rul = np.where(rep == 1, np.maximum(T - t, 0.0), np.nan)
    return rul


def _rul_backblaze(df: pd.DataFrame) -> np.ndarray:
    """使用 days_to_failure 列作为 RUL（天数）。"""
    if "days_to_failure" not in df.columns:
        raise ValueError("get_binned_labels(Backblaze): 需要列 'days_to_failure' 或传入 rul_col")
    return np.asarray(df["days_to_failure"], dtype=float)
