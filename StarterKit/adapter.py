"""
统一处理 vehicle_id (Scania) 与 serial_number (Backblaze) 的转换层。
使下游 pipeline 仅依赖统一列名 Asset_ID，与数据集无关。
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd

DatasetKind = Literal["scania", "backblaze"]

# 统一后的列名，供 features / model 使用
UNIT_ID = "unit_id"
TIME = "time"
ASSET_ID = "Asset_ID"

# 各数据集训练表文件名（与 demo 一致）
SCANIA_TRAIN_OPS = "train_operational_readouts.csv"
BACKBLAZE_TRAIN_SET = "train_set.csv"


class UniversalAdapter:
    """
    统一加载 Scania / Backblaze 数据并将单位标识列统一为 Asset_ID。
    与 .cursorrules 一致：一条可迁移 pipeline，统一列名。
    """

    def __init__(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        """
        data_dir: 可选的数据根目录。若为 None，load_data 时按数据集使用默认子路径。
        """
        self._data_dir: Optional[Path] = Path(data_dir) if data_dir else None

    def load_data(self, dataset_name: str, data_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        根据数据集名称加载训练表并统一单位标识列为 Asset_ID。

        - SCANIA: 读取 train_operational_readouts.csv（列 vehicle_id -> Asset_ID）
        - Backblaze: 读取 train_set.csv，date 解析为日期（列 serial_number -> Asset_ID）

        返回带有统一列 Asset_ID 的 DataFrame。
        """
        name = dataset_name.strip().lower()
        base = self._resolve_data_dir(dataset_name, data_dir)

        if name == "scania":
            path = base / SCANIA_TRAIN_OPS
            df = pd.read_csv(path)
            if "vehicle_id" not in df.columns:
                raise ValueError(f"SCANIA 表缺少 vehicle_id 列: {list(df.columns)[:5]}...")
            df = df.rename(columns={"vehicle_id": ASSET_ID})
        elif name == "backblaze":
            path = base / BACKBLAZE_TRAIN_SET
            df = pd.read_csv(path, parse_dates=["date"])
            if "serial_number" not in df.columns:
                raise ValueError(f"Backblaze 表缺少 serial_number 列: {list(df.columns)[:5]}...")
            df = df.rename(columns={"serial_number": ASSET_ID})
        else:
            raise ValueError(f"不支持的 dataset_name: {dataset_name}，应为 'scania' 或 'backblaze'")

        return df

    def _resolve_data_dir(self, dataset_name: str, data_dir: Optional[Union[str, Path]]) -> Path:
        """解析得到该数据集所在目录（用于拼文件路径）。默认路径相对 adapter.py 所在目录，与运行时的 cwd 无关。"""
        if data_dir is not None:
            return Path(data_dir)
        if self._data_dir is not None:
            return self._data_dir
        # 相对本文件所在目录（StarterKit）解析，这样从项目根或 StarterKit 运行都能找到数据
        _root = Path(__file__).resolve().parent.parent
        name = dataset_name.strip().lower()
        if name == "scania":
            return _root / "Datasets" / "SCANIA"
        if name == "backblaze":
            return _root / "Datasets" / "Backblaze"
        return _root


def to_canonical(
    df: pd.DataFrame,
    dataset: DatasetKind,
    *,
    unit_col: Optional[str] = None,
    time_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    将原始读入的 DataFrame 转为统一列名（Asset_ID, time）。
    若已是 Asset_ID + 时间列（如 load_data 之后），只需把时间列重命名为 time；若为原始列名则同时重命名单位列与时间列。
    """
    ucol, tcol = get_unit_and_time_columns(dataset)
    if unit_col is not None:
        ucol = unit_col
    if time_col is not None:
        tcol = time_col

    if ASSET_ID in df.columns and TIME in df.columns:
        return df.copy()
    if ASSET_ID in df.columns and tcol in df.columns:
        return df.rename(columns={tcol: TIME})
    if ucol in df.columns and tcol in df.columns:
        return df.rename(columns={ucol: ASSET_ID, tcol: TIME})
    raise ValueError(
        f"to_canonical: 需要 (Asset_ID 或 {ucol!r}) 且 (time 或 {tcol!r})，当前列: {list(df.columns)[:8]}..."
    )


def from_canonical(df: pd.DataFrame, dataset: DatasetKind) -> pd.DataFrame:
    """
    将统一列名的 DataFrame 还原为数据集原始列名（用于写回或生成 Kaggle 提交等）。
    """
    if ASSET_ID not in df.columns or TIME not in df.columns:
        raise ValueError(f"from_canonical: 需要列 {ASSET_ID!r} 和 {TIME!r}，当前: {list(df.columns)[:8]}...")
    if dataset == "scania":
        return df.rename(columns={ASSET_ID: "vehicle_id", TIME: "time_step"})
    return df.rename(columns={ASSET_ID: "serial_number", TIME: "date"})


def last_readout(df: pd.DataFrame, dataset: DatasetKind) -> pd.DataFrame:
    """
    先按 Asset_ID 与时间列排序，再为每个资产保留最后一行记录。
    入参可为原始列名或已 to_canonical 的 DataFrame；若为原始列名需传入 dataset。
    返回始终为统一列名（Asset_ID, TIME, ...），与 demo 中按 vehicle_id/time_step 或 serial_number/date 的 last_readout 行为一致。
    """
    if ASSET_ID in df.columns and TIME in df.columns:
        work = df.copy()
    else:
        work = to_canonical(df, dataset)
    work = work.sort_values([ASSET_ID, TIME])
    return work.drop_duplicates(subset=[ASSET_ID], keep="last").reset_index(drop=True)


def get_unit_and_time_columns(dataset: DatasetKind) -> tuple[str, str]:
    """返回 (单位列名, 时间列名)。"""
    if dataset == "scania":
        return "vehicle_id", "time_step"
    return "serial_number", "date"
