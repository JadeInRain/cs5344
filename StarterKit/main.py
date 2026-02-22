"""
Task 3 管道开发 - 集成原型：向小组成员演示 Adapter / Features / Pipeline 的自动化流转。
一键切换 DATASET_NAME，所有逻辑基于该变量自动跳转。代价矩阵与模型调优由负责模型的成员在框架上填充。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from adapter import ASSET_ID, TIME, UniversalAdapter, last_readout, to_canonical
from features import add_histogram_center, add_rolling_stats, get_binned_labels
from model_pipeline import create_generic_pipeline

# -----------------------------------------------------------------------------
# 一键切换：修改此处即可在 Scania / Backblaze 间切换
# -----------------------------------------------------------------------------
DATASET_NAME: str = "backblaze"  # 或 "backblaze"

# 数据根目录（与 adapter 默认一致）
_ROOT = Path(__file__).resolve().parent.parent
SCANIA_DIR = _ROOT / "Datasets" / "SCANIA"
BACKBLAZE_DIR = _ROOT / "Datasets" / "Backblaze"

# 演示用：滚动窗口列（取部分列以加快运行；可按需扩大）
ROLLING_COLS_SCANIA = ["171_0", "666_0", "427_0", "837_0", "167_0"]
ROLLING_COLS_BACKBLAZE = []  # 若为空则从数值列中取前几列，见下方逻辑
ROLLING_WINDOW = 5


def _load_and_stitch_scania() -> pd.DataFrame:
    """
    SCANIA：先算趋势，后取快照。
    在完整纵向 ops 上计算滚动统计与直方图重心，再 merge 规格与 TTE，最后 last_readout 保留每资产最后一行。
    """
    adapter = UniversalAdapter()
    ops = adapter.load_data("scania", data_dir=SCANIA_DIR)
    ops = to_canonical(ops, "scania")
    # 先在全量纵向数据上计算趋势特征（约 112 万行）
    ops = add_rolling_stats(ops, ROLLING_COLS_SCANIA, window=ROLLING_WINDOW, time_col=TIME)
    ops = add_histogram_center(ops)

    spec = pd.read_csv(SCANIA_DIR / "train_specifications.csv")
    tte = pd.read_csv(SCANIA_DIR / "train_tte.csv")
    ops = (
        ops.merge(spec, left_on=ASSET_ID, right_on="vehicle_id", how="inner")
        .drop(columns=["vehicle_id"])
        .merge(tte, left_on=ASSET_ID, right_on="vehicle_id", how="inner")
        .drop(columns=["vehicle_id"])
    )
    # 特征计算完成后再取每资产最后一行快照
    df = last_readout(ops, "scania")
    return df


def _load_and_stitch_backblaze() -> pd.DataFrame:
    """
    Backblaze：先算趋势与标签相关列，后取快照。
    在全量数据上计算 days_to_failure 与滚动特征，再 last_readout 保留每资产最后一行。
    """
    adapter = UniversalAdapter()
    df = adapter.load_data("backblaze", data_dir=BACKBLAZE_DIR)
    df = to_canonical(df, "backblaze")

    # 全量数据上计算 days_to_failure，保证标签准确
    failure_dates = (
        df.loc[df["failure"] == 1, [ASSET_ID, TIME]]
        .drop_duplicates(subset=[ASSET_ID], keep="last")
        .set_index(ASSET_ID)[TIME]
    )
    df["failure_date"] = df[ASSET_ID].map(failure_dates)
    df["days_to_failure"] = (df["failure_date"] - df[TIME]).dt.days

    # 在全量纵向数据上计算滚动特征
    rolling_cols = ROLLING_COLS_BACKBLAZE
    if not rolling_cols:
        numeric = [
            c for c in df.columns
            if c not in (ASSET_ID, TIME, "failure", "failure_date", "days_to_failure")
            and df[c].dtype in ("float64", "int64")
        ]
        rolling_cols = numeric[:5] if len(numeric) >= 5 else numeric
    if rolling_cols:
        df = add_rolling_stats(df, rolling_cols, window=ROLLING_WINDOW, time_col=TIME)

    # 特征与标签列就绪后再取每资产最后一行
    df = last_readout(df, "backblaze")
    return df


def _prepare_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """从缝合并做完特征的 df 中拆出特征矩阵 X 与标签 y。"""
    name = DATASET_NAME.strip().lower()
    y = get_binned_labels(df, name, time_col=TIME)

    drop = [ASSET_ID, TIME]
    if name == "scania":
        drop.extend(["length_of_study_time_step", "in_study_repair"])
    else:
        drop.extend(["failure", "failure_date", "days_to_failure"])
    feature_cols = [c for c in df.columns if c not in drop]
    X = df[feature_cols].copy()
    return X, y


def main() -> None:
    name = DATASET_NAME.strip().lower()
    if name not in ("scania", "backblaze"):
        raise ValueError(f"DATASET_NAME 应为 'scania' 或 'backblaze'，当前为 {DATASET_NAME!r}")

    # 1) 数据缝合（内含「先算趋势、后取快照」：全量纵向数据上滚动特征 + 直方图/标签，再 last_readout）
    if name == "scania":
        df = _load_and_stitch_scania()
    else:
        df = _load_and_stitch_backblaze()

    print(f"[{DATASET_NAME}] 缝合+特征+快照后行数: {len(df)}, 列数: {len(df.columns)}")

    # 2) 拆 X, y
    X, y = _prepare_X_y(df)
    # 丢弃训练集中全为 NaN 的列，避免 SimpleImputer(median) 无法计算并触发 UserWarning（Backblaze 中部分 SMART 列在子集中无观测值）
    all_nan = [c for c in X.columns if X[c].isna().all()]
    if all_nan:
        X = X.drop(columns=all_nan)
        print(f"[{DATASET_NAME}] 已丢弃 {len(all_nan)} 列全 NaN 特征")
    num_cols = [c for c in X.columns if X[c].dtype != "object"]
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    print(f"[{DATASET_NAME}] 特征维度: {X.shape[1]} (数值: {len(num_cols)}, 分类: {len(cat_cols)}), 样本数: {len(X)}")

    # 3) 管道：占位模型 + create_generic_pipeline
    estimator = RandomForestClassifier(n_estimators=50, random_state=0)
    pipe = create_generic_pipeline(estimator, DATASET_NAME, num_cols, cat_cols)

    # 4) 演示 fit 与 predict
    pipe.fit(X, y)
    y_pred = pipe.predict(X.iloc[: min(5, len(X))])
    print(f"[{DATASET_NAME}] 前 5 行预测结果: {y_pred.tolist()}")

    # 可选：由负责模型的成员在此处接入代价矩阵与 evaluate_cost_sensitive
    # from model_pipeline import get_cost_matrix, evaluate_cost_sensitive
    # cost = get_cost_matrix(name)
    # metrics = evaluate_cost_sensitive(y_true, y_pred, name)


if __name__ == "__main__":
    main()
