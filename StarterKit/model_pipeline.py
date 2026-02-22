"""
自动化工作流框架：抽象 Pipeline、预处理（SimpleImputer + RobustScaler）、
工厂函数 create_generic_pipeline(estimator)，以及按 dataset_name 的配置切换。
代价矩阵与具体模型由队友提供，本模块只提供管道封装。
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional, Protocol, TypedDict

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from adapter import DatasetKind

# -----------------------------------------------------------------------------
# 代价敏感评估（接口保留，具体矩阵由队友实现）
# -----------------------------------------------------------------------------

SCANIA_COST_SHAPE = (5, 5)
BACKBLAZE_COST_SHAPE = (3, 3)


def get_cost_matrix(dataset: DatasetKind) -> np.ndarray:
    """返回该数据集的代价矩阵 C；具体数值由队友提供。"""
    raise NotImplementedError(
        "get_cost_matrix: 由队友实现 Scania 5x5 / Backblaze 3x3 代价矩阵"
    )


def total_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost: Optional[np.ndarray] = None,
    dataset: Optional[DatasetKind] = None,
) -> int:
    """计算总代价；cost 或 dataset 由队友提供。"""
    raise NotImplementedError("total_cost: 由队友实现")

  
def evaluate_cost_sensitive(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset: DatasetKind,
) -> dict[str, float | int]:
    """返回 total_cost 等指标；由队友实现。"""
    raise NotImplementedError("evaluate_cost_sensitive: 由队友实现")


class CostSensitiveEvaluator(Protocol):
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset: DatasetKind,
    ) -> dict[str, float | int]:
        ...


# -----------------------------------------------------------------------------
# 预处理配置：按 dataset_name 切换
# -----------------------------------------------------------------------------


class PreprocessConfig(TypedDict):
    imputer_strategy: str
    scaler_quantile_range: tuple[int, int]


def get_preprocess_config(dataset_name: str) -> PreprocessConfig:
    """
    根据 dataset_name 返回预处理参数，用于 SimpleImputer 与 RobustScaler。
    可通过此函数扩展不同数据集的差异化配置。
    """
    name = dataset_name.strip().lower()
    if name == "scania":
        return PreprocessConfig(
            imputer_strategy="median",
            scaler_quantile_range=(25, 75),
        )
    if name == "backblaze":
        return PreprocessConfig(
            imputer_strategy="median",
            scaler_quantile_range=(25, 75),
        )
    return PreprocessConfig(
        imputer_strategy="median",
        scaler_quantile_range=(25, 75),
    )


def _build_preprocess(
    num_cols: List[str],
    cat_cols: List[str],
    config: PreprocessConfig,
) -> ColumnTransformer:
    """构建 ColumnTransformer：数值列 Imputer -> RobustScaler，分类列 Imputer -> OneHotEncoder。"""
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=config["imputer_strategy"])),
            ("scaler", RobustScaler(quantile_range=config["scaler_quantile_range"])),
        ]
    )
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformers: List[tuple[str, Any, List[str]]] = [
        ("num", num_pipeline, num_cols),
    ]
    if cat_cols:
        transformers.append(("cat", cat_pipeline, cat_cols))
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# -----------------------------------------------------------------------------
# 工厂函数：一键生成「预处理 + 任意估计器」的 Pipeline
# -----------------------------------------------------------------------------

DatasetName = Literal["scania", "backblaze"]


def create_generic_pipeline(
    estimator: Any,
    dataset_name: str,
    num_cols: List[str],
    cat_cols: Optional[List[str]] = None,
) -> Pipeline:
    """
    将队友提供的任意分类器包装进标准预处理管道，并通过 dataset_name 自动切换预处理配置。

    - 预处理：数值列 SimpleImputer -> RobustScaler，分类列 SimpleImputer -> OneHotEncoder。
    - 参数由 get_preprocess_config(dataset_name) 决定，满足 Task 1 标准化要求。

    Parameters
    ----------
    estimator : 任意实现 fit / predict 的估计器（如 RandomForestClassifier、XGBClassifier）。
    dataset_name : "scania" | "backblaze"，用于选择预处理参数。
    num_cols : 数值特征列名列表（用于 Imputer + RobustScaler）。
    cat_cols : 分类特征列名列表（用于 Imputer + OneHotEncoder）；可为空或 None。

    Returns
    -------
    sklearn.pipeline.Pipeline
        steps=[("preprocess", ColumnTransformer), ("model", estimator)]
    """
    name = dataset_name.strip().lower()
    if name not in ("scania", "backblaze"):
        raise ValueError(f"dataset_name 应为 'scania' 或 'backblaze'，得到 {dataset_name!r}")
    cat_cols = cat_cols or []
    config = get_preprocess_config(name)
    preprocess = _build_preprocess(num_cols, cat_cols, config)
    return Pipeline(steps=[("preprocess", preprocess), ("model", estimator)])
