"""测试 model_pipeline：需先有 X_train, y_train（及可选 X_val）。此处用 adapter 加载 Scania 并构造示例。"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from adapter import UniversalAdapter, to_canonical, last_readout, ASSET_ID, TIME
from model_pipeline import create_generic_pipeline

# 1) 加载数据并得到「每资产最后一行」的表格（带 Asset_ID, time）
try:
    adapter = UniversalAdapter()
    df = adapter.load_data("scania")
    df = to_canonical(df, "scania")
    df = last_readout(df, "scania")
    drop = [ASSET_ID, TIME]
    feature_cols = [c for c in df.columns if c not in drop and df[c].dtype in ("float64", "int64", "object")]
    X_train = df[feature_cols].head(500)
    y_train = pd.Series(0, index=X_train.index).astype(int)
except Exception as e:
    # 无数据时用合成数据仅测管道
    print("Scania 数据不可用，使用合成数据测试管道:", e)
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0] * 20, "b": [0, 1, 0] * 20, "c": ["x", "y", "x"] * 20})
    y_train = pd.Series([0, 1, 0] * 20)

# 3) 从训练集推断列类型
num_cols = [c for c in X_train.columns if X_train[c].dtype != "object"]
cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]

# 4) 创建管道并 fit / predict
clf = RandomForestClassifier(n_estimators=50, random_state=0)
pipe = create_generic_pipeline(clf, "scania", num_cols, cat_cols)
pipe.fit(X_train, y_train)

X_val = X_train.head(10)
y_pred = pipe.predict(X_val)
print("Pipeline fit/predict OK. Predictions shape:", y_pred.shape)
