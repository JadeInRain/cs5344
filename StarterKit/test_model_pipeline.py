from sklearn.ensemble import RandomForestClassifier
from model_pipeline import create_generic_pipeline, get_preprocess_config

# 从训练集推断列类型（与 demo 一致）
num_cols = [c for c in X_train.columns if X_train[c].dtype != "object"]
cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]

# 一键得到「预处理 + 任意估计器」的 Pipeline
clf = RandomForestClassifier(n_estimators=200, random_state=0)
pipe = create_generic_pipeline(clf, "scania", num_cols, cat_cols)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_val)