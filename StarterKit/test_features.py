import pandas as pd
import numpy as np
from adapter import UniversalAdapter, to_canonical, last_readout, ASSET_ID, TIME
from features import add_rolling_stats, get_binned_labels

# 1. 加载数据
adapter = UniversalAdapter()
# 注意：此时加载的是 train_set.csv
df = adapter.load_data("backblaze") 
df = to_canonical(df, "backblaze")
print(df[TIME].dtype)
print("--- 步骤 1: 检查 Backblaze 基础格式 ---")
print(f"原始列名: {df.columns[:5].tolist()}") # 应该包含 Asset_ID 和 time (即原 date)

# 2. 纵向特征测试 (Rolling)
# 选两个硬盘通用的 SMART 指标进行测试
smart_cols = ["smart_1_raw", "smart_5_raw"] 
df_feat = add_rolling_stats(df.head(10000), smart_cols, window=7) # 抽样前1万行快一点
df_feat.groupby(ASSET_ID).size().max()
print("\n--- 步骤 2: 验证滚动特征 ---")
new_col = "smart_5_raw_rolling_mean"
if new_col in df_feat.columns:
    print(f"✅ {new_col} 生成成功")
    # 检查是否有非空值 (因为 min_periods=1，第一行就该有值)
    print(f"非空值数量: {df_feat[new_col].notnull().sum()}")
else:
    print(f"❌ 滚动特征列丢失")

# 3. 标签生成测试 (关键：需要先算 RUL)
print("\n--- 步骤 3: 验证标签分桶 ---")
# 因为 train_set 原始表里可能没有 days_to_failure，我们模拟一个测试
# 在实际 Pipeline 中，你需要先像 demo 里的 prepare_training_set 那样算出这个列
if "failure" in df_feat.columns:
    # 简单模拟：假设 failure=1 的那行 RUL=0
    df_feat["days_to_failure"] = np.where(df_feat["failure"] == 1, 0, 100)
    y = get_binned_labels(df_feat, "backblaze", rul_col="days_to_failure")
    print("标签分布 (0-2):")
    print(y.value_counts().sort_index())
    print(y.unique())