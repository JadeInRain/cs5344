from adapter import UniversalAdapter, to_canonical, from_canonical, last_readout, ASSET_ID, TIME

adapter = UniversalAdapter()
df = adapter.load_data("backblaze", data_dir="Datasets/Backblaze")
df = to_canonical(df, "backblaze")    # time_step -> time (TIME)
df_last = last_readout(df, "backblaze")  # 每资产最后一行，列仍为 Asset_ID, TIME, ...
print(df.info())
print(df.columns)
print(df_last['Asset_ID'].is_unique)
# 写回 / 提交时还原列名
df_raw = from_canonical(df_last, "backblaze")  # vehicle_id, time_step
print(df_raw.head())