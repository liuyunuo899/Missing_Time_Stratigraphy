
import pandas as pd

raw_dir = r"F:\刘雨诺\论文投稿\投稿【nature geoscience】\data\raw"

units = pd.read_csv(
    raw_dir + r"\macrostrat_unit_boundaries_all.csv"
)

print("数据字段：")
print(units.columns.tolist())

# 查找可能使用地质时间区间ID的字段
id_columns = [
    c for c in units.columns
    if "int_id" in c.lower() or "interval" in c.lower()
]

print("\n与年代区间有关的字段：")
print(id_columns)

# 检查新增区间ID
for col in id_columns:
    values = units[col].astype(str)
    count = values.str.contains(
        r"(?<!\d)1802(?!\d)", regex=True
    ).sum()

    print(f"{col}: 引用1802的记录数 = {count}")