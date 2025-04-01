import pandas as pd

# 加载 methylation 数据
B_df = pd.read_csv('./data/B.tsv', sep='\t', index_col=0)

# 加载注释文件
anno = pd.read_csv('./anno/B_anno.csv', sep='\t', index_col=0)
anno = anno.rename(columns={"chrom": "CHR"})  # 如果你代码中用的是 "CHR"

# 检查 probe ID 匹配情况
missing_ids = B_df.index.difference(anno.index)
print(f"共有 {len(B_df)} 个 probes，其中 {len(missing_ids)} 个在 anno 中找不到。")

# 示例输出缺失的前几个
if len(missing_ids) > 0:
    print("缺失的 probe IDs（前 10 个）:")
    print(missing_ids[:10])
