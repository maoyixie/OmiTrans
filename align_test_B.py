import pandas as pd

# Setting the Path
anno_file = "./anno/B_anno.csv"
test_b_file = "./data_cancer/B.tsv"
output_file = "./data_cancer/B_aligned.tsv"

# Read the probe sequence for training
anno_df = pd.read_csv(anno_file, sep=None, engine='python')
if '#id' in anno_df.columns:
    probe_ids = anno_df['#id'].tolist()
elif 'Composite Element REF' in anno_df.columns:
    probe_ids = anno_df['Composite Element REF'].tolist()
elif 'id' in anno_df.columns:
    probe_ids = anno_df['id'].tolist()
else:
    raise ValueError("Cannot find probe id column in B_anno.csv")

# Read test's B.tsv
test_b = pd.read_csv(test_b_file, sep='\t', index_col=0)

# Re-index to ensure that the row order is consistent and missing items are filled with 0
aligned_b = test_b.reindex(probe_ids).fillna(0)

# Writing out files
aligned_b.to_csv(output_file, sep='\t')
print(f"✅ Finished: Output aligned test B.tsv to {output_file}")
