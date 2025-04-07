import pandas as pd

# Loading methylation data
B_df = pd.read_csv('./data/B.tsv', sep='\t', index_col=0)

# Loading annotation files
anno = pd.read_csv('./anno/B_anno.csv', sep='\t', index_col=0)
anno = anno.rename(columns={"chrom": "CHR"})  # If you use "CHR" in your code

# Check probe ID matching
missing_ids = B_df.index.difference(anno.index)
print(f"There are {len(B_df)} probes in total, {len(missing_ids)} are not found in anno.")

# The sample output is missing the first few
if len(missing_ids) > 0:
    print("Missing probe IDs(previous 10):")
    print(missing_ids[:10])
