import pandas as pd

# File Path
train_b_path = "./data_BRCA/B.tsv"
test_b_path = "./data_cancer/B.tsv"
output_path = "./data_cancer/B_fixed.tsv"

# Read data (index_col=0 is the first column as the index, i.e. probe ID)
train_b = pd.read_csv(train_b_path, sep='\t', index_col=0)
test_b = pd.read_csv(test_b_path, sep='\t', index_col=0)

# Rearrange the test data according to the training probe order
aligned_test_b = test_b.reindex(train_b.index)

# Fill missing values ​​(NA) with 0 or other values
aligned_test_b.fillna(0, inplace=True)  # Or change inplace=False to another value

# Add the probe ID back to the first column and write
aligned_test_b.to_csv(output_path, sep='\t')

print(f"✅ Aligned test B saved to: {output_path}")
