import pandas as pd

# === Step 1: Load original A and B files ===
A = pd.read_csv("./data/A_origin.tsv", sep="\t", index_col=0)
B = pd.read_csv("./data/B_origin.tsv", sep="\t", index_col=0)

# === Step 2: Get intersection of sample IDs ===
samples_A = set(A.columns)
samples_B = set(B.columns)
shared_samples = sorted(list(samples_A & samples_B))

print(f"🧬 Total shared samples: {len(shared_samples)}")
if len(shared_samples) == 0:
    raise ValueError("❌ No shared samples between A_origin.tsv and B_origin.tsv!")

# === Step 3: Filter and reorder columns ===
A_matched = A[shared_samples]
B_matched = B[shared_samples]

# === Step 4: Save to new files ===
A_matched.to_csv("./data/A.tsv", sep="\t")
B_matched.to_csv("./data/B.tsv", sep="\t")
