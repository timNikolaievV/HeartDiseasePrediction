import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/heart_cleveland_upload.csv")

print("DF shape:", df.shape)
print("\nColumn info:")
print(df.info())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values per column:")
print(df.isna().sum().sort_values(ascending=False))

print("\nSummary statistics (numeric columns):")
print(df.describe())

num_duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {num_duplicates}")

categorical_cols = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
    "condition"
]

print("\nUnique values in categorical columns:")
for col in categorical_cols:
    print(f"{col}: {df[col].unique()}")

for col in categorical_cols:
    if df[col].isna().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"Filled NaNs in {col} with {mode_val!r}")


for col in categorical_cols:
    counts = df[col].value_counts()

    plt.figure(figsize=(6, 4))
    bars = plt.bar(counts.index.astype(str), counts.values)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(height),
            ha='center',
            va='bottom'
        )

    plt.title(f"Count of Categories in {col}")
    plt.xlabel(col)
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.show()