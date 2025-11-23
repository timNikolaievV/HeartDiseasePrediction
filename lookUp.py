import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/heart_cleveland_upload.csv")

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
    plt.show()