
# Import libraries
import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi


# ==========================
# Config
# ==========================
HF_TOKEN = os.getenv("HF_TOKEN")

repo_id = "Sachinpp04/tourism-project"

save_path = (
    "/content/drive/MyDrive/"
    "PGP_AIML/Projects/Assignment10/"
    "tourism_project/data"
)

api = HfApi(token=HF_TOKEN)


# ==========================
# Load dataset from HF
# ==========================
dataset = load_dataset(
    repo_id,
    split="train",
    token=HF_TOKEN
)

df = dataset.to_pandas()

print(f"Dataset loaded | Shape: {df.shape}")


# ==========================
# Preprocess
# ==========================
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

df.drop(
    columns=["CustomerID"],
    errors="ignore",
    inplace=True
)

if "Gender" in df.columns:
    df["Gender"] = (
        df["Gender"]
        .replace(
            "Fe Male",
            "Female"
        )
    )


# ==========================
# Split
# ==========================
X = df.drop(
    "ProdTaken",
    axis=1
)

y = df["ProdTaken"]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# ==========================
# Save + Upload
# ==========================
files = {
    "Xtrain.csv": Xtrain,
    "Xtest.csv": Xtest,
    "ytrain.csv": ytrain,
    "ytest.csv": ytest
}

for name, data in files.items():

    path = f"{save_path}/{name}"

    data.to_csv(
        path,
        index=False
    )

    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=name,
        repo_id=repo_id,
        repo_type="dataset"
    )

    print(f"Uploaded: {name}")


# Verification
print("\nGenerated files:")
print(os.listdir(save_path))

print("\nCompleted")
