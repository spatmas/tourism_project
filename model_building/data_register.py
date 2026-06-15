
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError


# Read token from environment
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found")


repo_id = "Sachinpp04/tourism-project"
repo_type = "dataset"

folder_path = (
    "/content/drive/MyDrive/"
    "PGP_AIML/Projects/Assignment10/"
    "tourism_project/data"
)

api = HfApi(token=HF_TOKEN)


try:
    api.repo_info(
        repo_id=repo_id,
        repo_type=repo_type
    )
    print("Dataset exists")

except RepositoryNotFoundError:

    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        token=HF_TOKEN
    )

    print("Dataset created")


api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type=repo_type
)

print("Upload completed")
