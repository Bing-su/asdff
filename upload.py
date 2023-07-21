from huggingface_hub import CommitOperationDelete, HfApi

api = HfApi()
repo_id = "Bingsu/adetailer_pipeline"
pipeline_py = "pipeline.py"

opr = [
    CommitOperationDelete("asdff", is_folder=True),
    CommitOperationDelete(pipeline_py),
]

api.create_commit(repo_id, opr, commit_message="Delete files")
api.upload_folder(
    repo_id=repo_id, folder_path="asdff", path_in_repo="/asdff", allow_patterns="*.py"
)
api.upload_file(repo_id=repo_id, path_or_fileobj=pipeline_py, path_in_repo=pipeline_py)
