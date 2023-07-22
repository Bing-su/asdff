from contextlib import suppress
from pathlib import Path
from string import Template
from tempfile import TemporaryDirectory

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

from asdff import __version__

with Path(__file__).parent.joinpath("pipelines/readme.txt").open(
    "r", encoding="utf-8"
) as text:
    readme_template = Template(text.read())

api = HfApi()
pipeline_py = "pipeline.py"
py_files = Path("asdff").rglob("*.py")

with TemporaryDirectory() as tmp:
    for pipeline in Path("pipelines").glob("*.py"):
        repo_id = f"Bingsu/{pipeline.stem}"
        readme = Path(tmp).joinpath("README.md")
        readme.write_text(readme_template.substitute(repo_id=repo_id), encoding="utf-8")

        api.create_repo(repo_id, repo_type="model", exist_ok=True)

        opr = [
            CommitOperationDelete("asdff", is_folder=True),
            CommitOperationDelete(pipeline_py),
        ]
        with suppress(Exception):
            api.create_commit(repo_id, opr, commit_message="Delete files")

        opr = [CommitOperationAdd(file.as_posix(), file) for file in py_files]
        opr.append(CommitOperationAdd("README.md", readme))
        opr.append(CommitOperationAdd(pipeline_py, pipeline))
        api.create_commit(repo_id, opr, commit_message=f"Upload files: v{__version__}")
