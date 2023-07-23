from contextlib import suppress
from pathlib import Path
from string import Template
from tempfile import TemporaryDirectory

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

from asdff import __version__

root = Path(__file__).parent

with root.joinpath("pipelines/template.txt").open("r", encoding="utf-8") as text:
    readme_template = Template(text.read())

api = HfApi()
pipeline_py = "pipeline.py"
py_files = list(root.joinpath("asdff").rglob("*.py"))

m = {"adsd_pipeline": "stablediffusionapi/counterfeit-v30"}

with TemporaryDirectory() as tmp:
    for pipeline in root.joinpath("pipelines").glob("*.py"):
        repo_id = f"Bingsu/{pipeline.stem}"
        hf_id = m[pipeline.stem]
        text = readme_template.substitute(hf_id=hf_id, repo_id=repo_id)
        readme = Path(tmp, "README.md")
        readme.write_text(text, encoding="utf-8")

        api.create_repo(repo_id, repo_type="model", exist_ok=True)

        opr = [
            CommitOperationDelete("asdff", is_folder=True),
            CommitOperationDelete(pipeline_py),
        ]
        with suppress(Exception):
            api.create_commit(repo_id, opr, commit_message="Delete files")

        opr = [
            CommitOperationAdd(file.relative_to(root).as_posix(), file)
            for file in py_files
        ]
        opr.append(CommitOperationAdd("README.md", readme))
        opr.append(CommitOperationAdd(pipeline_py, pipeline))
        api.create_commit(repo_id, opr, commit_message=f"Upload files: v{__version__}")
