from contextlib import suppress
from pathlib import Path
from string import Template
from tempfile import TemporaryDirectory

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

from asdff import __version__

_readme_text = """---
license: agpl-3.0
---

# ADetailer Pipeline

```py
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "stablediffusionapi/counterfeit-v30",
    torch_dtype=torch.float16,
    custom_pipeline="$repo_id"
)
pipe.safety_checker = None
pipe.to("cuda")

common = {"prompt": "masterpiece, best quality, 1girl", "num_inference_steps": 28}
result = pipe(common=common)

images = result[0]
```

github: https://github.com/Bing-su/asdff
"""
readme_template = Template(_readme_text)

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
