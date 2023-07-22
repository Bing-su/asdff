---
license: agpl-3.0
tags:
- pytorch
- diffusers
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
