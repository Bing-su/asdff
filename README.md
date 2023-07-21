# asdff

Adetailer Stable Diffusion diFFusers pipeline

## 예시

```py
import torch
from asdff import AdPipeline

pipe = AdPipeline.from_pretrained("stablediffusionapi/counterfeit-v30", torch_dtype=torch.float16)
pipe.safety_checker = None
pipe = pipe.to("cuda")

common = {"prompt": "masterpiece, best quality, 1girl", "num_inference_steps": 28}
result = pipe(common=common)

images = result[0]
```
