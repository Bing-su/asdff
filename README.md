# asdff for face repairing

Adetailer Stable Diffusion diFFusers pipeline

This version supports fixes for input photos

### from pip install

```
pip install asdff
```

```py
import torch
from asdff import AdPipeline

pipe = AdPipeline.from_pretrained("stablediffusionapi/counterfeit-v30", torch_dtype=torch.float16)
pipe.safety_checker = None
pipe.to("cuda")

common = {"prompt": "masterpiece, best quality, 1girl", "num_inference_steps": 28}
result = pipe(common=common)

images = result[0]
```

### from custom pipeline


```py
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "stablediffusionapi/counterfeit-v30",
    torch_dtype=torch.float16,
    custom_pipeline="Bingsu/adsd_pipeline"
)
pipe.safety_checker = None
pipe.to("cuda")

common = {"prompt": "masterpiece, best quality, 1girl", "num_inference_steps": 28}
result = pipe(common=common)

images = result[0]
```
## example code for repairing faces

```py
import torch
import sys
from PIL import Image

sys.path.insert(0,'/adetailer/asdff')
from asdff import AdPipeline

pipe = AdPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",torch_dtype=torch.float16)
pipe.safety_checker = None
pipe.to("cuda")

img_path ='face_human.png'
res_path = 'save.png')

img=Image.open(img_path).convert("RGB")
result = pipe(common=common,images=[img])
e_time=time.time()

images.save(res_path)

```

## arguments

- `common: Mapping[str, Any] | None`

  Arguments used in txt2img_only

- `txt2img_only: Mapping[str, Any] | None`

  Arguments used in StableDiffusionPipeline

[StableDiffusionPipeline.__call__](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline.__call__)

- `inpaint_only: Mapping[str, Any] | None`

 Arguments to be used only by inpaint. Arguments that overlap with common are overwritten.

- `strength: 0.4`  Used to control the degree of change in the image before and after inpainting.

[StableDiffusionInpaintPipeline.__call__](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline.__call__)

- `detectors: DetectorType | Iterable[DetectorType] | None`

`DetectorType: Callable[[Image.Image], Optional[List[Image.Image]]]`

A Callable that takes a pil Image as input and returns a list (mask) of mask images, or None.

One such Callable, a list of Callables, or None.

If `None`, the `default_detector` is used.

```py
from asdff import AdPipeline

pipe = AdPipeline.from_pretrained(...)
pipe.default_detector
>>> <function asdff.yolo.yolo_detector(image: 'Image.Image', model_path: 'str | None' = None, confidence: 'float' = 0.3) -> 'list[Image.Image] | None'>
```

Usage examples

```py
from functools import partial

import torch
from asdff import AdPipeline, yolo_detector
from huggingface_hub import hf_hub_download

pipe = AdPipeline.from_pretrained("stablediffusionapi/counterfeit-v30", torch_dtype=torch.float16)
pipe.safety_checker = None
pipe.to("cuda")

person_model_path = hf_hub_download("Bingsu/adetailer", "person_yolov8s-seg.pt")
person_detector = partial(yolo_detector, model_path=person_model_path)
common = {"prompt": "masterpiece, best quality, 1girl", "num_inference_steps": 28}
result = pipe(common=common, detectors=[person_detector, pipe.default_detector])
result
```

- `mask_dilation: int, default = 4`
  
After detecting the mask, the cv2.dilate function is applied to grow the mask area, which is the size of the kernel to be applied.

- `mask_blur: int, default = 4`

The kernel size of the Gaussian blur to apply after dilation.

- `mask_padding: int, default = 32`
  
After applying dilation, the image will be cropped by adding this value to the bbox's width and height, and then inpaint will be attempted.
