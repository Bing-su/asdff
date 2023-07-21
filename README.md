# asdff

Adetailer Stable Diffusion diFFusers pipeline

## 예시

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

pip 설치 필요없음

```py
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("stablediffusionapi/counterfeit-v30", torch_dtype=torch.float16, custom_pipeline="Bingsu/adetailer_pipeline")
pipe.safety_checker = None
pipe.to("cuda")

common = {"prompt": "masterpiece, best quality, 1girl", "num_inference_steps": 28}
result = pipe(common=common)

images = result[0]
```

## arguments

- `common`: `Dict[str, Any] | None`

txt2img와 inpaint에서 공통적으로 사용할 인자들

- `txt2img_only`: `Dict[str, Any] | None`

txt2img에서만 사용할 인자. common과 겹치는 인자는 덮어씁니다.

[`StableDiffusionPipeline.__call__`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline.__call__)

- `inpaint_only`: `Dict[str, Any] | None`

inpaint에서만 사용할 인자. common과 겹치는 인자는 덮어씁니다.

`strength: 0.4`가 기본값으로 적용됩니다.

[`StableDiffusionInpaintPipeline.__call__`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline.__call__)

- `detector`: `Callable[[Image.Image], list[Image.Image] | None] | None`

pil Image를 입력으로 받아 마스크 이미지의 리스트(마스크), 또는 None을 반환하는 Callable

`None`일경우, `default_detector`가 사용됩니다.

```py
from asdff import AdPipeline

pipe = AdPipeline.from_pretrained(...)
pipe.default_detector
>>> <function asdff.yolo.yolo_detector(image: 'Image.Image', model_path: 'str | None' = None, confidence: 'float' = 0.3) -> 'list[Image.Image] | None'>
```

사용 예시

```py
import torch
from asdff import AdPipeline, yolo_detector
from huggingface_hub import hf_hub_download

pipe = AdPipeline.from_pretrained("stablediffusionapi/counterfeit-v30", torch_dtype=torch.float16)
pipe.safety_checker = None
pipe.to("cuda")

person_model_path = hf_hub_download("Bingsu/adetailer", "person_yolov8s-seg.pt")
common = {"prompt": "masterpiece, best quality, 1girl", "num_inference_steps": 28}
result = pipe(common=common, detector=yolo_detector, detector_kwargs={"model_path": person_model_path})
result
```

- `detector_kwargs`: `Dict[str, Any] | None`

`detector`의 키워드 인자

- `mask_dilation`: int, default = 4

마스크 감지 후, cv2.dilate 함수를 적용해 마스크 영역을 키우는 데, 이 때 적용할 커널의 크기.

- `mask_padding`: int, default = 32

dilation 적용 후 이 값만큼 bbox의 가로세로 영역을 더해서 이미지를 자른 뒤, inpaint를 시도하게 됩니다.
