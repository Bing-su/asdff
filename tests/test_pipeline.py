import torch
from diffusers import DiffusionPipeline
from PIL import Image

from asdff import AdPipeline

common = {"prompt": "masterpiece, best quality, 1girl", "num_inference_steps": 20}


def test_adpipeline():
    pipe = AdPipeline.from_pretrained(
        "stablediffusionapi/counterfeit-v30", torch_dtype=torch.float16
    )
    pipe.safety_checker = None
    pipe.to("cuda")

    result = pipe(common=common)
    images = result[0]
    init_images = result[1]

    assert images
    assert init_images
    assert isinstance(images[0], Image.Image)
    assert isinstance(init_images[0], Image.Image)
    assert images[0].mode == "RGB"
    assert init_images[0].mode == "RGB"


def test_diffusers_custom_pipeline():
    pipe = DiffusionPipeline.from_pretrained(
        "stablediffusionapi/counterfeit-v30",
        torch_dtype=torch.float16,
        custom_pipeline="Bingsu/adsd_pipeline",
    )
    pipe.safety_checker = None
    pipe.to("cuda")

    result = pipe(common=common)
    images = result[0]
    init_images = result[1]

    assert images
    assert init_images
    assert isinstance(images[0], Image.Image)
    assert isinstance(init_images[0], Image.Image)
    assert images[0].mode == "RGB"
    assert init_images[0].mode == "RGB"
