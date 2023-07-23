import torch
from diffusers import ControlNetModel, DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import load_image
from PIL import Image

from asdff import AdCnPipeline, AdPipeline


def test_adpipeline():
    pipe = AdPipeline.from_pretrained(
        "stablediffusionapi/counterfeit-v30", torch_dtype=torch.float16
    )
    pipe.safety_checker = None
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    common = {"prompt": "masterpiece, best quality, 1girl", "num_inference_steps": 20}
    result = pipe(common=common)
    images = result[0]
    init_images = result[1]

    assert len(images) == 1
    assert len(init_images) == 1
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
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    common = {"prompt": "masterpiece, best quality, 1girl", "num_inference_steps": 20}
    result = pipe(common=common)
    images = result[0]
    init_images = result[1]

    assert len(images) == 1
    assert len(init_images) == 1
    assert isinstance(images[0], Image.Image)
    assert isinstance(init_images[0], Image.Image)
    assert images[0].mode == "RGB"
    assert init_images[0].mode == "RGB"


def test_cn_pipeline():
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
    )
    pipe = AdCnPipeline.from_pretrained(
        "stablediffusionapi/counterfeit-v30",
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    pipe.to("cuda")

    image = load_image(
        "https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/converted/control_human_openpose.png"
    )
    common = {
        "prompt": "masterpiece, best quality, 1girl",
        "num_inference_steps": 20,
        "image": image,
    }

    result = pipe(common=common)
    images = result[0]
    init_images = result[1]

    assert len(images) == 1
    assert len(init_images) == 1
    assert isinstance(images[0], Image.Image)
    assert isinstance(init_images[0], Image.Image)
    assert images[0].mode == "RGB"
    assert init_images[0].mode == "RGB"
