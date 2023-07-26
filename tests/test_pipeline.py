import torch
from diffusers import ControlNetModel, DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import load_image
from PIL import Image

from asdff import AdCnPipeline, AdPipeline

common = {
    "prompt": "masterpiece, best quality, 1girl",
    "num_inference_steps": 10,
}
inpaint = {
    "prompt": "masterpiece, best quality, 1girl, red_eyes",
    "num_inference_steps": 10,
}
counterfeit = "stablediffusionapi/counterfeit-v30"


class Base:
    def test_pipeline(self):
        result = self.pipe(common=common, inpaint_only=inpaint)
        images = result[0]
        init_images = result[1]

        assert len(images) == 1
        assert len(init_images) == 1
        assert isinstance(images[0], Image.Image)
        assert isinstance(init_images[0], Image.Image)
        assert images[0].mode == "RGB"
        assert init_images[0].mode == "RGB"


class TestAdPipeline(Base):
    pipe = AdPipeline.from_pretrained(counterfeit, torch_dtype=torch.float16)
    pipe.safety_checker = None
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")


class TestDiffusersPipeline(Base):
    pipe = DiffusionPipeline.from_pretrained(
        counterfeit,
        torch_dtype=torch.float16,
        custom_pipeline="Bingsu/adsd_pipeline",
    )
    pipe.safety_checker = None
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")


class TestCnPipeline:
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
    )
    pipe = AdCnPipeline.from_pretrained(
        counterfeit,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    pipe.to("cuda")

    def test_cn_pipeline(self):
        image = load_image(
            "https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/converted/control_human_openpose.png"
        )
        common2 = common.copy()
        common2["image"] = image

        result = self.pipe(common=common2, inpaint_only=inpaint)
        images = result[0]
        init_images = result[1]

        assert len(images) == 1
        assert len(init_images) == 1
        assert isinstance(images[0], Image.Image)
        assert isinstance(init_images[0], Image.Image)
        assert images[0].mode == "RGB"
        assert init_images[0].mode == "RGB"
