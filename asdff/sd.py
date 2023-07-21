from __future__ import annotations

from functools import cached_property, partial
from typing import Any, Callable

import cv2
import numpy as np
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import logging
from PIL import Image, ImageChops

from asdff.yolo import yolo_detector

logger = logging.get_logger("diffusers")


def ordinal(n: int) -> str:
    d = {1: "st", 2: "nd", 3: "rd"}
    return str(n) + ("th" if 11 <= n % 100 <= 13 else d.get(n % 10, "th"))


class AdPipeline(StableDiffusionPipeline):
    @cached_property
    def inpaine_pipeline(self):
        return StableDiffusionInpaintPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            safety_checker=self.safety_checker,
            feature_extractor=self.feature_extractor,
            requires_safety_checker=self.requires_safety_checker,
        )

    def __call__(
        self,
        common: dict[str, Any] | None = None,
        txt2img_only: dict[str, Any] | None = None,
        inpaint_only: dict[str, Any] | None = None,
        detector: Callable[[Image.Image], list[Image.Image] | None] | None = None,
        detector_kwargs: dict[str, Any] | None = None,
        mask_dilation: int = 4,
        mask_padding: int = 32,
    ):
        if common is None:
            common = {}
        if txt2img_only is None:
            txt2img_only = {}
        if inpaint_only is None:
            inpaint_only = {}
        if detector_kwargs is None:
            detector_kwargs = {}
        if detector is None:
            detector = partial(self.default_detector, **detector_kwargs)
        else:
            detector = partial(detector, **detector_kwargs)

        txt2img_output = StableDiffusionPipeline.__call__(
            self, **common, **txt2img_only, output_type="pil"
        )
        txt2img_images: list[Image.Image] = txt2img_output[0]

        result_images = []

        for i, txt2img_image in enumerate(txt2img_images):
            masks = detector(txt2img_image)
            if masks is None:
                logger.info(f"No object detected on {ordinal(i + 1)} image.")
                continue

            for _j, mask in enumerate(masks):
                mask = mask.convert("L")
                mask = self.mask_dilate(mask, mask_dilation)
                bbox = mask.getbbox()
                if bbox is None:
                    # Never happens
                    continue
                bbox_padded = self.bbox_padding(bbox, txt2img_image.size, mask_padding)

                img_masked = Image.new("RGBa", txt2img_image.size)
                img_masked.paste(
                    txt2img_image.convert("RGBA").convert("RGBa"),
                    mask=ImageChops.invert(mask),
                )
                img_masked = img_masked.convert("RGBA")

                crop_image = txt2img_image.crop(bbox_padded)
                crop_mask = mask.crop(bbox_padded)

                inpaint_output = self.inpaine_pipeline(
                    **common,
                    **inpaint_only,
                    image=crop_image,
                    mask_image=crop_mask,
                    num_images_per_prompt=1,
                    output_type="pil",
                )
                inpaint_image: Image.Image = inpaint_output[0][0]
                resize = (
                    bbox_padded[2] - bbox_padded[0],
                    bbox_padded[3] - bbox_padded[1],
                )
                resized = inpaint_image.resize(resize)

                result_img = Image.new("RGBA", txt2img_image.size)
                result_img.paste(resized, bbox_padded)
                result_img.alpha_composite(img_masked)
                result_images.append(result_img.convert("RGB"))

        return StableDiffusionPipelineOutput(
            images=result_images, nsfw_content_detected=None
        )

    @property
    def default_detector(self) -> Callable[..., list[Image.Image] | None]:
        return yolo_detector

    @staticmethod
    def mask_dilate(image: Image.Image, value: int = 4) -> Image.Image:
        if value <= 0:
            return image

        arr = np.array(image)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
        dilated = cv2.dilate(arr, kernel, iterations=1)
        return Image.fromarray(dilated)

    @staticmethod
    def bbox_padding(
        bbox: tuple[int, int, int, int], image_size: tuple[int, int], value: int = 32
    ) -> tuple[int, int, int, int]:
        if value <= 0:
            return bbox

        arr = np.array(bbox).reshape(2, 2)
        arr[0] -= value
        arr[1] += value
        arr = np.clip(arr, (0, 0), image_size)
        return tuple(arr.flatten())
