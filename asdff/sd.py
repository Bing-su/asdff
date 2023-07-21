from __future__ import annotations

from functools import cached_property, partial
from typing import Any, Callable

from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import logging
from PIL import Image

from asdff.utils import bbox_padding, composite, mask_dilate
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
            requires_safety_checker=self.config.requires_safety_checker,
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
        """
        Call method for the StableDiffusionPipeline class.

        Parameters
        ----------
            common (dict[str, Any] | None, optional): Common parameters for the pipeline. Defaults to None.
            txt2img_only (dict[str, Any] | None, optional): Parameters for the txt2img step. Defaults to None.
            inpaint_only (dict[str, Any] | None, optional): Parameters for the inpaint step. Defaults to None.
            detector (Callable[[Image.Image], list[Image.Image] | None] | None, optional): Object detection function. Defaults to None.
            detector_kwargs (dict[str, Any] | None, optional): Parameters for the object detection function. Defaults to None.
            mask_dilation (int, optional): Dilation factor for the object mask. Defaults to 4.
            mask_padding (int, optional): Padding size for the object mask. Defaults to 32.
        Returns
        -------
            StableDiffusionPipelineOutput: Output of the StableDiffusionPipeline class.
        """
        if common is None:
            common = {}
        if txt2img_only is None:
            txt2img_only = {}
        if inpaint_only is None:
            inpaint_only = {}
        inpaint_only.setdefault("strength", 0.4)
        if detector_kwargs is None:
            detector_kwargs = {}
        if detector is None:
            detector = partial(self.default_detector, **detector_kwargs)
        else:
            detector = partial(detector, **detector_kwargs)

        txt2img_output = super().__call__(**common, **txt2img_only, output_type="pil")
        txt2img_images: list[Image.Image] = txt2img_output[0]

        result_images = []

        for i, init_image in enumerate(txt2img_images):
            masks = detector(init_image)
            if masks is None:
                logger.info(f"No object detected on {ordinal(i + 1)} image.")
                continue

            for j, mask in enumerate(masks):
                mask = mask.convert("L")
                mask = mask_dilate(mask, mask_dilation)
                bbox = mask.getbbox()
                if bbox is None:
                    logger.info(f"No object in {ordinal(j + 1)} mask.")
                    continue
                bbox_padded = bbox_padding(bbox, init_image.size, mask_padding)

                crop_image = init_image.crop(bbox_padded)
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
                final_image = composite(
                    init=init_image,
                    mask=mask,
                    gen=inpaint_image,
                    bbox_padded=bbox_padded,
                )
                result_images.append(final_image)

        return StableDiffusionPipelineOutput(
            images=result_images, nsfw_content_detected=None
        )

    @property
    def default_detector(self) -> Callable[..., list[Image.Image] | None]:
        return yolo_detector
