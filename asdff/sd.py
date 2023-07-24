from __future__ import annotations

from functools import cached_property
from typing import Any, Callable, Iterable, List, Mapping, Optional

from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline
from diffusers.utils import logging
from PIL import Image

from asdff.utils import (
    ADOutput,
    bbox_padding,
    composite,
    mask_dilate,
    mask_gaussian_blur,
)
from asdff.yolo import yolo_detector

logger = logging.get_logger("diffusers")


DetectorType = Callable[[Image.Image], Optional[List[Image.Image]]]


def ordinal(n: int) -> str:
    d = {1: "st", 2: "nd", 3: "rd"}
    return str(n) + ("th" if 11 <= n % 100 <= 13 else d.get(n % 10, "th"))


class AdPipeline(StableDiffusionPipeline):
    @cached_property
    def inpaint_pipeline(self):
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

    def __call__(  # noqa: C901
        self,
        common: Mapping[str, Any] | None = None,
        images:Union[None,list[Image.Image]]=None,
        txt2img_only: Mapping[str, Any] | None = None,
        inpaint_only: Mapping[str, Any] | None = None,
        detectors: DetectorType | Iterable[DetectorType] | None = None,
        mask_dilation: int = 4,
        mask_blur: int = 4,
        mask_padding: int = 32,
    ):
        if common is None:
            common = {}
        if txt2img_only is None:
            txt2img_only = {}
        if inpaint_only is None:
            inpaint_only = {}
        if "strength" not in inpaint_only:
            inpaint_only = {**inpaint_only, "strength": 0.4}

        if detectors is None:
            detectors = [self.default_detector]
        elif callable(detectors):
            detectors = [detectors]

        if images is None:
            txt2img_output = super().__call__(**common, **txt2img_only, output_type="pil")
            txt2img_images: list[Image.Image] = txt2img_output[0]
        else:
            txt2img_images = images

        init_images = []
        final_images = []

        for i, init_image in enumerate(txt2img_images):
            init_images.append(init_image.copy())
            final_image = None

            for j, detector in enumerate(detectors):
                masks = detector(init_image)
                if masks is None:
                    logger.info(
                        f"No object detected on {ordinal(i + 1)} image with {ordinal(j + 1)} detector."
                    )
                    continue

                for k, mask in enumerate(masks):
                    mask = mask.convert("L")
                    mask = mask_dilate(mask, mask_dilation)
                    bbox = mask.getbbox()
                    if bbox is None:
                        logger.info(f"No object in {ordinal(k + 1)} mask.")
                        continue
                    mask = mask_gaussian_blur(mask, mask_blur)
                    bbox_padded = bbox_padding(bbox, init_image.size, mask_padding)

                    crop_image = init_image.crop(bbox_padded)
                    crop_mask = mask.crop(bbox_padded)

                    inpaint_output = self.inpaint_pipeline(
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
                    init_image = final_image

            if final_image is not None:
                final_images.append(final_image)

        return ADOutput(images=final_images, init_images=init_images)

    @property
    def default_detector(self) -> Callable[..., list[Image.Image] | None]:
        return yolo_detector
