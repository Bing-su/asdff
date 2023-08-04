from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, List, Mapping, Optional

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


class AdPipelineBase(ABC):
    @property
    @abstractmethod
    def inpaint_pipeline(self) -> Callable:
        raise NotImplementedError

    @property
    @abstractmethod
    def txt2img_class(self) -> type:
        raise NotImplementedError

    def __call__(  # noqa: C901
        self,
        common: Mapping[str, Any] | None = None,
        txt2img_only: Mapping[str, Any] | None = None,
        inpaint_only: Mapping[str, Any] | None = None,
        images: Image.Image | Iterable[Image.Image] | None = None,
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
        elif not isinstance(detectors, Iterable):
            detectors = [detectors]

        if images is None:
            txt2img_output = self.process_txt2img(common, txt2img_only)
            txt2img_images = txt2img_output[0]
        else:
            if txt2img_only:
                msg = "Both `images` and `txt2img_only` are specified. if `images` is specified, `txt2img_only` is ignored."
                logger.warning(msg)

            txt2img_images = [images] if not isinstance(images, Iterable) else images

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

                    inpaint_output = self.process_inpainting(
                        common,
                        inpaint_only,
                        init_image,
                        mask,
                        bbox_padded,
                    )
                    inpaint_image = inpaint_output[0][0]

                    final_image = composite(
                        init_image,
                        mask,
                        inpaint_image,
                        bbox_padded,
                    )
                    init_image = final_image

            if final_image is not None:
                final_images.append(final_image)

        return ADOutput(images=final_images, init_images=init_images)

    @property
    def default_detector(self) -> Callable[..., list[Image.Image] | None]:
        return yolo_detector

    def _get_txt2img_args(
        self, common: Mapping[str, Any], txt2img_only: Mapping[str, Any]
    ):
        return {**common, **txt2img_only, "output_type": "pil"}

    def _get_inpaint_args(
        self, common: Mapping[str, Any], inpaint_only: Mapping[str, Any]
    ):
        common = dict(common)
        sig = inspect.signature(self.inpaint_pipeline)
        if (
            "control_image" in sig.parameters
            and "control_image" not in common
            and "image" in common
        ):
            common["control_image"] = common.pop("image")
        return {
            **common,
            **inpaint_only,
            "num_images_per_prompt": 1,
            "output_type": "pil",
        }

    def process_txt2img(
        self, common: Mapping[str, Any], txt2img_only: Mapping[str, Any]
    ):
        txt2img_args = self._get_txt2img_args(common, txt2img_only)
        return self.txt2img_class.__call__(self, **txt2img_args)

    def process_inpainting(
        self,
        common: Mapping[str, Any],
        inpaint_only: Mapping[str, Any],
        init_image: Image.Image,
        mask: Image.Image,
        bbox_padded: tuple[int, int, int, int],
    ):
        crop_image = init_image.crop(bbox_padded)
        crop_mask = mask.crop(bbox_padded)
        inpaint_args = self._get_inpaint_args(common, inpaint_only)
        inpaint_args["image"] = crop_image
        inpaint_args["mask_image"] = crop_mask
        return self.inpaint_pipeline(**inpaint_args)
