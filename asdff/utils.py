from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from diffusers.utils import BaseOutput
from PIL import Image, ImageFilter, ImageOps


@dataclass
class ADOutput(BaseOutput):
    images: list[Image.Image]
    init_images: list[Image.Image]


def mask_dilate(image: Image.Image, value: int = 4) -> Image.Image:
    if value <= 0:
        return image

    arr = np.array(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    dilated = cv2.dilate(arr, kernel, iterations=1)
    return Image.fromarray(dilated)


def mask_gaussian_blur(image: Image.Image, value: int = 4) -> Image.Image:
    if value <= 0:
        return image

    blur = ImageFilter.GaussianBlur(value)
    return image.filter(blur)


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


def composite(
    init: Image.Image,
    mask: Image.Image,
    gen: Image.Image,
    bbox_padded: tuple[int, int, int, int],
) -> Image.Image:
    img_masked = Image.new("RGBa", init.size)
    img_masked.paste(
        init.convert("RGBA").convert("RGBa"),
        mask=ImageOps.invert(mask),
    )
    img_masked = img_masked.convert("RGBA")

    size = (
        bbox_padded[2] - bbox_padded[0],
        bbox_padded[3] - bbox_padded[1],
    )
    resized = gen.resize(size)

    output = Image.new("RGBA", init.size)
    output.paste(resized, bbox_padded)
    output.alpha_composite(img_masked)
    return output.convert("RGB")
