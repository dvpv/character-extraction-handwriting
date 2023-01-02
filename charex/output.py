from typing import List
import numpy as np
import cv2
import os

DEFAULT_PREVIEW_WIDTH = 1000


def preview(
    image: np.array,
    title: str = "Preview",
    target_width: int = DEFAULT_PREVIEW_WIDTH,
) -> None:
    height: int = None
    width: int = None
    if len(image.shape) == 3:
        width, height, _ = image.shape
    elif len(image.shape) == 2:
        width, height = image.shape
    else:
        raise Exception(f"Shape of image ({image.shape}) is not supported")
    resized = cv2.resize(
        image,
        [target_width, int(width * target_width / height)],
    )
    cv2.imshow(title, resized)
    cv2.waitKey(0)


def save_list(
    letters: List[np.array],
    output_dir: str,
    prefix: str = "image",
) -> None:
    os.makedirs(output_dir)
    for index, letter in enumerate(letters):
        cv2.imwrite(f"{output_dir}/{prefix}_{index}.jpg", letter)
