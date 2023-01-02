import numpy as np
import cv2

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
