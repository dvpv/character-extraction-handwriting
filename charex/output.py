import numpy as np
import cv2

DEFAULT_PREVIEW_WIDTH = 1000


def preview(image: np.array, title: str = "Preview") -> None:
    width, height, _ = image.shape
    resized = cv2.resize(
        image,
        [DEFAULT_PREVIEW_WIDTH, int(width * DEFAULT_PREVIEW_WIDTH / height)],
    )
    cv2.imshow(title, resized)
    cv2.waitKey(0)
