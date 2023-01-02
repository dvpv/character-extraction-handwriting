import numpy as np
import cv2


def preview(image: np.array, title: str = "Preview") -> None:
    cv2.imshow(title, image)
    cv2.waitKey(0)
