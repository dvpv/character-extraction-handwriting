import numpy as np
import cv2


def preview(image: np.array) -> None:
    cv2.imshow("Preview", image)
    cv2.waitKey(0)
