import numpy as np
import cv2


def read(image_path: str) -> np.array:
    return cv2.imread(image_path, cv2.IMREAD_COLOR)
