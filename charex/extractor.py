import numpy as np
import cv2
from . import output
from .utils.colorcycle import ColorCycle
from typing import List

DEFAULT_LINE_CONTOUR_THRESH = 200
DEFAULT_WIDTH_DILATE_MULTIPLIER = 0.1
DEFAULT_BLUR_SIGMA = 0.7
DEFAULT_MASK_COLOR = (0xFF, 0xFF, 0xFF)  # #FFF


def extract(image_path: str) -> np.array:
    raise NotImplemented


def extract_rows(image: np.array, preview_flag: bool = False) -> List[np.array]:
    # Save an image copy
    original = np.copy(image)

    # Grayscale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Blur out noise
    image = cv2.GaussianBlur(image, (0, 0), DEFAULT_BLUR_SIGMA)
    # Grayscale to binary
    image = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=5,
    )

    # Dilate image horizontally
    _, width, _ = original.shape
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        ksize=(int(width * DEFAULT_WIDTH_DILATE_MULTIPLIER), 1),
    )
    dilated = cv2.dilate(image, kernel, iterations=1)
    # Get contours for each line
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract lines
    lines: List[np.array] = []
    for contour in contours:
        if cv2.contourArea(contour) > DEFAULT_LINE_CONTOUR_THRESH:
            (x, y, w, h) = cv2.boundingRect(contour)
            line = np.copy(original)
            # Build the mask
            mask = np.zeros(original.shape[:2], np.uint8)
            cv2.drawContours(mask, [contour], -1, DEFAULT_MASK_COLOR, -1, cv2.LINE_AA)
            # Crop line
            line = cv2.bitwise_and(original, original, mask=mask)
            # Apply white background
            background = np.ones(original.shape, np.uint8) * 255
            cv2.bitwise_not(background, background, mask=mask)
            line += background
            line = line[y : y + h, x : x + w]
            lines.append(line)

    # Show a nicely formatted image representing the lines with the lines
    if preview_flag:
        cc = ColorCycle()
        rectangles = np.copy(original)

        for contour in contours:
            if cv2.contourArea(contour) > DEFAULT_LINE_CONTOUR_THRESH:
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(
                    img=rectangles,
                    pt1=(x, y),
                    pt2=(x + w, y + h),
                    color=cc.next(),
                    thickness=-1,
                )
        overlay = np.copy(original)
        overlay = cv2.addWeighted(overlay, 0.6, rectangles, 0.5, 0)
        output.preview(overlay)

    return lines


def extract_letters(row: np.array, preview_flag: bool = False) -> List[np.array]:
    letters: List[np.array] = []

    return letters
