from . import extractor, output, reader
from typing import Optional
import cv2
import numpy as np
import math


def process(input_file: str, output_file: Optional[str], preview_flag: bool) -> None:
    original = reader.read(input_file)
    resized = cv2.resize(original, [1000, 1000])
    image = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    output.preview(image)
    image = cv2.GaussianBlur(image, (0, 0), 0.7)
    output.preview(image)
    image = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        10,
    )
    output.preview(image)
    kernel = np.ones((5, 5))
    dilated = cv2.dilate(image, kernel, iterations=1)
    output.preview(dilated)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    ## Split big horizontal contours

    # Calculate average contour size
    avr_contour_size = sum(cv2.boundingRect(c)[3] for c in contours) / len(contours)
    print(f"Average h size = {avr_contour_size}")

    # Split biggest blobs

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if h * 30 / 100 > avr_contour_size:
            print(dilated[x, y])
            dilated[int(y + h / 2) : int(y + h / 2 + 5), x : x + w] = 0
    output.preview(dilated)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(
            img=resized,
            pt1=(x, y),
            pt2=(x + w, y + h),
            color=(0, 255, 0),
            thickness=2,
        )
    output.preview(resized)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(
            img=resized,
            pt1=(x, y),
            pt2=(x + w, y + h),
            color=(0, 255, 0),
            thickness=2,
        )
    output.preview(resized)

    edges = cv2.Canny(image, 50, 200, None, 3)
    output.preview(edges)
    # WIP
