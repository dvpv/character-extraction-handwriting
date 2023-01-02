import numpy as np
import cv2
from . import output
from .utils.colorcycle import ColorCycle
from typing import List

DEFAULT_LETTER_IMAGE_SIZE = [50, 50]
DEFAULT_WIDTH_DILATE_MULTIPLIER = 0.1
DEFAULT_HEIGHT_DILATE_MULTIPLIER = 0.01
DEFAULT_WIDTH_ERODE_MULTIPLIER = 0.01
DEFAULT_HEIGHT_DILATE_MULTIPLIER = 0.1
DEFAULT_BLUR_SIGMA = 0.7
DEFAULT_MASK_COLOR = (0xFF, 0xFF, 0xFF)  # #FFF
DEFAULT_LETTER_WIDTH_THRESH_MULTIPLIER = 0.66
DEFAULT_LETTER_FRAGMENTATION_ROUNDS = 1


def extract(image_path: str) -> np.array:
    raise NotImplemented


def __cull_contours_thresh(contours: List[np.array], thresh: float) -> List[np.array]:
    avg_area = sum([cv2.contourArea(contour) for contour in contours]) / len(contours)
    return [
        contour for contour in contours if cv2.contourArea(contour) >= thresh * avg_area
    ]


def __average_contours_dimensions(contours: List[np.array]) -> tuple:
    w_h = [[w, h] for _, _, w, h in [cv2.boundingRect(contour) for contour in contours]]
    avr_w = int(sum([val[0] for val in w_h]) / len(w_h))
    avr_h = int(sum([val[1] for val in w_h]) / len(w_h))
    return (avr_w, avr_h)


def __average_contours_dimension_ratio(contours: List[np.array]) -> float:
    ratios = [
        w / h for _, _, w, h in [cv2.boundingRect(contour) for contour in contours]
    ]
    return sum(ratio for ratio in ratios) / len(ratios)


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
    width, _, _ = original.shape
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        ksize=(int(width * DEFAULT_WIDTH_DILATE_MULTIPLIER), 1),
    )
    dilated = cv2.dilate(image, kernel, iterations=1)
    # Get contours for each line
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = __cull_contours_thresh(contours, thresh=0.1)

    # Extract lines
    lines: List[np.array] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
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
            x, y, w, h = cv2.boundingRect(contour)
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


def __preprocess_row_image(image: np.array) -> np.array:
    original = np.copy(image)
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
    _, height, _ = original.shape
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        ksize=(1, 1),
    )
    eroded = cv2.erode(image, kernel, iterations=1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        ksize=(1, int(height * DEFAULT_HEIGHT_DILATE_MULTIPLIER)),
    )
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    return dilated


def __extract_letter_contours(image: np.array) -> List[np.array]:
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return __cull_contours_thresh(contours, thresh=0.1)


def __extract_letters(image: np.array, contours: List[np.array]) -> List[np.array]:
    letters: List[np.array] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        line = np.copy(image)
        # Build the mask
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.drawContours(mask, [contour], -1, DEFAULT_MASK_COLOR, -1, cv2.LINE_AA)
        # Crop line
        line = cv2.bitwise_and(image, image, mask=mask)
        # Apply white background
        background = np.ones(image.shape, np.uint8) * 255
        cv2.bitwise_not(background, background, mask=mask)
        line += background
        line = line[y : y + h, x : x + w]
        letters.append(line)


def extract_letters(image: np.array, preview_flag: bool = False) -> List[np.array]:
    original = np.copy(image)

    processed = __preprocess_row_image(image)
    contours = __extract_letter_contours(processed)

    # for _ in range(0, DEFAULT_LETTER_FRAGMENTATION_ROUNDS):
    #     avr_w, _ = __average_contours_dimensions(contours)
    #     print([cv2.boundingRect(contour)[2] for contour in contours])
    #     print(f"average: {avr_w}")
    #     good_contours = [
    #         contour
    #         for contour in contours
    #         if cv2.boundingRect(contour)[2] * DEFAULT_LETTER_WIDTH_THRESH_MULTIPLIER
    #         < avr_w
    #     ]
    #     # Break if there are no big blobs
    #     if len(good_contours) == len(contours):
    #         contours.extend(good_contours)
    #         break

    #     big_blobs = np.copy(original)
    #     for contour in good_contours:
    #         x, y, w, h = cv2.boundingRect(contour)
    #         cv2.rectangle(
    #             img=big_blobs,
    #             pt1=(x, y),
    #             pt2=(x + w, y + h),
    #             color=(255, 255, 255),
    #             thickness=-1,
    #         )
    #     output.preview(big_blobs)
    #     processed = __preprocess_row_image(big_blobs)
    #     output.preview(processed)
    #     smaller_contours = __extract_letter_contours(processed)
    #     for target in good_contours:
    #         for index, contour in enumerate(contours):
    #             if np.array_equal(target, index):
    #                 contour.pop(index)
    #                 break
    #     contours.extend(smaller_contours)
    #     contours = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[2])

    # Extract letters
    letters = __extract_letters(original, contours)

    if preview_flag:
        cc = ColorCycle()
        rectangles = np.copy(original)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
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
    return letters
