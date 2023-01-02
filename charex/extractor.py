import numpy as np
import cv2
from . import output
from .utils.colorcycle import ColorCycle
from typing import List

DEFAULT_LETTER_IMAGE_SIZE = [50, 50]
DEFAULT_WIDTH_DILATE_MULTIPLIER = 0.05
DEFAULT_HEIGHT_DILATE_MULTIPLIER = 0.01
DEFAULT_WIDTH_ERODE_MULTIPLIER = 0.01
DEFAULT_HEIGHT_DILATE_MULTIPLIER = 0.1
DEFAULT_BLUR_SIGMA = 0.7
DEFAULT_MASK_COLOR = (0xFF, 0xFF, 0xFF)  # #FFF
DEFAULT_LETTER_WIDTH_THRESH_MULTIPLIER = 0.66
DEFAULT_LETTER_FRAGMENTATION_ROUNDS = 1
DEFAULT_ELLIPSE_WIDTH_MULTIPLIER = 0.3
DEFAULT_ECLIPSE_ANGLE = 20
DEFAULT_AVERAGE_WIDTH_SPLIT_MULTIPLIER = 0.95


def extract(image_path: str) -> np.array:
    raise NotImplemented


def __cull_contours_thresh(contours: List[np.array], thresh: float) -> List[np.array]:
    if len(contours) == 0:
        return []
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
    contours = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[1])

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

        highlight = np.ones(original.shape, np.uint8) * 255
        for contour in contours:
            cv2.drawContours(highlight, [contour], -1, cc.next(), -1, cv2.LINE_AA)
        overlay = np.copy(original)
        overlay = cv2.addWeighted(overlay, 0.5, highlight, 0.5, 0)
        output.preview(overlay)

    return lines


def __extract_letter_contours(image: np.array) -> List[np.array]:
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = __cull_contours_thresh(contours, thresh=0.1)
    contours = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[0])
    return contours


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


def __preprocess_row_image_elliptic(image: np.array) -> np.array:
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
        ksize=(2, 2),
    )
    eroded = cv2.erode(image, kernel, iterations=1)
    contours = __extract_letter_contours(eroded)
    elliptic = np.copy(eroded)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        elliptic = cv2.ellipse(
            elliptic,
            center=(int(x + w / 2), int(y + h / 2)),
            axes=(int(DEFAULT_ELLIPSE_WIDTH_MULTIPLIER * w), height),
            angle=DEFAULT_ECLIPSE_ANGLE,
            startAngle=0,
            endAngle=360,
            color=255,
            thickness=-1,
        )

    return elliptic


def __extract_letters(image: np.array, contours: List[np.array]) -> List[np.array]:
    letters: List[np.array] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        letter = np.copy(image)
        # Build the mask
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.drawContours(mask, [contour], -1, DEFAULT_MASK_COLOR, -1, cv2.LINE_AA)
        # Crop line
        letter = cv2.bitwise_and(image, image, mask=mask)
        # Apply white background
        background = np.ones(image.shape, np.uint8) * 255
        cv2.bitwise_not(background, background, mask=mask)
        letter += background
        letter = letter[y : y + h, x : x + w]
        letters.append(letter)
    return letters


def __fragment_big_contours_high_erosion(
    image: np.array,
    contours: List[np.array],
) -> List[np.array]:
    original = np.copy(image)
    _, height, _ = original.shape
    avr_w, _ = __average_contours_dimensions(contours)
    good_contours = []
    bad_contours = []
    for contour in contours:
        if (
            cv2.boundingRect(contour)[2] * DEFAULT_LETTER_WIDTH_THRESH_MULTIPLIER
            < avr_w
        ):
            good_contours.append(contour)
        else:
            bad_contours.append(contour)
    big_blobs = np.copy(original)
    for contour in bad_contours:
        x, y, w, h = cv2.boundingRect(contour)
        splits = int(w / (avr_w * DEFAULT_AVERAGE_WIDTH_SPLIT_MULTIPLIER))
        split_size = w / (splits + 1)
        for index in range(1, splits + 1):
            big_blobs = cv2.ellipse(
                big_blobs,
                center=(int(x + split_size * index), int(y + h / 2)),
                axes=(2, height),
                angle=DEFAULT_ECLIPSE_ANGLE,
                startAngle=0,
                endAngle=360,
                color=(255, 255, 255),
                thickness=-1,
            )

    for contour in good_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(
            img=big_blobs,
            pt1=(x, y),
            pt2=(x + w, y + h),
            color=(255, 255, 255),
            thickness=-1,
        )
    processed = __preprocess_row_image_elliptic(big_blobs)
    smaller_contours = __extract_letter_contours(processed)
    contours = good_contours
    contours.extend(smaller_contours)
    contours = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[0])
    return contours


def extract_letters(
    image: np.array,
    preview_flag: bool = False,
    fragmentation_flag: bool = False,
) -> List[np.array]:
    original = np.copy(image)

    processed = __preprocess_row_image(image)
    contours = __extract_letter_contours(processed)

    if fragmentation_flag:
        contours = __fragment_big_contours_high_erosion(original, contours)
    # Extract letters
    letters = __extract_letters(original, contours)

    if preview_flag:
        cc = ColorCycle()

        highlight = np.ones(image.shape, np.uint8) * 255
        for contour in contours:
            cv2.drawContours(highlight, [contour], -1, cc.next(), -1, cv2.LINE_AA)
        overlay = np.copy(original)
        overlay = cv2.addWeighted(overlay, 0.5, highlight, 0.5, 0)
        output.preview(overlay)
    return letters
