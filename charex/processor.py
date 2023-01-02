from . import extractor, output, reader
from typing import Optional
import cv2

DEFAULT_HEIGHT_CROP_MULTIPLIER = 0.01
DEFAULT_WIDTH_CROP_MULTIPLIER = 0.01


def process(
    input_file: str,
    output_file: Optional[str],
    preview_flag: bool,
    fragmentation_flag: bool,
) -> None:
    image = reader.read(input_file)
    height, width, _ = image.shape
    crop_height = int(height * DEFAULT_HEIGHT_CROP_MULTIPLIER)
    crop_width = int(width * DEFAULT_WIDTH_CROP_MULTIPLIER)
    cropped = image[crop_height:-crop_height, crop_width:-crop_width]

    rows = extractor.extract_rows(cropped, preview_flag=preview_flag)

    # for row in rows:
    #     output.preview(row)

    letters_per_row = [
        extractor.extract_letters(
            row,
            preview_flag=preview_flag,
            fragmentation_flag=fragmentation_flag,
        )
        for row in rows
    ]
    for letters in letters_per_row:
        for letter in letters:
            output.preview(letter, target_width=300)

    # WIP
