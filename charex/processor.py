from . import extractor, output, reader
from typing import Optional
import cv2

DEFAULT_HEIGHT_CROP_MULTIPLIER = 0.01
DEFAULT_WIDTH_CROP_MULTIPLIER = 0.01


def process(
    input_file: str,
    output_dir: Optional[str],
    preview_flag: bool,
    fragmentation_flag: bool,
) -> None:
    if output_dir == None:
        preview_flag = True
    image = reader.read(input_file)
    height, width, _ = image.shape
    crop_height = int(height * DEFAULT_HEIGHT_CROP_MULTIPLIER)
    crop_width = int(width * DEFAULT_WIDTH_CROP_MULTIPLIER)
    cropped = image[crop_height:-crop_height, crop_width:-crop_width]

    rows = extractor.extract_rows(cropped, preview_flag=preview_flag)

    letters_per_row = [
        extractor.extract_letters(
            row,
            preview_flag=preview_flag,
            fragmentation_flag=fragmentation_flag,
        )
        for row in rows
    ]

    if output_dir:
        image_name = input_file.split(".")[0]
        output.save_list(rows, f"{output_dir}/rows", prefix=f"{image_name}_row")
        for index, letters in enumerate(letters_per_row):
            output.save_list(
                letters,
                f"{output_dir}/letters/row_{index}",
                prefix=f"{image_name}_letter",
            )
