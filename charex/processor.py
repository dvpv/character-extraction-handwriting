from . import extractor, output, reader
from typing import Optional
import cv2


def process(input_file: str, output_file: Optional[str], preview_flag: bool) -> None:
    original = reader.read(input_file)
    resized = cv2.resize(original, [1000, 1000])
    cropped = resized[10:-10, 10:-10]

    rows = extractor.extract_rows(cropped, preview_flag=preview_flag)

    for row in rows:
        output.preview(row)

    letters_per_row = [extractor.extract_letters(row) for row in rows]
    for letters in letters_per_row:
        for letter in letters:
            output.preview(letter)

    # WIP
