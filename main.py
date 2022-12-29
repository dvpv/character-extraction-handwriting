import argparse
import charex

parser = argparse.ArgumentParser(
    description="Character Extraction from Handwriting Tool",
)

parser.add_argument(
    "-p",
    "--preview",
    action="store_true",
    help="show preview even if [output] is set",
)

parser.add_argument(
    "input",
    type=str,
    help="path to the input image file",
)

parser.add_argument(
    "output",
    nargs="?",
    type=str,
    help="path to the output image file",
)


def main():
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    preview_flag = args.preview
    charex.processor.process(
        input_file=input_file,
        output_file=output_file,
        preview_flag=preview_flag,
    )


if __name__ == "__main__":
    main()
